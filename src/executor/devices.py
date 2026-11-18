"""执行器状态机：滴灌阀门 + 喷杆喷头（软件模拟，预留真实硬件转发）。

设备模型：
  - drip_valve（滴灌阀门）  : state=open/closed，流量 flow_rate(L/h)
  - spray_boom（喷杆喷头）  : state=on/off，喷头压力 nozzle_pressure(bar)

区域隔离（一个摄像头 = 一个区域）：
  每个区域（zone）拥有一套独立的滴灌阀门与喷杆喷头，互不影响。
  zone 默认 "global"（兼容旧调用：不传 zone 时仍操作全局默认区）。
  有摄像头时 zone 取 cam_id（如 "cam1"），区域名取摄像头名称。

每条动作产生一个事件 dict：
  {kind:"actuator_event", zone, device, operation, reason, duration_min, state, ts}

状态快照（供网页渲染）：
  {drip_valve:{state, flow_rate, since}, spray_boom:{state, nozzle_pressure, since}}
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Callable

# 设备默认参数
DRIP_FLOW_RATE_LH = 120.0      # 滴灌流量 L/h
SPRAY_PRESSURE_BAR = 3.5       # 喷杆喷头工作压力 bar

# 默认区域（无摄像头/旧调用方不指定区域时使用）
DEFAULT_ZONE = "global"

# ---- 硬件安全硬限制（由独立看门狗强制执行，不依赖决策循环存活）----
# 单次最长作业时长（分钟）：上层传入更长的时长也会被截断
MAX_OPEN_MIN = {"drip_valve": 120.0, "spray_boom": 30.0}
# 每日累计最长作业时长（分钟）：按「区域 + 设备 + 日期」累计
MAX_DAILY_MIN = {"drip_valve": 240.0, "spray_boom": 60.0}
WATCHDOG_INTERVAL_S = 1.0     # 看门狗巡检间隔（秒）


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _now_epoch() -> float:
    return time.time()


class DripValve:
    """滴灌阀门。"""

    def __init__(self):
        self.state = "closed"       # open / closed
        self.since = _ts()
        self.flow_rate = 0.0
        self.duration_min: float | None = None
        self._started_monotonic: float | None = None

    def open(self, duration_min: float | None = None, reason: str = "") -> dict:
        if self.state == "open":
            return self.event("open", reason, duration_min)
        self.state = "open"
        self.since = _ts()
        self.flow_rate = DRIP_FLOW_RATE_LH
        self.duration_min = duration_min
        self._started_monotonic = time.monotonic()
        return self.event("open", reason, duration_min)

    def close(self, reason: str = "") -> dict:
        if self.state == "closed":
            return self.event("close", reason, None)
        self.state = "closed"
        self.since = _ts()
        self.flow_rate = 0.0
        self.duration_min = None
        self._started_monotonic = None
        return self.event("close", reason, None)

    def event(self, operation: str, reason: str, duration_min: float | None) -> dict:
        return {
            "kind": "actuator_event",
            "zone": DEFAULT_ZONE,
            "device": "drip_valve",
            "operation": operation,
            "reason": reason,
            "duration_min": duration_min,
            "state": self.state,
            "flow_rate": self.flow_rate,
            "since": self.since,
            "ts": _ts(),
        }

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "flow_rate": self.flow_rate,
            "since": self.since,
            "duration_min": self.duration_min,
        }


class SprayBoom:
    """喷杆喷头（自动打药/喷灌降温共用，通过 duration_min 区分作业）。"""

    def __init__(self):
        self.state = "off"          # on / off
        self.since = _ts()
        self.nozzle_pressure = 0.0
        self.duration_min: float | None = None
        self._started_monotonic: float | None = None

    def on(self, duration_min: float | None = None, reason: str = "") -> dict:
        if self.state == "on":
            return self.event("on", reason, duration_min)
        self.state = "on"
        self.since = _ts()
        self.nozzle_pressure = SPRAY_PRESSURE_BAR
        self.duration_min = duration_min
        self._started_monotonic = time.monotonic()
        return self.event("on", reason, duration_min)

    def off(self, reason: str = "") -> dict:
        if self.state == "off":
            return self.event("off", reason, None)
        self.state = "off"
        self.since = _ts()
        self.nozzle_pressure = 0.0
        self.duration_min = None
        self._started_monotonic = None
        return self.event("off", reason, None)

    def event(self, operation: str, reason: str, duration_min: float | None) -> dict:
        return {
            "kind": "actuator_event",
            "zone": DEFAULT_ZONE,
            "device": "spray_boom",
            "operation": operation,
            "reason": reason,
            "duration_min": duration_min,
            "state": self.state,
            "nozzle_pressure": self.nozzle_pressure,
            "since": self.since,
            "ts": _ts(),
        }

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "nozzle_pressure": self.nozzle_pressure,
            "since": self.since,
            "duration_min": self.duration_min,
        }


def _new_devices() -> dict:
    return {"drip_valve": DripValve(), "spray_boom": SprayBoom()}


class ActuatorBus:
    """执行器总线（按区域隔离）：统一 apply(actions, zone) 并广播状态变化事件。

    区域划分：一个摄像头 = 一个区域。每个区域独立维护自己的滴灌阀门与喷杆喷头，
    A 区检出病害只会开启 A 区喷杆，B 区状态完全不受影响。

    兼容性：不传 zone 的旧调用一律落到默认区 "global"，`bus.devices` 也指向该区，
    因此旧代码（smoke_test / 页面）行为不变。
    """

    def __init__(self):
        # zone -> {"drip_valve": DripValve, "spray_boom": SprayBoom}
        self._zones: dict[str, dict] = {}
        self._listeners: list[Callable[[dict], None]] = []
        # 可重入锁：snapshot_all / auto_timeout_check 内部会再次取锁
        self._lock = threading.RLock()
        # 兼容旧属性：默认区设备字典（同一个对象，不会被替换）
        self.devices = self._zone_devices(DEFAULT_ZONE)
        # 每日累计用量：(zone, device, 日期) -> 分钟
        self._usage: dict[tuple[str, str, str], float] = {}
        # 独立看门狗：强制时长与每日累计上限，不依赖决策循环是否还活着
        self._watchdog: threading.Thread | None = None
        self._watchdog_stop = threading.Event()

    # ---------- 区域 ----------
    def _zone_devices(self, zone: str | None) -> dict:
        """取（必要时创建）某区域的设备组。调用方需自行保证线程安全。"""
        key = str(zone or DEFAULT_ZONE)
        devs = self._zones.get(key)
        if devs is None:
            devs = _new_devices()
            self._zones[key] = devs
        return devs

    def zones(self) -> list[str]:
        """已有设备的区域列表（默认区在前）。"""
        with self._lock:
            keys = [k for k in self._zones if k != DEFAULT_ZONE]
            return ([DEFAULT_ZONE] if DEFAULT_ZONE in self._zones else []) + keys

    # ---------- 每日用量与安全看门狗 ----------
    @staticmethod
    def _today() -> str:
        return datetime.now().date().isoformat()

    def _usage_key(self, zone: str, device: str) -> tuple[str, str, str]:
        return (str(zone), device, self._today())

    def daily_used(self, zone: str | None, device: str) -> float:
        """某区域某设备今日累计作业分钟数。"""
        with self._lock:
            return round(self._usage.get(
                self._usage_key(str(zone or DEFAULT_ZONE), device), 0.0), 3)

    def daily_remaining(self, zone: str | None, device: str) -> float:
        """今日剩余可用分钟数（无上限配置时返回 0，调用方需按有上限处理）。"""
        cap = float(MAX_DAILY_MIN.get(device, 0.0))
        return max(0.0, cap - self.daily_used(zone, device))

    def _start_watchdog(self) -> None:
        """首次执行时启动看门狗（守护线程，进程退出即结束）。"""
        with self._lock:
            if self._watchdog is not None and self._watchdog.is_alive():
                return
            self._watchdog = threading.Thread(
                target=self._watchdog_loop, name="actuator-watchdog", daemon=True)
            self._watchdog.start()

    def shutdown(self) -> None:
        """停止看门狗并关闭所有设备（测试/进程退出用）。"""
        self._watchdog_stop.set()
        with self._lock:
            self.reset()

    def _watchdog_loop(self) -> None:
        """独立巡检：累计用量 → 超时关闭 → 每日上限关闭。

        即使 Agent 决策循环卡死/崩溃，这里仍会按时关闭阀门与喷杆。
        """
        last = time.monotonic()
        while not self._watchdog_stop.wait(WATCHDOG_INTERVAL_S):
            now = time.monotonic()
            dt_min = (now - last) / 60.0
            last = now
            try:
                self._accrue_usage(dt_min)
                # 单次时长上限已按设备截断，这里用各设备最大值作为总兜底
                self.auto_timeout_check(max_minutes=max(MAX_OPEN_MIN.values()))
                self._enforce_daily_caps()
            except Exception:  # noqa: BLE001 - 看门狗自身异常不能杀死线程
                continue

    def _accrue_usage(self, dt_min: float) -> None:
        """把最近一个巡检周期内处于作业状态的设备计入当日累计。"""
        with self._lock:
            for zone, devices in self._zones.items():
                for name, dev in devices.items():
                    active = ((name == "drip_valve" and dev.state == "open")
                              or (name == "spray_boom" and dev.state == "on"))
                    if not active:
                        continue
                    key = self._usage_key(zone, name)
                    self._usage[key] = self._usage.get(key, 0.0) + dt_min

    def _enforce_daily_caps(self) -> None:
        """达到每日累计上限的设备强制关闭。"""
        events: list[dict] = []
        with self._lock:
            for zone, devices in self._zones.items():
                for name, dev in devices.items():
                    active = ((name == "drip_valve" and dev.state == "open")
                              or (name == "spray_boom" and dev.state == "on"))
                    cap = float(MAX_DAILY_MIN.get(name, 0.0))
                    if not active or cap <= 0:
                        continue
                    used = self._usage.get(self._usage_key(zone, name), 0.0)
                    if used < cap:
                        continue
                    reason = f"达到每日累计上限 {cap:.0f} 分钟，强制停止"
                    ev = dev.close(reason) if name == "drip_valve" else dev.off(reason)
                    ev["zone"] = zone
                    ev["daily_used_min"] = round(used, 2)
                    ev["daily_cap_min"] = cap
                    events.append(ev)
        for event in events:
            self._broadcast(event)

    def _clamp_open(self, zone: str, action: dict) -> tuple[float | None, dict | None]:
        """截断单次时长并校验每日余量；返回 (允许的时长, 拦截事件)。"""
        device = action.get("device")
        cap_open = float(MAX_OPEN_MIN.get(device, 30.0))
        req = action.get("duration_min")
        try:
            duration = float(req) if req is not None else cap_open
        except (TypeError, ValueError):
            duration = cap_open
        duration = max(0.0, min(duration, cap_open))
        remaining = self.daily_remaining(zone, device)
        used = self.daily_used(zone, device)
        cap_daily = float(MAX_DAILY_MIN.get(device, 0.0))
        if remaining <= 0:
            return None, {
                "kind": "actuator_blocked",
                "zone": zone,
                "device": device,
                "operation": action.get("op"),
                "reason": f"今日累计已达 {cap_daily:.0f} 分钟上限，拒绝开启",
                "requested_min": req,
                "max_duration_min": cap_open,
                "daily_used_min": used,
                "daily_cap_min": cap_daily,
                "ts": _ts(),
            }
        return min(duration, remaining), None

    # ---------- 事件订阅 ----------
    def subscribe(self, listener: Callable[[dict], None]) -> None:
        with self._lock:
            self._listeners.append(listener)

    def _broadcast(self, event: dict) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(event)
            except Exception:  # noqa: BLE001 - 单个监听器异常不影响执行器
                pass

    # ---------- 执行 ----------
    def apply(self, actions: list[dict], zone: str | None = None) -> list[dict]:
        """按 Agent 指令操作指定区域的执行器。

        action: {device, op, duration_min?, reason?}
        zone  : 区域标识（一个摄像头 = 一个区域）；缺省为默认区 "global"。
        """
        key = str(zone or DEFAULT_ZONE)
        self._start_watchdog()
        events: list[dict] = []
        with self._lock:
            devices = self._zone_devices(key)
            for a in actions or []:
                dev = devices.get(a.get("device"))
                if dev is None:
                    continue
                op = a.get("op")
                reason = a.get("reason", "")
                dur = a.get("duration_min")
                if op in ("open", "on"):
                    # 单次时长截断 + 每日累计上限校验，超限直接拒绝
                    dur, blocked = self._clamp_open(key, a)
                    if blocked is not None:
                        events.append(blocked)
                        continue
                if op == "open":
                    ev = dev.open(dur, reason)
                elif op == "close":
                    ev = dev.close(reason)
                elif op == "on":
                    ev = dev.on(dur, reason)
                elif op == "off":
                    ev = dev.off(reason)
                else:
                    continue
                # 事件里统一带上区域标识，前端据此只让该区域动画
                ev["zone"] = key
                if op in ("open", "on"):
                    # 记录安全限额与今日累计，便于审计「为什么只喷了这么久」
                    ev["max_duration_min"] = MAX_OPEN_MIN.get(a.get("device"))
                    ev["daily_cap_min"] = MAX_DAILY_MIN.get(a.get("device"))
                    ev["daily_used_min"] = self.daily_used(key, a.get("device"))
                events.append(ev)
        # 监听器可能执行任意用户代码，不能在持有执行器锁时回调；否则监听器
        # 内部若查询执行器状态，就会把其他线程一起卡住，形成锁顺序死锁。
        for event in events:
            self._broadcast(event)
        return events

    def snapshot(self, zone: str | None = None) -> dict:
        """某区域的状态快照（默认区）。"""
        key = str(zone or DEFAULT_ZONE)
        with self._lock:
            devices = self._zone_devices(key)
            return {name: dev.snapshot() for name, dev in devices.items()}

    def snapshot_all(self) -> dict[str, dict]:
        """所有区域的状态快照：{zone: {drip_valve:{...}, spray_boom:{...}}}。"""
        with self._lock:
            return {z: self.snapshot(z) for z in self.zones()}

    def reset(self, zone: str | None = None) -> None:
        """复位全部设备（停止作业）。zone=None 时复位所有区域。"""
        with self._lock:
            targets = [str(zone)] if zone else list(self._zones.keys())
            for key in targets:
                devices = self._zone_devices(key)
                devices["drip_valve"].close("系统复位")
                devices["spray_boom"].off("系统复位")

    def auto_timeout_check(self, max_minutes: float = 30.0,
                           zone: str | None = None) -> list[dict]:
        """安全检查：达到计划时长或全局安全上限时自动停止。

        max_minutes 是兜底上限，不应当延长设备请求的较短计划时长：
        duration_min=10 时应运行约 10 分钟，而不是被 max_minutes=30 拖到 30 分钟。

        zone=None 时检查所有区域（每个区域独立计时、独立停止）。
        """
        events: list[dict] = []
        with self._lock:
            targets = [str(zone)] if zone else list(self._zones.keys())
            for key in targets:
                devices = self._zone_devices(key)
                for name, dev in devices.items():
                    state = dev.state
                    active = (state == "open" and name == "drip_valve") or \
                             (state == "on" and name == "spray_boom")
                    if not active:
                        continue
                    started = getattr(dev, "_started_monotonic", None)
                    if started is None:
                        started = datetime.fromisoformat(dev.since).timestamp()
                        elapsed_min = (_now_epoch() - started) / 60.0
                    else:
                        elapsed_min = (time.monotonic() - started) / 60.0
                    requested = float(dev.duration_min) if dev.duration_min else float(max_minutes)
                    limit = min(max(requested, 0.01), float(max_minutes))
                    if elapsed_min >= limit:
                        if name == "drip_valve":
                            ev = dev.close(f"自动停止（已运行 {elapsed_min:.0f} 分钟）")
                        else:
                            ev = dev.off(f"自动停止（已运行 {elapsed_min:.0f} 分钟）")
                        ev["zone"] = key
                        events.append(ev)
        for event in events:
            self._broadcast(event)
        return events


DEFAULT_SNAPSHOT = {
    "drip_valve": {"state": "closed", "flow_rate": 0.0, "since": "", "duration_min": None},
    "spray_boom": {"state": "off", "nozzle_pressure": 0.0, "since": "", "duration_min": None},
}

# 模块级单例（供 Agent / WS 服务 / 页面复用）
bus = ActuatorBus()
