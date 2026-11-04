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
