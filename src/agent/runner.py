"""Agent 主循环：后台线程定时执行「采集→推理→分区决策→执行→落库→广播」。

典型一轮（演示链路）：
  多路摄像头/演示帧 → OpenCV → YOLO 推理 → 按区域分别决策
  → 各区域执行器独立动作 → 日志落库 → WebSocket 广播 → 网页分区动画展示

区域模型（一个摄像头 = 一个区域）：
  每一路摄像头画面覆盖一个独立作业区域（zone），区域各自维护病害防抖计数、
  喷雾冷却与滴灌/喷杆状态。A 区检出病害只开 A 区喷杆，B 区完全不受影响，
  从而实现「多区域分别执行、互不干扰」。

线程安全：start/stop/step 均可从 REST API / Streamlit 调用。
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import Callable

from .. import config, db, risk
from ..executor import bus
from ..executor.devices import DEFAULT_SNAPSHOT, DEFAULT_ZONE
from ..sensor import sensor
from ..vision import annotate, camera, camera_manager, detector
from ..weather import WeatherClient
from .daily_recorder import daily_recorder
from .llm_decision import decide

DEFAULT_GARDEN_KEY = "longjing"

# 决策类型/等级优先级：多区域聚合为全场视图时取最高优先级区域
_TYPE_PRIORITY = {"spray": 5, "cool": 5, "irrigate": 4, "irrigating": 3, "stop": 2, "noop": 1}
_LEVEL_PRIORITY = {"high": 3, "medium": 2, "low": 1}


class AgentRunner:
    """智能执行 Agent 主循环。"""

    def __init__(self, garden_key: str = DEFAULT_GARDEN_KEY, interval: float = 5.0,
                 refresh_mode: str = "interval", user_id: int = 0):
        self.garden_key = garden_key
        # user_id：当前 Agent 上下文归属的账号（0 = 游客，>0 = 登录用户）。
        # 传感器/执行器/决策/预警等历史数据按该账号落库与查询。
        self.user_id = int(user_id or 0)
        # refresh_mode: "interval" 自定义间隔整轮决策；"stream" 持续录像（画面连续推送，决策按间隔节流）
        self.refresh_mode = "stream" if str(refresh_mode).strip().lower() == "stream" else "interval"
        self.interval = max(0.5, float(interval))
        self._thread: threading.Thread | None = None          # 推流主循环线程
        self._step_thread: threading.Thread | None = None     # 决策线程（与推流解耦）
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        # 启动/停止/切换上下文串行执行，避免 stop 尚在等旧线程时 start 又创建一组线程。
        self._lifecycle_lock = threading.RLock()
        # REST 手动单步与后台决策线程可能同时触发，整轮决策操作必须串行。
        self._step_lock = threading.Lock()
        self._listeners: list[Callable[[dict], None]] = []
        self.running = False
        self.steps = 0
        self.latest: dict = {}
        self.latest_frame: dict = {}  # 最近一帧画面（stream_frame 与完整轮次都会更新，供 /api/frame 与主站实时画面读取）
        # 分区决策状态：zone_id -> 跨轮状态（防抖计数、喷雾冷却），区域之间完全隔离。
        # 一个摄像头 = 一个区域，A 区连续检出计数不会影响 B 区。
        self._zone_states: dict[str, dict] = {}
        # 兼容旧属性：默认区（无摄像头/旧调用场景）的跨轮状态
        self._decision_state: dict = self._zone_states.setdefault(DEFAULT_ZONE, {})
        # 视觉缓存：推流线程持续采集+推理，决策线程复用最新结果，两者互不阻塞
        self._cams_cache: dict | None = None

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
            except Exception:  # noqa: BLE001
                pass

    # ---------- 上下文（账号绑定） ----------
    def set_context(self, user_id: int = 0, garden_key: str | None = None) -> dict:
        """设置 Agent 归属账号与茶园（只能在未运行时切换）。

        user_id=0 表示游客（内置演示茶园）；登录用户使用自己的茶园 key。
        返回当前上下文信息。
        """
        with self._lifecycle_lock:
            with self._lock:
                if self.running or self._threads_alive():
                    return {
                        "ok": False,
                        "message": "Agent 正在运行或后台线程尚未退出，请停止后再切换账号",
                        **self._context_unlocked(),
                    }
                uid = int(user_id or 0)
                current_uid = self.user_id

            # 数据库查询不持有状态锁，避免网络抖动阻塞推流线程读取最新状态。
            if garden_key:
                next_garden_key = garden_key
            elif uid != current_uid:
                # 切换账号但未显式指定茶园时，先回默认茶园，再尝试采用该账号首个茶园；
                # 绝不能沿用上一个账号的 rdb:* 上下文。
                next_garden_key = DEFAULT_GARDEN_KEY
            else:
                next_garden_key = self.garden_key
            if uid and uid != current_uid:
                try:
                    gardens = db.user_gardens(uid)
                    if gardens:
                        next_garden_key = f"rdb:{gardens[0]['id']}"
                except Exception:  # noqa: BLE001
                    pass

            with self._lock:
                self.garden_key = next_garden_key
                self.user_id = uid
                ctx = self._context_unlocked()
        return {"ok": True, **ctx}

    def context(self) -> dict:
        """当前 Agent 上下文：归属账号、茶园、运行状态。"""
        with self._lock:
            return self._context_unlocked()

    def _context_unlocked(self) -> dict:
        return {
            "user_id": self.user_id,
            "garden_key": self.garden_key,
            "running": self.running,
            "steps": self.steps,
            "interval": self.interval,
            "refresh_mode": self.refresh_mode,
            "threads_alive": self._threads_alive(),
            "step_busy": self._step_lock.locked(),
        }

    def _threads_alive(self) -> bool:
        return any(t is not None and t.is_alive() for t in (self._thread, self._step_thread))

    def current_snapshot(self) -> dict:
        """返回最近一次完整快照的浅拷贝，供并发 REST 读取。"""
        with self._lock:
            return dict(self.latest) if self.latest else {}

    # ---------- 配置 ----------
    def configure(self, refresh_mode: str | None = None, interval: float | None = None) -> dict:
        """运行时更新刷新模式与间隔（线程安全）。

        refresh_mode: "interval"（自定义刷新时间） / "stream"（持续录像）
        interval: 刷新间隔秒数，最小 0.5 秒。
        """
        with self._lock:
            if refresh_mode is not None:
                self.refresh_mode = "stream" if str(refresh_mode).strip().lower() == "stream" else "interval"
            if interval is not None:
                self.interval = max(0.5, float(interval))
        return {"refresh_mode": self.refresh_mode, "interval": self.interval}

    # ---------- 生命周期 ----------
    def start(self) -> bool:
        with self._lifecycle_lock:
            return self._start_locked()

    def _start_locked(self) -> bool:
        with self._lock:
            if self.running:
                return False
            # 上一个循环线程还卡在采集/推理里未退出时拒绝再启动，
            # 避免两个线程并发抢摄像头/推理导致越卡越死
            if self._threads_alive():
                return False
            self.running = True
            self._stop_evt.clear()
            # 推流主循环线程：只做「采集缓存→推理→标注→WS 推流」，永不阻塞
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="tea-agent-runner")
            self._thread.start()
            # 决策线程：按 interval 节流执行完整决策（LLM/天气/写库慢操作
            # 只影响本线程，绝不冻结推流画面）
            self._step_thread = threading.Thread(target=self._step_loop, daemon=True,
                                                 name="tea-agent-decider")
            self._step_thread.start()
            return True

    def stop(self) -> bool:
        with self._lifecycle_lock:
            return self._stop_locked()

    def _stop_locked(self) -> bool:
        with self._lock:
            if not self.running:
                return False
            self.running = False
            self._stop_evt.set()
            threads = list(dict.fromkeys(
                t for t in (self._thread, self._step_thread) if t is not None))
        # 两个线程共享一个截止时间：总等待上限约 5 秒，而不是各等 5 秒。
        deadline = time.monotonic() + 5.0
        for t in threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if t.is_alive():
                t.join(timeout=remaining)
        with self._lock:
            if self._thread is not None and not self._thread.is_alive():
                self._thread = None
            if self._step_thread is not None and not self._step_thread.is_alive():
                self._step_thread = None
        return True

    def _loop(self) -> None:
        """推流主循环：无论哪种刷新模式，画面始终按 0.5s 推流；
        完整决策由 _step_loop 按 interval 节流，二者解耦互不阻塞。"""
        while not self._stop_evt.is_set():
            try:
                self._stream_tick()
                self._stop_evt.wait(0.5)
            except Exception as e:  # noqa: BLE001 - 单轮失败不中断循环
                self._broadcast({"kind": "agent_error", "message": str(e), "ts": _ts()})
                self._stop_evt.wait(0.5)

    def _step_loop(self) -> None:
        """决策线程：按 interval 节流执行完整决策（采集缓存→决策→执行→落库→广播）。"""
        while not self._stop_evt.is_set():
            if self._stop_evt.wait(self.interval):
                break
            try:
                self.step()
            except Exception as e:  # noqa: BLE001 - 单轮失败不中断决策循环
                self._broadcast({"kind": "agent_error", "message": str(e), "ts": _ts()})

    def _set_latest_frame(self, det_result: dict, image_b64: str) -> None:
        """写入最近一帧画面缓存（供 /api/frame 与主站实时画面读取，使用主源帧）。"""
        with self._lock:
            self.latest_frame = {
                "ts": _ts(),
                "source": det_result.get("source", "demo_frame"),
                "backend": det_result.get("backend", "sim"),
                "image": image_b64,
                "detections": det_result.get("detections", []),
                "summary": det_result.get("summary", {}),
            }

    # ---------- 多源采集 ----------
    def _collect_cams(self) -> list[dict]:
        """遍历所有摄像头源：采集→推理→标注，返回每路的展示数据（不含原始帧）。

        画面以 320px 缩略图 + JPEG q60 编码：持续录像高频推流时单路 payload
        从 20-50KB 降到 5-10KB，浏览器每 0.5s 解码多路大图的 CPU 压力大幅下降。
        raw_frame 字段仅供内部（决策落盘每日帧），广播前统一剔除。
        """
        cams = []
        # 显式传 Agent 归属账号，页面并发切换其他账号时也不会串用摄像头源。
        for item in camera_manager.read_all(self.user_id):
            frame = item.get("frame")
            det = detector.detect(frame, source_hint=item.get("last_source", "demo_frame"))
            image_b64 = ""
            if frame is not None:
                annotated = annotate.draw_detections(frame, det)
                image_b64 = annotate.frame_to_base64_thumb(annotated, max_w=320, quality=60)
            cams.append({
                "cam_id": item.get("cam_id"),
                "name": item.get("name", item.get("cam_id")),
                "source": item.get("source", ""),
                "kind": item.get("kind", "usb"),
                "last_source": item.get("last_source", "demo_frame"),
                "available": item.get("last_source") in ("usb_cam", "net_cam"),
                "ts": _ts(),
                "image": image_b64,
                "detections": det.get("detections", []),
                "summary": det.get("summary", {}),
                "det_source": det.get("source", "demo_frame"),
                "raw_frame": frame,
            })
        return cams

    @staticmethod
    def _strip_raw(cams: list[dict]) -> list[dict]:
        """剔除内部字段（raw_frame / frame），返回可广播的安全副本。"""
        return [{k: v for k, v in c.items() if k not in ("raw_frame", "frame")} for c in cams]

    @staticmethod
    def _merge_dets(cams: list[dict]) -> dict:
        """合并所有源的检测结果，仅用于「全场总览」（/api/frame、主站主画面）。

        重要：分区决策不再使用该合并结果 —— 合并会让 A 区病害触发全场打药。
        每个区域的决策改用 _zone_detections() 只取本区域这一路摄像头的结果。
        """
        dets = [c for c in cams if c.get("detections")]
        if not dets:
            return {"source": "demo_frame", "backend": "sim",
                    "detections": [], "summary": {"total": 0, "diseases": [], "max_conf": 0.0, "healthy": True},
                    "frame_shape": [480, 640]}
        diseases = sorted({d for c in dets for d in c.get("summary", {}).get("diseases", [])})
        total = sum(c.get("summary", {}).get("total", 0) for c in dets)
        max_conf = max([c.get("summary", {}).get("max_conf", 0.0) or 0.0 for c in dets] or [0.0])
        return {
            "source": dets[0].get("det_source", "demo_frame"),
            "backend": "ultralytics" if any(c.get("det_source") == "usb_cam" or c.get("last_source") in ("usb_cam", "net_cam") for c in dets) else "sim",
            "detections": [d for c in dets for d in c.get("detections", [])],
            "summary": {
                "total": total,
                "diseases": diseases,
                "max_conf": round(max_conf, 2),
                "healthy": not diseases,
            },
            "frame_shape": [480, 640],
        }

    # ---------- 区域（一个摄像头 = 一个区域） ----------
    @staticmethod
    def _zone_key(cam: dict) -> str:
        """区域标识：直接复用摄像头 id（cam1/cam2…），保证一机一区。"""
        return str(cam.get("cam_id") or DEFAULT_ZONE)

    @staticmethod
    def _zone_name(cam: dict) -> str:
        """区域展示名：优先摄像头名称（用户可命名成"东区""西坡"等）。"""
        return str(cam.get("name") or cam.get("cam_id") or DEFAULT_ZONE)

    @staticmethod
    def _zone_detections(cam: dict) -> dict:
        """把单路摄像头的检测结果整理成决策引擎输入（只含本区域，不含其他区域）。"""
        summary = cam.get("summary") or {}
        dets = cam.get("detections") or []
        diseases = list(summary.get("diseases") or [])
        last_source = cam.get("last_source", "demo_frame")
        return {
            "source": last_source,
            "backend": "ultralytics" if last_source in ("usb_cam", "net_cam") else "sim",
            "detections": dets,
            "summary": {
                "total": int(summary.get("total", len(dets)) or 0),
                "diseases": diseases,
                "max_conf": float(summary.get("max_conf", 0.0) or 0.0),
                "healthy": not diseases,
            },
            "frame_shape": [480, 640],
        }

    @staticmethod
    def _pick_actuator(zone_results: list[dict]) -> dict:
        """把多区域执行器状态聚合为全场视图（兼容旧页面/接口字段）。

        只要任一区域在作业，全场视图即显示为作业中，并附带 zone_id/zone_name，
        便于调用方知道到底是哪个区域在执行；各区域真实状态见快照的 zones 字段。
        """
        valve = dict(DEFAULT_SNAPSHOT["drip_valve"])
        boom = dict(DEFAULT_SNAPSHOT["spray_boom"])
        for z in zone_results:
            act = z.get("actuator") or {}
            v = act.get("drip_valve") or {}
            if v.get("state") == "open" and valve.get("state") != "open":
                valve = {**v, "zone_id": z["zone_id"], "zone_name": z["zone_name"]}
            b = act.get("spray_boom") or {}
            if b.get("state") == "on" and boom.get("state") != "on":
                boom = {**b, "zone_id": z["zone_id"], "zone_name": z["zone_name"]}
        return {"drip_valve": valve, "spray_boom": boom}

    @staticmethod
    def _aggregate_decision(zone_results: list[dict], ts: str) -> dict:
        """把各区域决策聚合成一条「全场决策」记录（落库 + 旧页面展示用）。

        - 决策类型/等级：取优先级最高的区域（有打药 > 灌溉 > 灌溉中 > 停止 > 无操作）
        - 摘要：各区域摘要拼接（已带 [区域名] 前缀），便于历史记录一眼看清分区结果
        - 动作：所有区域动作合并，每个动作带 zone_id/zone_name
        """
        decisions = [z.get("decision") or {} for z in zone_results]
        if not decisions:
            return {"ts": ts, "decision_type": "noop", "summary": "暂无区域",
                    "overall_level": "low", "actions": [],
                    "explanation": "", "zones": []}
        top = max(decisions, key=lambda d: (
            _TYPE_PRIORITY.get(d.get("decision_type"), 0),
            _LEVEL_PRIORITY.get(d.get("overall_level"), 0)))
        level = max((d.get("overall_level") or "low" for d in decisions),
                    key=lambda v: _LEVEL_PRIORITY.get(v, 0))
        actions = [a for d in decisions for a in (d.get("actions") or [])]
        summary = "；".join(d.get("summary", "") for d in decisions if d.get("summary"))
        return {
            "ts": ts,
            "decision_type": top.get("decision_type", "noop"),
            "summary": summary,
            "overall_level": level,
            "actions": actions,
            "explanation": top.get("explanation", ""),
            "zones": [{
                "zone_id": z.get("zone_id"),
                "zone_name": z.get("zone_name"),
                "decision_type": (z.get("decision") or {}).get("decision_type"),
                "overall_level": (z.get("decision") or {}).get("overall_level"),
                "summary": (z.get("decision") or {}).get("summary", ""),
                "actions": (z.get("decision") or {}).get("actions", []),
            } for z in zone_results],
        }

    def _stream_tick(self) -> None:
        """推流主循环单次 tick：采集所有源→推理→标注→更新视觉缓存→推送多路画面帧。

        完整快照（决策/执行器/传感器）由 _step_loop 按间隔生成（kind=state_update）。
        """
        cams = self._collect_cams()
        main = cams[0] if cams else None
        if main is not None:
            self._set_latest_frame(
                {"source": main["last_source"], "backend": main["det_source"],
                 "detections": main["detections"], "summary": main["summary"]},
                main["image"],
            )
        # 更新视觉缓存：决策线程复用本轮的采集+推理结果，不再重复读摄像头/推理
        with self._lock:
            self._cams_cache = {
                "cams": cams,
                "det_result": self._merge_dets(cams),
                "main_frame": cams[0]["raw_frame"] if cams else None,
                "main_image": cams[0]["image"] if cams else "",
                "ts": time.monotonic(),
            }
        self._broadcast({
            "kind": "stream_frame",
            "data": {
                "ts": _ts(),
                "cams": self._strip_raw(cams),
                # 各区域执行器实时状态：前端据此只在对应区域画喷药/滴灌动画
                "zone_actuators": bus.snapshot_all(),
                "steps": self.steps,
                "running": self.running,
                "interval": self.interval,
                "refresh_mode": self.refresh_mode,
            },
        })

    # ---------- 单轮决策 ----------
    def step(self, wait: bool = False) -> dict:
        """执行一轮完整决策，返回并广播最新状态快照。

        默认不排队：后台决策尚未结束时，手动触发直接返回当前快照并标记
        skipped=True，避免重复点击或 REST 重试堆出一长串过期决策请求。
        """
        if not self._step_lock.acquire(blocking=wait):
            snapshot = self.current_snapshot()
            snapshot.update({
                "busy": True,
                "skipped": True,
                "message": "上一轮决策仍在执行，已返回最近快照",
                **self.context(),
            })
            return snapshot
        try:
            return self._step_once()
        finally:
            self._step_lock.release()

    def _current_cams(self) -> tuple[list, dict, object, str]:
        """取最新视觉缓存（采集+推理结果）；缓存缺失/过期时自采一轮。

        决策与推流解耦：推流线程持续产帧，决策线程几乎总是能直接复用
        最新缓存，不重复读摄像头/跑推理，也不因 LLM/天气慢而冻结推流。
        """
        with self._lock:
            cc = self._cams_cache
        if cc is not None and (time.monotonic() - cc["ts"]) < 3.0:
            return cc["cams"], cc["det_result"], cc.get("main_frame"), cc.get("main_image", "")
        # 缓存为空或过期（如刚启动、决策间隔很大）：自采一轮
        cams = self._collect_cams()
        det_result = self._merge_dets(cams)
        with self._lock:
            self._cams_cache = {
                "cams": cams,
                "det_result": det_result,
                "main_frame": cams[0]["raw_frame"] if cams else None,
                "main_image": cams[0]["image"] if cams else "",
                "ts": time.monotonic(),
            }
        return cams, det_result, (cams[0]["raw_frame"] if cams else None), (cams[0]["image"] if cams else "")

    def _step_once(self) -> dict:
        """串行执行一次完整决策（由 step() 持锁调用）。

        区域化执行：每个摄像头 = 一个区域，逐区域独立决策 → 独立执行器 → 独立落库，
        区域之间不共享防抖计数、喷雾冷却与执行器状态（互不干扰）。
        全场共用的只有土壤传感器与气象数据。

        视觉部分复用推流线程的最新缓存（采集+推理已完成），
        LLM/天气/写库等慢操作只发生在本决策线程，不影响画面推流。
        """
        garden = self._garden()

        # 1) 传感器（全场共用一个土壤湿度传感器）
        soil = sensor.read()
        try:
            db.log_sensor(self.garden_key, soil["type"], soil["value"], soil["unit"],
                          soil.get("meta"), user_id=self.user_id)
        except Exception:  # noqa: BLE001 - 数据库不可用时降级跳过
            pass

        # 2) 视觉（复用推流缓存的多源采集+推理结果）
        cams, det_result, main_frame, image_b64 = self._current_cams()

        # 2.1) 土壤湿度历史（监控台趋势图数据，最近 60 条）
        soil_history = []
        try:
            soil_history = db.recent_sensor(self.garden_key, "soil_moisture", limit=60,
                                            user_id=self.user_id)
        except Exception:  # noqa: BLE001
            pass

        # 3) 气象（网络不可用/无 Key 时降级演示数据；全场共用）
        weather, risk_result = self._weather_and_risk(garden)

        # 4) 逐区域决策 + 执行（一个摄像头 = 一个区域）
        if not cams:
            # 无摄像头源：退化为「全园」单区域，行为与改造前一致
            cams = [{
                "cam_id": DEFAULT_ZONE, "name": "全园", "source": "", "kind": "usb",
                "last_source": "demo_frame", "available": False, "image": "",
                "detections": [], "summary": det_result.get("summary", {}),
                "det_source": "demo_frame", "raw_frame": None,
            }]

        # 多区域时默认不逐区调用大模型复核（N 个区域 = N 次外部调用会拖慢整轮决策），
        # 需要时可设 AGENT_LLM_MULTI_ZONE=1 打开。
        multi_zone = len(cams) > 1
        use_llm = (not multi_zone) or (os.getenv("AGENT_LLM_MULTI_ZONE", "0") == "1")

        zone_results: list[dict] = []
        events: list[dict] = []
        for cam in cams:
            zid = self._zone_key(cam)
            zname = self._zone_name(cam)
            zone_det = self._zone_detections(cam)
            # 本区域跨轮状态：防抖计数与喷雾冷却各自独立
            state = self._zone_states.setdefault(zid, {})
            inputs = {
                "soil": soil,
                "detections": zone_det,
                "weather": weather,
                "risk": risk_result,
                "actuators": bus.snapshot(zid),
                "zone": {"id": zid, "name": zname},
            }
            decision = decide(inputs, state, use_llm=use_llm)
            decision["zone_id"] = zid
            decision["zone_name"] = zname
            if decision.get("summary"):
                decision["summary"] = f"[{zname}] {decision['summary']}"
            for a in decision.get("actions") or []:
                a["zone_id"] = zid
                a["zone_name"] = zname
            # 只操作本区域执行器：A 区打药不会带动 B 区喷杆
            events.extend(bus.apply(decision.get("actions") or [], zone=zid))
            zone_results.append({
                "zone_id": zid,
                "zone_name": zname,
                "cam_id": cam.get("cam_id", zid),
                "available": bool(cam.get("available")),
                "source": cam.get("last_source", "demo_frame"),
                "detections": {
                    "total": zone_det["summary"]["total"],
                    "diseases": zone_det["summary"]["diseases"],
                    "max_conf": zone_det["summary"]["max_conf"],
                    "healthy": zone_det["summary"]["healthy"],
                },
                "decision": decision,
                "actuator": bus.snapshot(zid),
            })

        # 5) 超时安全停止（各区域独立计时、独立停止）
        timeout_events = bus.auto_timeout_check()
        events.extend(timeout_events)
        for event in timeout_events:
            if event.get("device") == "spray_boom" and event.get("operation") == "off":
                # 自动停止也属于该区域一次喷雾结束，需进入该区域自己的冷却期。
                zid = str(event.get("zone") or DEFAULT_ZONE)
                self._zone_states.setdefault(zid, {})["last_spray_end_at"] = event.get("ts")

        # 清理已移除摄像头对应的区域状态，避免残留计数影响后续同名区域
        alive_zones = {z["zone_id"] for z in zone_results}
        for zid in list(self._zone_states):
            if zid not in alive_zones and zid != DEFAULT_ZONE:
                self._zone_states.pop(zid, None)

        # 5.1) 灌溉响应：任一区域阀门开启时土壤湿度逐步回升（全场共用一个土壤传感器，
        #      多个区域同时滴灌也只按一次回水计入，避免湿度过冲）
        if any((z["actuator"].get("drip_valve") or {}).get("state") == "open" for z in zone_results):
            sensor.irrigate(0.6)

        # 6) 落库（全场一条聚合决策，动作带 zone_id；数据库不可用时降级跳过）
        agg = self._aggregate_decision(zone_results, _ts())
        try:
            for a in agg["actions"]:
                db.log_actuator(self.garden_key, a["device"], a["op"], a.get("duration_min"),
                                a.get("reason"), user_id=self.user_id)
            db.log_agent_decision(
                self.garden_key, agg["decision_type"], agg["summary"],
                inputs={
                    "soil": soil,
                    "weather": self._weather_light(weather),
                    "zones": [{
                        "zone_id": z["zone_id"],
                        "zone_name": z["zone_name"],
                        "cam_id": z["cam_id"],
                        "detections": z["detections"],
                        "decision_type": (z["decision"] or {}).get("decision_type"),
                        "actuator": z["actuator"],
                    } for z in zone_results],
                },
                actions=agg["actions"],
                overall_level=agg["overall_level"],
                user_id=self.user_id)
        except Exception:  # noqa: BLE001 - 落库失败不影响本轮执行与推送
            pass

        # 7) 标注帧（多路画面已随 cams 推送，此处缓存主源缩略帧供 /api/frame 使用）
        self._set_latest_frame(
            {"source": cams[0]["last_source"] if cams else "demo_frame",
             "backend": det_result.get("backend", "sim"),
             "detections": det_result.get("detections", []),
             "summary": det_result.get("summary", {})},
            image_b64,
        )

        # 7.1) 每日视频监测汇总（供风险预警「摄像头监测预警」每日推送：
        #      当日最新标注帧 + 病害检出计数 + 处理动作计数）
        try:
            if main_frame is not None:
                daily_recorder.save_frame(annotate.frame_to_jpeg_bytes(main_frame))
            daily_recorder.record(det_result.get("summary", {}), agg)
        except Exception:  # noqa: BLE001 - 汇总失败不影响主链路
            pass

        # 8) 状态快照 + 广播
        snapshot = {
            "ts": _ts(),
            "user_id": self.user_id,
            "garden": {"key": self.garden_key, "name": garden.get("name", self.garden_key)},
            "camera": {"available": bool(cams) and cams[0]["available"],
                       "backend": det_result.get("backend", "sim"),
                       "source": det_result.get("source", "demo_frame")},
            "cams": self._strip_raw(cams),
            # 分区结果：一个摄像头一个区域，含该区域检测、决策与执行器状态
            "zones": zone_results,
            "soil": soil,
            "soil_history": soil_history,
            "detections": {**det_result, "image": image_b64},
            # 全场聚合视图（任一区域作业即显示作业中，带 zone_name），兼容旧页面
            "actuator": self._pick_actuator(zone_results),
            "decision": agg,
            "events": events,
            "steps": self.steps,
            "running": self.running,
            "interval": self.interval,
            "refresh_mode": self.refresh_mode,
        }
        self.steps += 1
        snapshot["steps"] = self.steps
        with self._lock:
            self.latest = snapshot
        self._broadcast({"kind": "state_update", "data": snapshot})
        return snapshot

    # ---------- 辅助 ----------
    def _garden(self) -> dict:
        """当前茶园配置。用户自有茶园（rdb:{id} 或未知 key）不在内置配置时，
        直接以 key 作为展示名，避免错误借用第一个演示园的天气/位置。"""
        g = config.get_garden(self.garden_key)
        if g is None:
            return {"id": self.garden_key, "name": self.garden_key, "key": self.garden_key}
        return {**g, "key": self.garden_key}

    def _weather_and_risk(self, garden: dict) -> tuple[dict, dict]:
        weather: dict = {}
        risk_result: dict = {"items": [], "overall_level": "low"}
        try:
            client = WeatherClient()
            loc = garden.get("location_id") or ""
            if garden.get("lon") is not None and garden.get("lat") is not None:
                loc = f"{garden['lon']},{garden['lat']}"
            if loc:
                weather = {
                    "now": client.get_now(loc),
                    "daily": client.get_forecast(loc),
                }
                n = weather["now"].get("now", weather["now"])
                dl = weather["daily"].get("daily", [])
                if n and dl:
                    risk_result = risk.RiskEngine().assess_all(n, dl)
        except Exception:  # noqa: BLE001 - 气象不可用时决策仅依赖传感器与视觉
            pass
        return weather, risk_result

    @staticmethod
    def _weather_light(weather: dict) -> dict:
        n = (weather.get("now") or {}).get("now", {})
        dl = (weather.get("daily") or {}).get("daily", [])
        return {
            "temp": n.get("temp"), "humidity": n.get("humidity"),
            "text": n.get("text"), "windScale": n.get("windScale"),
            "precip_3d": [d.get("precip") for d in dl[:3]],
            "temp_3d": [f"{d.get('tempMin')}~{d.get('tempMax')}" for d in dl[:3]],
        }


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


# 模块级单例（供 WS 服务 / Streamlit 页面共享）
# 默认 5 秒刷新；最小 0.5 秒；可通过 /api/agent/config 或「智能执行」页运行时调整
runner = AgentRunner(interval=float(os.getenv("AGENT_INTERVAL", "5")))
