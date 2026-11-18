"""安全闸门测试：视觉来源校验（不凭假图打药）+ 执行器硬限制（时长/每日上限/看门狗）。"""
import time

from src.agent import decision as rule
from src.executor.devices import MAX_DAILY_MIN, MAX_OPEN_MIN, ActuatorBus

BASE = {
    "weather": {"now": {"now": {"temp": 26, "humidity": 70, "text": "多云", "windScale": 2}},
                "daily": {"daily": [{"precip": 0}]}},
    "actuators": {"drip_valve": {"state": "closed"}, "spray_boom": {"state": "off"}},
}


def _new_state() -> dict:
    return {"counters": {"spray_hits": 0, "spray_misses": 0}}


def _det(source: str, backend: str, conf: float = 0.92) -> dict:
    disease = "Leaf rust"
    return {
        "source": source,
        "backend": backend,
        "summary": {"total": 1, "diseases": [disease], "max_conf": conf, "healthy": False},
        "detections": [{"class": disease, "conf": conf}],
    }


def test_sim_detections_never_spray():
    """演示帧 / 模拟推理即使连续多轮高置信度，也绝不下发喷杆开启。"""
    state = _new_state()
    for _ in range(5):
        d = rule.decide({**BASE, "soil": {"value": 60.0},
                         "detections": _det("demo_frame", "sim")}, state)
        assert d["decision_type"] != "spray"
        assert not any(a["device"] == "spray_boom" and a["op"] == "on" for a in d["actions"])
    assert "不可信" in d["summary"]


def test_real_cam_and_backend_can_spray():
    state = _new_state()
    d1 = rule.decide({**BASE, "soil": {"value": 60.0},
                      "detections": _det("usb_cam", "ultralytics")}, state)
    assert d1["decision_type"] != "spray"          # 第一轮防抖
    d2 = rule.decide({**BASE, "soil": {"value": 60.0},
                      "detections": _det("usb_cam", "ultralytics")}, state)
    assert d2["decision_type"] == "spray"


def test_camera_loss_stops_running_boom():
    """喷药过程中视觉来源掉线 → 立即停喷，不凭假图继续作业。"""
    state = _new_state()
    base_on = {**BASE, "actuators": {"drip_valve": {"state": "closed"},
                                     "spray_boom": {"state": "on"}}}
    d = rule.decide({**base_on, "soil": {"value": 60.0},
                     "detections": _det("demo_frame", "sim")}, state)
    assert any(a["device"] == "spray_boom" and a["op"] == "off" for a in d["actions"])


def test_open_duration_is_clamped():
    bus = ActuatorBus()
    try:
        events = bus.apply([{"device": "spray_boom", "op": "on",
                             "duration_min": 999.0, "reason": "t"}])
        boom = bus.devices["spray_boom"]
        assert boom.state == "on"
        assert boom.duration_min == MAX_OPEN_MIN["spray_boom"]
        assert events[0]["daily_cap_min"] == MAX_DAILY_MIN["spray_boom"]
    finally:
        bus.shutdown()


def test_daily_cap_blocks_open():
    bus = ActuatorBus()
    try:
        cap = MAX_DAILY_MIN["spray_boom"]
        bus._usage[bus._usage_key("global", "spray_boom")] = cap
        events = bus.apply([{"device": "spray_boom", "op": "on",
                             "duration_min": 5.0, "reason": "t"}])
        assert bus.devices["spray_boom"].state == "off"
        assert events and events[0]["kind"] == "actuator_blocked"
    finally:
        bus.shutdown()


def test_watchdog_closes_without_decision_loop():
    """看门狗独立于决策循环：到时长自动关闭（决策循环完全不参与）。"""
    bus = ActuatorBus()
    try:
        bus.apply([{"device": "drip_valve", "op": "open",
                    "duration_min": 0.02, "reason": "t"}])  # 约 1.2 秒
        assert bus.devices["drip_valve"].state == "open"
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and bus.devices["drip_valve"].state == "open":
            time.sleep(0.2)
        assert bus.devices["drip_valve"].state == "closed"
    finally:
        bus.shutdown()
