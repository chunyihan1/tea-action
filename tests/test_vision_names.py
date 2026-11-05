"""视觉类别名归一化测试：权重长名 → 系统短名（健康叶不能被当成病害）。"""
from src.vision.detector import CLASS_NAMES, build_detection_result, canonical_class


def test_all_model_names_map_to_short_names():
    raw = {
        "Black rot of tea": "Black rot",
        "Brown blight of tea": "Brown blight",
        "Leaf rust of tea": "Leaf rust",
        "Red Spider infested tea leaf": "Red Spider",
        "Tea Mosquito bug infested leaf": "Tea Mosquito bug",
        "Tea leaf": "Healthy",
        "White spot of tea": "White spot",
    }
    for long_name, short_name in raw.items():
        assert canonical_class(long_name) == short_name
        assert short_name in CLASS_NAMES


def test_healthy_leaf_not_treated_as_disease():
    result = build_detection_result(
        [{"class": "Healthy", "conf": 0.9, "bbox": [0, 0, 10, 10]}],
        source="upload", backend="ultralytics", model="tea_yolo.pt",
    )
    assert result["summary"]["healthy"] is True
    assert result["summary"]["diseases"] == []


def test_unknown_class_passthrough():
    assert canonical_class("class_99") == "class_99"
