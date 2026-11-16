"""training/research 的共用底座：data.yaml 解析、标签读取、IoU 贪心匹配。

为什么单独拆一个文件：
  stats_eval.py（置信区间 + 显著性检验）与 error_analysis.py（FP/FN 定性分析）
  必须使用**完全一致**的图片列表、标签解析与匹配口径，否则论文里两个工具报出的
  TP/FP/FN 对不上，审稿人一眼就能看出矛盾。公共逻辑只在这里维护一份。

设计约束：
  - 只依赖 numpy / Pillow / pyyaml，不引入新依赖；
  - 不使用 scipy，统计量（分位数、卡方 p 值）由调用方用 numpy / math 自己算；
  - 路径解析兼容 ultralytics 的 data.yaml 写法：path 可绝对/相对，
    split 值可以是目录、"images" 下的子目录，也可以是图片清单 txt。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# 项目根目录：training/research/common.py -> 上溯两级
BASE = Path(__file__).resolve().parents[2]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# split 名称别名：data.yaml 里 val / valid / validation 混用很常见
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation"),
    "valid": ("valid", "val", "validation"),
    "test": ("test", "testing"),
}


# ============================================================
# data.yaml 解析
# ============================================================
def _names_from_config(raw) -> list[str]:
    """names 支持 dict / list / 逗号字符串三种写法。"""
    if isinstance(raw, dict):
        try:
            return [str(raw[k]) for k in sorted(raw, key=lambda x: int(x))]
        except (TypeError, ValueError):
            return [str(v) for v in raw.values()]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return []


def load_data_yaml(path: str | Path) -> tuple[dict, list[str], Path]:
    """读取 data.yaml，返回 (配置字典, 类别名, yaml 绝对路径)。"""
    yaml_path = Path(path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"data.yaml 不存在：{yaml_path}")
    text = yaml_path.read_text(encoding="utf-8", errors="ignore")
    try:
        import yaml

        cfg = yaml.safe_load(text) or {}
    except Exception as e:  # noqa: BLE001 - yaml 损坏时给出明确原因
        raise ValueError(f"无法解析 {yaml_path}：{e}") from e
    if not isinstance(cfg, dict):
        raise ValueError(f"{yaml_path} 顶层不是映射（mapping），不是合法的 data.yaml")
    return cfg, _names_from_config(cfg.get("names")), yaml_path.resolve()


def dataset_root(cfg: dict, yaml_path: Path) -> Path:
    """解析 data.yaml 的 path 字段：绝对路径直接用，相对路径依次尝试
    yaml 同级目录 / 当前工作目录 / 项目根目录（与 ultralytics 行为兼容）。"""
    raw = cfg.get("path")
    if raw:
        p = Path(str(raw))
        if p.is_absolute():
            if p.is_dir():
                return p
        else:
            for cand in (yaml_path.parent / p, Path.cwd() / p, BASE / p):
                if cand.is_dir():
                    return cand.resolve()
    return yaml_path.parent


def _find_images_dir(target: Path) -> Path | None:
    """split 值指向目录时，定位真正的图片目录（可能是该目录或其 images 子目录）。"""
    if not target.is_dir():
        return None
    if (target / "images").is_dir():
        return target / "images"
    # 目录本身直接放图片
    for p in sorted(target.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            return target
    # 兼容嵌套：取第一个含图片的子目录
    for p in sorted(target.iterdir()):
        if p.is_dir() and any(q.suffix.lower() in IMG_EXTS for q in p.iterdir() if q.is_file()):
            return p
    return target


def _collect_images(img_dir: Path) -> list[Path]:
    return sorted(p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def _split_entry(cfg: dict, split: str) -> object:
    for key in SPLIT_ALIASES.get(split, (split,)):
        if key in cfg:
            return cfg[key]
    return None


def label_path_for(img: Path, img_dir: Path) -> Path:
    """按 YOLO 约定由图片路径推出标签路径：.../images/a/b.jpg -> .../labels/a/b.txt。"""
    parts = list(img.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        return Path(*parts[:idx], "labels", *parts[idx + 1:]).with_suffix(".txt")
    try:
        rel = img.relative_to(img_dir)
    except ValueError:
        rel = Path(img.name)
    return img_dir.parent / "labels" / rel.with_suffix(".txt")


def resolve_split(cfg: dict, yaml_path: Path, split: str
                  ) -> tuple[list[Path], list[Path], Path]:
    """定位 split 的图片与标签路径。

    返回 (图片列表, 与之对齐的标签路径列表, split 根目录)。
    找不到 split 时抛 FileNotFoundError，由调用方给出清晰报错。
    """
    entry = _split_entry(cfg, split)
    root = dataset_root(cfg, yaml_path)
    images: list[Path] = []
    split_root = root / split

    if entry is None:
        # 没有显式配置时按标准目录结构兜底
        found = _find_images_dir(root / split)
        if found is None:
            raise FileNotFoundError(
                f"data.yaml 中没有 '{split}' 字段，且默认目录 {root / split} 不存在"
            )
        images = _collect_images(found)
        split_root = found.parent
    else:
        values = entry if isinstance(entry, (list, tuple)) else [entry]
        for value in values:
            target = Path(str(value))
            if not target.is_absolute():
                target = root / target
            if target.suffix.lower() == ".txt" and target.is_file():
                # 图片清单写法：一行一个图片路径
                for line in target.read_text(encoding="utf-8", errors="ignore").splitlines():
                    name = line.strip()
                    if not name:
                        continue
                    p = Path(name)
                    if not p.is_absolute():
                        p = target.parent / p
                    if p.is_file():
                        images.append(p)
                split_root = target.parent
                continue
            img_dir = _find_images_dir(target)
            if img_dir is None:
                raise FileNotFoundError(
                    f"split='{split}' 指向的目录不存在：{target}（data.yaml 的 path={cfg.get('path')}）"
                )
            images.extend(_collect_images(img_dir))
            split_root = img_dir.parent if img_dir.name == "images" else img_dir

    if not images:
        raise FileNotFoundError(f"split='{split}' 下没有找到任何图片（已搜索 {root}）")

    labels = [label_path_for(p, split_root / "images") for p in images]
    return images, labels, split_root


# ============================================================
# 标签读取与 IoU 匹配
# ============================================================
def read_gt(label_path: Path, img_w: int, img_h: int, nc: int = 0) -> list[dict]:
    """读取 YOLO 标签（class cx cy w h，归一化），转成绝对坐标 xyxy。

    非法行（列数不足、非数字、类别越界、宽高非正）直接跳过，不计入 GT。
    """
    gts: list[dict] = []
    if not label_path.is_file():
        return gts
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, bw, bh = (float(x) for x in parts[1:5])
        except ValueError:
            continue
        if nc and not 0 <= cid < nc:
            continue
        if bw <= 0 or bh <= 0:
            continue
        x1 = (cx - bw / 2.0) * img_w
        y1 = (cy - bh / 2.0) * img_h
        x2 = (cx + bw / 2.0) * img_w
        y2 = (cy + bh / 2.0) * img_h
        gts.append({
            "cls": cid,
            "box": (
                max(0.0, x1), max(0.0, y1),
                min(float(img_w), x2), min(float(img_h), y2),
            ),
        })
    return gts


def iou_xyxy(a: tuple[float, float, float, float],
             b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def match_image(preds: list[dict], gts: list[dict], iou_thr: float = 0.5) -> dict:
    """单图匹配：按置信度降序贪心，同类且 IoU ≥ iou_thr 记为 TP。

    preds: [{"cls": int, "conf": float, "box": (x1, y1, x2, y2)}, ...]
    gts:   [{"cls": int, "box": (x1, y1, x2, y2)}, ...]

    返回 {"tp": [(pred, gt 下标, iou)],
          "fp": [(pred, 最佳 iou, 最佳 gt 下标或 -1)],
          "fn": [未匹配的 gt 下标]}
    每个 GT 最多被一个预测匹配（一对一），符合 COCO/YOLO 的评估口径。
    """
    matched: set[int] = set()
    tp: list[tuple[dict, int, float]] = []
    fp: list[tuple[dict, float, int]] = []

    for pred in sorted(preds, key=lambda p: p["conf"], reverse=True):
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            if j in matched or gt["cls"] != pred["cls"]:
                continue
            value = iou_xyxy(pred["box"], gt["box"])
            if value > best_iou:
                best_iou, best_j = value, j
        if best_j >= 0 and best_iou >= iou_thr:
            matched.add(best_j)
            tp.append((pred, best_j, best_iou))
        else:
            fp.append((pred, best_iou, best_j))

    fn = [j for j in range(len(gts)) if j not in matched]
    return {"tp": tp, "fp": fp, "fn": fn, "n_gt": len(gts), "n_pred": len(preds)}


def best_iou_any(box: tuple[float, float, float, float],
                 boxes: list[dict]) -> tuple[float, int]:
    """与任意类别框的最大 IoU（用于误差分析：判断误报压在哪个真值上）。"""
    best, idx = 0.0, -1
    for j, other in enumerate(boxes):
        value = iou_xyxy(box, other["box"])
        if value > best:
            best, idx = value, j
    return best, idx


def micro_prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    """micro 口径的 Precision / Recall / F1；分母为 0 时该指标记 0。"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def get_preds(result, names_from_model: dict | None = None) -> list[dict]:
    """把 ultralytics 单图 Results 转成本模块的 preds 结构。"""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy().astype(int)
    conf = boxes.conf.detach().cpu().numpy()
    preds: list[dict] = []
    for i in range(xyxy.shape[0]):
        p = {
            "cls": int(cls[i]),
            "conf": float(conf[i]),
            "box": tuple(float(v) for v in xyxy[i][:4]),
        }
        if names_from_model and int(cls[i]) in names_from_model:
            p["name"] = names_from_model[int(cls[i])]
        preds.append(p)
    return preds


def class_name(names: list[str], cid: int, fallback: dict | None = None) -> str:
    """类别名查询：优先 data.yaml 的 names，其次模型自带的 names。"""
    if 0 <= cid < len(names):
        return names[cid]
    if fallback and cid in fallback:
        return str(fallback[cid])
    return f"class_{cid}"


def summarize_arrays(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> dict:
    """逐图数组 -> micro 指标（供 stats_eval 与 md 输出共用）。"""
    total_tp, total_fp, total_fn = float(tp.sum()), float(fp.sum()), float(fn.sum())
    precision, recall, f1 = micro_prf(total_tp, total_fp, total_fn)
    return {
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
