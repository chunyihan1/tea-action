"""数据集质量与泄漏审计：dHash / pHash 近重复检测 + 划分/标注统计。

为什么需要：
  训练集与验证集若存在近重复（同一张图轻微裁剪/调色/缩放），验证集指标会虚高。
  论文里必须给出"泄漏率"这一硬证据，否则审稿人会直接质疑 mAP。

实现要点：
  - 自己用 Pillow + numpy 实现 64 位 dHash（缩放到 9x8 灰度，比较水平相邻像素），
    不依赖 imagehash，避免新增依赖；
  - pHash 用 cv2.dct（32x32 灰度 -> DCT -> 左上 8x8 低频 -> 与中位数比较），
    对亮度/对比度/轻微几何变换更稳健；cv2 不可用时给出提示并自动回退 dHash；
  - 近重复搜索不做 O(n²)：把 64 位哈希从高位切成 threshold+1 段做精确分桶。
    依据鸽巢原理：若两张图 Hamming 距离 ≤ threshold，则至少有一段完全相同，
    因此分桶不会漏检，平均只需比较极少量候选；
  - 用 numpy 向量化 popcount 计算候选对的 Hamming 距离，1 万张图可在几分钟内完成。

用法：
  python training\\research\\audit_dataset.py
  python training\\research\\audit_dataset.py --data-root training\\datasets\\tea_leaf_disease --out research_out
  python training\\research\\audit_dataset.py --hash phash --hash-threshold 6
"""
from __future__ import annotations

import argparse
import heapq
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

# 项目根目录：training/research/audit_dataset.py -> 上溯两级
BASE = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = BASE / "training" / "datasets" / "tea_leaf_disease"
SPLITS = ("train", "valid", "test")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Pillow 9.1+ 用 Image.Resampling，老版本回退到 Image.BILINEAR
RESAMPLE = getattr(Image, "Resampling", Image).BILINEAR

try:  # numpy >= 2.0 自带位计数，速度最快
    _popcount = np.bitwise_count
except AttributeError:  # 兼容老版本 numpy
    _LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def _popcount(x: np.ndarray) -> np.ndarray:
        arr = np.ascontiguousarray(x, dtype=np.uint64)
        return _LUT[arr.view(np.uint8).reshape(-1, 8)].sum(axis=1)


# ============================================================
# 类别名解析（优先 pyyaml，失败时用正则兜底）
# ============================================================
def find_data_yaml(root: Path) -> Path | None:
    """在数据集根目录定位 data.yaml（兼容包一层目录的情况）。"""
    for name in ("data.yaml", "data_norm.yaml"):
        p = root / name
        if p.exists():
            return p
    hits = sorted(root.rglob("data.yaml"))
    return hits[0] if hits else None


def _names_from_regex(text: str) -> list[str]:
    """极简兜底解析：支持 names: [a, b] / {0: a} / 多行列表 / 多行映射。"""
    # 行内列表或行内字典
    m = re.search(r"(?m)^names\s*:\s*(\[.*?\]|\{.*?\})\s*$", text)
    if m:
        body = m.group(1).strip()
        if body.startswith("["):
            return [x.strip().strip("'\"") for x in body[1:-1].split(",") if x.strip()]
        pairs = re.findall(r"\d+\s*:\s*['\"]?([^,}'\"]+)", body)
        return [p.strip() for p in pairs]

    # 多行块：names: 之后到下一个顶层键
    m = re.search(r"(?ms)^names\s*:\s*\n(.*?)(?=^\S|\Z)", text)
    if not m:
        return []
    block = m.group(1)
    names: dict[int, str] = {}
    for line in block.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        mm = re.match(r"(\d+)\s*:\s*(.+)", s)
        if mm:
            names[int(mm.group(1))] = mm.group(2).strip().strip("'\"")
            continue
        if s.startswith("- "):
            names[len(names)] = s[2:].strip().strip("'\"")
    return [names[k] for k in sorted(names)]


def load_class_names(root: Path) -> tuple[list[str], Path | None]:
    """读取类别名：返回 (names, data.yaml 路径)。"""
    yaml_path = find_data_yaml(root)
    if yaml_path is None:
        return [], None
    text = yaml_path.read_text(encoding="utf-8", errors="ignore")
    names: list[str] = []
    try:  # pyyaml 是 ultralytics 的依赖，正常情况下可用
        import yaml

        cfg = yaml.safe_load(text) or {}
        raw = cfg.get("names")
        if isinstance(raw, dict):
            names = [str(raw[k]) for k in sorted(raw, key=lambda x: int(x))]
        elif isinstance(raw, list):
            names = [str(x) for x in raw]
        elif isinstance(raw, str):  # names: "a,b,c" 之类
            names = [x.strip() for x in raw.split(",") if x.strip()]
    except Exception:  # noqa: BLE001 - 坏 yaml / 无 pyyaml 时走正则
        names = []
    if not names:
        names = _names_from_regex(text)
    return names, yaml_path


# ============================================================
# 图像扫描：一次打开同时取哈希与宽高
# ============================================================
def phash_supported() -> bool:
    """pHash 依赖 cv2.dct，用于探测环境是否能走 pHash 分支。"""
    try:
        import cv2  # noqa: F401

        return True
    except ImportError:
        return False


def _dhash(gray: Image.Image, hash_size: int) -> int:
    """dHash：9x8 灰度，比较每行相邻像素，得到 64 bit。"""
    arr = np.asarray(gray.resize((hash_size + 1, hash_size), RESAMPLE), dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).ravel()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def _phash(image: Image.Image, hash_size: int, highfreq_factor: int = 4) -> int:
    """pHash：32x32 灰度 -> 二维 DCT -> 左上 8x8 低频 -> 与中位数比较。

    中位数取除直流分量（DCT 系数 [0,0]，代表整体亮度）之外的 63 个系数，
    否则直流项会把中位数抬高，几乎所有 bit 都变成 0，哈希就失去区分度。
    """
    import cv2

    size = hash_size * highfreq_factor  # 8 * 4 = 32
    arr = np.asarray(image.resize((size, size), RESAMPLE), dtype=np.float32)
    dct = cv2.dct(arr)
    low = np.asarray(dct[:hash_size, :hash_size], dtype=np.float32)
    median = float(np.median(low.flatten()[1:]))
    bits = (low > median).ravel()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def image_hash_and_size(path: Path, hash_size: int = 8,
                        method: str = "dhash") -> tuple[int, int, int]:
    """返回 (64 bit 哈希, 宽, 高)。method 取 dhash 或 phash。"""
    with Image.open(path) as im:
        width, height = im.size
        gray = im.convert("L")
        if method == "phash":
            digest = _phash(gray, hash_size)
        else:
            digest = _dhash(gray, hash_size)
    return digest, width, height


def resolve_split_dirs(root: Path, split: str) -> tuple[Path, Path | None] | None:
    """定位 split 的图片/标签目录；split 不存在返回 None。"""
    split_dir = root / split
    if not split_dir.is_dir():
        return None
    img_dir = split_dir / "images" if (split_dir / "images").is_dir() else split_dir
    lbl_dir = split_dir / "labels" if (split_dir / "labels").is_dir() else None
    return img_dir, lbl_dir


def read_label_file(path: Path, nc: int) -> tuple[dict[int, int], int, int]:
    """统计一个标签文件：返回 (每类框数, 有效框总数, 非法行数)。"""
    per_class: dict[int, int] = {}
    valid = 0
    invalid = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            if parts:
                invalid += 1
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, w, h = (float(x) for x in parts[1:5])
        except ValueError:
            invalid += 1
            continue
        if cid < 0 or (nc and cid >= nc):
            invalid += 1
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            invalid += 1
            continue
        per_class[cid] = per_class.get(cid, 0) + 1
        valid += 1
    return per_class, valid, invalid


def scan_split(root: Path, split: str, nc: int, hash_size: int = 8,
               hash_method: str = "dhash") -> dict:
    """扫描一个 split：图片数、标签、类别实例、尺寸、哈希。"""
    dirs = resolve_split_dirs(root, split)
    if dirs is None:
        return {}
    img_dir, lbl_dir = dirs
    images = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    label_map: dict[str, Path] = {}
    if lbl_dir is not None:
        label_map = {p.stem: p for p in lbl_dir.glob("*.txt")}

    hashes = np.zeros(len(images), dtype=np.uint64)
    names: list[str] = []
    widths: list[int] = []
    heights: list[int] = []
    class_counts: dict[int, int] = {}
    boxes = 0
    invalid_rows = 0
    unlabeled = 0
    used_label_stems: set[str] = set()
    t0 = time.time()

    for i, img in enumerate(images):
        try:
            digest, w, h = image_hash_and_size(img, hash_size, hash_method)
        except Exception as e:  # noqa: BLE001 - 损坏图片不应中断审计
            print(f"  [警告] 无法读取图像 {img.name}：{e}")
            digest, w, h = 0, 0, 0
        hashes[i] = np.uint64(digest)
        names.append(img.name)
        if w > 0:
            widths.append(w)
            heights.append(h)

        lbl = label_map.get(img.stem)
        if lbl is None:
            unlabeled += 1
            continue
        used_label_stems.add(img.stem)
        per_class, n_valid, n_invalid = read_label_file(lbl, nc)
        invalid_rows += n_invalid
        boxes += n_valid
        if n_valid == 0:  # 空标签视为背景图，计入无标签图片
            unlabeled += 1
        for cid, n in per_class.items():
            class_counts[cid] = class_counts.get(cid, 0) + n

        if (i + 1) % 500 == 0:
            print(f"  [{split}] 已处理 {i + 1}/{len(images)} 张（{time.time() - t0:.0f}s）")

    orphan = sorted(set(label_map) - used_label_stems)

    def _stat(values: list[int]) -> dict:
        if not values:
            return {"min": None, "median": None, "max": None}
        arr = np.asarray(values)
        return {
            "min": int(arr.min()),
            "median": round(float(np.median(arr)), 1),
            "max": int(arr.max()),
        }

    return {
        "images": len(images),
        "label_files": len(label_map),
        "boxes": boxes,
        "invalid_label_rows": invalid_rows,
        "unlabeled_images": unlabeled,
        "orphan_labels": len(orphan),
        "orphan_label_examples": orphan[:10],
        "class_instances": {str(k): int(v) for k, v in sorted(class_counts.items())},
        "width": _stat(widths),
        "height": _stat(heights),
        "_hashes": hashes,
        "_names": names,
    }


# ============================================================
# 近重复：分桶 + 向量化 popcount
# ============================================================
def find_near_duplicates(
    train_hashes: np.ndarray,
    valid_hashes: np.ndarray,
    threshold: int,
    top_k: int = 20,
) -> dict:
    """找出 train-valid 之间 Hamming 距离 ≤ threshold 的全部配对。

    分桶原理：64 位切成 k=threshold+1 段；若两哈希距离 ≤ threshold，
    则 k 段中至少有一段完全相同（鸽巢原理），故穷举"同段"候选不会漏。
    """
    if len(train_hashes) == 0 or len(valid_hashes) == 0:
        return {"pairs": 0, "valid_images_matched": 0, "leak_rate": 0.0, "top_pairs": []}

    k = max(1, threshold + 1)
    base_width, extra = divmod(64, k)
    segments: list[tuple[int, int]] = []  # (右移位数, 掩码)
    shift = 64
    for i in range(k):
        width = base_width + (1 if i < extra else 0)
        shift -= width
        segments.append((shift, (1 << width) - 1))

    buckets: list[dict[int, np.ndarray]] = []
    for shift, mask in segments:
        keys = ((train_hashes >> np.uint64(shift)) & np.uint64(mask)).tolist()
        mapping: dict[int, list[int]] = {}
        for idx, key in enumerate(keys):
            mapping.setdefault(int(key), []).append(idx)
        buckets.append({kk: np.asarray(vv, dtype=np.int64) for kk, vv in mapping.items()})

    total_pairs = 0
    matched = 0
    best_dist: list[int | None] = [None] * len(valid_hashes)
    top_heap: list[tuple[int, int, int]] = []  # (-distance, valid_idx, train_idx)

    for vi, hv in enumerate(valid_hashes):
        hv64 = np.uint64(hv)
        found: dict[int, int] = {}
        for (shift, mask), bucket in zip(segments, buckets):
            key = int((hv64 >> np.uint64(shift)) & np.uint64(mask))
            cand = bucket.get(key)
            if cand is None or cand.size == 0:
                continue
            dists = _popcount(train_hashes[cand] ^ hv64)
            hit = np.nonzero(dists <= threshold)[0]
            for s in hit:
                ti = int(cand[s])
                d = int(dists[s])
                prev = found.get(ti)
                if prev is None or d < prev:
                    found[ti] = d

        if found:
            matched += 1
            total_pairs += len(found)
            best_dist[vi] = min(found.values())
            for ti, d in found.items():
                item = (-d, vi, ti)
                if len(top_heap) < top_k:
                    heapq.heappush(top_heap, item)
                elif item > top_heap[0]:
                    heapq.heapreplace(top_heap, item)

    top_pairs = sorted(
        ({"distance": -neg_d, "valid_index": vi, "train_index": ti}
         for neg_d, vi, ti in top_heap),
        key=lambda x: (x["distance"], x["valid_index"], x["train_index"]),
    )
    return {
        "pairs": total_pairs,
        "valid_images_matched": matched,
        "leak_rate": round(matched / len(valid_hashes), 4),
        "mean_min_distance": round(float(np.mean([d for d in best_dist if d is not None])), 2)
        if matched else 0.0,
        "top_pairs": top_pairs,
    }


# ============================================================
# 报告输出
# ============================================================
def _fmt_num(value: float | int | None, ndigits: int = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{ndigits}f}"


def build_markdown(audit: dict, class_names: list[str]) -> str:
    ts = audit["generated_at"]
    lines: list[str] = [
        f"# 数据集审计报告",
        "",
        f"- 数据根目录：`{audit['data_root']}`",
        f"- data.yaml：`{audit['data_yaml'] or '未找到'}`",
        f"- 哈希方法：{audit['hash_method']}；近重复阈值：Hamming ≤ {audit['hash_threshold']}",
        f"- 生成时间：{ts}",
        "",
    ]

    # 1. split 概览
    lines += [
        "## 1. 划分概览",
        "",
        "| Split | 图片数 | 标签文件 | 标注框 | 无标签图片 | 孤儿标签 | 非法标签行 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        s = audit["splits"].get(split)
        if not s:
            continue
        lines.append(
            f"| {split} | {s['images']} | {s['label_files']} | {s['boxes']} | "
            f"{s['unlabeled_images']} | {s['orphan_labels']} | {s['invalid_label_rows']} |"
        )
    lines.append("")

    # 2. 类别实例分布表
    active = [s for s in SPLITS if audit["splits"].get(s)]
    label_names = list(class_names) or [f"class_{i}" for i in range(max([-1] + [int(k) for _s in audit["splits"].values() for k in _s["class_instances"]]) + 1)]  # data.yaml 缺 names 时兜底
    lines += ["## 2. 类别实例分布", "",
              "| 类别 | " + " | ".join(active) + " | 合计 |",
              "| --- |" + " ---: |" * (len(active) + 1)]
    grand = 0
    for cid, name in enumerate(label_names):
        row_total = 0
        cells: list[str] = []
        for split in active:
            n = audit["splits"][split]["class_instances"].get(str(cid), 0)
            cells.append(str(n))
            row_total += n
        grand += row_total
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {row_total} |")
    lines.append("| **合计** | " + " | ".join(
        str(audit["splits"][s]["boxes"]) for s in active) + f" | {grand} |")
    lines.append("")

    # 3. 尺寸
    lines += ["## 3. 图像尺寸统计（像素）", "",
              "| Split | 宽 min | 宽 median | 宽 max | 高 min | 高 median | 高 max |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for split in active:
        s = audit["splits"][split]
        w, h = s["width"], s["height"]
        lines.append(
            f"| {split} | {_fmt_num(w['min'])} | {_fmt_num(w['median'])} | {_fmt_num(w['max'])} | "
            f"{_fmt_num(h['min'])} | {_fmt_num(h['median'])} | {_fmt_num(h['max'])} |"
        )
    lines.append("")

    # 4. 近重复
    nd = audit["near_duplicate"]["train_vs_valid"]
    lines += [
        "## 4. train / valid 近重复检测",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 近重复配对数（Hamming ≤ {audit['hash_threshold']}） | {nd['pairs']} |",
        f"| 可匹配到 train 近重复的 valid 图片 | {nd['valid_images_matched']} / {nd['valid_images']} |",
        f"| 泄漏率（valid 中被 train 覆盖的比例） | {nd['leak_rate'] * 100:.1f}% |",
        f"| 匹配图片的平均最小距离 | {nd['mean_min_distance']} |",
        "",
    ]
    top = nd.get("top_pairs") or []
    if top:
        lines += ["### 泄漏最严重的前 20 对", "",
                  "| # | valid 图片 | train 图片 | Hamming 距离 |",
                  "| ---: | --- | --- | ---: |"]
        for rank, pair in enumerate(top[:20], 1):
            vname = nd["_valid_names"][pair["valid_index"]]
            tname = nd["_train_names"][pair["train_index"]]
            lines.append(f"| {rank} | {vname} | {tname} | {pair['distance']} |")
        lines.append("")

    lines += ["## 5. 结论", "", f"> {audit['conclusion']}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="数据集质量与近重复泄漏审计")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                        help="数据集根目录（含 train/valid/test）")
    parser.add_argument("--hash", choices=("dhash", "phash"), default="dhash",
                        help="近重复哈希算法，默认 dhash；phash 需要 opencv(cv2)")
    parser.add_argument("--out", default="research_out", help="输出目录")
    parser.add_argument("--hash-threshold", type=int, default=8,
                        help="哈希 Hamming 距离阈值，默认 8（0~64，dHash/pHash 通用）")
    args = parser.parse_args()

    root = Path(args.data_root)
    if not root.is_dir():
        print(f"[错误] 数据集目录不存在：{root}")
        print("       请先准备数据：python training\\download_data.py")
        sys.exit(1)
    if not 0 <= args.hash_threshold <= 64:
        print("[错误] --hash-threshold 必须在 0~64 之间")
        sys.exit(1)
    if args.hash == "phash" and not phash_supported():
        print("[警告] 未检测到 opencv(cv2)，pHash 不可用，已自动回退到 dHash")
        print("       如需 pHash：pip install opencv-python（本仓库环境通常已预装）")
        args.hash = "dhash"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    hash_method = ("pHash-64 (32x32 DCT, low-freq 8x8, vs median)"
                   if args.hash == "phash"
                   else "dHash-64 (9x8 grayscale, horizontal adjacent compare)")

    class_names, yaml_path = load_class_names(root)
    nc = len(class_names)
    print(f"[信息] 数据根目录：{root}")
    print(f"[信息] 哈希方法：{args.hash} ｜ 类别数：{nc} ｜ 阈值：{args.hash_threshold}")
    if not class_names:
        print("[警告] 未解析到类别名，类别将以 class_0/class_1... 显示")

    t_start = time.time()
    splits: dict[str, dict] = {}
    for split in SPLITS:
        s = scan_split(root, split, nc, hash_method=args.hash)
        if not s:
            print(f"[跳过] {split}/ 不存在")
            continue
        print(f"[完成] {split}/：{s['images']} 张图，{s['boxes']} 个框，"
              f"无标签 {s['unlabeled_images']}，孤儿标签 {s['orphan_labels']}")
        splits[split] = s

    if not splits:
        print("[错误] 未找到任何 train/valid/test 划分目录")
        sys.exit(1)

    # 近重复（仅 train vs valid，符合论文划分审计需要）
    nd: dict = {"train_vs_valid": {
        "pairs": 0, "valid_images": 0, "valid_images_matched": 0,
        "leak_rate": 0.0, "mean_min_distance": 0.0, "top_pairs": [],
    }}
    if "train" in splits and "valid" in splits:
        print("[计算] train / valid 近重复（分桶 + 向量化 popcount）……")
        t0 = time.time()
        res = find_near_duplicates(
            splits["train"]["_hashes"], splits["valid"]["_hashes"], args.hash_threshold
        )
        nd["train_vs_valid"].update(res)
        nd["train_vs_valid"]["valid_images"] = splits["valid"]["images"]
        nd["train_vs_valid"]["train_images"] = splits["train"]["images"]
        nd["train_vs_valid"]["_valid_names"] = splits["valid"]["_names"]
        nd["train_vs_valid"]["_train_names"] = splits["train"]["_names"]
        print(f"[完成] 近重复计算耗时 {time.time() - t0:.1f}s，"
              f"配对数 {res['pairs']}，泄漏率 {res['leak_rate'] * 100:.1f}%")
    else:
        print("[提示] 缺少 train 或 valid，跳过近重复检测")

    leak = nd["train_vs_valid"]["leak_rate"]
    hash_label = "pHash" if args.hash == "phash" else "dHash"
    if "train" in splits and "valid" in splits:
        if leak >= 0.05:
            conclusion = (f"验证集 {leak * 100:.1f}% 的图片与训练集存在近重复"
                          f"（{hash_label} 距离 ≤ {args.hash_threshold}），指标可能虚高；"
                          f"建议按茶园/拍摄批次重新划分数据后再报告结果。")
        else:
            conclusion = (f"验证集仅 {leak * 100:.1f}% 的图片与训练集存在近重复"
                          f"（{hash_label} 距离 ≤ {args.hash_threshold}），未见明显同源泄漏。")
    else:
        conclusion = "缺少 train 或 valid 划分，无法评估近重复泄漏。"

    audit = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(root),
        "data_yaml": str(yaml_path) if yaml_path else None,
        "hash_method": hash_method,
        "hash_type": args.hash,
        "hash_threshold": args.hash_threshold,
        "class_names": class_names,
        "splits": {},
        "total": {
            "images": sum(s["images"] for s in splits.values()),
            "boxes": sum(s["boxes"] for s in splits.values()),
            "unlabeled_images": sum(s["unlabeled_images"] for s in splits.values()),
            "orphan_labels": sum(s["orphan_labels"] for s in splits.values()),
        },
        "near_duplicate": nd,
        "conclusion": conclusion,
    }
    # 剥离内部字段（哈希数组/文件名列表），保持 JSON 可读
    for split, s in splits.items():
        audit["splits"][split] = {k: v for k, v in s.items() if not k.startswith("_")}

    json_path = out_dir / "audit.json"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    # markdown 里需要文件名列表用于 top20
    md_audit = dict(audit)
    md_audit["splits"] = splits
    md_path = out_dir / "audit.md"
    md_path.write_text(build_markdown(md_audit, class_names), encoding="utf-8")

    print(f"\n[输出] {json_path}")
    print(f"[输出] {md_path}")
    print(f"[结论] {conclusion}")
    print(f"[统计] 总图片 {audit['total']['images']} 张，标注框 {audit['total']['boxes']} 个，"
          f"近重复配对 {nd['train_vs_valid']['pairs']} 对，"
          f"耗时 {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
