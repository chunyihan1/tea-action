"""生成开放集（留一类）划分：S1~S6 的 known / unknown 目录与 data.yaml。

为什么需要：
  论文主实验是 6 组 leave-one-class-out（LOCO）：每次把一个具体病害/虫害类整体
  藏起来，模型只能见到其余类别；藏起来的类在验证集里充当"未知病害"。本脚本把
  现有 data.yaml 数据**可复现**地切成 known / unknown 两套目录，并把类别号重映射
  成无空洞的连续编号，避免训练阶段因空类别而报错。

关键规则（对应设计文档 2.2 / 2.4）：
  - 留出类的图像从 known/train 中**整体移除**（连图带标签，不是只删标签）；
  - known/val 中属于留出类的图像移到 unknown/val，其余留在 known/val；
  - unknown 的标签一律改写为 class 0（names: ["unknown"]）；
  - known 的类别号按"去掉留出类与排除类后的顺序"重映射，保证 0..nc-1 连续；
  - --exclude（默认 disease）在 known 与 unknown 中都不得出现，避免未知泄漏；
  - Healthy 永远留在 known 侧。

输出结构（每组）：
  <out>/S1/
    known/images/{train,val}、known/labels/{train,val}
    unknown/images/val、unknown/labels/val
    data_known.yaml、data_unknown.yaml、split_report.md

用法：
  python training\\research\\make_open_set_split.py --data-root training\\datasets\\tea_leaf_disease
  python training\\research\\make_open_set_split.py --data-root ... --splits S1,S5
  python training\\research\\make_open_set_split.py --data-root ... --splits "Leaf rust,Red Spider"
  python training\\research\\make_open_set_split.py --data-root ... --mode list --limit 20
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path

# 项目根目录：training/research/make_open_set_split.py -> 上溯两级
BASE = Path(__file__).resolve().parents[2]

from common import load_data_yaml, resolve_split  # noqa: E402

# 6 组 LOCO 的留出类（短名，与 src/vision/detector.py 的 canonical_class 输出一致）
HELDOUT = OrderedDict([
    ("S1", "Leaf rust"),
    ("S2", "Brown blight"),
    ("S3", "White spot"),
    ("S4", "Black rot"),
    ("S5", "Red Spider"),
    ("S6", "Tea Mosquito bug"),
])

# 权重内长名 -> 系统短名。与 src/vision/detector.py 的 _NAME_ALIASES 保持一致，
# 仅作为"没有 torch / 无法 import src"时的本地兜底。
_NAME_ALIASES = {
    "Black rot of tea": "Black rot",
    "Brown blight of tea": "Brown blight",
    "Leaf rust of tea": "Leaf rust",
    "Red Spider infested tea leaf": "Red Spider",
    "Tea Mosquito bug infested leaf": "Tea Mosquito bug",
    "Tea leaf": "Healthy",
    "White spot of tea": "White spot",
}

SPLIT_SOURCE_ALIASES = ("val", "valid", "validation", "test")


# ============================================================
# 类别名归一化
# ============================================================
def _load_canonical() -> "callable":
    """优先复用 src.vision.detector.canonical_class；失败时回退到本地映射表。

    脚本可能在没有 torch 的环境里只做划分，因此 import 失败不是错误。
    """
    try:
        if str(BASE) not in sys.path:
            sys.path.insert(0, str(BASE))
        from src.vision.detector import canonical_class

        return canonical_class
    except Exception:  # noqa: BLE001 - 依赖缺失时用本地表兜底
        return lambda name: _NAME_ALIASES.get(str(name), str(name)).strip()


canonical_class = _load_canonical()


def canon(name: str) -> str:
    """短名归一化：去掉首尾空白，长名走 alias 表。"""
    return canonical_class(str(name).strip())


# ============================================================
# 命令行
# ============================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成开放集（S1~S6 留一类）划分：known / unknown 目录 + data.yaml")
    parser.add_argument("--data-root", required=True,
                        help="数据集根目录（含 data.yaml 与 train/valid/test 的 images/labels）")
    parser.add_argument("--out", default="research_out/open_set",
                        help="输出目录，默认 research_out/open_set")
    parser.add_argument("--splits", default="all",
                        help="all（默认）或逗号列表，如 S1,S5，也接受类别短名（Leaf rust）")
    parser.add_argument("--exclude", default="disease",
                        help="要从 known / unknown 中整体排除的类，默认 disease（逗号分隔）")
    parser.add_argument("--mode", choices=("copy", "hardlink", "list"), default="copy",
                        help="copy（默认）| hardlink（失败自动回退 copy）| list（只统计不写文件）")
    parser.add_argument("--limit", type=int, default=0,
                        help="每类最多取多少张图（便于快速验证），0 表示不限制")
    parser.add_argument("--seed", type=int, default=42, help="抽样与打乱种子，默认 42")
    return parser.parse_args(argv)


def parse_names(text: str) -> list[str]:
    """把逗号/中文逗号分隔的类别列表归一化为短名，保持顺序去重。"""
    items = [x.strip() for x in str(text).replace("，", ",").split(",") if x.strip()]
    out: list[str] = []
    for item in items:
        name = canon(item)
        if name and name not in out:
            out.append(name)
    return out


def resolve_selection(text: str, dataset_names: list[str]) -> list[str]:
    """--splits 解析：all / S1..S6 / 类别短名（大小写不敏感），按数据集实际类别过滤。"""
    tokens = [x.strip() for x in str(text).replace("，", ",").split(",") if x.strip()]
    if len(tokens) == 1 and tokens[0].lower() == "all":
        selected = list(HELDOUT)
    else:
        selected = []
        for token in tokens:
            key = token.upper()
            if key in HELDOUT:
                if key not in selected:
                    selected.append(key)
                continue
            wanted = canon(token).lower()
            matches = [sid for sid, cls in HELDOUT.items() if cls.lower() == wanted]
            if not matches:
                raise SystemExit(
                    f"[错误] 无法识别的划分 '{token}'：只接受 all、S1~S6 或类别短名"
                    f"（{'、'.join(HELDOUT.values())}）")
            if matches[0] not in selected:
                selected.append(matches[0])
    known = {canon(n) for n in dataset_names}
    kept = [s for s in selected if HELDOUT[s] in known]
    skipped = [s for s in selected if s not in kept]
    if skipped:
        print(f"[警告] 数据集缺少以下留出类，已跳过："
              f"{'、'.join(f'{s}={HELDOUT[s]}' for s in skipped)}")
    return kept


# ============================================================
# 标签读取与目录扫描
# ============================================================
def read_instances(label_path: Path, nc: int) -> list[tuple[int, float, float, float, float]]:
    """读取 YOLO 标签，返回 (类别号, cx, cy, w, h) 列表；非法行跳过。"""
    insts: list[tuple[int, float, float, float, float]] = []
    if not label_path.is_file():
        return insts
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
        insts.append((cid, cx, cy, bw, bh))
    return insts


def split_source_dir(cfg: dict, yaml_path: Path) -> tuple[str, list[Path], list[Path], Path]:
    """定位验证集来源：val -> valid -> test，返回 (split 名, 图片, 标签, split 根目录)。"""
    errors: list[str] = []
    for split in SPLIT_SOURCE_ALIASES:
        try:
            images, labels, root = resolve_split(cfg, yaml_path, split)
            return split, images, labels, root
        except FileNotFoundError as e:
            errors.append(str(e))
    raise SystemExit("[错误] 找不到验证集（已尝试 val / valid / test）：\n  " + "\n  ".join(errors))


def collect_samples(images: list[Path], labels: list[Path], images_root: Path,
                    names: list[str], excluded: set[str]) -> list[dict]:
    """逐图收集：相对路径、有效实例、是否含留出类（留出类由调用方判定）。

    这里只做"排除类过滤"与原始实例保存，留出类判定放在 build_group 里，
    因为不同组留出的类不同。
    """
    nc = len(names)
    excluded_ids = {i for i, n in enumerate(names) if canon(n) in excluded}
    samples: list[dict] = []
    for img, lbl in zip(images, labels):
        try:
            rel = img.relative_to(images_root)
        except ValueError:
            rel = Path(img.name)
        raw = read_instances(lbl, nc)
        kept = [t for t in raw if t[0] not in excluded_ids]
        samples.append({
            "img": img,
            "label": lbl,
            "rel": rel,
            "raw": raw,
            "instances": kept,          # 已剔除 --exclude 类
            "dropped_excluded": len(raw) - len(kept),
            "classes": [canon(names[t[0]]) for t in kept],
            "has_label": lbl.is_file(),
        })
    return samples


# ============================================================
# 抽样（--limit）
# ============================================================
def apply_limit(samples: list[dict], key_fn, limit: int, rng: random.Random
                ) -> tuple[list[dict], bool]:
    """每类最多 limit 张图。返回 (保留样本, 是否触发限制)。

    一张图可能含多个类：只有它包含的所有类都超配额时才丢弃，
    否则保留并给这些类各记一次计数（与"每类最多 N 张"的直觉一致）。
    保留顺序仍按原始顺序，便于复现。
    """
    if limit <= 0 or not samples:
        return samples, False
    order = list(range(len(samples)))
    rng.shuffle(order)
    counts: Counter = Counter()
    keep_idx: set[int] = set()
    triggered = False
    for idx in order:
        keys = key_fn(samples[idx])
        if not keys:
            # 无实例的背景图不占任何类的配额，也不应被丢弃
            keep_idx.add(idx)
            continue
        if all(counts[k] >= limit for k in keys):
            triggered = True
            continue
        keep_idx.add(idx)
        for k in keys:
            counts[k] += 1
    kept = [s for i, s in enumerate(samples) if i in keep_idx]
    return kept, triggered


# ============================================================
# 单组划分
# ============================================================
def build_group(split_id: str, held_name: str, names: list[str],
                train_samples: list[dict], val_samples: list[dict],
                excluded: set[str], limit: int, seed: int) -> dict:
    """计算一组 known / unknown 的样本、标签重映射与统计（不落盘）。"""
    known_names = [canon(n) for n in names if canon(n) not in excluded and canon(n) != held_name]
    orig_to_known: dict[int, int] = {}
    for cid, name in enumerate(names):
        short = canon(name)
        if short in excluded or short == held_name:
            continue
        orig_to_known[cid] = known_names.index(short)

    def is_held(sample: dict) -> bool:
        return any(canon(names[t[0]]) == held_name for t in sample["raw"])

    def remap(sample: dict) -> list[tuple[int, float, float, float, float]]:
        """known 标签：过滤排除类并按已知类顺序重映射。"""
        return [(orig_to_known[t[0]], *t[1:]) for t in sample["instances"] if t[0] in orig_to_known]

    # ---- train：留出类图像整体移除 ----
    train_kept: list[dict] = []
    train_drop_held = 0
    train_drop_excluded = 0
    for s in train_samples:
        if is_held(s):
            train_drop_held += 1
            continue
        if s["raw"] and not s["instances"]:
            train_drop_excluded += 1
            continue
        s = dict(s)
        s["out_label"] = remap(s)
        train_kept.append(s)

    # ---- val：含留出类的整图移到 unknown，其余留 known ----
    val_known: list[dict] = []
    val_unknown: list[dict] = []
    val_drop_excluded = 0
    mixed_unknown = 0
    for s in val_samples:
        if is_held(s):
            s = dict(s)
            held_boxes = [t for t in s["instances"] if canon(names[t[0]]) == held_name]
            other_boxes = len(s["instances"]) - len(held_boxes)
            if other_boxes:
                mixed_unknown += 1
            # unknown：先剔除 disease 等排除类，其余框统一改写为 class 0
            s["out_label"] = [(0, *t[1:]) for t in s["instances"]]
            if s["out_label"]:
                val_unknown.append(s)
            else:
                val_drop_excluded += 1
            continue
        if s["raw"] and not s["instances"]:
            val_drop_excluded += 1
            continue
        s = dict(s)
        s["out_label"] = remap(s)
        val_known.append(s)

    # ---- --limit：train / known-val / unknown-val 各自独立配额 ----
    rng = random.Random(f"{seed}:{split_id}:train")
    train_kept, limit_train = apply_limit(
        train_kept, lambda s: sorted({t[0] for t in s["out_label"]}), limit, rng)
    rng = random.Random(f"{seed}:{split_id}:val")
    val_known, limit_val = apply_limit(
        val_known, lambda s: sorted({t[0] for t in s["out_label"]}), limit, rng)
    rng = random.Random(f"{seed}:{split_id}:unknown")
    val_unknown, limit_unknown = apply_limit(
        val_unknown, lambda s: ["unknown"], limit, rng)

    # ---- 统计 ----
    train_counts: Counter = Counter()
    for s in train_kept:
        train_counts.update(t[0] for t in s["out_label"])
    val_counts: Counter = Counter()
    for s in val_known:
        val_counts.update(t[0] for t in s["out_label"])
    unknown_boxes = sum(len(s["out_label"]) for s in val_unknown)

    checks = {
        "留出类不在 known/train": all(canon(names[cid]) != held_name for s in train_kept
                                    for cid, *_ in s["instances"]),
        "known 类别号连续无空洞": set(train_counts) | set(val_counts) <= set(range(len(known_names))),
        "unknown 标签全为 0": all(t[0] == 0 for s in val_unknown for t in s["out_label"]),
    }

    return {
        "split_id": split_id,
        "held": held_name,
        "known_names": known_names,
        "train": train_kept,
        "val_known": val_known,
        "val_unknown": val_unknown,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "unknown_boxes": unknown_boxes,
        "orig_to_known": orig_to_known,
        "stats": {
            "train_images": len(train_kept),
            "known_val_images": len(val_known),
            "unknown_val_images": len(val_unknown),
            "unknown_boxes": unknown_boxes,
            "train_drop_held_images": train_drop_held,
            "train_drop_excluded_images": train_drop_excluded,
            "val_drop_excluded_images": val_drop_excluded,
            "mixed_unknown_images": mixed_unknown,
            "limit": limit,
            "limit_triggered": {
                "train": limit_train, "known_val": limit_val, "unknown_val": limit_unknown,
            },
        },
        "checks": checks,
    }


# ============================================================
# 落盘
# ============================================================
def _place(src: Path, dst: Path, mode: str, fallback: Counter) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            fallback["hardlink"] += 1
    shutil.copy2(src, dst)


def _write_label(path: Path, instances: list[tuple[int, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cid, cx, cy, w, h in instances]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_group(group: dict, group_dir: Path, mode: str, fallback: Counter) -> None:
    """把一组的 known/unknown 图像与标签写到 group_dir。"""
    known = group_dir / "known"
    unknown = group_dir / "unknown"
    for sub in (known, unknown):
        if sub.exists():
            shutil.rmtree(sub)

    targets = [
        (group["train"], known / "images" / "train", known / "labels" / "train"),
        (group["val_known"], known / "images" / "val", known / "labels" / "val"),
        (group["val_unknown"], unknown / "images" / "val", unknown / "labels" / "val"),
    ]
    for samples, img_dir, lbl_dir in targets:
        for s in samples:
            dst_img = img_dir / s["rel"]
            _place(s["img"], dst_img, mode, fallback)
            _write_label(lbl_dir / s["rel"].with_suffix(".txt"), s["out_label"])

    write_yaml(group_dir / "data_known.yaml", known, {"train": "images/train", "val": "images/val"},
               group["known_names"])
    write_yaml(group_dir / "data_unknown.yaml", unknown, {"val": "images/val"}, ["unknown"])


def write_yaml(path: Path, root: Path, splits: dict[str, str], names: list[str]) -> None:
    lines = ["# 由 training/research/make_open_set_split.py 自动生成，请勿手工修改", ""]
    lines.append(f"path: {Path(root).resolve().as_posix()}")
    for key, value in splits.items():
        lines.append(f"{key}: {value}")
    lines += ["", "names:"]
    for i, name in enumerate(names):
        lines.append(f"  {i}: {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# 报告
# ============================================================
def build_report(group: dict, source_desc: dict) -> str:
    st = group["stats"]
    split_id = group["split_id"]
    lines = [
        f"# 开放集划分报告 {split_id}（leave-one-class-out）",
        "",
        f"- 留出类（unknown）：**{group['held']}**",
        f"- 已知类别（{len(group['known_names'])}）："
        + "、".join(group["known_names"]),
        f"- 排除类别：{'、'.join(source_desc['excluded']) or '无'}"
        f"（既不进 known 也不进 unknown）",
        f"- 数据根目录：`{source_desc['data_root']}`",
        f"- 验证集来源：`{source_desc['val_split']}` ｜ seed={source_desc['seed']}"
        f" ｜ mode={source_desc['mode']} ｜ limit={source_desc['limit']}",
        f"- 生成时间：{source_desc['generated_at']}",
        "",
        "## 1. 概况",
        "",
        "| 集合 | 图像数 | 说明 |",
        "| --- | ---: | --- |",
        f"| known/train | {st['train_images']} | 已彻底移除留出类图像 {st['train_drop_held_images']} 张 |",
        f"| known/val | {st['known_val_images']} | 仅含已知类 |",
        f"| unknown/val | {st['unknown_val_images']} | 含留出类图像，标签统一改写为 unknown |",
        "",
        "## 2. 每类实例数",
        "",
        "| 类别号 | 类别 | known/train | known/val | unknown/val |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for cid, name in enumerate(group["known_names"]):
        lines.append(
            f"| {cid} | {name} | {group['train_counts'].get(cid, 0)} | "
            f"{group['val_counts'].get(cid, 0)} | 0 |")
    lines.append(f"| 0 | unknown | 0 | 0 | {st['unknown_boxes']} |")

    lines += [
        "",
        "## 3. 规则校验",
        "",
    ]
    for name, ok in group["checks"].items():
        lines.append(f"- {'✅' if ok else '❌'} {name}")
    lines += [
        "",
        "## 4. 统计与提示",
        "",
        f"- --limit 是否触发：train={st['limit_triggered']['train']}，"
        f"known_val={st['limit_triggered']['known_val']}，"
        f"unknown_val={st['limit_triggered']['unknown_val']}",
        f"- train 中因只含排除类（如 disease）被丢弃：{st['train_drop_excluded_images']} 张",
        f"- val 中因只含排除类被丢弃：{st['val_drop_excluded_images']} 张",
        f"- unknown 中同时含已知类框的图像：{st['mixed_unknown_images']} 张"
        "（这些框按规则一并改写为 unknown，解读 unknown 召回时需注意）",
        "",
        "> 说明：unknown 目录的标签全部为 class 0，评估代码不区分未知病害的具体病种；"
        "known 标签的类别号已按去掉留出类与排除类后的顺序重映射，0..nc-1 连续。",
        "",
    ]
    return "\n".join(lines)


# ============================================================
# 主流程
# ============================================================
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        print(f"[错误] 数据集目录不存在：{data_root}")
        return 1
    if args.limit < 0:
        print("[错误] --limit 必须 ≥ 0")
        return 1

    yaml_candidates = [data_root / "data.yaml", data_root / "data_norm.yaml"]
    yaml_path = next((p for p in yaml_candidates if p.is_file()), None)
    if yaml_path is None:
        print(f"[错误] 在 {data_root} 下找不到 data.yaml / data_norm.yaml")
        return 1
    try:
        cfg, raw_names, yaml_path = load_data_yaml(yaml_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"[错误] {e}")
        return 1
    if not raw_names:
        print("[错误] data.yaml 缺少 names，无法确定类别；请先规范化数据集配置")
        return 1
    names = [canon(n) for n in raw_names]
    excluded = set(parse_names(args.exclude))
    missing_excluded = sorted(excluded - set(names))
    if missing_excluded:
        print(f"[警告] --exclude 中的类别不在数据集里：{'、'.join(missing_excluded)}")

    try:
        val_split, val_images, val_labels, val_root = split_source_dir(cfg, yaml_path)
        train_images, train_labels, train_root = resolve_split(cfg, yaml_path, "train")
    except FileNotFoundError as e:
        print(f"[错误] 无法定位 train：{e}")
        return 1
    except SystemExit as e:
        print(e)
        return 1

    selection = resolve_selection(args.splits, names)
    if not selection:
        print(f"[错误] 选中的划分在数据集中没有对应类别（数据集类别：{'、'.join(names)}）")
        return 1

    train_samples = collect_samples(train_images, train_labels,
                                    train_root / "images", names, excluded)
    val_samples = collect_samples(val_images, val_labels,
                                  val_root / "images", names, excluded)
    print(f"[信息] 数据根目录：{data_root}")
    print(f"[信息] 类别 {len(names)} 个：{'、'.join(names)} ｜ 排除：{'、'.join(sorted(excluded)) or '无'}")
    print(f"[信息] train {len(train_samples)} 张 ｜ {val_split} {len(val_samples)} 张 "
          f"｜ 选中划分：{'、'.join(selection)} ｜ mode={args.mode}")

    out_root = Path(args.out)
    out_resolved = out_root.resolve()
    if args.mode != "list" and (data_root.resolve() == out_resolved
                                or data_root.resolve() in out_resolved.parents):
        print("[错误] --out 不能位于 --data-root 内部（会污染原始数据集）")
        return 1
    if args.mode != "list":
        out_root.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []
    failed: list[str] = []
    for split_id in selection:
        held = HELDOUT[split_id]
        group = build_group(split_id, held, names, train_samples, val_samples,
                            excluded, args.limit, args.seed)
        st = group["stats"]
        print(f"\n[{split_id}] 留出 {held}：known/train {st['train_images']} 张"
              f"（移除留出类 {st['train_drop_held_images']} 张）｜ "
              f"known/val {st['known_val_images']} 张 ｜ "
              f"unknown/val {st['unknown_val_images']} 张 / {st['unknown_boxes']} 框")
        if st["limit_triggered"]["train"] or st["limit_triggered"]["known_val"] \
                or st["limit_triggered"]["unknown_val"]:
            print(f"  [提示] --limit={args.limit} 已触发")

        if not group["known_names"]:
            print(f"  [跳过] {split_id}：去掉留出类后没有剩余已知类")
            failed.append(split_id)
            continue
        if not group["val_unknown"]:
            print(f"  [警告] {split_id}：验证集中没有留出类图像，unknown 集为空")

        if args.mode == "list":
            print(f"  [list] 不写文件；known/train 实例 {sum(group['train_counts'].values())} 个，"
                  f"known/val 实例 {sum(group['val_counts'].values())} 个，"
                  f"unknown 实例 {st['unknown_boxes']} 个")
            generated.append(split_id)
            continue

        group_dir = out_root / split_id
        group_dir.mkdir(parents=True, exist_ok=True)
        fallback: Counter = Counter()
        write_group(group, group_dir, args.mode, fallback)
        report = build_report(group, {
            "excluded": sorted(excluded),
            "data_root": str(data_root.resolve()),
            "val_split": val_split,
            "seed": args.seed,
            "mode": args.mode,
            "limit": args.limit,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        })
        (group_dir / "split_report.md").write_text(report, encoding="utf-8")
        if fallback["hardlink"]:
            print(f"  [提示] {fallback['hardlink']} 个文件硬链接失败，已回退为复制")
        print(f"  [输出] {group_dir / 'data_known.yaml'}")
        print(f"  [输出] {group_dir / 'data_unknown.yaml'}")
        print(f"  [输出] {group_dir / 'split_report.md'}")
        generated.append(split_id)

    if not generated:
        print("\n[错误] 没有任何划分成功生成")
        return 1
    print(f"\n完成：{len(generated)} 组（{'、'.join(generated)}）"
          + (f"，失败 {len(failed)} 组（{'、'.join(failed)}）" if failed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
