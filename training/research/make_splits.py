"""按"组"（茶园 / 拍摄批次）切分 train/val/test，杜绝随机切分的数据泄漏。

为什么需要：
  同一片叶子连拍、同一批次同光照的多张图若被随机拆到 train 与 val，
  验证集里就有训练集的近重复，指标虚高（audit_dataset.py 的泄漏率就是证据）。
  Q1 审稿必问"划分键是什么、有没有跨 split 泄漏"，本脚本把划分键固化成
  可复查的 txt 清单与 json，正文可直接引用组数与各组归属。

设计要点：
  - 组名由 --group-regex 从**文件名**提取，默认 `^(.*?)[_-]`（g1_01.jpg -> g1）；
    提取不到时回退为文件名本身，并统计回退数量写进报告；
  - 以组为单位整体分配，分配后校验"任一图片不跨 split""各组互斥"；
  - 组数 < 3 时自动降级（2 组 -> 只切 train/val；1 组 -> 只有 train），
    在报告与终端明确提示，绝不静默产出空文件；
  - 输出绝对路径清单，Ultralytics 的 data.yaml 可直接写 `train: <out>/train.txt`。

用法：
  python training\\research\\make_splits.py --images training\\datasets\\tea_leaf_disease\\train\\images ^
      --labels training\\datasets\\tea_leaf_disease\\train\\labels
  python training\\research\\make_splits.py --images D:\\tea_raw --group-regex "^(\\w+?)_"
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path

from common import IMG_EXTS

SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按组切分 train/val/test（防数据泄漏）")
    parser.add_argument("--images", required=True, help="图片目录（递归收集）")
    parser.add_argument("--labels", default=None,
                        help="标签目录（递归；可用镜像路径对应图片，留空则只切图）")
    parser.add_argument("--group-regex", default=r"^(.*?)[_-]",
                        help=r"从文件名提取组名的正则（默认 ^(.*?)[_-]）")
    parser.add_argument("--ratios", default="0.7,0.2,0.1", help="train,val,test 比例")
    parser.add_argument("--seed", type=int, default=42, help="打乱组顺序的随机种子")
    parser.add_argument("--out", default="research_out/splits", help="输出目录")
    return parser.parse_args()


def parse_ratios(text: str) -> list[float]:
    parts = [x.strip() for x in text.replace("，", ",").split(",") if x.strip()]
    try:
        ratios = [float(x) for x in parts]
    except ValueError as e:
        raise SystemExit(f"[错误] --ratios 需要 3 个数字，收到：{text}") from e
    if len(ratios) != 3 or any(r < 0 for r in ratios) or sum(ratios) <= 0:
        raise SystemExit(f"[错误] --ratios 需要 3 个非负且和大于 0 的数字，收到：{text}")
    total = sum(ratios)
    return [r / total for r in ratios]


# ============================================================
# 扫描与分组
# ============================================================
def collect_images(root: Path) -> list[Path]:
    if not root.is_dir():
        raise SystemExit(f"[错误] 图片目录不存在：{root}")
    images = sorted(p for p in root.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"[错误] 目录下没有图片（支持 {', '.join(sorted(IMG_EXTS))}）：{root}")
    return images


def group_of(stem: str, pattern: re.Pattern) -> tuple[str, bool]:
    """返回 (组名, 是否命中正则)；未命中时用整段文件名兜底。"""
    m = pattern.match(stem)
    if m and m.group(1):
        return m.group(1), True
    return stem, False


def build_label_index(labels_root: Path) -> dict[str, Path]:
    """按文件名建立 标签stem -> 路径 的索引，兼容扁平目录与镜像目录。"""
    index: dict[str, Path] = {}
    for p in sorted(labels_root.rglob("*.txt")):
        index.setdefault(p.stem, p)
    return index


def label_for(img: Path, images_root: Path, labels_root: Path | None,
              index: dict[str, Path]) -> Path | None:
    if labels_root is None:
        return None
    try:
        rel = img.relative_to(images_root)
    except ValueError:
        rel = Path(img.name)
    mirrored = labels_root / rel.with_suffix(".txt")
    if mirrored.is_file():
        return mirrored
    return index.get(img.stem)


def read_classes(path: Path | None) -> list[int]:
    """读一行的首个 token 作为类别 id；非法行跳过。"""
    if path is None or not path.is_file():
        return []
    classes: list[int] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            classes.append(int(float(parts[0])))
        except ValueError:
            continue
    return classes


# ============================================================
# 以组为单位分配
# ============================================================
def allocate(groups: dict[str, list[Path]], ratios: list[float], seed: int
             ) -> tuple[dict[str, list[str]], list[str], bool]:
    """把组整体分配给各 split，返回 (split -> 组名列表, 警告, 是否降级)。"""
    warnings: list[str] = []
    names = list(groups)
    k = min(3, len(names))
    degraded = k < 3
    if degraded:
        if len(names) == 2:
            warnings.append("只找到 2 个组，无法三路切分：已降级为 train/val，"
                            "不生成 test.txt（需要独立测试集请补充更多批次）。")
        elif len(names) == 1:
            warnings.append("只找到 1 个组，全部图片进入 train，不生成 val/test；"
                            "该划分无法用于评估。")
        ratios = ratios[:k]
        norm = sum(ratios) or 1.0  # 降级后按剩余 split 重新归一化
        ratios = [r / norm for r in ratios]

    used = list(SPLIT_NAMES[:k])
    rng = random.Random(seed)
    order = names[:]
    rng.shuffle(order)
    order.sort(key=lambda g: -len(groups[g]))  # 大组优先，减少比例偏差

    total = sum(len(v) for v in groups.values())
    targets = {split: ratios[i] * total for i, split in enumerate(used)}
    counts = {split: 0 for split in used}
    assigned: dict[str, list[str]] = {split: [] for split in used}

    for i, g in enumerate(order):
        if i < k:
            split = used[i]
        else:
            split = max(used, key=lambda s: targets[s] - counts[s])
        assigned[split].append(g)
        counts[split] += len(groups[g])

    return assigned, warnings, degraded


# ============================================================
# 输出
# ============================================================
def write_list(path: Path, images: list[Path]) -> None:
    lines = [p.resolve().as_posix() for p in images]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_report(args: argparse.Namespace, ratios: list[float],
                 assigned: dict[str, list[str]], groups: dict[str, list[Path]],
                 split_images: dict[str, list[Path]], class_counts: dict[str, Counter],
                 missing_labels: int, fallback_names: int, warnings: list[str],
                 degraded: bool, checks: list[tuple[str, bool, str]]) -> str:
    total_images = sum(len(v) for v in split_images.values())
    lines = [
        "# 按组切分报告（防数据泄漏）",
        "",
        f"- 图片目录：`{Path(args.images).resolve()}`",
        f"- 标签目录：`{Path(args.labels).resolve()}`" if args.labels
        else "- 标签目录：未提供（只切图，不统计类别实例）",
        f"- 分组正则：`{args.group_regex}`（未命中时用文件名兜底，共 {fallback_names} 张）",
        f"- 目标比例：train/val/test = {ratios[0]:.2f}/{ratios[1]:.2f}/{ratios[2]:.2f}"
        f" ｜ seed={args.seed}",
        f"- 组数：{len(groups)} ｜ 图片总数：{total_images}"
        + (" ｜ **已降级**" if degraded else ""),
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    if warnings:
        lines += ["## 0. 提示", ""] + [f"- {w}" for w in warnings] + [""]

    lines += [
        "## 1. 各 split 概况",
        "",
        "| split | 图片数 | 实际占比 | 组数 | 组名 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for split in SPLIT_NAMES:
        if split not in split_images:
            lines.append(f"| {split} | 未生成 | — | — | — |")
            continue
        n = len(split_images[split])
        share = n / total_images if total_images else 0.0
        names = "、".join(assigned[split]) if assigned[split] else "—"
        lines.append(f"| {split} | {n} | {share:.1%} | {len(assigned[split])} | {names} |")

    if class_counts:
        all_ids = sorted({cid for c in class_counts.values() for cid in c})
        header = "| 类别 id | " + " | ".join(s for s in SPLIT_NAMES if s in class_counts) + " |"
        lines += [
            "",
            "## 2. 各类别实例数",
            "",
            header,
            "| ---: |" + " ---: |" * len(class_counts),
        ]
        for cid in all_ids:
            cells = [str(class_counts[s].get(cid, 0)) for s in SPLIT_NAMES if s in class_counts]
            lines.append(f"| {cid} | " + " | ".join(cells) + " |")
        if missing_labels:
            lines += ["", f"表注：{missing_labels} 张图片没有找到标签文件，按 0 个实例计。"]

    lines += ["", "## 3. 泄漏校验（组是否互斥）", ""]
    for name, ok, detail in checks:
        lines.append(f"- {'✅' if ok else '❌'} {name}：{detail}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ratios = parse_ratios(args.ratios)
    try:
        pattern = re.compile(args.group_regex)
    except re.error as e:
        raise SystemExit(f"[错误] --group-regex 不是合法正则：{e}") from e

    images_root = Path(args.images)
    images = collect_images(images_root)
    labels_root = Path(args.labels) if args.labels else None
    if labels_root is not None and not labels_root.is_dir():
        raise SystemExit(f"[错误] 标签目录不存在：{labels_root}")
    label_index = build_label_index(labels_root) if labels_root else {}

    groups: "OrderedDict[str, list[Path]]" = OrderedDict()
    fallback_names = 0
    label_map: dict[Path, Path | None] = {}
    for img in images:
        name, hit = group_of(img.stem, pattern)
        if not hit:
            fallback_names += 1
        groups.setdefault(name, []).append(img)
        label_map[img] = label_for(img, images_root, labels_root, label_index)

    print(f"[信息] 图片 {len(images)} 张 ｜ 组 {len(groups)} 个 "
          f"（正则未命中回退 {fallback_names} 张）")

    assigned, warnings, degraded = allocate(groups, ratios, args.seed)
    for w in warnings:
        print(f"[警告] {w}")

    split_images = {s: [img for g in gs for img in groups[g]] for s, gs in assigned.items()}
    for s in split_images:
        split_images[s].sort()

    class_counts: dict[str, Counter] = {}
    missing_labels = 0
    if labels_root is not None:
        for s, imgs in split_images.items():
            counter: Counter = Counter()
            for img in imgs:
                lp = label_map[img]
                if lp is None:
                    missing_labels += 1
                    continue
                counter.update(read_classes(lp))
            class_counts[s] = counter

    # 校验：组互斥 + 组全覆盖 + 图片不重复
    all_groups = set(groups)
    seen: set[str] = set()
    overlap: list[str] = []
    for s, gs in assigned.items():
        dup = seen & set(gs)
        overlap.extend(sorted(dup))
        seen |= set(gs)
    img_total = sum(len(v) for v in split_images.values())
    dup_images = img_total - len({img for v in split_images.values() for img in v})
    checks = [
        ("组不跨 split", not overlap,
         "所有组只出现在一个 split" if not overlap else f"重复出现的组：{sorted(set(overlap))}"),
        ("组全覆盖", seen == all_groups,
         f"已分配 {len(seen)}/{len(all_groups)} 组"),
        ("图片不重复", dup_images == 0, f"重复图片数 {dup_images}"),
        ("无空 split", all(len(v) > 0 for v in split_images.values()),
         "；".join(f"{s}={len(v)}" for s, v in split_images.items())),
    ]
    for name, ok, detail in checks:
        print(f"[校验] {'通过' if ok else '失败'} ｜ {name}：{detail}")
    if not all(ok for _, ok, _ in checks):
        print("[错误] 校验未通过，请检查输入目录或分组正则")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in SPLIT_NAMES:
        if s in split_images:
            write_list(out_dir / f"{s}.txt", split_images[s])
        else:
            stale = out_dir / f"{s}.txt"
            if stale.exists():  # 避免上一次运行的旧清单混进本次实验
                stale.unlink()

    report_path = out_dir / "split_report.md"
    report_path.write_text(
        build_report(args, ratios, assigned, groups, split_images, class_counts,
                     missing_labels, fallback_names, warnings, degraded, checks),
        encoding="utf-8")

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "images_root": str(images_root.resolve()),
        "labels_root": str(labels_root.resolve()) if labels_root else None,
        "group_regex": args.group_regex,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "seed": args.seed,
        "degraded": degraded,
        "warnings": warnings,
        "checks": {name: {"ok": ok, "detail": detail} for name, ok, detail in checks},
        "totals": {"images": img_total, "groups": len(groups),
                   "fallback_group_names": fallback_names,
                   "missing_labels": missing_labels},
        "splits": {
            s: {
                "images": len(split_images[s]),
                "groups": assigned[s],
                "class_instances": {str(k): int(v) for k, v in sorted(class_counts[s].items())}
                if class_counts else {},
                "image_list": str((out_dir / f"{s}.txt").resolve()),
            }
            for s in split_images
        },
    }
    (out_dir / "splits.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[输出] {out_dir / 'train.txt'}")
    for s in SPLIT_NAMES[1:]:
        if s in split_images:
            print(f"[输出] {out_dir / f'{s}.txt'}")
    print(f"[输出] {report_path}")
    print(f"[输出] {out_dir / 'splits.json'}")
    print(f"完成：{img_total} 张图 -> "
          + "，".join(f"{s} {len(v)}" for s, v in split_images.items())
          + f"（{len(groups)} 组）")


if __name__ == "__main__":
    main()
