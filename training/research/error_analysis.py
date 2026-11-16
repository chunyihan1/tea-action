"""FP / FN 定性分析：导出误报与漏检裁剪图，并按类别、尺寸、置信度统计。

为什么需要：
  mAP 是聚合指标，审稿人真正关心"错在哪"。本脚本把每个 FP / FN 落到具体图片、
  具体类别、具体尺寸上，并给出"最容易被误判成什么"的混淆来源统计，直接支撑论文
  的误差分析小节与图 5（FP/FN 示例）。

实现要点：
  - 匹配逻辑复用 common.py，与 stats_eval.py 口径完全一致（同类贪心 + IoU ≥ 阈值）；
  - FP 的 iou 取"与任意真值框的最大 IoU"（跨类别也算），用于判断误报压在了哪个类别上，
    这正是"最容易被误判成什么"的依据；
  - FN 没有置信度，表中填"与该漏检真值重叠度最高的预测框置信度"，无重叠则为空——
    这样按置信度分箱时，能区分"完全没检出"与"检出但框不准/类别错"；
  - 裁剪范围向外扩 20% 并夹紧到图像边界，避免坐标越界；
  - 无检测或无 GT 时仍输出只有表头的 csv 与全 0 的 md，不抛异常。

用法：
  python training\\research\\error_analysis.py --model data\\models\\tea_yolo.pt --data <data.yaml>
  python training\\research\\error_analysis.py --model runs\\best.pt --data <data.yaml> --max-samples 100
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

from common import (best_iou_any, class_name, get_preds, iou_xyxy,
                    load_data_yaml, match_image, read_gt, resolve_split)

CSV_FIELDS = ["type", "image", "class_id", "class_name", "conf",
              "iou", "area_ratio", "w", "h", "size_bin"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FP/FN 误差分析（裁剪图 + 分类统计）")
    parser.add_argument("--model", required=True, help="模型权重路径")
    parser.add_argument("--data", required=True, help="data.yaml 路径")
    parser.add_argument("--split", default="val", help="评估划分，默认 val")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS 与 TP 匹配的 IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理分辨率")
    parser.add_argument("--device", default="", help="cpu / 0 / 留空自动")
    parser.add_argument("--max-samples", type=int, default=60,
                        help="FP、FN 各自最多导出的裁剪图数量")
    parser.add_argument("--out", default="research_out/errors", help="输出目录")
    return parser.parse_args()


def size_bin(area: float) -> str:
    """COCO 口径：<32² 为 small，<96² 为 medium，其余 large。"""
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def safe_name(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in text]
    return ("".join(keep) or "unknown")[:40]


def expand_box(box: tuple[float, float, float, float], ratio: float,
               width: int, height: int) -> tuple[int, int, int, int]:
    """按比例向外扩框，并夹紧到 [0, w] x [0, h]。"""
    x1, y1, x2, y2 = box
    dx = (x2 - x1) * ratio
    dy = (y2 - y1) * ratio
    left = max(0, int(round(x1 - dx)))
    top = max(0, int(round(y1 - dy)))
    right = min(width, int(round(x2 + dx)))
    bottom = min(height, int(round(y2 + dy)))
    if right - left <= 0 or bottom - top <= 0:  # 退化框兜底
        left, top = max(0, left - 1), max(0, top - 1)
        right, bottom = min(width, right + 1), min(height, bottom + 1)
    return left, top, right, bottom


def save_crop(image_path: Path, box: tuple[float, float, float, float],
              out_path: Path, width: int, height: int) -> bool:
    try:
        with Image.open(image_path) as im:
            crop = im.convert("RGB").crop(expand_box(box, 0.2, width, height))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(out_path, quality=90)
        return True
    except Exception as e:  # noqa: BLE001 - 单张裁剪失败不应中断统计
        print(f"  [警告] 裁剪失败 {image_path.name}：{' '.join(str(e).split())[:100]}")
        return False


def collect(model_path: str, images: list[Path], labels: list[Path],
            names: list[str], args: argparse.Namespace) -> tuple[list[dict], int, int]:
    """逐图推理并收集 FP / FN 记录，返回 (记录列表, GT 总数, 预测总数)。"""
    from ultralytics import YOLO

    model = YOLO(model_path)
    stream = model.predict(
        source=[str(p) for p in images],
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device or None,
        verbose=False,
        stream=True,
    )
    records: list[dict] = []
    total_gt = 0
    total_pred = 0

    for i, (img_path, label_path, res) in enumerate(zip(images, labels, stream)):
        height, width = res.orig_shape[:2]
        area_img = float(width * height) or 1.0
        gts = read_gt(label_path, width, height, len(names))
        preds = get_preds(res)
        matched = match_image(preds, gts, args.iou)
        total_gt += len(gts)
        total_pred += len(preds)

        # ---- FP：被误报的框 ----
        for pred, _same_cls_iou, _gt_idx in matched["fp"]:
            x1, y1, x2, y2 = pred["box"]
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            any_iou, gt_j = best_iou_any(pred["box"], gts)
            gt_cls = gts[gt_j]["cls"] if gt_j >= 0 else -1
            records.append({
                "type": "FP",
                "image": str(img_path),
                "class_id": pred["cls"],
                "class_name": class_name(names, pred["cls"]),
                "conf": round(pred["conf"], 4),
                "iou": round(any_iou, 4),
                "area_ratio": round((w * h) / area_img, 6),
                "w": round(w, 1),
                "h": round(h, 1),
                "size_bin": size_bin(w * h),
                "box": pred["box"],
                "gt_cls": gt_cls,
                "width": width,
                "height": height,
            })

        # ---- FN：漏掉的真值框 ----
        for j in matched["fn"]:
            gt = gts[j]
            x1, y1, x2, y2 = gt["box"]
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            near_conf, near_iou, near_cls = None, 0.0, -1
            for pred in preds:  # 找与该真值重叠最多的预测，解释"为什么漏"
                value = iou_xyxy(pred["box"], gt["box"])
                if value > near_iou:
                    near_iou, near_conf, near_cls = value, pred["conf"], pred["cls"]
            records.append({
                "type": "FN",
                "image": str(img_path),
                "class_id": gt["cls"],
                "class_name": class_name(names, gt["cls"]),
                "conf": round(near_conf, 4) if near_conf is not None else "",
                "iou": round(near_iou, 4),
                "area_ratio": round((w * h) / area_img, 6),
                "w": round(w, 1),
                "h": round(h, 1),
                "size_bin": size_bin(w * h),
                "box": gt["box"],
                "gt_cls": near_cls,
                "width": width,
                "height": height,
            })

        if (i + 1) % 200 == 0:
            print(f"  已处理 {i + 1}/{len(images)} 张……")
    return records, total_gt, total_pred


def export_crops(records: list[dict], out_dir: Path, max_samples: int) -> dict:
    """按类型导出裁剪图，每种类型最多 max_samples 张（高置信度误报优先）。"""
    exported = {"FP": 0, "FN": 0}
    for kind in ("FP", "FN"):
        items = [r for r in records if r["type"] == kind]
        # FP 按置信度降序（高置信度误报最有害），FN 按面积升序（小目标漏检最常见）
        if kind == "FP":
            items.sort(key=lambda r: -float(r["conf"] or 0))
        else:
            items.sort(key=lambda r: r["w"] * r["h"])
        for idx, item in enumerate(items):
            if idx >= max_samples:
                break
            conf_txt = f"{float(item['conf']):.2f}" if item["conf"] != "" else "na"
            fname = (f"{kind.lower()}_{safe_name(item['class_name'])}"
                     f"_c{conf_txt}_{safe_name(Path(item['image']).stem)}_{idx:03d}.jpg")
            ok = save_crop(Path(item["image"]), item["box"], out_dir / kind.lower() / fname,
                           item["width"], item["height"])
            exported[kind] += int(ok)
    return exported


def write_csv(path: Path, records: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def _count(records: list[dict], kind: str, pred) -> int:
    return sum(1 for r in records if r["type"] == kind and pred(r))


def write_summary(path: Path, records: list[dict], names: list[str], total_gt: int,
                  total_pred: int, n_images: int, exported: dict,
                  args: argparse.Namespace, elapsed: float) -> None:
    fp_all = [r for r in records if r["type"] == "FP"]
    fn_all = [r for r in records if r["type"] == "FN"]
    lines = [
        "# FP / FN 误差分析",
        "",
        f"- 模型：`{args.model}`",
        f"- data.yaml：`{args.data}` ｜ split={args.split}",
        f"- 推理设置：imgsz={args.imgsz}，conf={args.conf}，IoU={args.iou}，device={args.device or 'auto'}",
        f"- 图片数：{n_images} ｜ GT 框：{total_gt} ｜ 预测框：{total_pred}",
        f"- 导出裁剪图：FP {exported['FP']} 张、FN {exported['FN']} 张"
        f"（每种类型上限 {args.max_samples}）",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')} ｜ 耗时 {elapsed:.1f}s",
        "",
        f"总计：**FP {len(fp_all)} 个，FN {len(fn_all)} 个**"
        f"（明细见 `errors.csv`，裁剪图见 `fp/`、`fn/`）。",
        "",
        "## 1. 按类别统计",
        "",
        "| 类别 | FP（误报） | FN（漏检） |",
        "| --- | ---: | ---: |",
    ]
    if not records:
        lines.append("| （无记录） | 0 | 0 |")
    else:
        for cid in sorted({r["class_id"] for r in records}):
            lines.append(
                f"| {cid} {class_name(names, cid)} | "
                f"{_count(fp_all, 'FP', lambda r: r['class_id'] == cid)} | "
                f"{_count(fn_all, 'FN', lambda r: r['class_id'] == cid)} |"
            )
    lines += ["", "## 2. 按目标尺寸（COCO 口径）", "",
              "| 尺寸 | 判定条件 | FP | FN |", "| --- | --- | ---: | ---: |"]
    for bin_name, cond in (("small", "area < 32²"), ("medium", "32² ≤ area < 96²"),
                           ("large", "area ≥ 96²")):
        lines.append(
            f"| {bin_name} | {cond} | "
            f"{_count(fp_all, 'FP', lambda r, b=bin_name: r['size_bin'] == b)} | "
            f"{_count(fn_all, 'FN', lambda r, b=bin_name: r['size_bin'] == b)} |"
        )
    lines += ["", "## 3. 按置信度分箱", "",
              "| 置信度 | FP | FN |", "| --- | ---: | ---: |"]
    low_label = f"{args.conf:.2f}-0.40"
    fp_low = _count(fp_all, "FP", lambda r: float(r["conf"] or 0) < 0.4)
    fn_low = _count(fn_all, "FN", lambda r: r["conf"] != "" and float(r["conf"]) < 0.4)
    fp_mid = _count(fp_all, "FP", lambda r: 0.4 <= float(r["conf"] or 0) < 0.6)
    fn_mid = _count(fn_all, "FN", lambda r: r["conf"] != "" and 0.4 <= float(r["conf"]) < 0.6)
    fp_high = _count(fp_all, "FP", lambda r: float(r["conf"] or 0) >= 0.6)
    fn_high = _count(fn_all, "FN", lambda r: r["conf"] != "" and float(r["conf"]) >= 0.6)
    fn_none = _count(fn_all, "FN", lambda r: r["conf"] == "")
    for label, fp_n, fn_n in ((low_label, fp_low, fn_low),
                              ("0.40-0.60", fp_mid, fn_mid),
                              ("0.60+", fp_high, fn_high),
                              ("无重叠预测", 0, fn_none)):
        lines.append(f"| {label} | {fp_n} | {fn_n} |")
    lines += ["", "表注：FN 的置信度为“与该漏检真值重叠最多的预测框”的置信度；"
              "若完全没有重叠预测则记为“无重叠预测”，说明该目标被彻底漏掉。", ""]

    # 4. 误报来源（最容易被误判成什么）
    lines += ["## 4. 误报来源（FP 压在了哪个类别的真值上）", ""]
    pairs: dict[tuple[str, str], int] = {}
    for r in fp_all:
        if r["iou"] > 0.1 and r["gt_cls"] >= 0:
            key = (r["class_name"], class_name(names, r["gt_cls"]))
            pairs[key] = pairs.get(key, 0) + 1
    if pairs:
        lines += ["| 预测类别 | 实际（真值）类别 | 次数 |", "| --- | --- | ---: |"]
        for (pred_name, gt_name), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:10]:
            lines.append(f"| {pred_name} | {gt_name} | {n} |")
        top_pred, top_gt = max(pairs.items(), key=lambda kv: kv[1])[0]
        lines += ["", f"最容易被误判成：**{top_gt}** 被预测为 **{top_pred}**（共 {max(pairs.values())} 次，"
                      f"IoU > 0.1 的跨类重叠）。", ""]
    else:
        lines += ["没有与真值存在重叠（IoU > 0.1）的误报，说明 FP 主要来自背景误检"
                  "（如叶片纹理、光斑、土壤），而非类别混淆。", ""]

    # 5. 一句话结论
    if records:
        small_fn = _count(fn_all, "FN", lambda r: r["size_bin"] == "small")
        conclusion = (f"共 {len(fp_all)} 个误报、{len(fn_all)} 个漏检；"
                      f"小目标漏检 {small_fn} 个，"
                      f"{'建议优先补小目标数据或提高输入分辨率' if small_fn > len(fn_all) / 2 else '漏检未集中在单一尺寸区间'}。")
    else:
        conclusion = "本划分上没有任何 FP/FN（或全部正确匹配），无需误差分析。"
    lines += ["## 5. 结论", "", f"> {conclusion}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.max_samples < 0:
        print("[错误] --max-samples 不能为负")
        sys.exit(1)

    try:
        cfg, names, yaml_path = load_data_yaml(args.data)
    except (FileNotFoundError, ValueError) as e:
        print(f"[错误] {e}")
        sys.exit(1)
    try:
        images, labels, split_root = resolve_split(cfg, yaml_path, args.split)
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        print("       请检查 data.yaml 的 path / split 字段，或用 --split 指定正确的划分。")
        sys.exit(1)

    out_dir = Path(args.out)
    for sub in ("fp", "fn"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    print(f"[信息] 数据：{yaml_path} ｜ split={args.split} ｜ 目录：{split_root}")
    print(f"[信息] 图片 {len(images)} 张 ｜ 类别 {len(names)} 个")
    print(f"[信息] imgsz={args.imgsz} conf={args.conf} iou={args.iou} "
          f"max-samples={args.max_samples}")

    t0 = time.time()
    try:
        records, total_gt, total_pred = collect(args.model, images, labels, names, args)
    except Exception as e:  # noqa: BLE001 - 推理失败给出清晰报错
        print(f"[错误] 推理失败：{' '.join(str(e).split())[:300]}")
        sys.exit(1)
    elapsed = time.time() - t0

    exported = export_crops(records, out_dir, args.max_samples)
    write_csv(out_dir / "errors.csv", records)
    write_summary(out_dir / "error_summary.md", records, names, total_gt, total_pred,
                  len(images), exported, args, elapsed)

    fp_n = sum(1 for r in records if r["type"] == "FP")
    fn_n = len(records) - fp_n
    print(f"[完成] FP {fp_n}，FN {fn_n}，导出裁剪图 FP {exported['FP']} / FN {exported['FN']}")
    print(f"[输出] {out_dir / 'errors.csv'}")
    print(f"[输出] {out_dir / 'error_summary.md'}")
    print(f"[输出] {out_dir / 'fp'}  |  {out_dir / 'fn'}")


if __name__ == "__main__":
    main()
