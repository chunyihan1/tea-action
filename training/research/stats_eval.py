"""统计显著性评估：bootstrap 95% 置信区间 + McNemar 配对检验。

为什么需要：
  SCI 期刊不认"单次跑出 mAP=0.812 就宣称更好"。至少要有
  (1) 指标的 95% 置信区间；(2) 两模型差异的显著性检验。
  本脚本用**逐图**TP/FP/FN 做 micro 聚合，再对图片做有放回重采样得到 CI，
  并对两个模型构造 2x2 配对表做 McNemar 检验。

实现要点：
  - 匹配口径与 error_analysis.py 共用 common.py：同类、置信度降序贪心、IoU ≥ --iou 记 TP；
  - bootstrap 用 np.random.multinomial 一次抽取"每张图被抽中几次"，等价于有放回重采样，
    但内存与耗时都远小于逐次 np.random.choice（固定 seed，结果可复现）；
  - McNemar 用连续性校正 χ² = (|b-c|-1)²/(b+c)，p = erfc(sqrt(χ²/2))，不依赖 scipy；
  - 单模型失败只报错不抛栈，仍有可用结果时照样落盘，最后以非 0 退出码提示。

用法：
  python training\\research\\stats_eval.py --model data\\models\\tea_yolo.pt --data training\\datasets\\tea_leaf_disease\\data.yaml
  python training\\research\\stats_eval.py --model runs\\a\\best.pt,runs\\b\\best.pt --bootstrap 2000
  python training\\research\\stats_eval.py --model data\\models\\tea_yolo.pt --imgsz 64 --conf 0.01 --device cpu --bootstrap 50
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from common import (class_name, get_preds, load_data_yaml, match_image,
                    micro_prf, read_gt, resolve_split, summarize_arrays)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="bootstrap 置信区间 + McNemar 显著性检验")
    parser.add_argument("--model", required=True,
                        help="模型权重路径，逗号分隔可给 1~2 个（2 个时做 McNemar 检验）")
    parser.add_argument("--data", required=True, help="data.yaml 路径")
    parser.add_argument("--split", default="val", help="评估划分，默认 val")
    parser.add_argument("--imgsz", type=int, default=640, help="推理分辨率")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS 与 TP 匹配的 IoU 阈值")
    parser.add_argument("--device", default="", help="cpu / 0 / 留空自动")
    parser.add_argument("--bootstrap", type=int, default=1000, help="bootstrap 重采样次数")
    parser.add_argument("--seed", type=int, default=42, help="bootstrap 随机种子")
    parser.add_argument("--out", default="research_out/stats", help="输出目录")
    return parser.parse_args()


# ============================================================
# 逐图推理与匹配
# ============================================================
def run_model(weights: str, images: list[Path], labels: list[Path],
              names: list[str], args: argparse.Namespace) -> dict:
    """跑一个模型，返回逐图 TP/FP/FN 数组与统计量；异常收敛成 status=error。"""
    result = {
        "model": Path(weights).name,
        "weights": str(weights),
        "status": "ok",
        "tp": np.zeros(len(images), dtype=np.int64),
        "fp": np.zeros(len(images), dtype=np.int64),
        "fn": np.zeros(len(images), dtype=np.int64),
        "gt": np.zeros(len(images), dtype=np.int64),
        "pred": np.zeros(len(images), dtype=np.int64),
        "n_gt": 0,
        "n_pred": 0,
        "error": "",
        "seconds": 0.0,
    }
    t0 = time.time()
    try:
        from ultralytics import YOLO

        model = YOLO(weights)
        model_names = getattr(model, "names", None) or {}
        stream = model.predict(
            source=[str(p) for p in images],
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device or None,
            verbose=False,
            stream=True,
        )
        for i, (img_path, label_path, res) in enumerate(zip(images, labels, stream)):
            height, width = res.orig_shape[:2]
            gts = read_gt(label_path, width, height, len(names))
            preds = get_preds(res)
            matched = match_image(preds, gts, args.iou)
            result["tp"][i] = len(matched["tp"])
            result["fp"][i] = len(matched["fp"])
            result["fn"][i] = len(matched["fn"])
            result["gt"][i] = len(gts)
            result["pred"][i] = len(preds)
            result["n_gt"] += len(gts)
            result["n_pred"] += len(preds)
            if (i + 1) % 200 == 0:
                print(f"  已推理 {i + 1}/{len(images)} 张……")
        if not names and model_names:  # data.yaml 缺 names 时用模型自带类别名
            names.extend(class_name([], k, model_names) for k in sorted(model_names))
    except Exception as e:  # noqa: BLE001 - 单模型失败不应中断整批
        result["status"] = "error"
        result["error"] = " ".join(str(e).split())[:200] or e.__class__.__name__
        print(f"  [失败] {result['model']}：{result['error']}")
    result["seconds"] = round(time.time() - t0, 1)
    return result


# ============================================================
# bootstrap 与 McNemar
# ============================================================
def _prf_arrays(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray):
    """向量化 micro P/R/F1（输入为逐图计数数组）。"""
    tp = tp.astype(np.float64)
    fp = fp.astype(np.float64)
    fn = fn.astype(np.float64)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(tp), where=total > 0)
    return precision, recall, f1


def bootstrap_metrics(arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                      n_boot: int, seed: int, chunk: int = 250) -> list[dict]:
    """对图片有放回重采样，返回每个模型的 P/R/F1 分布与 95% 区间。

    同一次抽样的计数矩阵被所有模型复用，因此两模型的差值也天然是"配对"的。
    """
    n = len(arrays[0][0])
    n_boot = max(1, int(n_boot))
    boot_p = [np.zeros(n_boot) for _ in arrays]
    boot_r = [np.zeros(n_boot) for _ in arrays]
    boot_f1 = [np.zeros(n_boot) for _ in arrays]

    if n > 0:
        rng = np.random.default_rng(seed)
        pvals = np.full(n, 1.0 / n)
        done = 0
        while done < n_boot:
            size = min(chunk, n_boot - done)
            # counts[k, i] = 第 k 次重采样中第 i 张图被抽中的次数
            counts = rng.multinomial(n, pvals, size=size).astype(np.float64)
            for m, (tp, fp, fn) in enumerate(arrays):
                tp_s = counts @ tp.astype(np.float64)
                fp_s = counts @ fp.astype(np.float64)
                fn_s = counts @ fn.astype(np.float64)
                p, r, f1 = _prf_arrays(tp_s, fp_s, fn_s)
                boot_p[m][done:done + size] = p
                boot_r[m][done:done + size] = r
                boot_f1[m][done:done + size] = f1
            done += size

    out: list[dict] = []
    for m in range(len(arrays)):
        item = {"n_boot": n_boot, "n_images": n}
        for key, arr in (("precision", boot_p[m]), ("recall", boot_r[m]), ("f1", boot_f1[m])):
            if n == 0:
                item[key] = {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
                continue
            lo, hi = np.percentile(arr, [2.5, 97.5])
            item[key] = {
                "mean": round(float(arr.mean()), 4),
                "ci_low": round(float(lo), 4),
                "ci_high": round(float(hi), 4),
            }
        out.append(item)
    return out, boot_f1


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """配对 McNemar 检验（连续性校正）。

    a11: 都正确；a10: A 对 B 错；a01: A 错 B 对；a00: 都错
    χ² = (|a10 - a01| - 1)² / (a10 + a01)，自由度 1；
    p = erfc(sqrt(χ²/2))（单侧卡方尾概率的解析式，避免引入 scipy）。
    """
    a11 = int(np.sum(correct_a & correct_b))
    a10 = int(np.sum(correct_a & ~correct_b))
    a01 = int(np.sum(~correct_a & correct_b))
    a00 = int(np.sum(~correct_a & ~correct_b))
    discordant = a10 + a01
    if discordant == 0:
        chi2, p = 0.0, 1.0
    else:
        chi2 = (abs(a10 - a01) - 1) ** 2 / discordant
        p = math.erfc(math.sqrt(chi2 / 2.0))
    return {
        "both_correct": a11,
        "only_a_correct": a10,
        "only_b_correct": a01,
        "both_wrong": a00,
        "discordant": discordant,
        "chi2": round(float(chi2), 4),
        "df": 1,
        "p_value": round(float(p), 6),
        "significant_0.05": bool(p < 0.05),
    }


# ============================================================
# 输出
# ============================================================
def write_per_image(path: Path, results: list[dict], images: list[Path]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "image", "gt", "pred", "tp", "fp", "fn", "status"])
        for r in results:
            for i, img in enumerate(images):
                writer.writerow([
                    r["model"], str(img), int(r["gt"][i]), int(r["pred"][i]),
                    int(r["tp"][i]), int(r["fp"][i]), int(r["fn"][i]), r["status"],
                ])


def fmt_ci(point: float, ci: dict) -> str:
    return f"{point:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# 统计显著性评估（bootstrap 95% CI + McNemar）",
        "",
        f"- data.yaml：`{report['data']}`",
        f"- 划分：{report['split']} ｜ 图片数：{report['n_images']} ｜ GT 框数：{report['n_gt']}",
        f"- 推理设置：imgsz={report['imgsz']}，conf={report['conf']}，IoU={report['iou']}，device={report['device'] or 'auto'}",
        f"- bootstrap：{report['bootstrap']} 次，seed={report['seed']}，2.5%/97.5% 分位",
        f"- 生成时间：{report['generated_at']}",
        "",
        "## 1. 主指标与 95% 置信区间（micro，逐图聚合）",
        "",
        "| 模型 | TP | FP | FN | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in report["models"]:
        if m["status"] != "ok":
            lines.append(f"| {m['model']} | — | — | — | 失败：{m['error']} | | |")
            continue
        lines.append(
            f"| {m['model']} | {m['tp']} | {m['fp']} | {m['fn']} | "
            f"{fmt_ci(m['precision'], m['ci']['precision'])} | "
            f"{fmt_ci(m['recall'], m['ci']['recall'])} | "
            f"{fmt_ci(m['f1'], m['ci']['f1'])} |"
        )
    lines += ["", "表注：方括号内为 95% bootstrap 置信区间；F1 由 micro P/R 计算。", ""]

    comp = report.get("comparison")
    if comp:
        lines += [
            "## 2. 配对 McNemar 检验",
            "",
            "判定口径：某张图“检测正确”= 该图无 FP 且无 FN（整图级配对）。",
            "",
            "| | 模型 B 正确 | 模型 B 错误 |",
            "| --- | ---: | ---: |",
            f"| **模型 A 正确** | {comp['mcnemar']['both_correct']} | {comp['mcnemar']['only_a_correct']} |",
            f"| **模型 A 错误** | {comp['mcnemar']['only_b_correct']} | {comp['mcnemar']['both_wrong']} |",
            "",
            f"- A = `{comp['model_a']}`，B = `{comp['model_b']}`",
            f"- 不一致对数 b={comp['mcnemar']['only_a_correct']}，c={comp['mcnemar']['only_b_correct']}",
            f"- χ²（连续性校正）= {comp['mcnemar']['chi2']:.3f}，df=1，p = {comp['mcnemar']['p_value']:.6f}",
            f"- 显著性水平 α=0.05：{'存在显著差异' if comp['mcnemar']['significant_0.05'] else '无显著差异'}",
            "",
            "## 3. F1 差值（配对 bootstrap）",
            "",
            f"- F1(A) − F1(B) = {comp['f1_diff']['point']:+.4f}，"
            f"95% CI [{comp['f1_diff']['ci_low']:+.4f}, {comp['f1_diff']['ci_high']:+.4f}]",
            f"- 区间{'不包含 0，差异可认为稳定' if comp['f1_diff']['excludes_zero'] else '包含 0，差异可能只是重采样噪声'}",
            "",
        ]

    lines += ["## 4. 结论", "", f"> {report['conclusion']}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_conclusion(report: dict) -> str:
    ok = [m for m in report["models"] if m["status"] == "ok"]
    if not ok:
        return "所有模型均运行失败，未得到有效指标。"
    if len(ok) == 1:
        m = ok[0]
        return (f"{m['model']} 在 {report['split']} 上 F1={m['f1']:.4f}"
                f"（95% CI {m['ci']['f1']['ci_low']:.4f}–{m['ci']['f1']['ci_high']:.4f}）；"
                f"置信区间宽度反映样本量，区间越窄结论越稳。")
    comp = report.get("comparison") or {}
    a, b = ok[0], ok[1]
    mc = comp.get("mcnemar", {})
    diff = comp.get("f1_diff", {})
    if mc.get("significant_0.05"):
        verdict = (f"两模型整图级正确率差异显著（McNemar p={mc['p_value']:.4f} < 0.05），"
                   f"F1 差值 {diff.get('point', 0):+.4f}（95% CI 已给出）")
    else:
        verdict = (f"两模型整图级正确率差异不显著（McNemar p={mc['p_value']:.4f} ≥ 0.05），"
                   f"不足以宣称 {a['model']} 优于 {b['model']}")
    return f"{verdict}；建议结合多 seed 的 mean±std 一起讨论。"


def main() -> None:
    args = parse_args()
    models = [x.strip() for x in args.model.replace("，", ",").split(",") if x.strip()]
    if not models:
        print("[错误] --model 至少需要一个权重路径")
        sys.exit(1)
    if len(models) > 2:
        print(f"[错误] 最多支持 2 个模型做配对检验，当前给 {len(models)} 个")
        sys.exit(1)
    if args.bootstrap < 1:
        print("[错误] --bootstrap 必须 ≥ 1")
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
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[信息] 数据：{yaml_path} ｜ split={args.split} ｜ 目录：{split_root}")
    print(f"[信息] 图片 {len(images)} 张 ｜ 类别 {len(names)} 个 ｜ 模型 {len(models)} 个")
    print(f"[信息] imgsz={args.imgsz} conf={args.conf} iou={args.iou} "
          f"bootstrap={args.bootstrap} seed={args.seed}")

    results: list[dict] = []
    for i, weights in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {weights}")
        r = run_model(weights, images, labels, names, args)
        results.append(r)
        if r["status"] == "ok":
            p, rec, f1 = micro_prf(r["tp"].sum(), r["fp"].sum(), r["fn"].sum())
            print(f"  [完成] TP={int(r['tp'].sum())} FP={int(r['fp'].sum())} "
                  f"FN={int(r['fn'].sum())} P={p:.4f} R={rec:.4f} F1={f1:.4f}")

    ok = [r for r in results if r["status"] == "ok"]
    report: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data": str(yaml_path),
        "split": args.split,
        "split_dir": str(split_root),
        "n_images": len(images),
        "n_gt": int(sum(r["n_gt"] for r in ok)) if ok else 0,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "class_names": names,
        "models": [],
        "comparison": None,
    }

    boot_cis: list[dict] = []
    boot_f1: list[np.ndarray] = []
    if ok:
        arrays = [(r["tp"], r["fp"], r["fn"]) for r in ok]
        boot_cis, boot_f1 = bootstrap_metrics(arrays, args.bootstrap, args.seed)

    ok_counter = 0
    for r in results:
        entry = {
            "model": r["model"],
            "weights": r["weights"],
            "status": r["status"],
            "error": r["error"],
            "seconds": r["seconds"],
            "n_pred": int(r["n_pred"]),
        }
        if r["status"] == "ok":
            summary = summarize_arrays(r["tp"], r["fp"], r["fn"])
            entry.update(summary)
            entry["ci"] = boot_cis[ok_counter]
            ok_counter += 1
            entry["per_image_errors"] = int(np.sum(
                (r["fp"] > 0) | (r["fn"] > 0)))
            entry["correct_images"] = int(np.sum((r["fp"] == 0) & (r["fn"] == 0)))
        report["models"].append(entry)

    # 两模型：配对 McNemar + 配对 bootstrap 的 F1 差值
    if len(ok) == 2:
        corrects = [(r["fp"] == 0) & (r["fn"] == 0) for r in ok]
        mc = mcnemar_test(corrects[0], corrects[1])
        f1_a = micro_prf(ok[0]["tp"].sum(), ok[0]["fp"].sum(), ok[0]["fn"].sum())[2]
        f1_b = micro_prf(ok[1]["tp"].sum(), ok[1]["fp"].sum(), ok[1]["fn"].sum())[2]
        diff_boot = boot_f1[0] - boot_f1[1]
        lo, hi = np.percentile(diff_boot, [2.5, 97.5])
        point = float(f1_a - f1_b)
        report["comparison"] = {
            "model_a": ok[0]["model"],
            "model_b": ok[1]["model"],
            "mcnemar": mc,
            "f1_diff": {
                "point": round(point, 4),
                "ci_low": round(float(lo), 4),
                "ci_high": round(float(hi), 4),
                "excludes_zero": bool(lo > 0 or hi < 0),
                "bootstrap": args.bootstrap,
            },
        }
    elif len(ok) == 1 and len(results) == 2:
        report["comparison"] = None

    report["conclusion"] = build_conclusion(report)

    json_path = out_dir / "stats.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = out_dir / "per_image.csv"
    write_per_image(csv_path, results, images)
    md_path = out_dir / "stats.md"
    write_markdown(md_path, report)

    print(f"\n[输出] {json_path}")
    print(f"[输出] {csv_path}")
    print(f"[输出] {md_path}")
    print(f"[结论] {report['conclusion']}")

    if len(ok) < len(results):
        print(f"[错误] {len(results) - len(ok)} 个模型运行失败，见 stats.json 的 error 字段")
        sys.exit(1)


if __name__ == "__main__":
    main()
