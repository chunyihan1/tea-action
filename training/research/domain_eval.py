"""跨域评估：in-domain 与 out-of-domain 各报一组 P/R/F1 + 95% CI，并量化域偏移。

为什么需要：
  Q1 论文不能只报同分布指标。审稿人要求"外部域不参与训练"的独立验证，
  并把指标下降（ΔF1）作为泛化性的直接证据。本脚本把多个域放在**同一个模型、
  同一套 conf/iou/imgsz** 下评估，避免手工汇总时口径不一致。

设计要点：
  - 域的指定格式为 `名称=路径`，--data 可重复传，第一个域视为 in-domain 基准；
  - 图片/标签读取、匹配全部复用 common.py（与 stats_eval.py 完全一致的口径），
    逐图推理与 bootstrap 直接复用 stats_eval.py，保证两处数字对得上；
  - 单个域失败（路径错、无图片、标签全缺）只跳过该域并在表里留一行 status=error，
    不中断其他域；结论段会点名失败的域；
  - bootstrap 使用固定 seed，所有域共用同一个重采样次数，CI 可比。

用法：
  python training\\research\\domain_eval.py --model data\\models\\tea_yolo.pt ^
      --data in=training\\datasets\\tea_leaf_disease\\data.yaml ^
      --data bd=data\\external\\tea_leaf_bd\\data.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from common import load_data_yaml, resolve_split
from stats_eval import bootstrap_metrics, run_model

FIELDNAMES = [
    "status", "domain", "data", "split", "images", "gt", "pred", "tp", "fp", "fn",
    "precision", "recall", "f1", "f1_ci_low", "f1_ci_high",
    "f1_drop_abs", "f1_drop_pct", "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="跨域评估（in-domain vs out-of-domain + ΔF1）")
    parser.add_argument("--model", required=True, help="模型权重路径")
    parser.add_argument("--data", nargs="+", action="append", required=True,
                        metavar="名称=路径",
                        help="可重复传：`名称=data.yaml`，第一个域作为 in-domain 基准")
    parser.add_argument("--split", default="val", help="各域评估划分，默认 val")
    parser.add_argument("--imgsz", type=int, default=640, help="推理分辨率")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS 与 TP 匹配的 IoU 阈值")
    parser.add_argument("--device", default="", help="cpu / 0 / 留空自动")
    parser.add_argument("--bootstrap", type=int, default=1000, help="bootstrap 重采样次数")
    parser.add_argument("--seed", type=int, default=42, help="bootstrap 随机种子（固定）")
    parser.add_argument("--out", default="research_out/domain", help="输出目录")
    return parser.parse_args()


def parse_domains(raw: list[list[str]]) -> list[tuple[str, str]]:
    """把 --data 的多次/多值写法摊平成 [(名称, 路径), ...]，保持输入顺序。"""
    flat = [item for group in raw for item in group]
    domains: list[tuple[str, str]] = []
    for i, item in enumerate(flat, 1):
        text = item.strip()
        if not text:
            continue
        if "=" in text:
            name, path = text.split("=", 1)
            name = name.strip() or f"domain{i}"
            path = path.strip()
        else:
            name, path = Path(text).parent.name or f"domain{i}", text
        if not path:
            raise SystemExit(f"[错误] 域名 `{name}` 没有给出路径：{text}")
        domains.append((name, path))
    if not domains:
        raise SystemExit("[错误] --data 至少需要一个 `名称=data.yaml`")
    return domains


# ============================================================
# 单域评估
# ============================================================
def eval_domain(name: str, data_path: str, weights: str,
                args: argparse.Namespace) -> dict:
    """评估一个域；任何可预期错误都收敛成 status=error，交给调用方跳过。"""
    row: dict = {k: "" for k in FIELDNAMES}
    row.update({"status": "error", "domain": name, "data": data_path,
                "split": args.split})

    try:
        cfg, names, yaml_path = load_data_yaml(data_path)
    except (FileNotFoundError, ValueError) as e:
        row["error"] = str(e)
        return row
    try:
        images, labels, _ = resolve_split(cfg, yaml_path, args.split)
    except FileNotFoundError as e:
        row["error"] = str(e)
        return row

    missing = [p for p in labels if not p.is_file()]
    if len(missing) == len(labels):
        row["error"] = (f"{len(images)} 张图片全部缺标签（已查找 {yaml_path.parent} 的 "
                        f"labels/ 目录），无法计算召回")
        return row
    if missing:
        print(f"  [提示] {len(missing)}/{len(labels)} 张图片无标签文件，按无 GT（背景图）处理")

    # 复用 stats_eval 的逐图推理与匹配，确保与 stats.md 数字同源
    sub_args = argparse.Namespace(conf=args.conf, iou=args.iou,
                                  imgsz=args.imgsz, device=args.device)
    result = run_model(weights, images, labels, names, sub_args)
    if result["status"] != "ok":
        row["error"] = result["error"]
        return row

    row.update({
        "status": "ok",
        "images": len(images),
        "gt": int(result["gt"].sum()),
        "pred": int(result["pred"].sum()),
    })
    if row["gt"] == 0:
        row["status"] = "error"
        row["error"] = "该域 GT 框数为 0（标签为空或全部非法），无法计算召回"
        return row

    # 单模型的逐图 TP/FP/FN，直接复用 bootstrap 逻辑（固定 seed）
    boot_cis, _ = bootstrap_metrics(
        [(result["tp"], result["fp"], result["fn"])], args.bootstrap, args.seed)
    ci = boot_cis[0]

    tp, fp, fn = int(result["tp"].sum()), int(result["fp"].sum()), int(result["fn"].sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    row.update({
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "f1_ci_low": ci["f1"]["ci_low"],
        "f1_ci_high": ci["f1"]["ci_high"],
        "tp": tp, "fp": fp, "fn": fn,
    })
    return row


# ============================================================
# 输出
# ============================================================
def apply_drops(rows: list[dict]) -> tuple[dict | None, list[str]]:
    """以第一个成功域为基准计算 F1 绝对/相对下降，返回 (基准行, 警告列表)。"""
    warnings: list[str] = []
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        return None, ["没有任何域评估成功，无法计算域偏移。"]
    ref = ok[0]
    if rows[0]["status"] != "ok":
        warnings.append(f"第一个域 `{rows[0]['domain']}` 评估失败，"
                        f"改以 `{ref['domain']}` 作为域偏移基准。")
    for r in rows:
        if r["status"] != "ok":
            continue
        drop = ref["f1"] - r["f1"]
        r["f1_drop_abs"] = round(drop, 4)
        r["f1_drop_pct"] = round(drop / ref["f1"] * 100.0, 2) if ref["f1"] > 0 else 0.0
    return ref, warnings


def build_conclusion(rows: list[dict], ref: dict | None) -> str:
    ok = [r for r in rows if r["status"] == "ok"]
    failed = [r["domain"] for r in rows if r["status"] != "ok"]
    if ref is None:
        return "没有可用的域结果，请检查 --data 路径与 split 设置。"
    if len(ok) == 1:
        text = (f"仅 `{ref['domain']}` 评估成功（F1={ref['f1']:.4f}），"
                f"缺少外部域，无法给出域偏移结论。")
    else:
        worst = max(ok, key=lambda r: r.get("f1_drop_abs") or 0.0)
        text = (f"以 `{ref['domain']}`（F1={ref['f1']:.4f}）为 in-domain 基准，"
                f"最差外部域 `{worst['domain']}` 的 F1 为 {worst['f1']:.4f}，"
                f"绝对下降 {worst['f1_drop_abs']:.4f}，相对下降 {worst['f1_drop_pct']:.2f}%。")
        stable = [r for r in ok[1:] if abs(r.get("f1_drop_abs") or 0.0) < 0.05]
        if stable:
            names = "、".join(f"`{r['domain']}`" for r in stable)
            text += f" {names} 的下降小于 5 个百分点，域偏移相对可控。"
    if failed:
        text += f" 失败并已跳过的域：{'、'.join(f'`{x}`' for x in failed)}（原因见表）。"
    return text


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def write_markdown(path: Path, rows: list[dict], ref: dict | None,
                   warnings: list[str], report: dict) -> None:
    lines = [
        "# 跨域评估（in-domain vs out-of-domain）",
        "",
        f"- 模型：`{report['model']}`",
        f"- 推理设置：imgsz={report['imgsz']}，conf={report['conf']}，"
        f"IoU={report['iou']}，device={report['device'] or 'auto'}",
        f"- bootstrap：{report['bootstrap']} 次，seed={report['seed']}（固定），2.5%/97.5% 分位",
        f"- 基准域：`{ref['domain']}`" if ref else "- 基准域：无（全部失败）",
        f"- 生成时间：{report['generated_at']}",
        "",
        "## 1. 各域指标（micro，逐图聚合）",
        "",
        "| 域 | 图片数 | GT 框数 | Precision | Recall | F1 [95% CI] | F1 下降(绝对) | F1 下降(相对) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"| {r['domain']} | — | — | — | — | 失败：{r['error']} | — | — |")
            continue
        lines.append(
            f"| {r['domain']} | {r['images']} | {r['gt']} | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | {r['f1']:.4f} [{r['f1_ci_low']:.4f}, {r['f1_ci_high']:.4f}] | "
            f"{r['f1_drop_abs']:.4f} | {r['f1_drop_pct']:.2f}% |"
        )
    lines += ["", "表注：下降幅度相对第一个成功域（in-domain）计算，正值表示跨域后性能变差。", ""]

    if warnings:
        lines += ["## 2. 提示", ""]
        lines += [f"- {w}" for w in warnings]
        lines.append("")

    lines += [f"## {3 if warnings else 2}. 域偏移结论", "", f"> {report['conclusion']}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1:
        print("[错误] --bootstrap 必须 ≥ 1")
        sys.exit(1)
    domains = parse_domains(args.data)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[信息] 模型：{args.model}")
    print(f"[信息] 域 {len(domains)} 个：{', '.join(n for n, _ in domains)}")
    print(f"[信息] imgsz={args.imgsz} conf={args.conf} iou={args.iou} "
          f"bootstrap={args.bootstrap} seed={args.seed}\n")

    rows: list[dict] = []
    for i, (name, data_path) in enumerate(domains, 1):
        print(f"[{i}/{len(domains)}] 域 `{name}`：{data_path}")
        row = eval_domain(name, data_path, args.model, args)
        rows.append(row)
        if row["status"] == "ok":
            print(f"  [完成] 图片 {row['images']} 张，GT {row['gt']}，"
                  f"P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f}")
        else:
            print(f"  [跳过] {row['error']}")

    ref, warnings = apply_drops(rows)
    for w in warnings:
        print(f"[警告] {w}")

    report: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "reference": ref["domain"] if ref else None,
        "domains": [],
    }
    report["conclusion"] = build_conclusion(rows, ref)

    for r in rows:
        report["domains"].append({k: r.get(k, "") for k in FIELDNAMES})

    write_csv(out_dir / "domain_eval.csv", rows)
    write_markdown(out_dir / "domain_eval.md", rows, ref, warnings, report)
    (out_dir / "domain_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[输出] {out_dir / 'domain_eval.csv'}")
    print(f"[输出] {out_dir / 'domain_eval.md'}")
    print(f"[输出] {out_dir / 'domain_eval.json'}")
    print(f"[结论] {report['conclusion']}")

    if all(r["status"] != "ok" for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
