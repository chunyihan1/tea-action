"""把 benchmark.csv 转成论文可直接粘贴的表格（Markdown + LaTeX）。

为什么需要：
  论文表 1 的数值格式、粗体最优值、基线增量必须统一，手工整理极易出错且不可复现。

输出：
  - paper_table.md / paper_table.tex      主表（TTA=False 的主结果）
  - paper_table_tta.md                    若 csv 同时含 augment=False/True，生成 TTA 消融表
  - paper_table_imgsz.md                  若 csv 含多个 imgsz，生成分辨率消融表
  - paper_table_seeds.md                  若 csv 含 seed 列（或有同模型多行重复实验），
                                          生成 mean ± std 表

用法：
  python training\\research\\make_tables.py
  python training\\research\\make_tables.py --csv research_out\\benchmark.csv --baseline yolov8n.pt
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

FALLBACK_BASELINE = "yolov8n.pt"


def to_float(value) -> float | None:
    """空值/NaN/破折号统一转成 None，其余转 float。"""
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "—", "-", "na"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_bool(value) -> bool | None:
    """把 csv 中 True/False/1/0 统一成布尔值。"""
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def load_rows(csv_path: Path) -> tuple[list[dict], list[str], int, bool]:
    """读取 csv，返回 (成功行, 失败模型名, 失败数, 是否含 seed 列)。"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    has_seed = "seed" in df.columns
    rows: list[dict] = []
    failed: list[str] = []
    for _, rec in df.iterrows():
        status = str(rec.get("status", "ok")).strip().lower()
        model = str(rec.get("model", "")).strip() or "unknown"
        if status in {"", "nan"}:
            status = "ok"
        if status not in {"ok", "success", "done"}:
            failed.append(model)
            continue
        params = to_float(rec.get("params"))
        rows.append({
            "model": model,
            "params": params,
            "params_m": params / 1e6 if params is not None else None,
            "precision": to_float(rec.get("precision")),
            "recall": to_float(rec.get("recall")),
            "map50": to_float(rec.get("map50")),
            "map5095": to_float(rec.get("map50_95")),
            "latency": to_float(rec.get("latency_ms")),
            "augment": to_bool(rec.get("augment")),
            "imgsz": to_float(rec.get("imgsz")),
            "seed": to_float(rec.get("seed")) if has_seed else None,
        })
    return rows, failed, len(failed), has_seed


def pick_baseline(rows: list[dict], requested: str | None) -> dict | None:
    """选基线：显式指定 > yolov8n.pt > 第一个成功模型。"""

    def match(row: dict, name: str) -> bool:
        return row["model"] == name or Path(row["model"]).stem == Path(name).stem

    if requested:
        for row in rows:
            if match(row, requested):
                return row
        print(f"[警告] 基线 {requested} 不在成功结果中，改用第一个成功模型")
    for row in rows:
        if match(row, FALLBACK_BASELINE):
            return row
    return rows[0] if rows else None


def fmt(value: float | None, ndigits: int) -> str:
    return f"{value:.{ndigits}f}" if value is not None else "—"


def md_wrap(text: str, bold: bool) -> str:
    return f"**{text}**" if bold else text


def tex_escape(text: str) -> str:
    return (text.replace("—", "--").replace("\\", r"\textbackslash{}")
            .replace("_", r"\_").replace("%", r"\%").replace("&", r"\&"))


def tex_wrap(text: str, bold: bool) -> str:
    return f"\\textbf{{{text}}}" if bold else text


def main_rows(rows: list[dict]) -> list[dict]:
    """主表只取主设置：若同时存在 TTA 行，只用 augment=False 的行，避免同一模型重复。"""
    no_tta = [r for r in rows if r["augment"] is False]
    return no_tta if no_tta else rows


def write_main_table(rows: list[dict], csv_name: str, out_dir: Path,
                     failed: list[str], baseline: dict, total_success: int) -> None:
    """主表：降序按 mAP50-95；粗体标出各项最优。"""
    rows = main_rows([r for r in rows if r["map5095"] is not None])
    rows.sort(key=lambda r: r["map5095"], reverse=True)
    if not rows:
        print("[警告] 没有可用的成功结果，跳过主表")
        return

    map50_values = [r["map50"] for r in rows if r["map50"] is not None]
    best_map50 = max(map50_values) if map50_values else None
    best_map5095 = max(r["map5095"] for r in rows)
    latencies = [r["latency"] for r in rows if r["latency"] is not None]
    best_latency = min(latencies) if latencies else None
    params = [r["params_m"] for r in rows if r["params_m"] is not None]
    best_params = min(params) if params else None
    base_value = baseline["map5095"] if baseline and baseline.get("map5095") is not None else None
    now = datetime.now().isoformat(sep=" ", timespec="seconds")

    # ---- Markdown ----
    lines = [
        f"<!-- 数据来源：{csv_name} ｜ 生成时间：{now} -->",
        "# 表 1 多模型检测性能对比",
        "",
        "| Model | Params(M) | mAP@0.5 | mAP@0.5:0.95 | Latency(ms) | ΔmAP@0.5:0.95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        is_base = baseline is not None and r["model"] == baseline["model"]
        if is_base or base_value is None:
            delta = "—"
        else:
            delta = f"{r['map5095'] - base_value:+.3f}"
        lines.append(
            "| " + " | ".join([
                r["model"],
                md_wrap(fmt(r["params_m"], 2),
                        r["params_m"] is not None and r["params_m"] == best_params),
                md_wrap(fmt(r["map50"], 3),
                        r["map50"] is not None and r["map50"] == best_map50),
                md_wrap(fmt(r["map5095"], 3), r["map5095"] == best_map5095),
                md_wrap(fmt(r["latency"], 2), r["latency"] is not None and r["latency"] == best_latency),
                delta,
            ]) + " |"
        )
    note = f"表注：{len(failed)} 个模型运行失败"
    if failed:
        note += f"（{'、'.join(failed)}）"
    note += f"；主表为非 TTA 主结果 {len(rows)} 行"
    if total_success > len(rows):
        note += f"（另有 {total_success - len(rows)} 行 TTA 结果见 paper_table_tta.md）"
    note += f"；Δ 为相对基线 {baseline['model'] if baseline else '—'} 的 mAP@0.5:0.95 差值。"
    lines += ["", note, ""]
    (out_dir / "paper_table.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- LaTeX ----
    tex = [
        f"% 数据来源：{csv_name} ｜ 生成时间：{now}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{多模型检测性能对比（同一数据集、同一输入分辨率、同一验证集）}",
        r"\label{tab:benchmark}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & Params (M) & mAP@0.5 & mAP@0.5:0.95 & Latency (ms) & $\Delta$mAP@0.5:0.95 \\",
        r"\midrule",
    ]
    for r in rows:
        is_base = baseline is not None and r["model"] == baseline["model"]
        if is_base or base_value is None:
            delta = "--"
        else:
            delta = f"${r['map5095'] - base_value:+.3f}$"
        tex.append(" & ".join([
            tex_wrap(tex_escape(r["model"]), False),
            tex_wrap(fmt(r["params_m"], 2),
                     r["params_m"] is not None and r["params_m"] == best_params),
            tex_wrap(fmt(r["map50"], 3),
                     r["map50"] is not None and r["map50"] == best_map50),
            tex_wrap(fmt(r["map5095"], 3), r["map5095"] == best_map5095),
            tex_wrap(fmt(r["latency"], 2).replace("—", "--"),
                     r["latency"] is not None and r["latency"] == best_latency),
            delta,
        ]) + r" \\")
    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{2pt}",
        r"{\footnotesize " + tex_escape(note) + "}",
        r"\end{table}",
        "",
    ]
    (out_dir / "paper_table.tex").write_text("\n".join(tex), encoding="utf-8")
    print(f"[输出] {out_dir / 'paper_table.md'}")
    print(f"[输出] {out_dir / 'paper_table.tex'}")


def write_tta_table(rows: list[dict], csv_name: str, out_dir: Path) -> None:
    """TTA 消融表：同一模型 augment=False vs True 的 mAP 与增量。"""
    grouped: dict[str, dict[bool, dict]] = {}
    for r in rows:
        if r["augment"] is None or r["map5095"] is None:
            continue
        slot = grouped.setdefault(r["model"], {})
        old = slot.get(r["augment"])
        if old is None or r["map5095"] > old["map5095"]:
            slot[r["augment"]] = r

    entries = [m for m, v in grouped.items() if True in v and False in v]
    if not entries:
        return

    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    lines = [
        f"<!-- 数据来源：{csv_name} ｜ 生成时间：{now} -->",
        "# 表 TTA 测试时增强消融",
        "",
        "| Model | mAP@0.5 (w/o TTA) | mAP@0.5 (TTA) | mAP@0.5:0.95 (w/o TTA) | mAP@0.5:0.95 (TTA) | ΔmAP@0.5:0.95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in sorted(entries):
        off, on = grouped[model][False], grouped[model][True]
        delta = on["map5095"] - off["map5095"]
        better_on = delta > 0
        lines.append("| " + " | ".join([
            model,
            fmt(off["map50"], 3),
            md_wrap(fmt(on["map50"], 3), (on["map50"] or 0) > (off["map50"] or 0)),
            fmt(off["map5095"], 3),
            md_wrap(fmt(on["map5095"], 3), better_on),
            f"{delta:+.3f}",
        ]) + " |")
    lines += ["", "表注：每行只改变 TTA 一个变量，其余设置（数据、imgsz、阈值）完全相同。", ""]
    path = out_dir / "paper_table_tta.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[输出] {path}")


def write_imgsz_table(rows: list[dict], csv_name: str, out_dir: Path) -> None:
    """分辨率消融表：行=模型，列=imgsz，单元格=mAP@0.5:0.95。"""
    values = sorted({int(r["imgsz"]) for r in main_rows(rows) if r["imgsz"] is not None})
    if len(values) < 2:
        return
    grouped: dict[str, dict[int, dict]] = {}
    for r in main_rows(rows):
        if r["imgsz"] is None or r["map5095"] is None:
            continue
        grouped.setdefault(r["model"], {})[int(r["imgsz"])] = r

    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    lines = [
        f"<!-- 数据来源：{csv_name} ｜ 生成时间：{now} -->",
        "# 表 输入分辨率消融",
        "",
        "| Model | " + " | ".join(f"imgsz={v}" for v in values) + " |",
        "| --- |" + " ---: |" * len(values),
    ]
    for model in sorted(grouped):
        cells = []
        best = max((row["map5095"] for row in grouped[model].values()), default=None)
        for v in values:
            row = grouped[model].get(v)
            text = fmt(row["map5095"], 3) if row else "—"
            cells.append(md_wrap(text, row is not None and row["map5095"] == best))
        lines.append("| " + model + " | " + " | ".join(cells) + " |")
    lines += ["", "表注：单元格为 mAP@0.5:0.95，粗体为该模型最优分辨率；每行只改变 imgsz 一个变量。", ""]
    path = out_dir / "paper_table_imgsz.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[输出] {path}")


def _mean_std(values: list[float]) -> tuple[float, float | None]:
    """均值与样本标准差（ddof=1）；样本少于 2 个时标准差无定义。"""
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, var ** 0.5


def write_seeds_table(rows: list[dict], csv_name: str, out_dir: Path,
                      has_seed_col: bool) -> None:
    """同模型多行（多种子重复实验）聚合成 mean ± std 表。

    触发条件：csv 含 seed 列；或同模型在同一 imgsz、非 TTA 条件下出现多行
    （视为重复实验）。分辨率消融（同模型不同 imgsz）不会被误聚合。
    """
    base = main_rows(rows)
    grouped: dict[str, list[dict]] = {}
    for r in base:
        grouped.setdefault(r["model"], []).append(r)

    multi: dict[str, list[dict]] = {}
    for model, group in grouped.items():
        if len(group) < 2:
            continue
        if not has_seed_col and len({r["imgsz"] for r in group}) > 1:
            continue  # 同模型多 imgsz = 分辨率消融，不是重复实验
        multi[model] = group
    if not multi:
        return

    entries: list[dict] = []
    for model, group in multi.items():
        item: dict = {"model": model, "n": len(group)}
        for key in ("map50", "map5095", "precision", "recall"):
            values = [r[key] for r in group if r[key] is not None]
            item[key] = _mean_std(values) if values else (None, None)
        item["seeds"] = sorted({int(r["seed"]) for r in group if r["seed"] is not None})
        entries.append(item)
    entries.sort(key=lambda e: (e["map5095"][0] is not None, e["map5095"][0] or -1.0),
                 reverse=True)

    def cell(pair: tuple[float | None, float | None]) -> str:
        mean, std = pair
        if mean is None:
            return "—"
        return f"{mean:.3f} ± {std:.3f}" if std is not None else f"{mean:.3f} ± —"

    best = max((e["map5095"][0] for e in entries if e["map5095"][0] is not None), default=None)
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    lines = [
        f"<!-- 数据来源：{csv_name} ｜ 生成时间：{now} -->",
        "# 表 多种子重复实验（mean ± std）",
        "",
        "| Model | n | mAP@0.5 | mAP@0.5:0.95 | P | R |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for e in entries:
        map5095_text = cell(e["map5095"])
        lines.append("| " + " | ".join([
            e["model"],
            str(e["n"]),
            cell(e["map50"]),
            md_wrap(map5095_text, best is not None and e["map5095"][0] == best),
            cell(e["precision"]),
            cell(e["recall"]),
        ]) + " |")

    seeds_desc = "；".join(
        f"{e['model']}: seed={e['seeds']}" for e in entries if e["seeds"]) if has_seed_col else ""
    note = ("表注：std 为样本标准差（ddof=1）；n 为参与聚合的重复次数。"
            + (f"使用的随机种子：{seeds_desc}。" if seeds_desc else
               "csv 无 seed 列，按“同模型 + 同 imgsz 多行”识别为重复实验。"))
    lines += ["", note, ""]
    path = out_dir / "paper_table_seeds.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[输出] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="benchmark.csv -> 论文表格（Markdown/LaTeX）")
    parser.add_argument("--csv", default=str(Path("research_out") / "benchmark.csv"),
                        help="benchmark.csv 路径")
    parser.add_argument("--out-dir", default="research_out", help="输出目录")
    parser.add_argument("--baseline", default=None,
                        help="基线模型名（默认 yolov8n.pt；不存在时取第一个成功模型）")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[错误] 找不到 {csv_path}")
        print("       请先运行：python training\\research\\benchmark.py")
        sys.exit(1)

    rows, failed, failed_count, has_seed = load_rows(csv_path)
    if not rows:
        print("[错误] csv 中没有成功结果（status 全部为 error？）")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = pick_baseline(rows, args.baseline)
    if baseline:
        print(f"[信息] 基线：{baseline['model']}（mAP50-95={fmt(baseline['map5095'], 3)}）")

    write_main_table(rows, csv_path.name, out_dir, failed, baseline, total_success=len(rows))
    write_tta_table(rows, csv_path.name, out_dir)
    write_imgsz_table(rows, csv_path.name, out_dir)
    write_seeds_table(rows, csv_path.name, out_dir, has_seed)
    print(f"完成：成功 {len(rows)} 行，失败 {failed_count} 行")


if __name__ == "__main__":
    main()
