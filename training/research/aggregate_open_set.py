"""把多次开放集评估 run 的 metrics.json 汇总成论文表（mean ± std）。

为什么需要：
  open_set_eval.py 每次只产出单个划分（如 S1 / S5）与单个 seed 的指标，
  论文表格需要跨 run 的均值、样本标准差（n−1）、最小/最大值，
  否则无法说明结论是否稳定。本脚本不重跑评估，只做只读聚合。

用法：
  python training\\research\\aggregate_open_set.py ^
      --pattern "research_out\\open_set\\S*\\eval\\metrics.json" --out-dir research_out\\aggregate
  python training\\research\\aggregate_open_set.py --inputs a\\metrics.json b\\metrics.json
  python training\\research\\aggregate_open_set.py --pattern "runs\\*\\metrics.json" --group-regex "(seed\\d+)"

输出：
  - aggregate.csv：长表（group, scorer, metric, mean, std, min, max, n）；
  - aggregate.md：每个 group 一张表，行=打分器、列=指标，格式 `0.9545 ± 0.0119`。
  缺少某字段时跳过而不是崩溃；某指标不足 2 个有效 run 时 std 输出 None（md 显示 μ ± —）。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import statistics
import sys
from pathlib import Path

# 指标名 → metrics.json 中的取值位置（auprc 对外统一叫 aupr）
METRIC_ORDER = [
    "auroc",
    "aupr",
    "fpr_at_95tpr",
    "aurc",
    "e_aurc",
    "coverage@5%",
    "coverage@10%",
    "unknown_recall",
    "misdiagnosis_rate",
    "unknown_detection_miss_rate",
    "referral_rate",
    "referral_rate_known",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="汇总多次 open_set_eval 的 metrics.json（mean / std / min / max / n）")
    parser.add_argument("--inputs", nargs="+", default=None,
                        help="多个 metrics.json 路径，空格分隔")
    parser.add_argument("--pattern", default=None,
                        help="通配符，如 research_out/open_set/S*/eval/metrics.json")
    parser.add_argument("--out-dir", default="research_out/aggregate",
                        help="输出目录，默认 research_out/aggregate")
    parser.add_argument("--group-regex", default=r"(S\d+)",
                        help="从路径中提取分组名的正则，默认 (S\\d+)；无捕获组时用整个匹配")
    return parser.parse_args(argv)


def resolve_inputs(inputs: list[str] | None, pattern: str | None) -> list[Path]:
    """合并 --inputs 与 --pattern（去重并按路径排序，保证输出可复现）。"""
    paths: list[Path] = []
    if inputs:
        paths.extend(Path(x) for x in inputs)
    if pattern:
        paths.extend(Path(x) for x in glob.glob(pattern, recursive=True))
    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path.resolve())] = path
    return sorted(unique.values(), key=lambda p: str(p))


def group_of(path: Path, pattern: str) -> str:
    """按路径（目录/文件名）中的模式提取分组名；无匹配归入 all。"""
    text = str(path)
    match = re.search(pattern, text)
    if not match:
        return "all"
    return match.group(1) if match.groups() else match.group(0)


def load_report(path: Path) -> dict | None:
    """读取单个 metrics.json；损坏或非 JSON 时返回 None（由调用方跳过）。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"[警告] 跳过无法读取的文件 {path}：{e}")
        return None


def scorers_of(report: dict) -> list[str]:
    """按 metrics.json 中实际存在的打分器顺序返回，不假设固定 6 个。"""
    available = report.get("scorers_available")
    if isinstance(available, list) and available:
        return [str(x) for x in available]
    names: list[str] = []
    for key in ("rq2", "rejection"):
        block = report.get(key)
        if isinstance(block, dict):
            names.extend(str(x) for x in block if x not in names)
    selective = ((report.get("selective_prediction") or {}).get("scorers") or {})
    names.extend(str(x) for x in selective if x not in names)
    return names


def metric_values(report: dict, scorer: str) -> dict:
    """从单个 run 提取该打分器的 12 个指标；缺失键返回 None。"""
    rq2 = report.get("rq2") or {}
    selective = ((report.get("selective_prediction") or {}).get("scorers") or {})
    rejection = ((report.get("rejection") or {}).get(scorer) or {})
    calib = rejection.get("calib") or {}
    row2 = rq2.get(scorer) or {}
    row_sp = selective.get(scorer) or {}
    return {
        "auroc": row2.get("auroc"),
        "aupr": row2.get("auprc"),
        "fpr_at_95tpr": row2.get("fpr_at_95tpr"),
        "aurc": row_sp.get("aurc"),
        "e_aurc": row_sp.get("e_aurc"),
        "coverage@5%": row_sp.get("coverage_at_5pct"),
        "coverage@10%": row_sp.get("coverage_at_10pct"),
        "unknown_recall": calib.get("unknown_recall"),
        "misdiagnosis_rate": calib.get("misdiagnosis_rate"),
        "unknown_detection_miss_rate": calib.get("unknown_detection_miss_rate"),
        "referral_rate": calib.get("referral_rate"),
        "referral_rate_known": calib.get("referral_rate_known"),
    }


def summarize(values) -> dict:
    """mean / 样本标准差（n−1）/ min / max / n；非数值与 NaN 一律跳过。"""
    clean: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        f = float(value)
        if math.isnan(f):
            continue
        clean.append(f)
    n = len(clean)
    if n == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": round(statistics.fmean(clean), 4),
        "std": round(statistics.stdev(clean), 4) if n >= 2 else None,
        "min": round(min(clean), 4),
        "max": round(max(clean), 4),
        "n": n,
    }


def build_rows(items: list[tuple[str, dict]]) -> tuple[list[dict], list[str], dict]:
    """把 (group, report) 列表聚合成长表行，并返回 group 顺序与 scorer 顺序。

    某 run 缺少某字段时该字段只贡献 None，被 summarize 跳过，不会影响其它指标。
    """
    groups: list[str] = []
    scorer_order: dict[str, list[str]] = {}
    buckets: dict[tuple[str, str, str], list] = {}
    for group, report in items:
        if group not in groups:
            groups.append(group)
            scorer_order[group] = []
        for scorer in scorers_of(report):
            if scorer not in scorer_order[group]:
                scorer_order[group].append(scorer)
            metrics = metric_values(report, scorer)
            for metric in METRIC_ORDER:
                buckets.setdefault((group, scorer, metric), []).append(metrics.get(metric))

    rows: list[dict] = []
    for group in groups:
        for scorer in scorer_order[group]:
            for metric in METRIC_ORDER:
                stats = summarize(buckets.get((group, scorer, metric), []))
                rows.append({"group": group, "scorer": scorer, "metric": metric, **stats})
    return rows, groups, scorer_order


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["group", "scorer", "metric", "mean", "std", "min", "max", "n"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row.get(key) is None else row.get(key))
                             for key in fields})


def _fmt_cell(stats: dict) -> str:
    """md 单元格：`0.9545 ± 0.0119`；std 缺失（单 run）显示 μ ± —。"""
    if stats.get("mean") is None:
        return "—"
    mean = f"{stats['mean']:.4f}"
    if stats.get("std") is None:
        return f"{mean} ± —"
    return f"{mean} ± {stats['std']:.4f}"


def write_markdown(path: Path, rows: list[dict], groups: list[str],
                   scorer_order: dict, n_runs: dict[str, int]) -> None:
    lookup = {(row["group"], row["scorer"], row["metric"]): row for row in rows}
    lines = [
        "# 开放集评估多 run 汇总（mean ± std）",
        "",
        f"- 分组：{'、'.join(groups) if groups else '—'} ｜ 正则：见命令行 `--group-regex`",
        f"- 总 run 数：{sum(n_runs.values())}",
        "- std 为样本标准差（n−1）；同一指标的有效 run 数不足 2 时记为 `μ ± —`。",
        "- 缺失字段按跳过处理，不会中断聚合。",
        "",
    ]
    for group in groups:
        lines += [f"## {group}", "", f"- run 数：{n_runs.get(group, 0)}", ""]
        header = "| 打分器 | " + " | ".join(METRIC_ORDER) + " |"
        sep = "| --- | " + " | ".join("---:" for _ in METRIC_ORDER) + " |"
        lines += [header, sep]
        for scorer in scorer_order.get(group, []):
            cells = [_fmt_cell(lookup.get((group, scorer, metric)) or {})
                     for metric in METRIC_ORDER]
            lines.append(f"| {scorer} | " + " | ".join(cells) + " |")
        lines.append("")
        if n_runs.get(group, 0) < 2:
            lines += [f"> {group} 只有 {n_runs.get(group, 0)} 个 run，"
                      "样本标准差无法计算（std 为 None，表格显示 μ ± —）。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = resolve_inputs(args.inputs, args.pattern)
    if not paths:
        print("[错误] 未提供输入：请用 --inputs 或 --pattern 指定 metrics.json")
        return 1

    items: list[tuple[str, dict]] = []
    n_runs: dict[str, int] = {}
    for path in paths:
        report = load_report(path)
        if report is None:
            continue
        group = group_of(path, args.group_regex)
        items.append((group, report))
        n_runs[group] = n_runs.get(group, 0) + 1
    if not items:
        print("[错误] 没有可用的 metrics.json")
        return 1

    rows, groups, scorer_order = build_rows(items)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "aggregate.csv"
    md_path = out_dir / "aggregate.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, groups, scorer_order, n_runs)

    print(f"[信息] 读入 {len(items)} 个 run，分组："
          + "，".join(f"{g}={n}" for g, n in n_runs.items()))
    print(f"[输出] {csv_path}")
    print(f"[输出] {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
