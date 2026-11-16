"""多种子重复实验：训练 + 验证，输出 mean ± std（SCI 的基本要求）。

为什么需要：
  单次训练 0.740 与基线 0.735 的差距可能完全落在随机初始化的噪声里。
  期刊要求至少 5 个 seed 的均值与标准差，本脚本把这套流程批量化并自动汇总。

实现要点：
  - 每个 seed 依次执行 `yolo detect train ... seed=S deterministic=True`，
    再对 `best.pt` 执行 `yolo detect val`；
  - 命令一律用**列表**形式交给 subprocess，Windows 下不会被引号/空格坑到；
  - val 输出解析失败时该 seed 记 status=error 并继续下一个，保证一次跑完能拿到尽可能多的数据；
  - --dry-run 只打印命令（每个 seed 一行，train && val），不加载权重也不训练，可离线验证。

用法：
  python training\\research\\seed_study.py --seeds 1,2,3,4,5 --data <data.yaml> --dry-run
  python training\\research\\seed_study.py --seeds 42,43,44,45,46 --data <data.yaml> --epochs 100
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ultralytics 验证输出里 "all" 行的四列指标
VAL_RE = re.compile(
    r"^\s*all\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", re.M
)
FIELDNAMES = ["seed", "status", "best_weights", "precision", "recall",
              "map50", "map50_95", "epochs", "imgsz", "batch", "seconds",
              "error", "timestamp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多种子重复实验（mean ± std）")
    parser.add_argument("--seeds", default="1,2,3,4,5", help="逗号分隔的随机种子")
    parser.add_argument("--data", required=True, help="data.yaml 路径")
    parser.add_argument("--base-model", default="yolov8s.pt", help="基础权重")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入分辨率")
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--device", default="0", help="cpu / 0 / 0,1")
    parser.add_argument("--out", default="research_out/seeds", help="输出目录")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的命令")
    return parser.parse_args()


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            seeds.append(int(part))
        except ValueError as e:
            raise SystemExit(f"[错误] 无法解析 seed：{part}") from e
    if not seeds:
        raise SystemExit("[错误] --seeds 至少需要一个整数")
    return seeds


def yolo_prefix() -> list[str]:
    """优先用 PATH 里的 yolo 可执行文件；找不到时回退到 python -m ultralytics。"""
    exe = shutil.which("yolo")
    if exe:
        return [exe]
    print("[警告] 未在 PATH 中找到 yolo，改用 `python -m ultralytics` 调用")
    return [sys.executable, "-m", "ultralytics"]


def build_commands(seed: int, args: argparse.Namespace,
                   prefix: list[str]) -> tuple[list[str], list[str], Path]:
    """返回 (train 命令, val 命令, best.pt 路径)，均为列表形式。"""
    runs_dir = Path(args.out) / "runs"
    name = f"seed_{seed}"
    best = runs_dir / name / "weights" / "best.pt"
    train = prefix + [
        "detect", "train",
        f"model={args.base_model}",
        f"data={args.data}",
        f"epochs={args.epochs}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"device={args.device}",
        f"seed={seed}",
        "deterministic=True",
        f"project={runs_dir}",
        f"name={name}",
        "exist_ok=True",
    ]
    val = prefix + [
        "detect", "val",
        f"model={best}",
        f"data={args.data}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"device={args.device}",
        "split=val",
        "plots=False",
    ]
    return train, val, best


def parse_val_metrics(text: str) -> dict | None:
    """从 val 输出解析 all 行的 P / R / mAP50 / mAP50-95。"""
    matches = VAL_RE.findall(text or "")
    if not matches:
        return None
    _images, _instances, precision, recall, map50, map50_95 = matches[-1]
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "map50": round(float(map50), 4),
        "map50_95": round(float(map50_95), 4),
    }


def run_seed(seed: int, args: argparse.Namespace, prefix: list[str]) -> dict:
    """跑完一个 seed 的 train + val，异常收敛成 status=error 并继续。"""
    row: dict = {k: "" for k in FIELDNAMES}
    row.update({"seed": seed, "status": "ok", "epochs": args.epochs,
                "imgsz": args.imgsz, "batch": args.batch,
                "timestamp": datetime.now().isoformat(timespec="seconds")})
    train_cmd, val_cmd, best = build_commands(seed, args, prefix)
    row["best_weights"] = str(best)
    t0 = time.time()

    try:
        print(f"  [训练] {' '.join(train_cmd)}")
        proc = subprocess.run(train_cmd, capture_output=True, text=True,
                             encoding="utf-8", errors="ignore")
        if proc.returncode != 0:
            tail = " ".join((proc.stderr or proc.stdout or "").strip().split())[-300:]
            row["status"] = "error"
            row["error"] = f"train 退出码 {proc.returncode}：{tail}"
            print(f"  [失败] seed={seed} 训练失败：{row['error']}")
            return row
        if not best.is_file():
            row["status"] = "error"
            row["error"] = f"训练结束但找不到 {best}"
            print(f"  [失败] seed={seed}：{row['error']}")
            return row

        print(f"  [验证] {' '.join(val_cmd)}")
        proc = subprocess.run(val_cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="ignore")
        metrics = parse_val_metrics((proc.stdout or "") + "\n" + (proc.stderr or ""))
        if proc.returncode != 0 or metrics is None:
            row["status"] = "error"
            tail = " ".join((proc.stderr or proc.stdout or "").strip().split())[-300:]
            row["error"] = (f"val 解析失败（退出码 {proc.returncode}）：{tail}"
                            if metrics is None else f"val 退出码 {proc.returncode}")
            print(f"  [失败] seed={seed}：{row['error']}")
            return row
        row.update(metrics)
    except Exception as e:  # noqa: BLE001 - 单个 seed 失败不能中断整批
        row["status"] = "error"
        row["error"] = " ".join(str(e).split())[:200] or e.__class__.__name__
        print(f"  [失败] seed={seed}：{row['error']}")
    finally:
        row["seconds"] = round(time.time() - t0, 1)
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def summarize(values: list[float]) -> tuple[float, float] | tuple[None, None]:
    """样本均值与样本标准差（ddof=1）；少于 2 个样本时标准差无定义。"""
    if not values:
        return None, None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) >= 2 else None
    return mean, std


def write_markdown(path: Path, rows: list[dict], args: argparse.Namespace) -> None:
    ok = [r for r in rows if r["status"] == "ok"]
    lines = [
        "# 多种子重复实验",
        "",
        f"- data.yaml：`{args.data}` ｜ 基础权重：`{args.base_model}`",
        f"- 训练设置：epochs={args.epochs}，imgsz={args.imgsz}，batch={args.batch}，"
        f"device={args.device}，deterministic=True",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 1. 逐 seed 结果",
        "",
        "| Seed | 状态 | P | R | mAP@0.5 | mAP@0.5:0.95 | 耗时(s) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        if r["status"] == "ok":
            lines.append(
                f"| {r['seed']} | ok | {r['precision']:.4f} | {r['recall']:.4f} | "
                f"{r['map50']:.4f} | {r['map50_95']:.4f} | {r['seconds']} |"
            )
        else:
            lines.append(f"| {r['seed']} | error | — | — | — | — | {r['seconds']} |")
    lines += ["", "## 2. 汇总（mean ± std）", ""]
    if ok:
        lines += ["| 指标 | mean ± std | n |", "| --- | --- | ---: |"]
        for key, label in (("precision", "P"), ("recall", "R"),
                           ("map50", "mAP@0.5"), ("map50_95", "mAP@0.5:0.95")):
            values = [float(r[key]) for r in ok]
            mean, std = summarize(values)
            text = f"{mean:.4f} ± {std:.4f}" if std is not None else f"{mean:.4f} ± —"
            lines.append(f"| {label} | {text} | {len(values)} |")
        lines += ["", "表注：std 为样本标准差（ddof=1），n 为成功训练的 seed 数。", ""]
    else:
        lines += ["没有成功完成的 seed，无法汇总。", ""]

    failed = [r for r in rows if r["status"] != "ok"]
    if failed:
        lines += ["## 3. 失败记录", "", "| Seed | 原因 |", "| ---: | --- |"]
        for r in failed:
            lines.append(f"| {r['seed']} | {r['error']} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    prefix = yolo_prefix()

    if args.dry_run:
        print(f"[dry-run] 共 {len(seeds)} 个 seed，每个 seed 执行 1 次 train + 1 次 val：\n")
        for i, seed in enumerate(seeds, 1):
            train_cmd, val_cmd, _best = build_commands(seed, args, prefix)
            # 一行一个 seed，用 && 连接，方便直接复制到 cmd/powershell 中手工复现
            line = (subprocess.list2cmdline(train_cmd) + " && "
                    + subprocess.list2cmdline(val_cmd))
            print(f"[{i}/{len(seeds)}] {line}\n")
        print("[dry-run] 未执行任何训练；去掉 --dry-run 即开始正式实验。")
        return

    data = Path(args.data)
    if not data.is_file():
        print(f"[错误] data.yaml 不存在：{data}")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)

    print(f"[信息] seeds={seeds} ｜ data={data} ｜ base={args.base_model}")
    print(f"[信息] epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} device={args.device}")
    print(f"[信息] 输出目录：{out_dir}\n")

    rows: list[dict] = []
    for i, seed in enumerate(seeds, 1):
        print(f"[{i}/{len(seeds)}] seed={seed}")
        row = run_seed(seed, args, prefix)
        rows.append(row)
        if row["status"] == "ok":
            print(f"  [完成] mAP50={row['map50']:.4f}  mAP50-95={row['map50_95']:.4f}  "
                  f"耗时 {row['seconds']}s")
        # 每个 seed 结束就落盘，中途中断也不丢已完成的结果
        write_csv(out_dir / "seeds.csv", rows)
        write_markdown(out_dir / "seeds.md", rows, args)

    ok = [r for r in rows if r["status"] == "ok"]
    print(f"\n[输出] {out_dir / 'seeds.csv'}")
    print(f"[输出] {out_dir / 'seeds.md'}")
    print(f"完成 {len(rows)} 个 seed，成功 {len(ok)}，失败 {len(rows) - len(ok)}")
    if ok:
        mean, std = summarize([float(r["map50_95"]) for r in ok])
        std_txt = f"{std:.4f}" if std is not None else "—"
        print(f"[汇总] mAP@0.5:0.95 = {mean:.4f} ± {std_txt}（n={len(ok)}）")
    if len(ok) < len(rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
