"""多模型横向对比：产出论文表 1（同数据、同 imgsz、同验证集）。

为什么需要：
  只有单一模型的 mAP 无法说明方法好坏，审稿人要求与同量级基线对比（YOLOv8n/s/m、YOLO11n/s）。

设计要点：
  - 每个模型独立 try/except：权重下载失败、CUDA 不可用、验证报错都只记一行 status=error，
    然后继续下一个，保证一次运行能拿全所有可跑的结果；
  - 参数量直接数参数（sum(p.numel())），GFLOPs 尽力从 model.info() 取，取不到就留空；
  - --augment 打开即 TTA 消融，与默认行一起交给 make_tables.py 生成消融表；
  - --latency-image 可选：单图 warmup 3 次 + 计时 20 次取中位数，模拟端到端延迟。

用法：
  python training\\research\\benchmark.py
  python training\\research\\benchmark.py --models yolov8n.pt,yolov8s.pt --augment
  python training\\research\\benchmark.py --latency-image data\\samples\\tea.jpg --device cpu
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# 项目根目录：training/research/benchmark.py -> 上溯两级
BASE = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE / "training" / "datasets" / "tea_leaf_disease"
DEFAULT_MODELS = "yolov8n.pt,yolov8s.pt,yolov8m.pt,yolo11n.pt,yolo11s.pt"
FIELDNAMES = [
    "status", "model", "params", "gflops", "precision", "recall",
    "map50", "map50_95", "preprocess_ms", "inference_ms", "postprocess_ms",
    "latency_ms", "imgsz", "augment", "timestamp", "error",
]


def find_data_yaml(dataset_dir: Path) -> Path | None:
    """自动定位数据集配置：优先 data.yaml，其次训练脚本生成的 data_norm.yaml。"""
    for name in ("data.yaml", "data_norm.yaml"):
        p = dataset_dir / name
        if p.exists():
            return p
    for pattern in ("data.yaml", "data_norm.yaml"):
        hits = sorted(dataset_dir.rglob(pattern))
        if hits:
            return hits[0]
    return None


def parse_models(text: str) -> list[str]:
    models = [x.strip() for x in text.replace("，", ",").split(",") if x.strip()]
    if not models:
        raise argparse.ArgumentTypeError("至少需要一个模型权重")
    return models


def _safe_float(value) -> float | None:
    try:
        f = float(value)
        return f if f == f else None  # 过滤 NaN
    except (TypeError, ValueError):
        return None


def measure_latency(model, image: str, imgsz: int, device: str | None,
                    warmup: int = 3, runs: int = 20) -> float | None:
    """单图端到端延迟：warmup 后计时取中位数（ms）。失败返回 None。"""
    try:
        for _ in range(warmup):
            model.predict(source=image, imgsz=imgsz, device=device, verbose=False)
        costs: list[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model.predict(source=image, imgsz=imgsz, device=device, verbose=False)
            costs.append((time.perf_counter() - t0) * 1000.0)
        return round(statistics.median(costs), 2)
    except Exception as e:  # noqa: BLE001 - 延迟只是附加指标，失败不影响主表
        print(f"  [警告] 延迟测试失败：{' '.join(str(e).split())[:120]}")
        return None


def run_one(weights: str, data: Path, imgsz: int, device: str | None,
            augment: bool, latency_image: str | None, out_dir: Path) -> dict:
    """跑通一个模型并返回一行结果；任何异常都收敛成 status=error。"""
    row: dict = {k: "" for k in FIELDNAMES}
    row.update({
        "status": "ok",
        "model": Path(weights).name,
        "imgsz": imgsz,
        "augment": augment,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    try:
        from ultralytics import YOLO

        model = YOLO(weights)

        # 参数量：直接数，避免版本差异
        try:
            row["params"] = int(sum(p.numel() for p in model.model.parameters()))
        except Exception:  # noqa: BLE001
            row["params"] = ""

        # GFLOPs：不同版本返回格式不一，取不到就留空
        try:
            info = model.info(verbose=False)
            if isinstance(info, (tuple, list)) and len(info) >= 4:
                gf = _safe_float(info[3])
                if gf:
                    row["gflops"] = round(gf, 2)
        except Exception:  # noqa: BLE001
            pass

        metrics = model.val(
            data=str(data),
            imgsz=imgsz,
            device=device,
            split="val",
            augment=augment,
            plots=False,
            verbose=False,
            project=str(out_dir / "runs"),
            name="bench_val",
            exist_ok=True,
        )
        box = metrics.box
        row["precision"] = round(float(box.mp), 4)
        row["recall"] = round(float(box.mr), 4)
        row["map50"] = round(float(box.map50), 4)
        row["map50_95"] = round(float(box.map), 4)

        speed = getattr(metrics, "speed", None) or {}
        for key, col in (("preprocess", "preprocess_ms"),
                         ("inference", "inference_ms"),
                         ("postprocess", "postprocess_ms")):
            val = _safe_float(speed.get(key))
            if val is not None:
                row[col] = round(val, 3)

        if latency_image:
            latency = measure_latency(model, latency_image, imgsz, device)
            if latency is not None:
                row["latency_ms"] = latency
    except Exception as e:  # noqa: BLE001 - 单模型失败不能中断整批
        row["status"] = "error"
        row["error"] = " ".join(str(e).split())[:200] or e.__class__.__name__
        print(f"  [失败] {row['model']}：{row['error']}")
        traceback.print_exc(limit=1)
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def write_markdown(path: Path, rows: list[dict], data: Path, imgsz: int,
                   augment: bool, error_count: int) -> None:
    def sort_key(row: dict):
        value = _safe_float(row.get("map50_95"))
        return (row.get("status") == "ok" and value is not None, value or -1.0)

    ok_rows = sorted([r for r in rows if r.get("status") == "ok"], key=sort_key, reverse=True)
    lines = [
        "<!-- 数据来源：benchmark.csv（本目录） -->",
        "# 表 1 多模型检测性能对比",
        "",
        f"- 数据集：`{data}`",
        f"- 输入分辨率：{imgsz}；TTA（augment）：{augment}",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "| 排名 | 模型 | 参数量(M) | GFLOPs | P | R | mAP@0.5 | mAP@0.5:0.95 | 推理(ms/img) | 延迟(ms) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, r in enumerate(ok_rows, 1):
        params = _safe_float(r.get("params"))
        params_txt = f"{params / 1e6:.2f}" if params else "—"
        gflops = _safe_float(r.get("gflops"))
        latency = _safe_float(r.get("latency_ms"))
        lines.append(
            f"| {rank} | {r['model']} | {params_txt} | "
            f"{f'{gflops:.2f}' if gflops else '—'} | "
            f"{_safe_float(r.get('precision')) or 0:.3f} | "
            f"{_safe_float(r.get('recall')) or 0:.3f} | "
            f"{_safe_float(r.get('map50')) or 0:.3f} | "
            f"{_safe_float(r.get('map50_95')) or 0:.3f} | "
            f"{_safe_float(r.get('inference_ms')) or 0:.2f} | "
            f"{f'{latency:.2f}' if latency is not None else '—'} |"
        )
    if error_count:
        failed = [r["model"] for r in rows if r.get("status") != "ok"]
        lines += ["", f"表注：{error_count} 个模型运行失败（{'、'.join(failed)}），未计入排名。"]
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="多模型横向对比（论文表 1）")
    parser.add_argument("--data", default=None,
                        help="data.yaml 路径（默认自动查找 training/datasets/tea_leaf_disease）")
    parser.add_argument("--models", default=DEFAULT_MODELS, type=parse_models,
                        help=f"逗号分隔的模型权重，默认 {DEFAULT_MODELS}")
    parser.add_argument("--imgsz", type=int, default=640, help="验证分辨率")
    parser.add_argument("--device", default="", help="cpu / 0 / 留空自动")
    parser.add_argument("--augment", action="store_true", help="开启 TTA 验证（消融用）")
    parser.add_argument("--out", default="research_out", help="输出目录")
    parser.add_argument("--latency-image", default=None, help="用于测端到端延迟的单张图片")
    args = parser.parse_args()

    data = Path(args.data) if args.data else find_data_yaml(DATASET_DIR)
    if data is None or not data.exists():
        print(f"[错误] 找不到 data.yaml（默认目录：{DATASET_DIR}）")
        print("       请先准备数据集：python training\\download_data.py")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or None

    if args.latency_image and not Path(args.latency_image).exists():
        print(f"[警告] 延迟测试图片不存在：{args.latency_image}，将跳过延迟测试")
        args.latency_image = None

    print(f"[信息] 数据：{data}")
    print(f"[信息] 模型：{args.models}")
    print(f"[信息] imgsz={args.imgsz} ｜ device={device or 'auto'} ｜ TTA={args.augment}")
    print("[提示] 首次运行会下载权重（需网络）；单模型失败会自动跳过。\n")

    rows: list[dict] = []
    for i, weights in enumerate(args.models, 1):
        print(f"[{i}/{len(args.models)}] {weights}")
        t0 = time.time()
        row = run_one(weights, data, args.imgsz, device, args.augment,
                      args.latency_image, out_dir)
        rows.append(row)
        if row["status"] == "ok":
            print(f"  [完成] mAP50={row['map50']:.4f}  mAP50-95={row['map50_95']:.4f}  "
                  f"P={row['precision']:.3f}  R={row['recall']:.3f}  "
                  f"耗时 {time.time() - t0:.0f}s")
        # 每跑完一个就落盘，避免中途崩溃丢失已完成结果
        write_csv(out_dir / "benchmark.csv", rows)
        write_markdown(out_dir / "benchmark.md", rows, data, args.imgsz,
                       args.augment, error_count=sum(r["status"] != "ok" for r in rows))

    success = sum(r["status"] == "ok" for r in rows)
    failed = len(rows) - success
    print(f"\n[输出] {out_dir / 'benchmark.csv'}")
    print(f"[输出] {out_dir / 'benchmark.md'}")
    print(f"完成 {len(rows)} 个模型，成功 {success}，失败 {failed}")


if __name__ == "__main__":
    main()
