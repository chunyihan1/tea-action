"""置信度阈值自动标定：在验证集上扫描多个 conf，输出 P/R/F1/mAP，自动推荐 F1 最高的阈值。

为什么需要：页面默认阈值 0.25 是经验值，未必适合当前权重。用验证集扫一遍，
通常能提升 1~3 个点的实际可用性，且不需要任何新增标注、不需要重新训练。

用法：
  python training/calibrate_conf.py
  python training/calibrate_conf.py --confs 0.10,0.15,0.20,0.25,0.30,0.35,0.40
  python training/calibrate_conf.py --imgsz 320 --device cpu

前置：数据集已下载（python training/download_data.py）、权重已训练（data/models/tea_yolo.pt）。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = BASE / "data" / "models" / "tea_yolo.pt"
DATASET_DIR = BASE / "training" / "datasets" / "tea_leaf_disease"


def find_data_yaml(dataset_dir: Path) -> Path | None:
    """定位可用的 data.yaml：优先训练脚本产出的 data_norm.yaml。"""
    for name in ("data_norm.yaml", "data.yaml"):
        p = dataset_dir / name
        if p.exists():
            return p
    hits = sorted(dataset_dir.rglob("data_norm.yaml")) or sorted(dataset_dir.rglob("data.yaml"))
    return hits[0] if hits else None


def parse_confs(text: str) -> list[float]:
    confs = []
    for part in text.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        val = float(part)
        if not 0.0 < val < 1.0:
            raise argparse.ArgumentTypeError(f"阈值必须在 (0, 1) 之间：{val}")
        confs.append(round(val, 3))
    if not confs:
        raise argparse.ArgumentTypeError("至少需要一个阈值")
    return sorted(set(confs))


def main() -> None:
    parser = argparse.ArgumentParser(description="扫描并推荐最佳置信度阈值")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="模型权重路径")
    parser.add_argument("--data", default=None, help="data.yaml 路径（默认自动查找）")
    parser.add_argument("--imgsz", type=int, default=640, help="验证分辨率，需与部署一致")
    parser.add_argument("--iou", type=float, default=0.5, help="评估 IoU 阈值")
    parser.add_argument("--confs", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50",
                        help="逗号分隔的阈值列表")
    parser.add_argument("--device", default=None, help="cpu / 0 / 留空自动")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[错误] 找不到权重：{model_path}")
        print("       请先训练：python training/train_yolo.py --full")
        sys.exit(1)

    data_yaml = Path(args.data) if args.data else find_data_yaml(DATASET_DIR)
    if data_yaml is None or not data_yaml.exists():
        print(f"[错误] 找不到数据集配置（{DATASET_DIR}）")
        print("       请先下载：python training/download_data.py")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except Exception as e:  # noqa: BLE001
        print(f"[错误] 需要 ultralytics：pip install ultralytics\n{e}")
        sys.exit(1)

    confs = parse_confs(args.confs)
    print(f"[信息] 权重：{model_path}")
    print(f"[信息] 数据：{data_yaml}")
    print(f"[信息] 分辨率：{args.imgsz} ｜ 评估 IoU：{args.iou} ｜ 阈值：{confs}")
    print("[信息] 每个阈值都会完整跑一遍验证集，请耐心等待……\n")

    model = YOLO(str(model_path))
    rows = []
    for conf in confs:
        metrics = model.val(
            data=str(data_yaml),
            imgsz=args.imgsz,
            iou=args.iou,
            conf=conf,
            device=args.device,
            verbose=False,
            plots=False,
            project=str(BASE / "training" / "runs"),
            name="conf_calib",
            exist_ok=True,
        )
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append((conf, float(metrics.box.map50), precision, recall, f1))
        print(f"  conf={conf:.2f}  mAP50={rows[-1][1]:.4f}  "
              f"P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}")

    best = max(rows, key=lambda x: x[4])
    print("\n" + "=" * 62)
    print(f"{'conf':>6} {'mAP50':>9} {'Precision':>11} {'Recall':>9} {'F1':>8}")
    print("-" * 62)
    for conf, map50, p, r, f1 in rows:
        mark = "  ← 推荐" if (conf, map50, p, r, f1) == best else ""
        print(f"{conf:>6.2f} {map50:>9.4f} {p:>11.4f} {r:>9.4f} {f1:>8.4f}{mark}")
    print("=" * 62)
    print(f"\n[结论] 推荐阈值：{best[0]:.2f}（F1={best[4]:.4f}，"
          f"P={best[2]:.4f}，R={best[3]:.4f}）")

    if best[0] in (confs[0], confs[-1]):
        print("[提醒] 最优值落在扫描区间边界，建议把 --confs 范围向该方向再扩一档。")

    # 写入文件供上传页读取默认阈值（找不到该文件时页面仍用 0.25）
    out = BASE / "data" / "models" / "conf_threshold.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "conf": best[0],
        "f1": round(best[4], 4),
        "precision": round(best[2], 4),
        "recall": round(best[3], 4),
        "map50": round(best[1], 4),
        "imgsz": args.imgsz,
        "model": model_path.name,
        "time": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[记录] 已写入 {out}，「病虫害识别」页会自动把默认阈值设为 {best[0]:.2f}")

    print("[下一步] 重启网页即可生效；也可在页面上手动微调。")
    print("[提示] 若更在意漏检（宁可多报），选 F1 相近但 Recall 更高的阈值。")


if __name__ == "__main__":
    main()
