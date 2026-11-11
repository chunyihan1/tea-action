"""伪标签自训练：用当前模型给未标注照片自动打框，过滤后并入训练集（不产生人工标注）。

安全设计：
  1. 默认只做「试运行」，只打印统计，不动任何文件；确认后再加 --apply。
  2. 只接受置信度 >= --conf（默认 0.6）且检出数在 1~--max-det 之间的图片，
     高置信度多目标/低置信度图片一律丢弃，避免把误检学进模型。
  3. 扩充量不超过原训练集的 --max-ratio（默认 20%），防止伪标签淹没真实标签。
  4. 文件名加 pseudo_ 前缀，并写入 pseudo_manifest.jsonl；--revert 可完整回滚。
  5. 自动跳过与训练/验证/测试集重复的图片（按文件 MD5）。
  6. 类别名与数据集不一致时直接终止，避免把类别号写错。

用法：
  # 1) 把没标注的茶园照片放进 data/incoming/
  python training/pseudo_label.py                  # 试运行，看会接受多少张
  python training/pseudo_label.py --apply          # 真正写入训练集
  python training/pseudo_label.py --revert         # 反悔：删掉上次写入的全部伪标签
  # 2) 重新训练（伪标签与真实标签一起参与）
  python training/train_yolo.py --full

注意：伪标签不能替代真实标注；迭代 2~3 轮后收益递减。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = BASE / "data" / "models" / "tea_yolo.pt"
DEFAULT_SRC = BASE / "data" / "incoming"
DATASET_DIR = BASE / "training" / "datasets" / "tea_leaf_disease"
IMG_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_dataset_root(dataset_dir: Path) -> Path | None:
    """定位包含 train/images 的数据集根目录（兼容多一层嵌套）。"""
    for cand in [dataset_dir, *dataset_dir.glob("*")]:
        if (cand / "train" / "images").is_dir():
            return cand
    return None


def load_names(dataset_root: Path) -> list[str] | None:
    """从 data.yaml / data_norm.yaml 读取类别名。"""
    try:
        import yaml
    except Exception:  # noqa: BLE001
        return None
    for name in ("data_norm.yaml", "data.yaml"):
        p = dataset_root / name
        if not p.exists():
            continue
        cfg = yaml.safe_load(p.read_text(encoding="utf-8", errors="ignore")) or {}
        names = cfg.get("names")
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
        if isinstance(names, list):
            return [str(x) for x in names]
    return None


def norm(name: str) -> str:
    return "".join(str(name).lower().split())


def existing_hashes(dataset_root: Path) -> set[str]:
    hashes: set[str] = set()
    for split in ("train", "valid", "test"):
        img_dir = dataset_root / split / "images"
        if not img_dir.is_dir():
            continue
        for p in img_dir.iterdir():
            if p.suffix.lower() in IMG_SUFFIX:
                hashes.add(md5_of(p))
    return hashes


def revert(dataset_root: Path) -> None:
    manifest = dataset_root / "pseudo_manifest.jsonl"
    if not manifest.exists():
        print("[信息] 没有找到伪标签记录，无需回滚。")
        return
    removed = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        for key in ("image", "label"):
            p = dataset_root / item[key]
            if p.exists():
                p.unlink()
                removed += 1
    manifest.unlink()
    print(f"[完成] 已回滚：删除 {removed} 个文件，并移除 pseudo_manifest.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="用现有模型给未标注照片自动打伪标签")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="模型权重路径")
    parser.add_argument("--src", default=str(DEFAULT_SRC), help="未标注图片目录")
    parser.add_argument("--dataset", default=str(DATASET_DIR), help="数据集根目录（含 train/images）")
    parser.add_argument("--conf", type=float, default=0.6, help="接受伪标签的最低置信度")
    parser.add_argument("--imgsz", type=int, default=640, help="推理分辨率")
    parser.add_argument("--max-det", type=int, default=8, help="单图最多接受的目标数，超过则丢弃")
    parser.add_argument("--max-ratio", type=float, default=0.2, help="伪标签图最多占训练集比例")
    parser.add_argument("--apply", action="store_true", help="真正写入数据集（默认只试运行）")
    parser.add_argument("--revert", action="store_true", help="回滚上次写入的伪标签")
    parser.add_argument("--force", action="store_true", help="类别名不一致时仍然继续（不推荐）")
    args = parser.parse_args()

    if args.conf < 0.5 and not args.force:
        print(f"[错误] 置信度 {args.conf} 过低，会把误检写进训练集；请使用 >= 0.5 的值。")
        print("       确需测试可用 --force（不建议用于真实数据）。")
        sys.exit(1)

    dataset_root = find_dataset_root(Path(args.dataset))
    if dataset_root is None:
        print(f"[错误] 找不到数据集（{args.dataset} 下没有 train/images）")
        print("       请先下载：python training/download_data.py")
        sys.exit(1)

    if args.revert:
        revert(dataset_root)
        return

    model_path = Path(args.model)
    src_dir = Path(args.src)
    if not model_path.exists():
        print(f"[错误] 找不到权重：{model_path}")
        sys.exit(1)
    if not src_dir.is_dir():
        print(f"[错误] 找不到图片目录：{src_dir}")
        print(f"       请先新建该目录，把未标注的茶园照片放进去。")
        sys.exit(1)

    images = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMG_SUFFIX)
    if not images:
        print(f"[信息] {src_dir} 里没有图片，先放几张再运行。")
        return

    try:
        from ultralytics import YOLO
    except Exception as e:  # noqa: BLE001
        print(f"[错误] 需要 ultralytics：pip install ultralytics\n{e}")
        sys.exit(1)

    names = load_names(dataset_root)
    model = YOLO(str(model_path))
    model_names = [str(v) for v in model.names.values()] if isinstance(model.names, dict) else list(model.names)

    if names and [norm(x) for x in names] != [norm(x) for x in model_names]:
        print("[警告] 模型类别与数据集类别不一致：")
        print(f"  模型：{model_names}")
        print(f"  数据：{names}")
        if not args.force:
            print("[终止] 为避免类别号错位，已停止。确认无误可加 --force。")
            sys.exit(1)

    train_imgs = [p for p in (dataset_root / "train" / "images").iterdir()
                  if p.suffix.lower() in IMG_SUFFIX]
    train_count = len(train_imgs)
    limit = max(1, int(train_count * args.max_ratio))
    print(f"[信息] 训练集 {train_count} 张 ｜ 本次最多接受 {limit} 张（{args.max_ratio:.0%}）")
    print(f"[信息] 阈值 {args.conf} ｜ 分辨率 {args.imgsz} ｜ 单图上限 {args.max_det} 个目标")

    known = existing_hashes(dataset_root)
    accepted: list[tuple[Path, list, float]] = []
    stats = {"重复": 0, "无目标": 0, "目标过多": 0, "已接受": 0, "超配额": 0}

    for img in images:
        digest = md5_of(img)
        if digest in known:
            stats["重复"] += 1
            continue
        result = model.predict(source=str(img), conf=args.conf, iou=0.5,
                               imgsz=args.imgsz, verbose=False)[0]
        boxes = result.boxes
        count = 0 if boxes is None else len(boxes)
        if count == 0:
            stats["无目标"] += 1
            continue
        if count > args.max_det:
            stats["目标过多"] += 1
            continue
        if len(accepted) >= limit:
            stats["超配额"] += 1
            continue
        best = float(boxes.conf.max().item())
        accepted.append((img, boxes, best))
        known.add(digest)
        stats["已接受"] += 1

    print("\n" + "=" * 74)
    for img, boxes, best in accepted[:20]:
        print(f"  {img.name[:40]:42s} 目标={len(boxes):2d}  最高置信度={best:.2f}")
    if len(accepted) > 20:
        print(f"  …… 其余 {len(accepted) - 20} 张省略")
    print("=" * 74)
    print("  ".join(f"{k}={v}" for k, v in stats.items()))

    if not accepted:
        print("\n[结论] 没有可接受的图片。可尝试降低 --conf（但不低于 0.5）或换更清晰的照片。")
        return

    if not args.apply:
        print(f"\n[试运行] 将写入 {len(accepted)} 张到 train/，未改动任何文件。")
        print("         确认无误后执行：python training/pseudo_label.py --apply")
        return

    # 类别名 → 数据集类别号（双保险）
    index_by_name = {norm(n): i for i, n in enumerate(names)} if names else None
    img_out_dir = dataset_root / "train" / "images"
    lbl_out_dir = dataset_root / "train" / "labels"
    lbl_out_dir.mkdir(parents=True, exist_ok=True)
    manifest = dataset_root / "pseudo_manifest.jsonl"
    written = 0

    with open(manifest, "a", encoding="utf-8") as mf:
        for img, boxes, _ in accepted:
            digest = md5_of(img)
            stem = f"pseudo_{digest[:8]}_{img.stem}"
            img_dst = img_out_dir / f"{stem}{img.suffix.lower()}"
            lbl_dst = lbl_out_dir / f"{stem}.txt"
            lines = []
            for xywhn, cls, conf in zip(boxes.xywhn.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
                cls = int(cls)
                if index_by_name is not None:
                    cls_name = model_names[cls]
                    if norm(cls_name) not in index_by_name:
                        continue
                    cls = index_by_name[norm(cls_name)]
                cx, cy, w, h = xywhn
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            if not lines:
                continue
            shutil.copy2(img, img_dst)
            lbl_dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
            mf.write(json.dumps({
                "image": str(img_dst.relative_to(dataset_root)).replace("\\", "/"),
                "label": str(lbl_dst.relative_to(dataset_root)).replace("\\", "/"),
                "src": str(img),
                "detections": len(lines),
                "conf": args.conf,
                "time": datetime.now().isoformat(timespec="seconds"),
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n[完成] 已写入 {written} 张伪标签图片（训练集 {train_count} → {train_count + written}）")
    print(f"[记录] {manifest}")
    print("[下一步] 重新训练：python training\\train_yolo.py --full")
    print("[回滚] 反悔执行：python training/pseudo_label.py --revert")


if __name__ == "__main__":
    main()
