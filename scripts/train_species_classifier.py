"""Train a YOLOv8-cls species classifier on the synthetic dataset.

Usage::

    python scripts/train_species_classifier.py \
        --data data/species_cls --epochs 30 --imgsz 96

Copies the best checkpoint to src/cv_bridge/assets/species_cls.pt for inference.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

_ASSET_WEIGHTS = Path("src/cv_bridge/assets/species_cls.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train YOLOv8-cls species classifier.")
    ap.add_argument("--data", type=Path, default=Path("data/species_cls"))
    ap.add_argument("--model", default="yolov8n-cls.pt", help="Base cls checkpoint.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=96)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="0", help="CUDA index or 'cpu'.")
    ap.add_argument("--out", type=Path, default=_ASSET_WEIGHTS)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/species_cls",
        name="train",
        exist_ok=True,
        # mild on-the-fly aug on top of our synthetic variety
        hsv_h=0.02,
        hsv_s=0.4,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.0,
        erasing=0.2,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.is_file():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, args.out)
        print(f"[train] copied best weights -> {args.out}")
    else:
        print(f"[train][warn] best.pt not found at {best}")

    top1 = getattr(results, "top1", None)
    top5 = getattr(results, "top5", None)
    print(f"[train] done. val top1={top1} top5={top5}")


if __name__ == "__main__":
    main()
