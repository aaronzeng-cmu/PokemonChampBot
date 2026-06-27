#!/usr/bin/env python3
"""Export bc_dataset.pt to memory-mapped .npy files for low-RAM training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import BC_DATASET_PATH
from src.core.training.mmap_dataset import export_pt_to_mmap, has_mmap_dataset, mmap_dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=BC_DATASET_PATH)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or mmap_dataset_dir(args.dataset)
    if has_mmap_dataset(out_dir):
        print(f"Mmap dataset already exists: {out_dir}", flush=True)
        return

    if not args.dataset.is_file():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    out = export_pt_to_mmap(args.dataset, out_dir)
    print(f"Exported -> {out}", flush=True)


if __name__ == "__main__":
    main()
