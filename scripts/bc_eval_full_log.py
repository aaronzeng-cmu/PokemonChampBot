#!/usr/bin/env python3
"""Full validation-set BC eval with log-reconstructed masking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from config.settings import BC_DATASET_PATH, BC_EVAL_LOG_DIR, BC_MODEL_PATH, RAW_LOGS_DIR
from src.doubles.evaluation.bc_full_log_eval import evaluate_bc_full_log, write_full_log_eval_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Full val BC eval with log masking")
    parser.add_argument("--model", type=Path, default=BC_MODEL_PATH)
    parser.add_argument("--dataset", type=Path, default=BC_DATASET_PATH)
    parser.add_argument("--logs", type=Path, default=RAW_LOGS_DIR)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=BC_EVAL_LOG_DIR,
    )
    args = parser.parse_args()

    metrics = evaluate_bc_full_log(
        model_path=args.model,
        dataset_path=args.dataset,
        log_dir=args.logs,
        val_frac=args.val_frac,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
    )
    out_path = write_full_log_eval_report(
        metrics,
        args.out_dir,
        model_path=args.model,
        dataset_path=args.dataset,
    )

    print(f"Val samples: {metrics.n_val}")
    print(f"With log view: {metrics.n_with_log} | missing log: {metrics.n_missing_log}")
    print(f"Slot top-1 (masked+log): {100 * metrics.slot_top1:.2f}%")
    print(f"Joint top-1 (masked+log): {100 * metrics.joint_top1:.2f}%")
    print(f"Masking overrides (raw argmax != masked): {metrics.masking_overrides}")
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
