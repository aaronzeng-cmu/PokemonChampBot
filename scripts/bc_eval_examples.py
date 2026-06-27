#!/usr/bin/env python3
"""Print BC validation examples: model prediction vs logged human action."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from config.settings import BC_DATASET_PATH, BC_EVAL_LOG_DIR, BC_MODEL_PATH, RAW_LOGS_DIR
from src.doubles.evaluation.bc_examples import generate_bc_examples, write_bc_examples_report


def main() -> None:
    parser = argparse.ArgumentParser(description="BC prediction vs ground-truth examples")
    parser.add_argument("--model", type=Path, default=BC_MODEL_PATH)
    parser.add_argument("--dataset", type=Path, default=BC_DATASET_PATH)
    parser.add_argument("--logs", type=Path, default=RAW_LOGS_DIR)
    parser.add_argument("--n", type=int, default=50, help="Number of examples")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k predictions per slot")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mix",
        choices=("random", "correct", "incorrect"),
        default="random",
        help="Sample mix: all random val, or only joint-correct / joint-wrong",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=BC_EVAL_LOG_DIR / "bc_examples",
    )
    args = parser.parse_args()

    examples = generate_bc_examples(
        model_path=args.model,
        dataset_path=args.dataset,
        log_dir=args.logs,
        n_examples=args.n,
        val_frac=args.val_frac,
        seed=args.seed,
        device=args.device,
        mix=args.mix,
        top_k=args.top_k,
    )
    txt_path, json_path = write_bc_examples_report(
        examples,
        args.out_dir,
        model_path=args.model,
        dataset_path=args.dataset,
        mix=args.mix,
    )

    if not examples:
        print("No examples matched the requested mix.")
        return

    n = len(examples)
    joint = sum(1 for e in examples if e.correct_joint)
    top3_avg = sum(e.top3_slot0_hit + e.top3_slot1_hit for e in examples) / (2 * n) if n else 0.0
    raw_legal = sum(int(e.raw_slot0_legal) + int(e.raw_slot1_legal) for e in examples)
    print(f"Examples: {n} (mix={args.mix})")
    print(f"Joint top-1 on sample: {joint}/{n} ({100 * joint / n:.1f}%)")
    print(f"Raw top-1 legal: {raw_legal}/{2 * n} ({100 * raw_legal / (2 * n):.1f}%)")
    print(f"Avg top-{args.top_k} hit rate: {100 * top3_avg:.1f}%")
    print(f"Text report: {txt_path}")
    print(f"JSON report: {json_path}")
    print()
    print(examples[0].to_text_block().rstrip())
    if len(examples) > 1:
        print(f"... ({len(examples) - 1} more in {txt_path.name})")


if __name__ == "__main__":
    main()
