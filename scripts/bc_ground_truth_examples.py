#!/usr/bin/env python3
"""Print ground-truth-only BC examples from the parsed dataset (no model)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import BC_DATASET_PATH, BC_EVAL_LOG_DIR, RAW_LOGS_DIR
from src.doubles.evaluation.bc_ground_truth import (
    generate_ground_truth_examples,
    write_ground_truth_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ground-truth-only parser audit (BC dataset, no model)",
    )
    parser.add_argument("--dataset", type=Path, default=BC_DATASET_PATH)
    parser.add_argument("--logs", type=Path, default=RAW_LOGS_DIR)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--filter",
        choices=("random", "diverse", "protect", "trickroom", "knockoff", "spread", "offensive"),
        default="diverse",
        help="Sample filter (default: diverse mix of move types)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=BC_EVAL_LOG_DIR / "bc_ground_truth",
    )
    parser.add_argument(
        "--print-first",
        type=int,
        default=5,
        help="Print first N examples to console",
    )
    args = parser.parse_args()

    examples = generate_ground_truth_examples(
        dataset_path=args.dataset,
        log_dir=args.logs,
        n_examples=args.n,
        seed=args.seed,
        sample_filter=args.filter,
    )
    txt_path, json_path = write_ground_truth_report(
        examples,
        args.out_dir,
        dataset_path=args.dataset,
        sample_filter=args.filter,
    )

    if not examples:
        print(f"No examples matched filter={args.filter!r}.")
        return

    warned = sum(1 for e in examples if e.warnings)
    print(f"Examples: {len(examples)} (filter={args.filter})")
    print(f"Warnings: {warned}")
    print(f"Text report: {txt_path}")
    print(f"JSON report: {json_path}")
    print()

    for ex in examples[: args.print_first]:
        print(ex.to_text_block().rstrip())
        print()

    remaining = len(examples) - min(args.print_first, len(examples))
    if remaining > 0:
        print(f"... ({remaining} more in {txt_path.name})")


if __name__ == "__main__":
    main()
