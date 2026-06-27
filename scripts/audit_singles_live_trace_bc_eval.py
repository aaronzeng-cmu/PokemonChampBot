#!/usr/bin/env python3
"""Parse a singles live inference trace through BC eval and compare with live decisions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from config.settings import SINGLES_BC_DATASET_PATH, SINGLES_BC_MODEL_PATH
from src.singles.evaluation.live_bc_alignment import run_alignment_checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-json", type=Path, required=True)
    parser.add_argument("--side", default="p1")
    parser.add_argument("--model", type=Path, default=SINGLES_BC_MODEL_PATH)
    parser.add_argument("--dataset", type=Path, default=SINGLES_BC_DATASET_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or Path("logs/eval/singles") / f"alignment_{args.trace_json.stem}"
    report = run_alignment_checks(
        args.trace_json,
        out_dir,
        model_path=args.model,
        dataset_path=args.dataset,
        device=args.device,
        top_k=args.top_k,
        side=args.side,
    )

    print(report.audit_text)
    print("\n" + "=" * 72 + "\n")
    print(report.dataset_text)
    print(
        f"\nAlignment: trajectory {100 * report.traj_match_rate:.0f}%, "
        f"pred {100 * report.pred_match_rate:.0f}% "
        f"({report.n_compared} live decisions)"
    )
    print(f"Saved -> {report.audit_path.resolve()}")
    print(f"Saved -> {report.dataset_path.resolve()}")


if __name__ == "__main__":
    main()
