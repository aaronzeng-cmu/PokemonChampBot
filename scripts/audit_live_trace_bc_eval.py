#!/usr/bin/env python3
"""Parse a live inference trace protocol log through BC eval and compare with live decisions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from config.settings import BC_MODEL_PATH
from src.doubles.evaluation.live_bc_alignment import build_bc_audit_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-json", type=Path, required=True)
    parser.add_argument("--side", default="p1")
    parser.add_argument("--model", type=Path, default=BC_MODEL_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = build_bc_audit_report(
        args.trace_json,
        side=args.side,
        model_path=args.model,
        device=args.device,
        top_k=args.top_k,
    )
    out = args.out or Path("logs/eval") / f"live_trace_bc_audit_{args.trace_json.stem}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved -> {out.resolve()}")


if __name__ == "__main__":
    main()
