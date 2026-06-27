#!/usr/bin/env python3
"""Audit RL env observation encoding and action decode vs BC live bridge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from config.settings import BC_MODEL_PATH
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_LOGS = ROOT / "logs" / "eval"
from src.doubles.evaluation.rl_env_alignment import (
    run_live_rl_alignment,
    run_trace_rl_alignment,
    save_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-json",
        type=Path,
        default=EVAL_LOGS
        / "pipeline_20260614_024848"
        / "inference_trace"
        / "inference_trace_latest.json",
        help="Inference trace for offline tensor/pred/decode audit",
    )
    parser.add_argument("--live", action="store_true", help="Also run live VGCRLEnv steps")
    parser.add_argument("--live-steps", type=int, default=40)
    parser.add_argument("--model", type=Path, default=BC_MODEL_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=EVAL_LOGS / "rl_alignment",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.trace_json.is_file():
        print(f"Trace audit: {args.trace_json}", flush=True)
        trace_report = run_trace_rl_alignment(
            args.trace_json, model_path=args.model, device=args.device
        )
        save_report(trace_report, args.out_dir, "rl_trace_alignment_latest")
        print("\n".join(trace_report.lines[-8:]))
        print(f"\nSaved -> {trace_report.txt_path}")
    else:
        print(f"Trace not found (skip): {args.trace_json}", flush=True)
        trace_report = None

    if args.live:
        print(f"\nLive audit: {args.live_steps} steps", flush=True)
        live_report = run_live_rl_alignment(
            n_steps=args.live_steps, model_path=args.model, device=args.device
        )
        save_report(live_report, args.out_dir, "rl_live_alignment_latest")
        print("\n".join(live_report.lines[-10:]))
        print(f"\nSaved -> {live_report.txt_path}")


if __name__ == "__main__":
    main()
