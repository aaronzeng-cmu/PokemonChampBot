#!/usr/bin/env python3
"""Run the standard BC Transformer evaluation pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from config.settings import BC_EVAL_LOG_DIR, BC_MODEL_PATH
from src.doubles.evaluation.eval_pipeline import EvalPipelineConfig, run_eval_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standard post-training eval: BC examples, 100b live, replays, trace, alignment"
    )
    parser.add_argument("--model", type=Path, default=BC_MODEL_PATH)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--eval-battles", type=int, default=100)
    parser.add_argument("--replay-battles", type=int, default=10)
    parser.add_argument("--trace-battles", type=int, default=1)
    parser.add_argument("--bc-examples", type=int, default=50)
    parser.add_argument("--out-dir", type=Path, default=BC_EVAL_LOG_DIR)
    parser.add_argument("--skip-eval", action="store_true", help="Skip 100-battle live eval")
    parser.add_argument("--skip-replays", action="store_true")
    parser.add_argument("--skip-trace", action="store_true")
    parser.add_argument("--skip-bc-examples", action="store_true")
    parser.add_argument("--skip-alignment", action="store_true")
    args = parser.parse_args()

    cfg = EvalPipelineConfig(
        model_path=args.model,
        device=args.device,
        out_root=args.out_dir,
        eval_battles=args.eval_battles,
        replay_battles=args.replay_battles,
        trace_battles=args.trace_battles,
        bc_examples_n=args.bc_examples,
        skip_eval=args.skip_eval,
        skip_replays=args.skip_replays,
        skip_trace=args.skip_trace,
        skip_bc_examples=args.skip_bc_examples,
        skip_alignment=args.skip_alignment,
    )
    run_eval_pipeline(cfg)


if __name__ == "__main__":
    main()
