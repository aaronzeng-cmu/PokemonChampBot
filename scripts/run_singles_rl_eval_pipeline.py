#!/usr/bin/env python3
"""Run the Singles MaskablePPO evaluation pipeline (policy examples, replays, traces)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from config.settings import (
    BC_EVAL_LOG_DIR,
    SINGLES_BC_MODEL_PATH,
    SINGLES_PREVIEW_MODEL_PATH,
)
from src.singles.evaluation.rl_eval_pipeline import RLEvalPipelineConfig, run_rl_eval_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Singles RL post-training eval: policy examples, replays, inference trace"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="MaskablePPO .zip (default: models/rl_checkpoints_singles/best_model.zip)",
    )
    parser.add_argument("--bc-model", type=Path, default=SINGLES_BC_MODEL_PATH)
    parser.add_argument("--preview-model", type=Path, default=SINGLES_PREVIEW_MODEL_PATH)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--eval-battles", type=int, default=100)
    parser.add_argument("--replay-battles", type=int, default=10)
    parser.add_argument("--trace-battles", type=int, default=1)
    parser.add_argument("--policy-examples", type=int, default=50)
    parser.add_argument("--out-dir", type=Path, default=BC_EVAL_LOG_DIR / "singles")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-replays", action="store_true")
    parser.add_argument("--skip-trace", action="store_true")
    parser.add_argument("--skip-policy-examples", action="store_true")
    parser.add_argument("--skip-alignment", action="store_true")
    parser.add_argument(
        "--fixed-team",
        action="store_true",
        help="Use fixed agent team instead of meta pool (default: meta pool)",
    )
    args = parser.parse_args()

    cfg = RLEvalPipelineConfig(
        rl_checkpoint=args.checkpoint,
        bc_model_path=args.bc_model,
        preview_model_path=args.preview_model,
        device=args.device,
        out_root=args.out_dir,
        eval_battles=args.eval_battles,
        replay_battles=args.replay_battles,
        trace_battles=args.trace_battles,
        policy_examples_n=args.policy_examples,
        use_meta_pool=not args.fixed_team,
        skip_eval=args.skip_eval,
        skip_replays=args.skip_replays,
        skip_trace=args.skip_trace,
        skip_policy_examples=args.skip_policy_examples,
        skip_alignment=args.skip_alignment,
    )
    run_rl_eval_pipeline(cfg)


if __name__ == "__main__":
    main()
