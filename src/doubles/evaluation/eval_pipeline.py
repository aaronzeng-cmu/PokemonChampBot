"""Standard post-training evaluation pipeline for the BC Transformer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from config.settings import (
    BC_DATASET_PATH,
    BC_EVAL_LOG_DIR,
    BC_MODEL_PATH,
    RAW_LOGS_DIR,
)
from src.doubles.evaluation.bc_examples import generate_bc_examples, write_bc_examples_report
from src.doubles.evaluation.inference_trace_run import InferenceTraceReport, run_inference_trace
from src.doubles.evaluation.live_bc_alignment import AlignmentReport, run_alignment_checks
from src.doubles.evaluation.maxdamage_eval import LiveEvalReport, run_maxdamage_eval
from src.doubles.evaluation.replay_batch import ReplayBatchReport, run_replay_batch


@dataclass
class EvalPipelineConfig:
    model_path: Path = BC_MODEL_PATH
    device: str = "cuda"
    out_root: Path = BC_EVAL_LOG_DIR
    dataset_path: Path = BC_DATASET_PATH
    log_dir: Path = RAW_LOGS_DIR
    eval_battles: int = 100
    replay_battles: int = 10
    trace_battles: int = 1
    bc_examples_n: int = 50
    bc_examples_top_k: int = 3
    trace_top_k: int = 5
    skip_eval: bool = False
    skip_replays: bool = False
    skip_trace: bool = False
    skip_bc_examples: bool = False
    skip_alignment: bool = False


@dataclass
class BcExamplesStep:
    n_examples: int
    joint_top1: int
    joint_top1_rate: float
    top3_hit_rate: float
    txt_path: Path
    json_path: Path


@dataclass
class EvalPipelineResult:
    stamp: str
    out_dir: Path
    model_path: Path
    bc_examples: BcExamplesStep | None = None
    live_eval: LiveEvalReport | None = None
    replays: ReplayBatchReport | None = None
    trace: InferenceTraceReport | None = None
    alignment: AlignmentReport | None = None
    summary_path: Path | None = None


def _bc_examples_step(cfg: EvalPipelineConfig, out_dir: Path) -> BcExamplesStep:
    examples_dir = out_dir / "bc_examples"
    examples = generate_bc_examples(
        model_path=cfg.model_path,
        dataset_path=cfg.dataset_path,
        log_dir=cfg.log_dir,
        n_examples=cfg.bc_examples_n,
        device=cfg.device,
        top_k=cfg.bc_examples_top_k,
    )
    txt_path, json_path = write_bc_examples_report(
        examples,
        examples_dir,
        model_path=cfg.model_path,
        dataset_path=cfg.dataset_path,
        mix="random",
    )
    n = len(examples)
    joint = sum(1 for e in examples if e.correct_joint)
    top3_avg = (
        sum(e.top3_slot0_hit + e.top3_slot1_hit for e in examples) / (2 * n) if n else 0.0
    )
    print(
        f"BC examples: {joint}/{n} joint top-1 ({100 * joint / max(1, n):.1f}%), "
        f"top-{cfg.bc_examples_top_k} hit {100 * top3_avg:.1f}%",
        flush=True,
    )
    return BcExamplesStep(
        n_examples=n,
        joint_top1=joint,
        joint_top1_rate=joint / max(1, n),
        top3_hit_rate=top3_avg,
        txt_path=txt_path,
        json_path=json_path,
    )


def run_eval_pipeline(cfg: EvalPipelineConfig | None = None) -> EvalPipelineResult:
    """
    Standard evaluation after training:

    1. Offline BC examples (val-set predictions vs human logs)
    2. Live win-rate vs MaxDamage (default 100 battles)
    3. HTML replay batch (default 10 battles)
    4. Full inference trace (default 1 battle + protocol log)
    5. BC/live alignment audit on the trace
    """
    cfg = cfg or EvalPipelineConfig()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = cfg.out_root / f"pipeline_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = EvalPipelineResult(stamp=stamp, out_dir=out_dir, model_path=cfg.model_path)

    print(f"=== Eval pipeline {stamp} ===", flush=True)
    print(f"Model: {cfg.model_path}", flush=True)
    print(f"Output: {out_dir}", flush=True)

    if not cfg.skip_bc_examples:
        print("\n--- Step 1: BC examples ---", flush=True)
        result.bc_examples = _bc_examples_step(cfg, out_dir)

    if not cfg.skip_eval:
        print(f"\n--- Step 2: Live eval ({cfg.eval_battles} battles) ---", flush=True)
        result.live_eval = run_maxdamage_eval(
            n_battles=cfg.eval_battles,
            model_path=cfg.model_path,
            device=cfg.device,
            out_dir=out_dir / "live_eval",
        )
        print(
            f"Win rate: {result.live_eval.wins}/{result.live_eval.battles} "
            f"({100 * result.live_eval.win_rate:.1f}%), "
            f"illegal top-1: {result.live_eval.illegal_top1_count}",
            flush=True,
        )

    if not cfg.skip_replays:
        print(f"\n--- Step 3: Replay batch ({cfg.replay_battles} battles) ---", flush=True)
        result.replays = run_replay_batch(
            n_battles=cfg.replay_battles,
            model_path=cfg.model_path,
            device=cfg.device,
            out_dir=out_dir / "replays",
            opponent="maxdamage",
        )
        print(
            f"Replays: {result.replays.wins}/{result.replays.battles} "
            f"({100 * result.replays.win_rate:.1f}%), "
            f"dir={result.replays.out_dir}",
            flush=True,
        )

    if not cfg.skip_trace:
        print(f"\n--- Step 4: Inference trace ({cfg.trace_battles} battle) ---", flush=True)
        result.trace = run_inference_trace(
            n_battles=cfg.trace_battles,
            model_path=cfg.model_path,
            device=cfg.device,
            out_dir=out_dir / "inference_trace",
            opponent="maxdamage",
            top_k=cfg.trace_top_k,
            save_replays=True,
        )
        print(
            f"Trace: {result.trace.n_decisions} decisions, "
            f"{result.trace.n_fallbacks} fallbacks, "
            f"json={result.trace.trace_json}",
            flush=True,
        )

    if not cfg.skip_alignment and result.trace is not None:
        print("\n--- Step 5: BC/live alignment ---", flush=True)
        result.alignment = run_alignment_checks(
            result.trace.trace_json,
            out_dir / "alignment",
            model_path=cfg.model_path,
            device=cfg.device,
            top_k=cfg.trace_top_k,
        )
        print(
            f"Alignment: tensor {100 * result.alignment.tensor_match_rate:.1f}%, "
            f"pred {100 * result.alignment.pred_match_rate:.1f}%",
            flush=True,
        )

    summary = _build_summary(result, cfg)
    summary_path = out_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    latest = cfg.out_root / "pipeline_latest.json"
    latest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    result.summary_path = summary_path

    print(f"\n=== Pipeline complete ===", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    return result


def _build_summary(result: EvalPipelineResult, cfg: EvalPipelineConfig) -> dict:
    summary: dict = {
        "stamp": result.stamp,
        "model": str(result.model_path),
        "device": cfg.device,
        "out_dir": str(result.out_dir),
    }
    if result.bc_examples:
        summary["bc_examples"] = {
            "n": result.bc_examples.n_examples,
            "joint_top1": result.bc_examples.joint_top1,
            "joint_top1_rate": result.bc_examples.joint_top1_rate,
            "top3_hit_rate": result.bc_examples.top3_hit_rate,
            "txt": str(result.bc_examples.txt_path),
        }
    if result.live_eval:
        summary["live_eval"] = {
            "battles": result.live_eval.battles,
            "wins": result.live_eval.wins,
            "win_rate": result.live_eval.win_rate,
            "illegal_top1": result.live_eval.illegal_top1_count,
            "report": str(result.live_eval.report_path),
        }
    if result.replays:
        summary["replays"] = {
            "battles": result.replays.battles,
            "wins": result.replays.wins,
            "win_rate": result.replays.win_rate,
            "illegal_top1": result.replays.illegal_top1_count,
            "dir": str(result.replays.out_dir),
        }
    if result.trace:
        summary["trace"] = {
            "decisions": result.trace.n_decisions,
            "fallbacks": result.trace.n_fallbacks,
            "json": str(result.trace.trace_json),
            "txt": str(result.trace.trace_txt),
            "replay_dir": str(result.trace.replay_dir) if result.trace.replay_dir else None,
        }
    if result.alignment:
        summary["alignment"] = {
            "tensor_match_rate": result.alignment.tensor_match_rate,
            "pred_match_rate": result.alignment.pred_match_rate,
            "parity": str(result.alignment.parity_path),
            "audit": str(result.alignment.audit_path),
        }
    return summary
