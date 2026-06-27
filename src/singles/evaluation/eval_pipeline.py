"""Standard post-training evaluation pipeline for the Singles BC Transformer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from config.settings import (
    BC_EVAL_LOG_DIR,
    SINGLES_BC_DATASET_PATH,
    SINGLES_BC_MODEL_PATH,
    SINGLES_PREVIEW_MODEL_PATH,
    SINGLES_RAW_LOGS_DIR,
)
from src.singles.evaluation.bc_examples import generate_bc_examples, write_bc_examples_report
from src.singles.evaluation.inference_trace_run import InferenceTraceReport, run_inference_trace
from src.singles.evaluation.live_bc_alignment import AlignmentReport, run_alignment_checks
from src.singles.evaluation.eval_package_alignment import (
    EvalPackageAlignmentReport,
    run_eval_package_alignment,
)
from src.singles.evaluation.maxdamage_eval import LiveEvalReport, run_maxdamage_eval
from src.singles.evaluation.replay_batch import ReplayBatchReport, run_replay_batch
from src.singles.evaluation.rl_trace_run import RLTraceReport, run_rl_trace_alignment


@dataclass
class EvalPipelineConfig:
    model_path: Path = SINGLES_BC_MODEL_PATH
    preview_model_path: Path = SINGLES_PREVIEW_MODEL_PATH
    device: str = "cuda"
    out_root: Path = BC_EVAL_LOG_DIR / "singles"
    dataset_path: Path = SINGLES_BC_DATASET_PATH
    log_dir: Path = SINGLES_RAW_LOGS_DIR
    eval_battles: int = 100
    replay_battles: int = 10
    trace_battles: int = 1
    bc_examples_n: int = 50
    bc_examples_top_k: int = 3
    skip_eval: bool = False
    skip_replays: bool = False
    skip_trace: bool = False
    skip_bc_examples: bool = False
    skip_alignment: bool = False
    skip_rl_alignment: bool = False
    rl_trace_battles: int = 1


@dataclass
class BcExamplesStep:
    n_examples: int
    top1: int
    top1_rate: float
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
    package_alignment: EvalPackageAlignmentReport | None = None
    rl_trace: RLTraceReport | None = None
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
    top1 = sum(1 for e in examples if e.correct)
    top3 = sum(1 for e in examples if e.top3_hit)
    print(
        f"BC examples: {top1}/{n} top-1 ({100 * top1 / max(1, n):.1f}%), "
        f"top-{cfg.bc_examples_top_k} hit {100 * top3 / max(1, n):.1f}%",
        flush=True,
    )
    return BcExamplesStep(
        n_examples=n,
        top1=top1,
        top1_rate=top1 / max(1, n),
        top3_hit_rate=top3 / max(1, n),
        txt_path=txt_path,
        json_path=json_path,
    )


def run_bc_examples_step(
    cfg: EvalPipelineConfig,
    out_dir: Path,
) -> BcExamplesStep:
    """Offline BC examples vs human logs (shared by pipeline, train, and live eval)."""
    return _bc_examples_step(cfg, out_dir)


def run_eval_pipeline(cfg: EvalPipelineConfig | None = None) -> EvalPipelineResult:
    """
    Standard singles evaluation after training:

    1. Offline BC examples (val-set predictions vs human logs)
    2. HTML replay batch (default 10 battles)
    3. Full inference trace (default 1 battle + protocol log)
    """
    cfg = cfg or EvalPipelineConfig()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = cfg.out_root / f"pipeline_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = EvalPipelineResult(stamp=stamp, out_dir=out_dir, model_path=cfg.model_path)

    print(f"=== Singles eval pipeline {stamp} ===", flush=True)
    print(f"Model: {cfg.model_path}", flush=True)
    print(f"Output: {out_dir}", flush=True)

    if not cfg.skip_bc_examples:
        print("\n--- Step 1: BC examples ---", flush=True)
        result.bc_examples = run_bc_examples_step(cfg, out_dir)

    if not cfg.skip_eval:
        print(f"\n--- Step 2: Live eval ({cfg.eval_battles} battles) ---", flush=True)
        result.live_eval = run_maxdamage_eval(
            n_battles=cfg.eval_battles,
            model_path=cfg.model_path,
            preview_model_path=cfg.preview_model_path,
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
            preview_model_path=cfg.preview_model_path,
            device=cfg.device,
            out_dir=out_dir / "replays",
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
            preview_model_path=cfg.preview_model_path,
            device=cfg.device,
            out_dir=out_dir / "inference_trace",
            save_replays=True,
        )
        print(
            f"Trace: {result.trace.n_decisions} decisions, "
            f"{result.trace.n_fallbacks} fallbacks, "
            f"json={result.trace.trace_json}",
            flush=True,
        )

        if not cfg.skip_alignment and result.trace.trace_json is not None:
            print("\n--- Step 5: BC/live alignment ---", flush=True)
            result.alignment = run_alignment_checks(
                result.trace.trace_json,
                out_dir / "alignment",
                model_path=cfg.model_path,
                dataset_path=cfg.dataset_path,
                device=cfg.device,
            )
            print(
                f"Alignment: trajectory {100 * result.alignment.traj_match_rate:.1f}%, "
                f"tensor (recomputed) {100 * result.alignment.recomputed_digest_match_rate:.1f}%, "
                f"pred {100 * result.alignment.pred_match_rate:.1f}% "
                f"({result.alignment.n_compared} decisions)",
                flush=True,
            )

    if not cfg.skip_rl_alignment:
        print(f"\n--- Step 6: RL encoding trace ({cfg.rl_trace_battles} battle) ---", flush=True)
        result.rl_trace = run_rl_trace_alignment(
            n_battles=cfg.rl_trace_battles,
            model_path=cfg.model_path,
            preview_model_path=cfg.preview_model_path,
            device=cfg.device,
            out_dir=out_dir / "rl_trace",
        )
        if result.rl_trace.alignment is not None:
            enc = result.rl_trace.alignment
            print(
                f"RL trace: {result.rl_trace.n_decisions} decisions, "
                f"tensor (recomputed) {100 * enc.recomputed_digest_match_rate:.1f}%, "
                f"trajectory {100 * enc.traj_match_rate:.1f}%",
                flush=True,
            )

    if not cfg.skip_alignment:
        print("\n--- Step 7: Package alignment (BC + RL traces) ---", flush=True)
        result.package_alignment = run_eval_package_alignment(
            out_dir,
            model_path=cfg.model_path,
            dataset_path=cfg.dataset_path,
            device=cfg.device,
        )
        print(result.package_alignment.summary_text, flush=True)
        status = "PASS" if result.package_alignment.all_pass else "FAIL"
        print(f"Package alignment: {status}", flush=True)

    summary = _build_summary(result, cfg)
    summary_path = out_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    latest = cfg.out_root / "pipeline_latest.json"
    latest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    result.summary_path = summary_path

    print("\n=== Pipeline complete ===", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    return result


def _build_summary(result: EvalPipelineResult, cfg: EvalPipelineConfig) -> dict:
    summary: dict = {
        "stamp": result.stamp,
        "format": "singles",
        "model": str(result.model_path),
        "preview_model": str(cfg.preview_model_path),
        "device": cfg.device,
        "out_dir": str(result.out_dir),
    }
    if result.bc_examples:
        summary["bc_examples"] = {
            "n": result.bc_examples.n_examples,
            "top1": result.bc_examples.top1,
            "top1_rate": result.bc_examples.top1_rate,
            "top3_hit_rate": result.bc_examples.top3_hit_rate,
            "txt": str(result.bc_examples.txt_path),
            "json": str(result.bc_examples.json_path),
        }
    if result.live_eval:
        summary["live_eval"] = {
            "battles": result.live_eval.battles,
            "wins": result.live_eval.wins,
            "win_rate": result.live_eval.win_rate,
            "illegal_top1": result.live_eval.illegal_top1_count,
            "json": str(result.live_eval.report_path),
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
            "n_compared": result.alignment.n_compared,
            "traj_match_rate": result.alignment.traj_match_rate,
            "digest_match_rate": result.alignment.digest_match_rate,
            "recomputed_digest_match_rate": result.alignment.recomputed_digest_match_rate,
            "pred_match_rate": result.alignment.pred_match_rate,
            "audit": str(result.alignment.audit_path),
            "dataset": str(result.alignment.dataset_path),
        }
    if result.rl_trace:
        summary["rl_trace"] = {
            "decisions": result.rl_trace.n_decisions,
            "json": str(result.rl_trace.trace_json),
            "txt": str(result.rl_trace.trace_txt),
        }
        if result.rl_trace.alignment:
            summary["rl_trace"]["traj_match_rate"] = result.rl_trace.alignment.traj_match_rate
            summary["rl_trace"]["recomputed_digest_match_rate"] = (
                result.rl_trace.alignment.recomputed_digest_match_rate
            )
            summary["rl_trace"]["pred_match_rate"] = result.rl_trace.alignment.pred_match_rate
            summary["rl_trace"]["audit"] = str(result.rl_trace.alignment.audit_path)
    if result.package_alignment:
        summary["package_alignment"] = {
            "all_pass": result.package_alignment.all_pass,
            "report": str(result.package_alignment.report_path),
            "sources": [
                {
                    "source": s.source,
                    "n_live_decisions": s.n_live_decisions,
                    "n_compared": s.n_compared,
                    "unmatched_live": s.unmatched_live,
                    "pred_match_rate": s.pred_match_rate,
                    "recomputed_digest_match_rate": s.recomputed_digest_match_rate,
                    "encoding_only": s.encoding_only,
                }
                for s in result.package_alignment.sources
            ],
        }
    return summary
