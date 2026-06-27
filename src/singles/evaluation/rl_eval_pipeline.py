"""Post-training evaluation pipeline for a trained Singles MaskablePPO policy."""

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
from src.singles.evaluation.eval_package_alignment import (
    EvalPackageAlignmentReport,
    run_eval_package_alignment,
)
from src.singles.evaluation.inference_trace_run import InferenceTraceReport, run_inference_trace
from src.singles.evaluation.live_bc_alignment import AlignmentReport, run_alignment_checks
from src.singles.evaluation.maxdamage_eval import LiveEvalReport, run_maxdamage_eval
from src.singles.evaluation.replay_batch import ReplayBatchReport, run_replay_batch
from src.singles.evaluation.rl_policy_examples import (
    generate_rl_policy_examples,
    write_rl_policy_examples_report,
)
from src.singles.rl.checkpoints import resolve_rl_checkpoint


@dataclass
class RLEvalPipelineConfig:
    rl_checkpoint: Path | None = None
    bc_model_path: Path = SINGLES_BC_MODEL_PATH
    preview_model_path: Path = SINGLES_PREVIEW_MODEL_PATH
    device: str = "cuda"
    out_root: Path = BC_EVAL_LOG_DIR / "singles"
    dataset_path: Path = SINGLES_BC_DATASET_PATH
    log_dir: Path = SINGLES_RAW_LOGS_DIR
    eval_battles: int = 100
    replay_battles: int = 10
    trace_battles: int = 1
    policy_examples_n: int = 50
    use_meta_pool: bool = True
    skip_eval: bool = False
    skip_replays: bool = False
    skip_trace: bool = False
    skip_policy_examples: bool = False
    skip_alignment: bool = False


@dataclass
class PolicyExamplesStep:
    n_examples: int
    top1: int
    top1_rate: float
    top3_hit_rate: float
    txt_path: Path
    json_path: Path


@dataclass
class RLEvalPipelineResult:
    stamp: str
    out_dir: Path
    rl_checkpoint: Path
    policy_examples: PolicyExamplesStep | None = None
    live_eval: LiveEvalReport | None = None
    replays: ReplayBatchReport | None = None
    trace: InferenceTraceReport | None = None
    alignment: AlignmentReport | None = None
    package_alignment: EvalPackageAlignmentReport | None = None
    summary_path: Path | None = None


def run_rl_eval_pipeline(cfg: RLEvalPipelineConfig | None = None) -> RLEvalPipelineResult:
    cfg = cfg or RLEvalPipelineConfig()
    rl_checkpoint = resolve_rl_checkpoint(cfg.rl_checkpoint)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = cfg.out_root / f"rl_pipeline_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = RLEvalPipelineResult(
        stamp=stamp,
        out_dir=out_dir,
        rl_checkpoint=rl_checkpoint,
    )

    print(f"=== Singles RL eval pipeline {stamp} ===", flush=True)
    print(f"RL checkpoint: {rl_checkpoint}", flush=True)
    print(f"Meta-pool agent teams: {cfg.use_meta_pool}", flush=True)
    print(f"Output: {out_dir}", flush=True)

    common = dict(
        rl_checkpoint=rl_checkpoint,
        model_path=cfg.bc_model_path,
        preview_model_path=cfg.preview_model_path,
        device=cfg.device,
        use_meta_pool=cfg.use_meta_pool,
    )

    if not cfg.skip_policy_examples:
        print("\n--- Step 1: RL policy examples (offline dataset) ---", flush=True)
        examples = generate_rl_policy_examples(
            rl_checkpoint=rl_checkpoint,
            dataset_path=cfg.dataset_path,
            log_dir=cfg.log_dir,
            n_examples=cfg.policy_examples_n,
            device=cfg.device,
        )
        txt_path, json_path = write_rl_policy_examples_report(
            examples,
            out_dir / "policy_examples",
            rl_checkpoint=rl_checkpoint,
            dataset_path=cfg.dataset_path,
            mix="random",
        )
        n = len(examples)
        top1 = sum(1 for e in examples if e.correct)
        top3 = sum(1 for e in examples if e.top3_hit)
        print(
            f"Policy examples: {top1}/{n} top-1 ({100 * top1 / max(1, n):.1f}%), "
            f"top-3 hit {100 * top3 / max(1, n):.1f}%",
            flush=True,
        )
        result.policy_examples = PolicyExamplesStep(
            n_examples=n,
            top1=top1,
            top1_rate=top1 / max(1, n),
            top3_hit_rate=top3 / max(1, n),
            txt_path=txt_path,
            json_path=json_path,
        )

    if not cfg.skip_eval:
        print(f"\n--- Step 2: Live eval ({cfg.eval_battles} battles) ---", flush=True)
        result.live_eval = run_maxdamage_eval(
            n_battles=cfg.eval_battles,
            out_dir=out_dir / "live_eval",
            **common,
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
            out_dir=out_dir / "replays",
            **common,
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
            out_dir=out_dir / "inference_trace",
            save_replays=True,
            **common,
        )
        print(
            f"Trace: {result.trace.n_decisions} decisions, "
            f"{result.trace.n_fallbacks} fallbacks, "
            f"json={result.trace.trace_json}",
            flush=True,
        )

        if not cfg.skip_alignment and result.trace.trace_json is not None:
            print("\n--- Step 5: Encoding alignment (RL trace) ---", flush=True)
            result.alignment = run_alignment_checks(
                result.trace.trace_json,
                out_dir / "alignment",
                model_path=cfg.bc_model_path,
                dataset_path=cfg.dataset_path,
                device=cfg.device,
            )
            print(
                f"Alignment: trajectory {100 * result.alignment.traj_match_rate:.1f}%, "
                f"tensor (recomputed) {100 * result.alignment.recomputed_digest_match_rate:.1f}% "
                f"({result.alignment.n_compared} decisions)",
                flush=True,
            )

    if not cfg.skip_alignment:
        print("\n--- Step 6: Package alignment (encoding only) ---", flush=True)
        result.package_alignment = run_eval_package_alignment(
            out_dir,
            model_path=cfg.bc_model_path,
            dataset_path=cfg.dataset_path,
            device=cfg.device,
            encoding_only_all=True,
        )
        print(result.package_alignment.summary_text, flush=True)
        status = "PASS" if result.package_alignment.all_pass else "FAIL"
        print(f"Package alignment: {status}", flush=True)

    summary = _build_summary(result, cfg)
    summary_path = out_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    latest = cfg.out_root / "rl_pipeline_latest.json"
    latest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    result.summary_path = summary_path

    print("\n=== RL pipeline complete ===", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    return result


def _build_summary(result: RLEvalPipelineResult, cfg: RLEvalPipelineConfig) -> dict:
    summary: dict = {
        "stamp": result.stamp,
        "format": "singles",
        "agent": "maskable_ppo",
        "rl_checkpoint": str(result.rl_checkpoint),
        "bc_model": str(cfg.bc_model_path),
        "preview_model": str(cfg.preview_model_path),
        "device": cfg.device,
        "use_meta_pool": cfg.use_meta_pool,
        "out_dir": str(result.out_dir),
    }
    if result.policy_examples:
        summary["policy_examples"] = {
            "n": result.policy_examples.n_examples,
            "top1": result.policy_examples.top1,
            "top1_rate": result.policy_examples.top1_rate,
            "top3_hit_rate": result.policy_examples.top3_hit_rate,
            "txt": str(result.policy_examples.txt_path),
            "json": str(result.policy_examples.json_path),
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
