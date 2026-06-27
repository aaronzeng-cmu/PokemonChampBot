"""Cross-check BC alignment across inference trace, live eval, and replay batch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from config.settings import SINGLES_BC_DATASET_PATH, SINGLES_BC_MODEL_PATH
from src.singles.evaluation.live_bc_alignment import run_alignment_checks


@dataclass
class SourceAlignmentResult:
    source: str
    trace_path: Path
    n_live_decisions: int
    n_compared: int
    unmatched_live: int
    pred_match_rate: float
    traj_match_rate: float
    digest_match_rate: float
    recomputed_digest_match_rate: float
    audit_path: Path
    encoding_only: bool = False


@dataclass
class EvalPackageAlignmentReport:
    sources: list[SourceAlignmentResult]
    all_pass: bool
    report_path: Path
    summary_text: str


def _count_inference_decisions(trace_path: Path) -> int:
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    battles = data.get("battles") or [data]
    total = 0
    for battle in battles:
        for dec in battle.get("decisions") or []:
            if dec.get("kind") == "inference":
                total += 1
    return total


def _parse_unmatched(audit_text: str) -> int:
    for line in audit_text.splitlines():
        if line.startswith("Unmatched live decisions:"):
            return int(line.split(":")[1].strip())
    return 0


def _align_trace_source(
    *,
    source: str,
    trace_path: Path,
    out_dir: Path,
    model_path: Path,
    dataset_path: Path,
    device: str,
    encoding_only: bool = False,
) -> SourceAlignmentResult | None:
    if not trace_path.is_file():
        return None
    report = run_alignment_checks(
        trace_path,
        out_dir / source,
        model_path=model_path,
        dataset_path=dataset_path,
        device=device,
    )
    return SourceAlignmentResult(
        source=source,
        trace_path=trace_path,
        n_live_decisions=_count_inference_decisions(trace_path),
        n_compared=report.n_compared,
        unmatched_live=_parse_unmatched(report.audit_text),
        pred_match_rate=report.pred_match_rate,
        traj_match_rate=report.traj_match_rate,
        digest_match_rate=report.digest_match_rate,
        recomputed_digest_match_rate=report.recomputed_digest_match_rate,
        audit_path=report.audit_path,
        encoding_only=encoding_only,
    )


def _trace_battle_paths(trace_path: Path) -> list[Path]:
    """Expand a trace JSON into per-battle files when needed."""
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    battles = data.get("battles") or [data]
    if len(battles) <= 1:
        return [trace_path]
    out: list[Path] = []
    stem = trace_path.stem
    for i, battle in enumerate(battles, start=1):
        part = trace_path.parent / f"{stem}_battle{i}.json"
        part.write_text(json.dumps({"battles": [battle]}, indent=2), encoding="utf-8")
        out.append(part)
    return out


def run_eval_package_alignment(
    pipeline_dir: Path,
    *,
    model_path: Path = SINGLES_BC_MODEL_PATH,
    dataset_path: Path = SINGLES_BC_DATASET_PATH,
    device: str = "cpu",
    encoding_only_all: bool = False,
) -> EvalPackageAlignmentReport:
    """Run BC/live alignment on trace, live-eval sample, and replay-batch traces."""
    alignment_dir = pipeline_dir / "package_alignment"
    alignment_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[tuple[str, Path]] = []

    trace_dir = pipeline_dir / "inference_trace"
    if trace_dir.is_dir():
        jsons = sorted(trace_dir.glob("inference_trace_*.json"), reverse=True)
        if jsons:
            candidates.append(("inference_trace", jsons[0]))

    live_trace = pipeline_dir / "live_eval" / "trace_battle_1.json"
    if live_trace.is_file():
        candidates.append(("live_eval_battle_1", live_trace))

    replay_trace = pipeline_dir / "replays" / "replay_traces.json"
    if replay_trace.is_file():
        candidates.append(("replay_batch", replay_trace))

    rl_trace_dir = pipeline_dir / "rl_trace"
    if rl_trace_dir.is_dir():
        rl_jsons = sorted(rl_trace_dir.glob("inference_trace_*.json"), reverse=True)
        if rl_jsons:
            candidates.append(("rl_trace", rl_jsons[0]))

    encoding_only_sources = {"rl_trace"}
    if encoding_only_all:
        encoding_only_sources = {source for source, _ in candidates}

    sources: list[SourceAlignmentResult] = []
    for source, path in candidates:
        for part_path in _trace_battle_paths(path):
            label = source
            if part_path != path:
                label = f"{source}_{part_path.stem.split('_')[-1]}"
            result = _align_trace_source(
                source=label,
                trace_path=part_path,
                out_dir=alignment_dir,
                model_path=model_path,
                dataset_path=dataset_path,
                device=device,
                encoding_only=source in encoding_only_sources,
            )
            if result is not None:
                sources.append(result)

    lines = [
        "=== Singles eval package BC alignment ===",
        f"Pipeline: {pipeline_dir}",
        "",
    ]
    all_pass = True
    for row in sources:
        ok = (
            row.unmatched_live == 0
            and row.n_compared == row.n_live_decisions
            and row.recomputed_digest_match_rate >= 1.0
            and row.traj_match_rate >= 1.0
            and (row.encoding_only or row.pred_match_rate >= 1.0)
        )
        all_pass = all_pass and ok
        status = "PASS" if ok else "FAIL"
        lines.extend(
            [
                f"--- {row.source} [{status}] ---",
                f"trace: {row.trace_path}",
                f"live decisions: {row.n_live_decisions}",
                f"matched parser samples: {row.n_compared}",
                f"unmatched live: {row.unmatched_live}",
                f"trajectory match: {100 * row.traj_match_rate:.1f}%",
                f"tensor digest (recorded): {100 * row.digest_match_rate:.1f}%",
                f"tensor digest (recomputed): {100 * row.recomputed_digest_match_rate:.1f}%",
                f"BC pred == live picked: {100 * row.pred_match_rate:.1f}%",
                f"audit: {row.audit_path}",
                "",
            ]
        )

    lines.append(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    summary_text = "\n".join(lines)
    report_path = alignment_dir / "package_alignment_summary.txt"
    report_path.write_text(summary_text, encoding="utf-8")

    payload = {
        "pipeline_dir": str(pipeline_dir),
        "all_pass": all_pass,
        "sources": [
            {
                "source": r.source,
                "trace": str(r.trace_path),
                "n_live_decisions": r.n_live_decisions,
                "n_compared": r.n_compared,
                "unmatched_live": r.unmatched_live,
                "pred_match_rate": r.pred_match_rate,
                "traj_match_rate": r.traj_match_rate,
                "digest_match_rate": r.digest_match_rate,
                "recomputed_digest_match_rate": r.recomputed_digest_match_rate,
            }
            for r in sources
        ],
    }
    (alignment_dir / "package_alignment_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    return EvalPackageAlignmentReport(
        sources=sources,
        all_pass=all_pass,
        report_path=report_path,
        summary_text=summary_text,
    )
