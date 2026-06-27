"""Run RL env inference traces and verify BC-parser encoding alignment."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from poke_env.ps_client.account_configuration import AccountConfiguration

from config.settings import (
    BC_EVAL_LOG_DIR,
    SINGLES_BATTLE_FORMAT,
    SINGLES_BC_MODEL_PATH,
    SINGLES_PREVIEW_MODEL_PATH,
)
from src.singles.evaluation.inference_trace import write_trace_report
from src.singles.evaluation.live_bc_alignment import AlignmentReport, run_alignment_checks
from src.singles.max_damage_player import SinglesMaxDamagePlayer
from src.singles.preview_orchestrator import SinglesPreviewOrchestrator
from src.singles.rl_eval_player import SinglesRLEvalPlayer, start_bc_action_feeder
from src.singles.teams.team_pool import load_agent_team, load_opponent_team_builder
from src.singles.teampreview import battle_team_summary, opponent_team_summary


async def _run_one_rl_trace_battle(agent: SinglesRLEvalPlayer, opponent) -> dict:
    await agent.battle_against(opponent, n_battles=1)
    battle = next(iter(agent.battles.values()))
    trace = agent.drain_rl_trace(battle.battle_tag)
    team_info = battle_team_summary(battle)
    opp_info = opponent_team_summary(battle)
    trace.update(
        {
            "won": bool(battle.won),
            "turn": battle.turn,
            "lead": team_info.get("lead"),
            "brought": team_info.get("brought"),
            "opponent_brought": opp_info.get("brought"),
        }
    )
    return trace


@dataclass
class RLTraceReport:
    out_dir: Path
    trace_json: Path
    trace_txt: Path
    battles: list[dict]
    n_decisions: int
    alignment: AlignmentReport | None = None


def run_rl_trace_alignment(
    *,
    n_battles: int = 1,
    model_path: Path = SINGLES_BC_MODEL_PATH,
    preview_model_path: Path = SINGLES_PREVIEW_MODEL_PATH,
    device: str = "cpu",
    out_dir: Path | None = None,
) -> RLTraceReport:
    """
    Battle with SinglesRLEvalPlayer (BC-aligned encoding), capture protocol trace,
    and audit tensor/trajectory parity vs the replay parser.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    trace_dir = out_dir or (BC_EVAL_LOG_DIR / "singles" / "rl_trace" / stamp)
    trace_dir.mkdir(parents=True, exist_ok=True)

    agent_team = load_agent_team()
    opponent_team = load_opponent_team_builder(use_pool=True)
    preview = SinglesPreviewOrchestrator(model_path=preview_model_path, device=device)

    agent = SinglesRLEvalPlayer(
        battle_format=SINGLES_BATTLE_FORMAT,
        team=agent_team,
        device=device,
        preview=preview,
        trace_decisions=True,
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration.generate("SinglesRLTrace", rand=True),
    )
    opponent = SinglesMaxDamagePlayer(
        battle_format=SINGLES_BATTLE_FORMAT,
        team=opponent_team,
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration.generate("SinglesMaxDmgRLTrace", rand=True),
    )

    battle_traces: list[dict] = []
    feeder = start_bc_action_feeder(agent, model_path=model_path, device=device)
    try:
        for i in range(n_battles):
            agent.reset_battles()
            opponent.reset_battles()
            agent.reset_rl_state()
            trace = asyncio.run(_run_one_rl_trace_battle(agent, opponent))
            trace["index"] = i + 1
            battle_traces.append(trace)
            n_decisions = len(
                [d for d in trace.get("decisions", []) if d.get("kind") == "inference"]
            )
            print(
                f"  [{i + 1}/{n_battles}] "
                f"{'WIN' if trace.get('won') else 'LOSS'} "
                f"turns={trace.get('turn')} decisions={n_decisions} "
                f"tag={trace.get('battle_tag')}",
                flush=True,
            )
    finally:
        getattr(feeder, "_stop_event").set()

    trace_txt, trace_json = write_trace_report(
        battle_traces,
        trace_dir,
        model_path=model_path,
        opponent="SinglesRLEvalPlayer",
    )
    n_decisions = sum(
        len([d for d in b.get("decisions", []) if d.get("kind") == "inference"])
        for b in battle_traces
    )

    alignment: AlignmentReport | None = None
    if trace_json is not None and n_decisions > 0:
        alignment = run_alignment_checks(
            trace_json,
            trace_dir / "alignment",
            model_path=model_path,
            device=device,
        )
        print(
            f"RL encoding alignment: trajectory {100 * alignment.traj_match_rate:.1f}%, "
            f"tensor (recomputed) {100 * alignment.recomputed_digest_match_rate:.1f}% "
            f"({alignment.n_compared} decisions)",
            flush=True,
        )

    return RLTraceReport(
        out_dir=trace_dir,
        trace_json=trace_json,
        trace_txt=trace_txt,
        battles=battle_traces,
        n_decisions=n_decisions,
        alignment=alignment,
    )


def encoding_alignment_pass(report: AlignmentReport) -> bool:
    """RL traces must match parser tensors; action picks may differ from BC."""
    unmatched_live = 0
    n_live = 0
    for line in report.audit_text.splitlines():
        if line.startswith("Unmatched live decisions:"):
            unmatched_live = int(line.split(":")[1].strip())
        elif line.startswith("Live inference decisions:"):
            n_live = int(line.split(":")[1].strip())
    return (
        unmatched_live == 0
        and n_live > 0
        and report.n_compared == n_live
        and report.traj_match_rate >= 1.0
        and report.recomputed_digest_match_rate >= 1.0
    )
