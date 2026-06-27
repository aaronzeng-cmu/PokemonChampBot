"""Run live inference trace battles with protocol capture."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from poke_env.ps_client.account_configuration import AccountConfiguration

from config.settings import (
    BATTLE_FORMAT,
    BC_EVAL_LOG_DIR,
    BC_MODEL_PATH,
    REPLAYS_DIR,
    USE_OPPONENT_TEAM_POOL,
)
from src.core.battle.replay_utils import keep_agent_replays
from src.doubles.evaluation.battle_inference_trace import format_trace_text, write_trace_report
from src.doubles.players.max_damage_player import MaxDamagePlayer
from src.doubles.players.transformer_player import TransformerPlayer
from src.doubles.teams.team_pool import load_agent_team, load_opponent_team_builder, opponent_pool_description
from src.doubles.teams.teampreview import (
    battle_team_summary,
    opponent_full_team_summary,
    opponent_team_summary,
)


async def _run_one_trace_battle(agent: TransformerPlayer, opponent) -> dict:
    await agent.battle_against(opponent, n_battles=1)
    battle = next(iter(agent.battles.values()))
    trace = agent.drain_inference_trace(battle.battle_tag)
    team_info = battle_team_summary(battle)
    opp_info = opponent_team_summary(battle)
    full_info = opponent_full_team_summary(battle)
    trace.update(
        {
            "won": bool(battle.won),
            "turn": battle.turn,
            "leads": team_info["leads"],
            "brought": team_info["brought"],
            "opponent_brought": opp_info["brought"],
            "opponent_full_team": full_info["full_team"],
        }
    )
    return trace


@dataclass
class InferenceTraceReport:
    out_dir: Path
    trace_json: Path
    trace_txt: Path
    replay_dir: Path | None
    battles: list[dict]
    n_decisions: int
    n_fallbacks: int


def run_inference_trace(
    *,
    n_battles: int = 1,
    model_path: Path = BC_MODEL_PATH,
    device: str = "cpu",
    out_dir: Path | None = None,
    opponent: str = "maxdamage",
    top_k: int = 5,
    save_replays: bool = True,
    mirror: bool = False,
) -> InferenceTraceReport:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    trace_dir = out_dir or (BC_EVAL_LOG_DIR / "inference_trace" / stamp)
    trace_dir.mkdir(parents=True, exist_ok=True)
    replay_dir = (REPLAYS_DIR / f"trace_{stamp}") if save_replays else None
    if replay_dir is not None:
        replay_dir.mkdir(parents=True, exist_ok=True)

    agent_team = load_agent_team()
    use_pool = USE_OPPONENT_TEAM_POOL and not mirror
    opponent_team = load_opponent_team_builder(use_pool=use_pool, use_curriculum=False)
    pool_info = opponent_pool_description(opponent_team)

    agent = TransformerPlayer(
        model_path=model_path,
        battle_format=BATTLE_FORMAT,
        team=agent_team,
        device=device,
        trace_inference=True,
        trace_top_k=top_k,
        capture_battle_log=True,
        log_illegal_top1=True,
        max_concurrent_battles=1,
        save_replays=str(replay_dir) if replay_dir else False,
        account_configuration=AccountConfiguration.generate("TransformerTrace", rand=True),
    )
    if opponent == "maxdamage":
        opp = MaxDamagePlayer(
            battle_format=BATTLE_FORMAT,
            team=opponent_team,
            max_concurrent_battles=1,
            account_configuration=AccountConfiguration.generate("MaxDamageTrace", rand=True),
        )
    else:
        from src.doubles.players.vgc_random_player import VGCRandomPlayer

        opp = VGCRandomPlayer(
            battle_format=BATTLE_FORMAT,
            team=opponent_team,
            max_concurrent_battles=1,
            account_configuration=AccountConfiguration.generate("RandomTrace", rand=True),
        )

    battle_traces: list[dict] = []
    for i in range(n_battles):
        agent.reset_battles()
        opp.reset_battles()
        trace = asyncio.run(_run_one_trace_battle(agent, opp))
        trace["index"] = i + 1
        battle_traces.append(trace)
        if replay_dir is not None:
            keep_agent_replays(replay_dir, agent.username)
        n_decisions = len(trace.get("decisions", []))
        n_fallback = sum(1 for d in trace.get("decisions", []) if d.get("any_fallback"))
        print(
            f"  [{i + 1}/{n_battles}] "
            f"{'WIN' if trace.get('won') else 'LOSS'} "
            f"turns={trace.get('turn')} decisions={n_decisions} "
            f"fallbacks={n_fallback} tag={trace.get('battle_tag')}",
            flush=True,
        )

    txt_path, json_path = write_trace_report(
        battle_traces,
        trace_dir,
        model_path=model_path,
        opponent=opponent,
    )
    meta = {
        "timestamp_utc": stamp,
        "format": BATTLE_FORMAT,
        "model": str(model_path.resolve()),
        "opponent": opponent,
        "opponent_team_mode": pool_info,
        "battles": n_battles,
        "top_k": top_k,
        "trace_text": str(txt_path.resolve()),
        "trace_json": str(json_path.resolve()),
        "replay_dir": str(replay_dir.resolve()) if replay_dir else None,
    }
    (trace_dir / "trace_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    total_decisions = sum(len(t.get("decisions", [])) for t in battle_traces)
    total_fallbacks = sum(
        sum(1 for d in t.get("decisions", []) if d.get("any_fallback"))
        for t in battle_traces
    )

    return InferenceTraceReport(
        out_dir=trace_dir,
        trace_json=json_path,
        trace_txt=txt_path,
        replay_dir=replay_dir,
        battles=battle_traces,
        n_decisions=total_decisions,
        n_fallbacks=total_fallbacks,
    )
