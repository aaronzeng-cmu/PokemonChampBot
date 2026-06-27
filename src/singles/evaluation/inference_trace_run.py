"""Run live singles inference trace battles with protocol capture."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from poke_env.ps_client.account_configuration import AccountConfiguration

from config.settings import (
    BC_EVAL_LOG_DIR,
    REPLAYS_DIR,
    SINGLES_BATTLE_FORMAT,
    SINGLES_BC_MODEL_PATH,
    SINGLES_PREVIEW_MODEL_PATH,
)
from src.core.battle.replay_utils import keep_agent_replays
from src.singles.evaluation.eval_agent import (
    build_opponent_team,
    build_singles_eval_agent,
    describe_agent_team,
    load_eval_agent_team,
)
from src.singles.evaluation.inference_trace import write_trace_report
from src.singles.max_damage_player import SinglesMaxDamagePlayer
from src.singles.teampreview import (
    battle_team_summary,
    opponent_team_summary,
)


async def _run_one_trace_battle(agent, opponent) -> dict:
    await agent.battle_against(opponent, n_battles=1)
    battle = next(iter(agent.battles.values()))
    trace = agent.drain_inference_trace(battle.battle_tag)
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
    model_path: Path = SINGLES_BC_MODEL_PATH,
    rl_checkpoint: Path | None = None,
    preview_model_path: Path = SINGLES_PREVIEW_MODEL_PATH,
    device: str = "cpu",
    out_dir: Path | None = None,
    save_replays: bool = True,
    mirror: bool = False,
    use_meta_pool: bool = False,
) -> InferenceTraceReport:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    trace_dir = out_dir or (BC_EVAL_LOG_DIR / "singles" / "inference_trace" / stamp)
    trace_dir.mkdir(parents=True, exist_ok=True)
    replay_dir = (REPLAYS_DIR / f"singles_trace_{stamp}") if save_replays else None
    if replay_dir is not None:
        replay_dir.mkdir(parents=True, exist_ok=True)

    agent_team = load_eval_agent_team(use_meta_pool=use_meta_pool)
    opponent_team = build_opponent_team(mirror=mirror)
    agent_pool_info = describe_agent_team(agent_team)
    pool_info = describe_agent_team(opponent_team)

    agent = build_singles_eval_agent(
        rl_checkpoint=rl_checkpoint,
        bc_model_path=model_path,
        preview_model_path=preview_model_path,
        device=device,
        team=agent_team,
        trace_inference=True,
        trace_top_k=5,
        capture_battle_log=True,
        log_illegal_top1=True,
        save_replays=str(replay_dir) if replay_dir else False,
        account_name="SinglesRLTrace" if rl_checkpoint else "SinglesTransformerTrace",
    )
    opponent = SinglesMaxDamagePlayer(
        battle_format=SINGLES_BATTLE_FORMAT,
        team=opponent_team,
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration.generate("SinglesMaxDmgTrace", rand=True),
    )

    battle_traces: list[dict] = []
    for i in range(n_battles):
        agent.reset_battles()
        opponent.reset_battles()
        trace = asyncio.run(_run_one_trace_battle(agent, opponent))
        trace["index"] = i + 1
        battle_traces.append(trace)
        if replay_dir is not None:
            keep_agent_replays(replay_dir, agent.username)
        n_decisions = len(trace.get("decisions", []))
        n_fallback = sum(1 for d in trace.get("decisions", []) if d.get("fallback"))
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
        model_path=rl_checkpoint or model_path,
        opponent="SinglesMaxDamagePlayer",
    )
    meta = {
        "timestamp_utc": stamp,
        "format": SINGLES_BATTLE_FORMAT,
        "agent": "maskable_ppo" if rl_checkpoint else "singles_transformer",
        "model": str((rl_checkpoint or model_path).resolve()),
        "rl_checkpoint": str(rl_checkpoint.resolve()) if rl_checkpoint else None,
        "bc_model": str(model_path.resolve()),
        "preview_model": str(preview_model_path.resolve()),
        "opponent": "SinglesMaxDamagePlayer",
        "agent_team_mode": agent_pool_info,
        "opponent_team_mode": pool_info,
        "use_meta_pool": use_meta_pool,
        "battles": n_battles,
        "trace_text": str(txt_path.resolve()),
        "trace_json": str(json_path.resolve()),
        "replay_dir": str(replay_dir.resolve()) if replay_dir else None,
    }
    (trace_dir / "trace_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    total_decisions = sum(len(t.get("decisions", [])) for t in battle_traces)
    total_fallbacks = sum(1 for d in sum((t.get("decisions", []) for t in battle_traces), []) if d.get("fallback"))

    return InferenceTraceReport(
        out_dir=trace_dir,
        trace_json=json_path,
        trace_txt=txt_path,
        replay_dir=replay_dir,
        battles=battle_traces,
        n_decisions=total_decisions,
        n_fallbacks=total_fallbacks,
    )
