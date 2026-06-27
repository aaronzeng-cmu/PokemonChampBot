"""Record HTML replay batches for the Singles BC Transformer agent."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from poke_env.ps_client.account_configuration import AccountConfiguration

from config.settings import (
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
from src.singles.max_damage_player import SinglesMaxDamagePlayer
from src.singles.teampreview import (
    battle_team_summary,
    opponent_full_team_summary,
    opponent_team_summary,
)


async def _run_one_battle(agent, opponent) -> dict:
    await agent.battle_against(opponent, n_battles=1)
    battle = next(iter(agent.battles.values()))
    team_info = battle_team_summary(battle)
    opp_info = opponent_team_summary(battle)
    full_info = opponent_full_team_summary(battle)
    row = {
        "battle_tag": battle.battle_tag,
        "won": bool(battle.won),
        "turn": battle.turn,
        "lead": team_info.get("lead"),
        "brought": team_info.get("brought"),
        "opponent_brought": opp_info.get("brought"),
        "opponent_full_team": full_info.get("full_team"),
    }
    if getattr(agent, "trace_inference", False):
        row["_trace"] = agent.drain_inference_trace(battle.battle_tag)
    return row


@dataclass
class ReplayBatchReport:
    battles: int
    wins: int
    win_rate: float
    illegal_top1_count: int
    out_dir: Path
    summary_path: Path
    payload: dict


def run_replay_batch(
    *,
    n_battles: int = 10,
    model_path: Path = SINGLES_BC_MODEL_PATH,
    rl_checkpoint: Path | None = None,
    preview_model_path: Path = SINGLES_PREVIEW_MODEL_PATH,
    device: str = "cpu",
    out_dir: Path | None = None,
    mirror: bool = False,
    use_meta_pool: bool = False,
) -> ReplayBatchReport:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = out_dir or (REPLAYS_DIR / f"singles_{stamp}")
    out.mkdir(parents=True, exist_ok=True)

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
        capture_battle_log=True,
        log_illegal_top1=True,
        save_replays=str(out),
        account_name="SinglesRLReplay" if rl_checkpoint else "SinglesTransformerBC",
    )
    opponent = SinglesMaxDamagePlayer(
        battle_format=SINGLES_BATTLE_FORMAT,
        team=opponent_team,
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration.generate("SinglesMaxDmgBC", rand=True),
    )

    results: list[dict] = []
    battle_traces: list[dict] = []
    illegal_top1: list[dict] = []
    for i in range(n_battles):
        agent.reset_battles()
        opponent.reset_battles()
        row = asyncio.run(_run_one_battle(agent, opponent))
        trace = row.pop("_trace", None)
        if trace is not None:
            trace["index"] = i + 1
            battle_traces.append(trace)
        row["index"] = i + 1
        results.append(row)
        for ev in agent.drain_illegal_top1_events():
            ev["battle_index"] = i + 1
            illegal_top1.append(ev)
        keep_agent_replays(out, agent.username)
        print(
            f"  [{i + 1}/{n_battles}] "
            f"{'WIN' if row['won'] else 'LOSS'} turn={row['turn']} "
            f"brought={row['brought']} tag={row['battle_tag']}",
            flush=True,
        )

    wins = sum(1 for r in results if r["won"])
    replay_files = sorted(out.glob("*.html"))
    payload = {
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
        "mirror": mirror,
        "unique_opponent_brought_lineups": len(
            {tuple(r["opponent_brought"]) for r in results}
        ),
        "unique_opponent_pool_teams": len(
            {tuple(r["opponent_full_team"]) for r in results}
        ),
        "battles": n_battles,
        "wins": wins,
        "losses": n_battles - wins,
        "win_rate": wins / n_battles if n_battles else 0.0,
        "device": device,
        "out_dir": str(out.resolve()),
        "replay_files": [str(p.resolve()) for p in replay_files],
        "battles_detail": results,
        "illegal_top1_count": len(illegal_top1),
        "illegal_top1_events": illegal_top1,
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if battle_traces:
        trace_path = out / "replay_traces.json"
        trace_path.write_text(
            json.dumps({"battles": battle_traces}, indent=2),
            encoding="utf-8",
        )

    return ReplayBatchReport(
        battles=n_battles,
        wins=wins,
        win_rate=payload["win_rate"],
        illegal_top1_count=len(illegal_top1),
        out_dir=out,
        summary_path=summary_path,
        payload=payload,
    )
