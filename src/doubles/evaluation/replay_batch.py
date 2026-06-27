"""Record HTML replay batches for the BC Transformer agent."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from poke_env.ps_client.account_configuration import AccountConfiguration

from config.settings import BATTLE_FORMAT, BC_MODEL_PATH, REPLAYS_DIR, USE_OPPONENT_TEAM_POOL
from src.core.battle.replay_utils import keep_agent_replays
from src.doubles.players.max_damage_player import MaxDamagePlayer
from src.doubles.players.transformer_player import TransformerPlayer
from src.doubles.teams.team_pool import load_agent_team, load_opponent_team_builder, opponent_pool_description
from src.doubles.teams.teampreview import (
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
    return {
        "battle_tag": battle.battle_tag,
        "won": bool(battle.won),
        "turn": battle.turn,
        "leads": team_info["leads"],
        "brought": team_info["brought"],
        "opponent_brought": opp_info["brought"],
        "opponent_full_team": full_info["full_team"],
    }


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
    model_path: Path = BC_MODEL_PATH,
    device: str = "cpu",
    out_dir: Path | None = None,
    opponent: str = "maxdamage",
    mirror: bool = False,
) -> ReplayBatchReport:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = out_dir or (REPLAYS_DIR / stamp)
    out.mkdir(parents=True, exist_ok=True)

    agent_team = load_agent_team()
    use_pool = USE_OPPONENT_TEAM_POOL and not mirror
    opponent_team = load_opponent_team_builder(use_pool=use_pool, use_curriculum=False)
    pool_info = opponent_pool_description(opponent_team)

    agent = TransformerPlayer(
        model_path=model_path,
        battle_format=BATTLE_FORMAT,
        team=agent_team,
        device=device,
        capture_battle_log=True,
        max_concurrent_battles=1,
        save_replays=str(out),
        account_configuration=AccountConfiguration.generate("TransformerBC", rand=True),
    )
    if opponent == "maxdamage":
        opp = MaxDamagePlayer(
            battle_format=BATTLE_FORMAT,
            team=opponent_team,
            max_concurrent_battles=1,
            account_configuration=AccountConfiguration.generate("OppBC", rand=True),
        )
    else:
        from src.doubles.players.vgc_random_player import VGCRandomPlayer

        opp = VGCRandomPlayer(
            battle_format=BATTLE_FORMAT,
            team=opponent_team,
            max_concurrent_battles=1,
            account_configuration=AccountConfiguration.generate("OppBC", rand=True),
        )

    results: list[dict] = []
    illegal_top1: list[dict] = []
    for i in range(n_battles):
        agent.reset_battles()
        opp.reset_battles()
        row = asyncio.run(_run_one_battle(agent, opp))
        row["index"] = i + 1
        results.append(row)
        for ev in agent.drain_illegal_top1_events():
            ev["battle_index"] = i + 1
            illegal_top1.append(ev)
        keep_agent_replays(out, agent.username)
        print(
            f"  [{i + 1}/{n_battles}] "
            f"{'WIN' if row['won'] else 'LOSS'} turn={row['turn']} "
            f"opp={row['opponent_brought']} tag={row['battle_tag']}",
            flush=True,
        )

    wins = sum(1 for r in results if r["won"])
    replay_files = sorted(out.glob("*.html"))
    payload = {
        "timestamp_utc": stamp,
        "format": BATTLE_FORMAT,
        "agent": "transformer",
        "model": str(model_path.resolve()),
        "opponent": opponent,
        "opponent_team_mode": pool_info,
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
    if illegal_top1:
        (out / "illegal_top1_log.json").write_text(
            json.dumps(
                {
                    "timestamp_utc": stamp,
                    "model": str(model_path.resolve()),
                    "opponent": opponent,
                    "count": len(illegal_top1),
                    "events": illegal_top1,
                },
                indent=2,
            ),
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
