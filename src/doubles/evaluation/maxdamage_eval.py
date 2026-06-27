"""Win-rate evaluation vs MaxDamage on the opponent team pool."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from poke_env.ps_client.account_configuration import AccountConfiguration

from config.settings import BATTLE_FORMAT, BC_MODEL_PATH, USE_OPPONENT_TEAM_POOL
from src.doubles.players.max_damage_player import MaxDamagePlayer
from src.doubles.players.transformer_player import TransformerPlayer
from src.doubles.teams.team_pool import load_agent_team, load_opponent_team_builder, opponent_pool_description
from src.doubles.teams.teampreview import opponent_full_team_summary, opponent_team_summary


async def _run_batch(agent, opponent, *, n_battles: int) -> list[dict]:
    await agent.battle_against(opponent, n_battles=n_battles)
    rows: list[dict] = []
    for battle in agent.battles.values():
        opp = opponent_team_summary(battle)
        full = opponent_full_team_summary(battle)
        rows.append(
            {
                "battle_tag": battle.battle_tag,
                "won": bool(battle.won),
                "turn": battle.turn,
                "opponent_brought": opp["brought"],
                "opponent_full_team": full["full_team"],
            }
        )
    return rows


@dataclass
class LiveEvalReport:
    battles: int
    wins: int
    win_rate: float
    illegal_top1_count: int
    report_path: Path
    payload: dict


def run_maxdamage_eval(
    *,
    n_battles: int = 100,
    model_path: Path = BC_MODEL_PATH,
    device: str = "cpu",
    out_dir: Path,
    mirror: bool = False,
    batch_size: int = 10,
) -> LiveEvalReport:
    out_dir.mkdir(parents=True, exist_ok=True)
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
        account_configuration=AccountConfiguration.generate("TransformerBC", rand=True),
    )
    opponent = MaxDamagePlayer(
        battle_format=BATTLE_FORMAT,
        team=opponent_team,
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration.generate("MaxDamageBC", rand=True),
    )

    all_rows: list[dict] = []
    illegal_top1: list[dict] = []
    wins = 0
    done = 0
    batch_size = min(batch_size, n_battles)

    while done < n_battles:
        k = min(batch_size, n_battles - done)
        agent.reset_battles()
        opponent.reset_battles()
        rows = asyncio.run(_run_batch(agent, opponent, n_battles=k))
        for row in rows:
            row["index"] = len(all_rows) + 1
            all_rows.append(row)
            wins += int(row["won"])
        illegal_top1.extend(agent.drain_illegal_top1_events())
        done += k
        print(f"Progress: {done}/{n_battles} battles, wins={wins}", flush=True)

    payload = {
        "battles": n_battles,
        "wins": wins,
        "losses": n_battles - wins,
        "draws": 0,
        "win_rate": wins / n_battles if n_battles else 0.0,
        "model": str(model_path),
        "opponent": "MaxDamagePlayer",
        "opponent_team_mode": pool_info,
        "unique_opponent_brought_lineups": len(
            {tuple(r["opponent_brought"]) for r in all_rows}
        ),
        "unique_opponent_pool_teams": len(
            {tuple(r["opponent_full_team"]) for r in all_rows}
        ),
        "format": BATTLE_FORMAT,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "battles_detail": all_rows,
        "illegal_top1_count": len(illegal_top1),
        "illegal_top1_events": illegal_top1,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"bc_eval_{stamp}.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest = out_dir / "bc_eval_latest.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return LiveEvalReport(
        battles=n_battles,
        wins=wins,
        win_rate=payload["win_rate"],
        illegal_top1_count=len(illegal_top1),
        report_path=report_path,
        payload=payload,
    )
