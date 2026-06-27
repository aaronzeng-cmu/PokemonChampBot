#!/usr/bin/env python3
"""Evaluate TransformerPlayer vs MaxDamagePlayer on local Showdown."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poke_env.ps_client.account_configuration import AccountConfiguration

from config.settings import (
    BATTLE_FORMAT,
    BC_EVAL_LOG_DIR,
    BC_MODEL_PATH,
    USE_OPPONENT_TEAM_POOL,
)
from src.doubles.players.max_damage_player import MaxDamagePlayer
from src.doubles.players.transformer_player import TransformerPlayer
from src.doubles.teams.team_pool import (
    PoolTeambuilder,
    load_agent_team,
    load_opponent_team_builder,
    opponent_pool_description,
)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BC Transformer bot")
    parser.add_argument("--battles", type=int, default=100)
    parser.add_argument("--model", type=Path, default=BC_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Force mirror match (same team both sides) instead of opponent pool",
    )
    args = parser.parse_args()

    agent_team = load_agent_team()
    use_pool = USE_OPPONENT_TEAM_POOL and not args.mirror
    opponent_team = load_opponent_team_builder(
        use_pool=use_pool,
        use_curriculum=False,
    )
    pool_info = opponent_pool_description(opponent_team)

    agent = TransformerPlayer(
        model_path=args.model,
        battle_format=BATTLE_FORMAT,
        team=agent_team,
        device=args.device,
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

    print(f"Format: {BATTLE_FORMAT}")
    print(f"Agent team: {Path('teams/reg_ma_team.txt').name}")
    print(
        f"Opponent teams: {pool_info['mode']} "
        f"(active={pool_info['active_teams']}, pool={pool_info['pool_size']})"
    )
    print(f"Battles: {args.battles}")

    all_rows: list[dict] = []
    illegal_top1: list[dict] = []
    wins = 0
    batch = min(10, args.battles)
    done = 0
    while done < args.battles:
        k = min(batch, args.battles - done)
        agent.reset_battles()
        opponent.reset_battles()
        rows = asyncio.run(_run_batch(agent, opponent, n_battles=k))
        for row in rows:
            row["index"] = len(all_rows) + 1
            all_rows.append(row)
            wins += int(row["won"])
        events = agent.drain_illegal_top1_events()
        illegal_top1.extend(events)
        done += k
        print(f"Progress: {done}/{args.battles} battles, wins={wins}")

    unique_brought_lineups = len({tuple(r["opponent_brought"]) for r in all_rows})
    unique_pool_teams = len({tuple(r["opponent_full_team"]) for r in all_rows})

    report = {
        "battles": args.battles,
        "wins": wins,
        "losses": args.battles - wins,
        "draws": 0,
        "win_rate": wins / args.battles if args.battles else 0.0,
        "model": str(args.model),
        "opponent": "MaxDamagePlayer",
        "opponent_team_mode": pool_info,
        "unique_opponent_brought_lineups": unique_brought_lineups,
        "unique_opponent_pool_teams": unique_pool_teams,
        "format": BATTLE_FORMAT,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "battles_detail": all_rows,
        "illegal_top1_count": len(illegal_top1),
        "illegal_top1_events": illegal_top1,
    }
    print(f"Win rate: {report['win_rate']:.1%} ({wins}/{args.battles})")
    if illegal_top1:
        print(f"Illegal raw top-1 fallbacks: {len(illegal_top1)}", flush=True)
    print(
        f"Unique opponent pool teams: {unique_pool_teams} | "
        f"unique brought lineups (4 mon): {unique_brought_lineups}"
    )

    BC_EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = BC_EVAL_LOG_DIR / f"bc_eval_{stamp}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    main()
