#!/usr/bin/env python3
"""Record HTML battle replays for the active BC pipeline (Transformer / random)."""

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
    BC_MODEL_PATH,
    EVAL_N_BATTLES,
    REPLAYS_DIR,
    USE_OPPONENT_TEAM_POOL,
)
from src.core.battle.replay_utils import keep_agent_replays
from src.doubles.players.max_damage_player import MaxDamagePlayer
from src.doubles.players.transformer_player import TransformerPlayer
from src.doubles.players.vgc_random_player import VGCRandomPlayer
from src.doubles.teams.team_pool import (
    load_agent_team,
    load_opponent_team_builder,
    opponent_pool_description,
)
from src.doubles.teams.teampreview import (
    battle_team_summary,
    opponent_full_team_summary,
    opponent_team_summary,
)

OPPONENT_CHOICES = ("random", "maxdamage")
AGENT_CHOICES = ("transformer", "random")


def _make_agent(kind: str, *, model_path: Path, team: str, device: str, out_dir: Path):
    common = dict(
        battle_format=BATTLE_FORMAT,
        team=team,
        max_concurrent_battles=1,
        save_replays=str(out_dir),
    )
    if kind == "transformer":
        return TransformerPlayer(
            model_path=model_path,
            device=device,
            capture_battle_log=True,
            account_configuration=AccountConfiguration.generate("TransformerBC", rand=True),
            **common,
        )
    if kind == "random":
        return VGCRandomPlayer(
            account_configuration=AccountConfiguration.generate("RandomBC", rand=True),
            **common,
        )
    raise ValueError(f"Unknown agent {kind!r}")


def _make_opponent(kind: str, *, team):
    common = dict(
        battle_format=BATTLE_FORMAT,
        team=team,
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration.generate("OppBC", rand=True),
    )
    if kind == "random":
        return VGCRandomPlayer(**common)
    if kind == "maxdamage":
        return MaxDamagePlayer(**common)
    raise ValueError(f"Unknown opponent {kind!r}")


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


def _record_battles(agent, opponent, *, n_battles: int, out_dir: Path) -> tuple[list[dict], list[dict]]:
    results: list[dict] = []
    illegal_top1: list[dict] = []
    for i in range(n_battles):
        agent.reset_battles()
        opponent.reset_battles()
        row = asyncio.run(_run_one_battle(agent, opponent))
        row["index"] = i + 1
        results.append(row)
        if hasattr(agent, "drain_illegal_top1_events"):
            events = agent.drain_illegal_top1_events()
            for ev in events:
                ev["battle_index"] = i + 1
            illegal_top1.extend(events)
        keep_agent_replays(out_dir, agent.username)
        print(
            f"  [{i + 1}/{n_battles}] "
            f"{'WIN' if row['won'] else 'LOSS'} turn={row['turn']} "
            f"opp={row['opponent_brought']} tag={row['battle_tag']}",
            flush=True,
        )
    return results, illegal_top1


def main() -> None:
    parser = argparse.ArgumentParser(description="Record HTML replays on local Showdown")
    parser.add_argument(
        "--agent",
        choices=AGENT_CHOICES,
        default="transformer",
        help="Agent to record from (default: transformer)",
    )
    parser.add_argument(
        "--opponent",
        choices=OPPONENT_CHOICES,
        default="random",
        help="Opponent type (default: random)",
    )
    parser.add_argument("--model", type=Path, default=BC_MODEL_PATH)
    parser.add_argument(
        "--battles",
        type=int,
        default=min(10, EVAL_N_BATTLES),
        help=f"Number of battles (default: {min(10, EVAL_N_BATTLES)})",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Force mirror match instead of sampling opponent pool",
    )
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (REPLAYS_DIR / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    agent_team = load_agent_team()
    use_pool = USE_OPPONENT_TEAM_POOL and not args.mirror
    opponent_team = load_opponent_team_builder(
        use_pool=use_pool,
        use_curriculum=False,
    )
    pool_info = opponent_pool_description(opponent_team)

    print(f"Format: {BATTLE_FORMAT}", flush=True)
    print(f"Agent: {args.agent}", flush=True)
    if args.agent == "transformer":
        print(f"Model: {args.model}", flush=True)
    print(f"Opponent: {args.opponent}", flush=True)
    print(
        f"Opponent teams: {pool_info['mode']} "
        f"(active={pool_info['active_teams']}, pool={pool_info['pool_size']})",
        flush=True,
    )
    print(f"Battles: {args.battles}", flush=True)
    print(f"Replays -> {out_dir.resolve()}", flush=True)

    agent = _make_agent(
        args.agent,
        model_path=args.model,
        team=agent_team,
        device=args.device,
        out_dir=out_dir,
    )
    opponent = _make_opponent(args.opponent, team=opponent_team)

    results, illegal_top1 = _record_battles(agent, opponent, n_battles=args.battles, out_dir=out_dir)

    wins = sum(1 for r in results if r["won"])
    unique_brought_lineups = len({tuple(r["opponent_brought"]) for r in results})
    unique_pool_teams = len({tuple(r["opponent_full_team"]) for r in results})
    replay_files = sorted(out_dir.glob("*.html"))
    summary = {
        "timestamp_utc": stamp,
        "format": BATTLE_FORMAT,
        "agent": args.agent,
        "model": str(args.model.resolve()) if args.agent == "transformer" else None,
        "opponent": args.opponent,
        "opponent_team_mode": pool_info,
        "unique_opponent_brought_lineups": unique_brought_lineups,
        "unique_opponent_pool_teams": unique_pool_teams,
        "battles": args.battles,
        "wins": wins,
        "losses": args.battles - wins,
        "win_rate": wins / args.battles if args.battles else 0.0,
        "device": args.device,
        "out_dir": str(out_dir.resolve()),
        "replay_files": [str(p.resolve()) for p in replay_files],
        "battles_detail": results,
        "illegal_top1_count": len(illegal_top1),
        "illegal_top1_events": illegal_top1,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if illegal_top1:
        illegal_path = out_dir / "illegal_top1_log.json"
        illegal_path.write_text(
            json.dumps(
                {
                    "timestamp_utc": stamp,
                    "model": str(args.model.resolve()) if args.agent == "transformer" else None,
                    "opponent": args.opponent,
                    "count": len(illegal_top1),
                    "events": illegal_top1,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Illegal top-1 fallbacks: {len(illegal_top1)} (see {illegal_path.name})", flush=True)

    print(f"\nWin rate: {wins}/{args.battles} ({100 * summary['win_rate']:.1f}%)", flush=True)
    print(
        f"Unique pool teams: {unique_pool_teams} | "
        f"unique brought lineups: {unique_brought_lineups}",
        flush=True,
    )
    print(f"Replays saved: {len(replay_files)} HTML file(s)", flush=True)
    print(f"Summary: {summary_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
