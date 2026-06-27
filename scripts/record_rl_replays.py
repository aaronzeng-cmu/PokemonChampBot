#!/usr/bin/env python3
"""Record HTML battle replays for a trained MaskablePPO RL policy."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from poke_env.ps_client.account_configuration import AccountConfiguration
from sb3_contrib import MaskablePPO

from config.settings import BATTLE_FORMAT, REPLAYS_DIR, USE_OPPONENT_TEAM_POOL
from src.core.battle.replay_utils import keep_agent_replays
from src.doubles.players.max_damage_player import MaxDamagePlayer
from src.doubles.players.rl_replay_player import RLReplayPlayer
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

RL_CHECKPOINTS_DIR = Path("models/rl_checkpoints")
DEFAULT_BEST = RL_CHECKPOINTS_DIR / "best_wr86_steps200000_20260615_112649.zip"


def _resolve_checkpoint(path: Path | None) -> Path:
    if path is not None:
        p = path if path.suffix == ".zip" else path.with_suffix(".zip")
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p
    if DEFAULT_BEST.is_file():
        return DEFAULT_BEST
    candidates = sorted(RL_CHECKPOINTS_DIR.glob("best_wr*.zip"))
    if not candidates:
        raise FileNotFoundError(f"No RL checkpoints in {RL_CHECKPOINTS_DIR}")
    return candidates[-1]


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None, help="MaskablePPO .zip")
    parser.add_argument("--battles", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of deterministic argmax",
    )
    args = parser.parse_args()

    ckpt = _resolve_checkpoint(args.checkpoint)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (REPLAYS_DIR / f"rl_eval_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    agent_team = load_agent_team()
    opponent_team = load_opponent_team_builder(use_pool=USE_OPPONENT_TEAM_POOL)
    pool_info = opponent_pool_description(opponent_team)

    print(f"Checkpoint: {ckpt.resolve()}", flush=True)
    print(f"Format: {BATTLE_FORMAT}", flush=True)
    print(f"Opponent: MaxDamagePlayer", flush=True)
    print(f"Opponent pool: {pool_info['mode']}", flush=True)
    print(f"Battles: {args.battles}", flush=True)
    print(f"Replays -> {out_dir.resolve()}", flush=True)

    model = MaskablePPO.load(str(ckpt), device=args.device)
    agent = RLReplayPlayer(
        model,
        deterministic=not args.stochastic,
        battle_format=BATTLE_FORMAT,
        team=agent_team,
        device="cpu",
        max_concurrent_battles=1,
        save_replays=str(out_dir),
        account_configuration=AccountConfiguration.generate("RLReplay", rand=True),
    )
    opponent = MaxDamagePlayer(
        battle_format=BATTLE_FORMAT,
        team=opponent_team,
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration.generate("MaxDamageRL", rand=True),
    )

    results: list[dict] = []
    for i in range(args.battles):
        agent.reset_battles()
        opponent.reset_battles()
        row = asyncio.run(_run_one_battle(agent, opponent))
        row["index"] = i + 1
        results.append(row)
        keep_agent_replays(out_dir, agent.username)
        print(
            f"  [{i + 1}/{args.battles}] "
            f"{'WIN' if row['won'] else 'LOSS'} turn={row['turn']} "
            f"opp={row['opponent_brought']} tag={row['battle_tag']}",
            flush=True,
        )

    wins = sum(1 for r in results if r["won"])
    replay_files = sorted(out_dir.glob("*.html"))
    summary = {
        "timestamp_utc": stamp,
        "checkpoint": str(ckpt.resolve()),
        "deterministic": not args.stochastic,
        "format": BATTLE_FORMAT,
        "opponent": "MaxDamagePlayer",
        "battles": args.battles,
        "wins": wins,
        "losses": args.battles - wins,
        "win_rate": wins / args.battles if args.battles else 0.0,
        "out_dir": str(out_dir.resolve()),
        "replay_files": [str(p.resolve()) for p in replay_files],
        "battles_detail": results,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nWin rate: {wins}/{args.battles} ({100 * summary['win_rate']:.1f}%)", flush=True)
    print(f"Replays saved: {len(replay_files)} HTML file(s)", flush=True)
    print(f"Summary: {summary_path.resolve()}", flush=True)
    for p in replay_files:
        print(f"  {p.resolve()}", flush=True)


if __name__ == "__main__":
    main()
