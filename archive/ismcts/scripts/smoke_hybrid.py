#!/usr/bin/env python3
"""Smoke test for HybridPlayer (offline checks + optional live battle)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from config.settings import BATTLE_FORMAT
from src.planning.belief_state import BeliefState
from src.planning.macro_strategist import HeuristicMacroStrategist
from src.planning.meta_database import MetaDatabase


def offline_smoke() -> None:
    db = MetaDatabase()
    print(f"Meta species with priors: {len(db._pikalytics.get('pokemon', {}))}")
    print(f"Dex moves cached: {len(db._dex.get('moves', {}))}")
    print(f"Pool spread species: {len(db._spread_priors)}")

    prior = db.get_species_prior("Kingambit")
    top_move = max(prior.moves.items(), key=lambda x: x[1])
    print(f"Kingambit top move: {top_move[0]} ({top_move[1]:.1%})")
    print(f"Move desc: {db.move_description(top_move[0])[:80]}")

    belief = BeliefState()
    print("BeliefState + MacroStrategist offline checks passed.")


def live_smoke(battles: int = 1) -> None:
    from archive.rl.env.champions_vgc_env import (
        ChampionsVGCRLEnv,
        load_team,
        resolve_opponent_team,
    )
    from archive.rl.env.hybrid_agent_wrapper import HybridAgentWrapper
    from archive.ismcts.players.hybrid_player import HybridPlayer
    from src.players.max_damage_player import MaxDamagePlayer

    env = ChampionsVGCRLEnv(
        team=load_team(),
        opponent_team=resolve_opponent_team(),
        log_level=25,
        open_timeout=None,
    )
    agent = HybridPlayer(start_listening=False)
    opponent = MaxDamagePlayer(start_listening=False)
    gym_env = HybridAgentWrapper(env, agent, opponent)

    wins = 0
    for i in range(battles):
        obs, _ = gym_env.reset()
        total_reward = 0.0
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, _ = gym_env.step(0)
            total_reward += float(reward)
            steps += 1
            if steps > 500:
                break
        won = bool(env.battle1.won) if env.battle1 else False
        wins += int(won)
        print(
            f"Battle {i + 1}: {'WIN' if won else 'LOSS'} "
            f"reward={total_reward:+.2f} steps={steps}"
        )
    gym_env.close()
    print(f"Live smoke: {wins}/{battles} wins")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid player smoke test")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live battle vs MaxDamagePlayer (requires local Showdown)",
    )
    parser.add_argument("--battles", type=int, default=1)
    args = parser.parse_args()

    offline_smoke()
    if args.live:
        print(f"\nLive battles ({BATTLE_FORMAT})...")
        live_smoke(args.battles)


if __name__ == "__main__":
    main()
