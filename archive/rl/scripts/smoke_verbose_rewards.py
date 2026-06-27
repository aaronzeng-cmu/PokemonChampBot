#!/usr/bin/env python3
"""Run a few battles and print per-step reward breakdown."""

import random

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from src.players.vgc_random_player import VGCRandomPlayer

from config.settings import BATTLE_FORMAT
from archive.rl.env.champions_vgc_env import ChampionsVGCRLEnv, load_team
from archive.rl.env.gym_wrappers import wrap_for_sb3


def main(n_episodes: int = 3):
    env = ChampionsVGCRLEnv(team=load_team(), log_level=40, open_timeout=None)
    opponent = VGCRandomPlayer(
        battle_format=BATTLE_FORMAT,
        team=load_team(),
        start_listening=False,
    )
    gym_env = wrap_for_sb3(SingleAgentWrapper(env, opponent))

    for ep in range(n_episodes):
        gym_env.reset()
        total = 0.0
        step = 0
        print(f"\n=== Episode {ep + 1} ===")
        while True:
            masks = gym_env.action_masks()
            legal = [i for i, v in enumerate(masks) if v]
            action = random.choice(legal)
            _, reward, term, trunc, _ = gym_env.step(action)
            total += reward
            step += 1
            if reward != 0.0:
                print(f"  step {step}: reward={reward:+.4f} (cum={total:+.4f})")
            if term or trunc or step > 300:
                won = env.battle1.won if env.battle1 else False
                print(f"  done in {step} steps, total={total:+.4f}, won={won}")
                break

    gym_env.close()


if __name__ == "__main__":
    main()
