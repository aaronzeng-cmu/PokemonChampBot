#!/usr/bin/env python3
"""Log combinatorial action space sizes over random-play battles."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import random
from collections import Counter

from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from src.players.vgc_random_player import VGCRandomPlayer

from config.settings import BATTLE_FORMAT, MAX_COMBOS
from archive.rl.env.champions_vgc_env import ChampionsVGCRLEnv, load_team
from archive.rl.env.combo_action_space import enumerate_legal_combos
from archive.rl.env.gym_wrappers import wrap_for_sb3


def main(n_battles: int = 5):
    env = ChampionsVGCRLEnv(team=load_team(), log_level=40, open_timeout=None)
    opponent = VGCRandomPlayer(
        battle_format=BATTLE_FORMAT,
        team=load_team(),
        start_listening=False,
    )
    gym_env = wrap_for_sb3(SingleAgentWrapper(env, opponent))
    counts: list[int] = []

    for battle_i in range(n_battles):
        gym_env.reset()
        steps = 0
        while True:
            battle = env.battle1
            if battle and not battle.finished and env.agent1_to_move:
                combos = enumerate_legal_combos(battle)
                counts.append(len(combos))
            masks = gym_env.action_masks()
            legal = [i for i, v in enumerate(masks) if v]
            action = random.choice(legal) if legal else 0
            _, _, term, trunc, _ = gym_env.step(action)
            steps += 1
            if term or trunc or steps > 400:
                break
        print(f"Battle {battle_i + 1}/{n_battles} finished in {steps} steps")

    if not counts:
        print("No decision points recorded.")
        return

    print(f"\nMAX_COMBOS setting: {MAX_COMBOS}")
    print(f"Samples: {len(counts)}")
    print(f"max(K): {max(counts)}")
    print(f"min(K): {min(counts)}")
    print(f"mean(K): {sum(counts) / len(counts):.1f}")
    print(f"K > 512: {sum(1 for c in counts if c > 512)}")
    print(f"K > 1024: {sum(1 for c in counts if c > 1024)}")
    dist = Counter(counts)
    print("Top 10 K values:", dist.most_common(10))
    if max(counts) > MAX_COMBOS:
        print("WARNING: Increase MAX_COMBOS!")
    else:
        print("OK: MAX_COMBOS is sufficient.")

    gym_env.close()


if __name__ == "__main__":
    main()
