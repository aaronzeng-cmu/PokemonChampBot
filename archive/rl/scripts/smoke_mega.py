#!/usr/bin/env python3
"""Verify mega evolution is legal and reachable in training battles."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from poke_env.environment.doubles_env import DoublesEnv

from config.settings import BATTLE_FORMAT
from archive.rl.env.champions_vgc_env import ChampionsVGCRLEnv, load_team, resolve_opponent_team
from archive.rl.env.combo_action_space import enumerate_legal_combos
from archive.rl.env.gym_wrappers import wrap_for_sb3
from archive.rl.env.single_agent_wrapper import SingleAgentWrapper
from src.players.vgc_random_player import VGCRandomPlayer


def _combo_has_mega(battle, combo_idx: int) -> bool:
    combos = enumerate_legal_combos(battle)
    if combo_idx >= len(combos):
        return False
    action = np.array(list(combos[combo_idx]), dtype=np.int64)
    order = DoublesEnv.action_to_order(action, battle, fake=True, strict=False)
    for slot in (order.first_order, order.second_order):
        if getattr(slot, "mega", False):
            return True
    return False


def main(n_battles: int = 4) -> None:
    env = ChampionsVGCRLEnv(
        team=load_team(),
        opponent_team=resolve_opponent_team(),
        log_level=50,
        open_timeout=None,
    )
    opponent = VGCRandomPlayer(battle_format=BATTLE_FORMAT, start_listening=False)
    gym = wrap_for_sb3(SingleAgentWrapper(env, opponent))

    stats = {
        "battles": 0,
        "charizard_active_turns": 0,
        "can_mega_steps": 0,
        "mega_legal_combos": 0,
        "mega_taken": 0,
    }

    for _ in range(n_battles):
        gym.reset()
        battle = env.battle1
        for _ in range(150):
            battle = env.battle1
            if battle is None or battle.finished:
                break
            if not battle.teampreview and battle.turn >= 1:
                for i, mon in enumerate(battle.active_pokemon):
                    if mon and "charizard" in mon.species:
                        stats["charizard_active_turns"] += 1
                        if battle.can_mega_evolve[i]:
                            stats["can_mega_steps"] += 1
                            legal = [j for j, v in enumerate(gym.action_masks()) if v]
                            if any(_combo_has_mega(battle, j) for j in legal):
                                stats["mega_legal_combos"] += 1
            if battle.teampreview and env.agent1_to_move:
                legal = [i for i, v in enumerate(gym.action_masks()) if v]
                action = random.choice(legal) if legal else 0
            else:
                action = 0
                if (
                    battle
                    and not battle.teampreview
                    and any(battle.can_mega_evolve)
                    and env.agent1_to_move
                ):
                    legal = [i for i, v in enumerate(gym.action_masks()) if v]
                    mega_legal = [i for i in legal if _combo_has_mega(battle, i)]
                    if mega_legal and random.random() < 0.5:
                        action = random.choice(mega_legal)
                        stats["mega_taken"] += 1
            _, _, term, trunc, _ = gym.step(action)
            if term or trunc:
                break
        stats["battles"] += 1

    gym.close()
    print(json.dumps(stats, indent=2))
    out = Path("logs") / "mega_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote {out.resolve()}")

    if stats["can_mega_steps"] == 0:
        raise SystemExit(
            "No can_mega steps: check reg_ma_team.txt slot order (Charizard must be slots 3–6)."
        )
    if stats["mega_legal_combos"] == 0:
        raise SystemExit("Mega never appeared in legal action masks.")


if __name__ == "__main__":
    main()
