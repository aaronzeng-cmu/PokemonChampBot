"""Opponent that greedily picks the highest estimated-damage legal combo."""

from __future__ import annotations

import numpy as np
from poke_env.battle.double_battle import DoubleBattle
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder
from poke_env.player.player import Player

from src.doubles.battle.action_space import enumerate_legal_combos
from src.doubles.planning.damage_eval import combo_damage_score
from src.doubles.teams.teampreview import random_teampreview_command


class MaxDamagePlayer(Player):
    def teampreview(self, battle: DoubleBattle) -> str:
        return random_teampreview_command(battle)

    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        if battle.wait:
            return DefaultBattleOrder()
        combos = enumerate_legal_combos(battle)
        best_idx = 0
        best_score = -1.0
        for i, (a0, a1) in enumerate(combos):
            score = combo_damage_score(battle, a0, a1)
            if score > best_score:
                best_score = score
                best_idx = i
        action = np.array(list(combos[best_idx]), dtype=np.int64)
        return DoublesEnv.action_to_order(action, battle, strict=False)
