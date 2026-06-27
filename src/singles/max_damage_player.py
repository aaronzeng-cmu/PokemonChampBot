"""Greedy singles opponent: pick the highest estimated-damage legal move."""

from __future__ import annotations

import numpy as np
from poke_env.battle.battle import Battle
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder
from poke_env.player.player import Player

from src.singles.planning.damage_eval import singles_action_damage_score
from src.singles.teampreview import random_teampreview_command

_MOVE_ACTION_MIN = 6


def _candidate_actions(battle: Battle, legal: list[int]) -> list[int]:
    if battle.force_switch or battle.active_pokemon is None:
        return [i for i in legal if i < _MOVE_ACTION_MIN]
    moves = [i for i in legal if i >= _MOVE_ACTION_MIN]
    if moves:
        return moves
    return [i for i in legal if i < _MOVE_ACTION_MIN]


class SinglesMaxDamagePlayer(Player):
    def teampreview(self, battle: Battle) -> str:
        return random_teampreview_command(battle)

    def choose_move(self, battle: Battle) -> BattleOrder:
        if battle.wait and not battle.force_switch:
            return DefaultBattleOrder()

        mask = SinglesEnv.get_action_mask(battle)
        legal = [i for i, allowed in enumerate(mask) if allowed]
        if not legal:
            return DefaultBattleOrder()

        candidates = _candidate_actions(battle, legal)
        best_action = candidates[0]
        best_score = -1.0
        for action in candidates:
            score = singles_action_damage_score(battle, action)
            if score > best_score:
                best_score = score
                best_action = action
        return SinglesEnv.action_to_order(np.int64(best_action), battle, strict=False)
