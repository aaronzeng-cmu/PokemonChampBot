"""Gauntlet opponent: MaxDamage with occasional status / Protect heuristics."""

from __future__ import annotations

import random

import numpy as np
from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.move import Move
from poke_env.battle.move_category import MoveCategory
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder

from src.doubles.battle.action_space import enumerate_legal_combos
from src.doubles.planning.damage_eval import combo_damage_score
from src.doubles.players.max_damage_player import MaxDamagePlayer

_STATUS_MOVE_CHANCE = 0.10
_PROTECT_LOW_HP_CHANCE = 0.15
_LOW_HP_RATIO = 0.35


def _combo_orders(battle: DoubleBattle, combo: tuple[int, int]):
    action = np.array(list(combo), dtype=np.int64)
    return DoublesEnv.action_to_order(action, battle, fake=True, strict=False)


def _slot_uses_status(slot_order) -> bool:
    move = slot_order.order
    return isinstance(move, Move) and move.category == MoveCategory.STATUS


def _slot_uses_protect(slot_order) -> bool:
    move = slot_order.order
    return isinstance(move, Move) and move.id == "protect"


def _combo_has_status(battle: DoubleBattle, combo: tuple[int, int]) -> bool:
    try:
        order = _combo_orders(battle, combo)
    except (ValueError, TypeError):
        return False
    return _slot_uses_status(order.first_order) or _slot_uses_status(order.second_order)


def _combo_has_protect(battle: DoubleBattle, combo: tuple[int, int]) -> bool:
    try:
        order = _combo_orders(battle, combo)
    except (ValueError, TypeError):
        return False
    return _slot_uses_protect(order.first_order) or _slot_uses_protect(order.second_order)


def _best_combo(
    battle: DoubleBattle,
    combos: list[tuple[int, int]],
    *,
    predicate,
) -> tuple[int, int] | None:
    filtered = [c for c in combos if predicate(battle, c)]
    if not filtered:
        return None
    best = filtered[0]
    best_score = combo_damage_score(battle, best[0], best[1])
    for combo in filtered[1:]:
        score = combo_damage_score(battle, combo[0], combo[1])
        if score > best_score:
            best, best_score = combo, score
    return best


class GauntletOpponentPlayer(MaxDamagePlayer):
    """MaxDamage baseline with 10% status bias and low-HP Protect."""

    def __init__(self, *, status_chance: float = _STATUS_MOVE_CHANCE, **kwargs):
        super().__init__(**kwargs)
        self.status_chance = status_chance
        self._rng = random.Random()

    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        if battle.wait:
            return DefaultBattleOrder()

        combos = enumerate_legal_combos(battle)
        if not combos:
            return DefaultBattleOrder()

        low_hp = any(
            mon
            and not mon.fainted
            and mon.max_hp
            and (mon.current_hp or 0) / mon.max_hp <= _LOW_HP_RATIO
            for mon in battle.active_pokemon
        )
        if low_hp and self._rng.random() < _PROTECT_LOW_HP_CHANCE:
            protect_combo = _best_combo(battle, combos, predicate=_combo_has_protect)
            if protect_combo is not None:
                action = np.array(list(protect_combo), dtype=np.int64)
                return DoublesEnv.action_to_order(action, battle, strict=False)

        if self._rng.random() < self.status_chance:
            status_combo = _best_combo(battle, combos, predicate=_combo_has_status)
            if status_combo is not None:
                action = np.array(list(status_combo), dtype=np.int64)
                return DoublesEnv.action_to_order(action, battle, strict=False)

        return super().choose_move(battle)
