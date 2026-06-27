"""Fast math-only damage estimates for ISMCTS (no poke-env calculate_damage)."""

from __future__ import annotations

import numpy as np
from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.move import Move
from poke_env.battle.move_category import MoveCategory
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player.battle_order import SingleBattleOrder

from src.doubles.planning.damage_eval import (
    MEGA_DAMAGE_BOOST,
    defender_pokemon,
    fallback_damage_estimate,
    living_opponent_pokemon,
)

_MEGA_OFFENSE_MULT = 1.25
_MEGA_DEFENSE_MULT = 1.1


def _stat(mon, key: str, *, mega: bool, offense: bool) -> float:
    raw = mon.stats.get(key) if mon.stats else None
    if raw is None and mon.base_stats:
        raw = mon.base_stats.get(key)
    val = float(raw or 100)
    if not mega:
        return val
    if offense and key in ("atk", "spa"):
        return val * _MEGA_OFFENSE_MULT
    if not offense and key in ("def", "spd"):
        return val * _MEGA_DEFENSE_MULT
    return val


def fast_estimated_damage_to_defender(
    battle: DoubleBattle,
    move: Move,
    attacker,
    defender,
    *,
    mega: bool,
) -> float:
    if attacker is None or defender is None or defender.fainted:
        return 0.0
    if move is None or move.category == MoveCategory.STATUS or move.base_power <= 1:
        return 0.0

    scale = MEGA_DAMAGE_BOOST if mega else 1.0
    if move.category == MoveCategory.SPECIAL:
        atk = _stat(attacker, "spa", mega=mega, offense=True)
        df = _stat(defender, "spd", mega=False, offense=False)
    else:
        atk = _stat(attacker, "atk", mega=mega, offense=True)
        df = _stat(defender, "def", mega=False, offense=False)

    mult = attacker.damage_multiplier(defender.type_1)
    if defender.type_2 is not None:
        mult = max(mult, attacker.damage_multiplier(defender.type_2))
    stab = 1.5 if move.type in attacker.types else 1.0
    core = float(move.base_power) * (atk / max(df, 1.0)) * mult * stab * 0.05
    if core <= 0:
        return scale * fallback_damage_estimate(battle, move, attacker, defender)
    return scale * core


def fast_slot_order_damage(
    battle: DoubleBattle,
    slot_order: SingleBattleOrder,
    attacker_pos: int,
) -> float:
    move = slot_order.order
    if not isinstance(move, Move):
        return 0.0
    attacker = battle.active_pokemon[attacker_pos]
    if attacker is None:
        return 0.0

    defenders = defender_pokemon(battle, move, slot_order.move_target)
    if not defenders and move.deduced_target in {None}:
        defenders = living_opponent_pokemon(battle)

    mega = bool(getattr(slot_order, "mega", False))
    return sum(
        fast_estimated_damage_to_defender(battle, move, attacker, defender, mega=mega)
        for defender in defenders
    )


def fast_combo_damage_score(battle: DoubleBattle, a0: int, a1: int) -> float:
    action = np.array([a0, a1], dtype=np.int64)
    try:
        order = DoublesEnv.action_to_order(action, battle, fake=True, strict=False)
    except (ValueError, TypeError):
        return -1.0
    return fast_slot_order_damage(battle, order.first_order, 0) + fast_slot_order_damage(
        battle, order.second_order, 1
    )
