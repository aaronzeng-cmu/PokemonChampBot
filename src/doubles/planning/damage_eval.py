"""Shared damage evaluation for MaxDamagePlayer and ISMCTS."""

from __future__ import annotations

import numpy as np
from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.move import Move
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.target import Target
from poke_env.calc import calculate_damage
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player.battle_order import SingleBattleOrder

MEGA_DAMAGE_BOOST = 1.4
_STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")

_SPREAD_FOE_TARGETS = frozenset(
    {
        Target.ALL_ADJACENT_FOES,
        Target.ALL,
        Target.FOE_SIDE,
        Target.RANDOM_NORMAL,
    }
)
_SPREAD_ALL_ADJACENT = frozenset({Target.ALL_ADJACENT})


def stats_ready(pokemon) -> bool:
    if pokemon is None:
        return False
    stats = pokemon.stats
    return all(isinstance(stats.get(key), (int, float)) for key in _STAT_KEYS)


def team_identifier(battle: DoubleBattle, pokemon) -> str | None:
    if pokemon is None:
        return None
    for ident, mon in battle.team.items():
        if mon is pokemon:
            return ident
    for ident, mon in battle.opponent_team.items():
        if mon is pokemon:
            return ident
    return None


def pokemon_at_target(battle: DoubleBattle, target_pos: int):
    if target_pos == DoubleBattle.POKEMON_1_POSITION:
        return battle.active_pokemon[0]
    if target_pos == DoubleBattle.POKEMON_2_POSITION:
        return battle.active_pokemon[1]
    if target_pos == DoubleBattle.OPPONENT_1_POSITION:
        return battle.opponent_active_pokemon[0]
    if target_pos == DoubleBattle.OPPONENT_2_POSITION:
        return battle.opponent_active_pokemon[1]
    return None


def living_opponent_pokemon(battle: DoubleBattle) -> list:
    mons: list = []
    for pos in (DoubleBattle.OPPONENT_1_POSITION, DoubleBattle.OPPONENT_2_POSITION):
        mon = pokemon_at_target(battle, pos)
        if mon is not None and not mon.fainted:
            mons.append(mon)
    return mons


def living_ally_pokemon(battle: DoubleBattle) -> list:
    mons: list = []
    for pos in (DoubleBattle.POKEMON_1_POSITION, DoubleBattle.POKEMON_2_POSITION):
        mon = pokemon_at_target(battle, pos)
        if mon is not None and not mon.fainted:
            mons.append(mon)
    return mons


def defender_pokemon(battle: DoubleBattle, move: Move, move_target: int) -> list:
    if move_target in (
        DoubleBattle.POKEMON_1_POSITION,
        DoubleBattle.POKEMON_2_POSITION,
    ):
        return []

    if move_target in (
        DoubleBattle.OPPONENT_1_POSITION,
        DoubleBattle.OPPONENT_2_POSITION,
    ):
        mon = pokemon_at_target(battle, move_target)
        return [mon] if mon is not None and not mon.fainted else []

    if move_target != DoubleBattle.EMPTY_TARGET_POSITION:
        return []

    deduced = move.deduced_target
    if deduced in _SPREAD_FOE_TARGETS:
        return living_opponent_pokemon(battle)
    if deduced in _SPREAD_ALL_ADJACENT:
        return living_opponent_pokemon(battle)
    if deduced in {Target.SELF, Target.ADJACENT_ALLY, Target.ADJACENT_ALLY_OR_SELF}:
        return []
    if deduced in {Target.ADJACENT_FOE, Target.NORMAL, Target.ANY, None}:
        return living_opponent_pokemon(battle)
    return []


def fallback_damage_estimate(
    battle: DoubleBattle,
    move: Move,
    attacker,
    defender,
) -> float:
    if move is None or move.category == MoveCategory.STATUS or move.base_power <= 1:
        return 0.0
    if defender is None or defender.fainted:
        return 0.0
    mult = attacker.damage_multiplier(defender.type_1)
    if defender.type_2 is not None:
        mult = max(mult, attacker.damage_multiplier(defender.type_2))
    stab = 1.5 if move.type in attacker.types else 1.0
    return float(move.base_power) * mult * stab


def estimated_damage_to_defender(
    battle: DoubleBattle,
    move: Move,
    attacker,
    defender,
    *,
    mega: bool,
) -> float:
    if attacker is None or defender is None or defender.fainted:
        return 0.0

    scale = MEGA_DAMAGE_BOOST if mega else 1.0
    attacker_id = team_identifier(battle, attacker)
    defender_id = team_identifier(battle, defender)
    if (
        attacker_id is not None
        and defender_id is not None
        and stats_ready(attacker)
        and stats_ready(defender)
    ):
        try:
            lo, hi = calculate_damage(
                attacker_id,
                defender_id,
                move,
                battle,
            )
            return scale * (float(lo) + float(hi)) / 2.0
        except (AssertionError, ValueError, TypeError, KeyError, IndexError):
            pass
    return scale * fallback_damage_estimate(battle, move, attacker, defender)


def slot_order_damage(
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
    if not defenders and move.deduced_target in _SPREAD_ALL_ADJACENT:
        defenders = living_opponent_pokemon(battle)

    return sum(
        estimated_damage_to_defender(
            battle,
            move,
            attacker,
            defender,
            mega=bool(getattr(slot_order, "mega", False)),
        )
        for defender in defenders
    )


def combo_damage_score(battle: DoubleBattle, a0: int, a1: int) -> float:
    action = np.array([a0, a1], dtype=np.int64)
    try:
        order = DoublesEnv.action_to_order(action, battle, fake=True, strict=False)
    except (ValueError, TypeError):
        return -1.0
    return slot_order_damage(battle, order.first_order, 0) + slot_order_damage(
        battle, order.second_order, 1
    )


def infer_defense_stat_range(
    observed_damage: float,
    move: Move,
    attacker_stats: dict,
    *,
    defender_hp: int,
) -> tuple[str, int, int]:
    """Rough reverse inference of defender def/spd EV range from damage taken."""
    if observed_damage <= 0 or move is None:
        return ("def", 0, 32)
    if move.category == MoveCategory.SPECIAL:
        stat = "spd"
        atk = attacker_stats.get("spa", 100)
    else:
        stat = "def"
        atk = attacker_stats.get("atk", 100)
    ratio = observed_damage / max(defender_hp, 1)
    if ratio > 0.5:
        return (stat, 0, 8)
    if ratio > 0.25:
        return (stat, 4, 20)
    return (stat, 0, 32)
