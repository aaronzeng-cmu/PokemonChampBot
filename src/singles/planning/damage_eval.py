"""Damage evaluation for singles MaxDamagePlayer."""

from __future__ import annotations

import numpy as np
from poke_env.battle.battle import Battle
from poke_env.battle.move import Move
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon import Pokemon
from poke_env.calc import calculate_damage
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player.battle_order import DefaultBattleOrder, SingleBattleOrder

MEGA_DAMAGE_BOOST = 1.4
_TERA_DAMAGE_BOOST = 1.0
_STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")


def stats_ready(pokemon: Pokemon | None) -> bool:
    if pokemon is None:
        return False
    stats = pokemon.stats
    return all(isinstance(stats.get(key), (int, float)) for key in _STAT_KEYS)


def team_identifier(battle: Battle, pokemon: Pokemon | None) -> str | None:
    if pokemon is None:
        return None
    for ident, mon in battle.team.items():
        if mon is pokemon:
            return ident
    for ident, mon in battle.opponent_team.items():
        if mon is pokemon:
            return ident
    return None


def fallback_damage_estimate(
    battle: Battle,
    move: Move,
    attacker: Pokemon,
    defender: Pokemon,
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
    battle: Battle,
    move: Move,
    attacker: Pokemon,
    defender: Pokemon,
    *,
    mega: bool = False,
    terastallize: bool = False,
) -> float:
    if attacker is None or defender is None or defender.fainted:
        return 0.0
    if move.category == MoveCategory.STATUS:
        return 0.0

    scale = 1.0
    if mega:
        scale *= MEGA_DAMAGE_BOOST
    if terastallize:
        scale *= _TERA_DAMAGE_BOOST

    attacker_id = team_identifier(battle, attacker)
    defender_id = team_identifier(battle, defender)
    if (
        attacker_id is not None
        and defender_id is not None
        and stats_ready(attacker)
        and stats_ready(defender)
    ):
        try:
            lo, hi = calculate_damage(attacker_id, defender_id, move, battle)
            return scale * (float(lo) + float(hi)) / 2.0
        except (AssertionError, ValueError, TypeError, KeyError, IndexError):
            pass
    return scale * fallback_damage_estimate(battle, move, attacker, defender)


def singles_order_damage(battle: Battle, order: SingleBattleOrder) -> float:
    move = order.order
    if not isinstance(move, Move):
        return -1.0
    attacker = battle.active_pokemon
    defender = battle.opponent_active_pokemon
    if attacker is None or defender is None or defender.fainted:
        return 0.0
    return estimated_damage_to_defender(
        battle,
        move,
        attacker,
        defender,
        mega=bool(order.mega),
        terastallize=bool(order.terastallize),
    )


def singles_action_damage_score(battle: Battle, action: int) -> float:
    order = SinglesEnv.action_to_order(np.int64(action), battle, strict=False)
    if isinstance(order, DefaultBattleOrder) or not isinstance(order, SingleBattleOrder):
        return -1.0
    if isinstance(order.order, Pokemon):
        return -1.0
    return singles_order_damage(battle, order)
