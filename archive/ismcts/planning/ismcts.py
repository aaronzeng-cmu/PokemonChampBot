"""Information Set MCTS (lightweight determinization + damage heuristic)."""

from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Callable

import numpy as np
from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.move import Move
from poke_env.environment.doubles_env import DoublesEnv

from config.settings import (
    ISMCTS_DETERMINIZATIONS,
    ISMCTS_RL_VALUE_WEIGHT,
    ISMCTS_TIME_BUDGET_MS,
    ISMCTS_USE_FAST_DAMAGE,
)
from archive.rl.env.combo_action_space import enumerate_legal_combos
from src.doubles.planning.belief_state import BeliefState, ConcreteSet
from src.doubles.planning.damage_eval import (
    combo_damage_score,
    estimated_damage_to_defender,
    living_ally_pokemon,
    living_opponent_pokemon,
)
from archive.ismcts.planning.fast_damage import fast_combo_damage_score, fast_estimated_damage_to_defender
from src.doubles.planning.macro_strategist import GamePlan
from src.core.planning.species_normalize import opponent_belief_key

if ISMCTS_USE_FAST_DAMAGE:
    _combo_damage = fast_combo_damage_score
    _damage_to_defender = fast_estimated_damage_to_defender
else:
    _combo_damage = combo_damage_score
    _damage_to_defender = estimated_damage_to_defender


def _targets_priority_species(battle: DoubleBattle, priority: list[str]) -> float:
    bonus = 0.0
    for mon in living_opponent_pokemon(battle):
        if mon.species in priority:
            bonus += 15.0
    return bonus


def _opponent_is_mega(
    belief: BeliefState,
    concrete: dict[str, ConcreteSet],
    opp,
) -> bool:
    key = opponent_belief_key(opp)
    if belief.is_mega_confirmed(key):
        return True
    sampled = concrete.get(key)
    return bool(sampled and sampled.mega)


def _threat_penalty(
    battle: DoubleBattle,
    concrete: dict[str, ConcreteSet],
    game_plan: GamePlan | None,
    belief: BeliefState,
) -> float:
    penalty = 0.0
    threats = set(game_plan.primary_threats if game_plan else [])
    for mon in living_ally_pokemon(battle):
        for opp in living_opponent_pokemon(battle):
            key = opponent_belief_key(opp)
            sampled = concrete.get(key)
            if sampled is None:
                continue
            mega = _opponent_is_mega(belief, concrete, opp)
            for move_name in sampled.moves[:2]:
                try:
                    move = Move(move_name.replace(" ", "").lower(), gen=9)
                except Exception:
                    continue
                dmg = _damage_to_defender(
                    battle, move, opp, mon, mega=mega
                )
                weight = 1.5 if key in threats or opp.species in threats else 1.0
                penalty += dmg * weight * 0.01
    return penalty


def evaluate_combo(
    battle: DoubleBattle,
    a0: int,
    a1: int,
    concrete: dict[str, ConcreteSet],
    game_plan: GamePlan | None,
    belief: BeliefState,
) -> float:
    damage = _combo_damage(battle, a0, a1)
    score = damage

    if game_plan:
        score += _targets_priority_species(battle, game_plan.priority_kos) * 0.5
        for mon in living_opponent_pokemon(battle):
            if mon.species in game_plan.primary_threats:
                score += 5.0

    score -= _threat_penalty(battle, concrete, game_plan, belief)

    try:
        order = DoublesEnv.action_to_order(
            np.array([a0, a1], dtype=np.int64),
            battle,
            fake=True,
            strict=False,
        )
        for slot_order in (order.first_order, order.second_order):
            move = slot_order.order
            if isinstance(move, Move) and move.id == "protect":
                score += 8.0
    except (ValueError, TypeError):
        pass

    return score


def search(
    battle: DoubleBattle,
    belief: BeliefState,
    game_plan: GamePlan | None,
    *,
    n_determinizations: int | None = None,
    time_budget_ms: int | None = None,
    rng: random.Random | None = None,
    value_fn: Callable[[DoubleBattle], float] | None = None,
    value_weight: float | None = None,
) -> int:
    n_det = n_determinizations or ISMCTS_DETERMINIZATIONS
    budget_ms = time_budget_ms or ISMCTS_TIME_BUDGET_MS
    rng = rng or random.Random()
    v_weight = ISMCTS_RL_VALUE_WEIGHT if value_weight is None else value_weight

    combos = enumerate_legal_combos(battle)
    if not combos:
        return 0

    deadline = time.monotonic() + budget_ms / 1000.0
    scores: dict[int, float] = defaultdict(float)
    completed = 0
    state_value = value_fn(battle) * v_weight if value_fn else 0.0

    for _d in range(n_det):
        if time.monotonic() >= deadline:
            break
        concrete = belief.sample_determinization(rng)
        for i, (a0, a1) in enumerate(combos):
            if time.monotonic() >= deadline:
                break
            scores[i] += evaluate_combo(battle, a0, a1, concrete, game_plan, belief)
            if state_value:
                scores[i] += state_value
        completed += 1

    if not scores:
        return 0

    if completed > 0:
        return max(scores, key=lambda idx: scores[idx] / completed)
    return max(scores, key=scores.get)
