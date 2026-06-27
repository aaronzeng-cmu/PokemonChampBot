"""Combinatorial joint-legal action enumeration for VGC doubles."""

from __future__ import annotations

import numpy as np
from poke_env.battle.double_battle import DoubleBattle
from poke_env.environment.doubles_env import DoublesEnv

from config.settings import MAX_COMBOS


def _legal_individual_indices(battle: DoubleBattle, pos: int) -> list[int]:
    mask = DoublesEnv.get_action_mask_individual(battle, pos)
    return [i for i, v in enumerate(mask) if v]


def enumerate_legal_combos(battle: DoubleBattle) -> list[tuple[int, int]]:
    """Return sorted list of joint-legal (slot0, slot1) action index pairs."""
    if battle.finished:
        return [(0, 0)]

    if battle._wait or all(
        not any(DoublesEnv.get_action_mask_individual(battle, pos))
        for pos in (0, 1)
    ):
        return [(0, 0)]

    indices_0 = _legal_individual_indices(battle, 0)
    indices_1 = _legal_individual_indices(battle, 1)
    combos: list[tuple[int, int]] = []

    for a0 in indices_0:
        for a1 in indices_1:
            action = np.array([a0, a1], dtype=np.int64)
            try:
                DoublesEnv.action_to_order(action, battle, fake=False, strict=True)
            except ValueError:
                continue
            combos.append((int(a0), int(a1)))

    combos = sorted(set(combos))
    if len(combos) > MAX_COMBOS:
        raise ValueError(
            f"Legal combo count {len(combos)} exceeds MAX_COMBOS={MAX_COMBOS}. "
            "Increase MAX_COMBOS in config/settings.py."
        )
    if not combos:
        raise ValueError(
            f"No legal action combos for battle {battle.battle_tag} turn {battle.turn}"
        )
    return combos


def combo_to_ndarray(combo_idx: int, combos: list[tuple[int, int]]) -> np.ndarray:
    if combo_idx < 0 or combo_idx >= len(combos):
        raise IndexError(
            f"Combo index {combo_idx} out of range for {len(combos)} legal combos"
        )
    a0, a1 = combos[combo_idx]
    return np.array([a0, a1], dtype=np.int64)


def combo_action_mask(combos: list[tuple[int, int]]) -> np.ndarray:
    """Boolean mask of length MAX_COMBOS; True for legal combo indices."""
    mask = np.zeros(MAX_COMBOS, dtype=bool)
    mask[: len(combos)] = True
    return mask
