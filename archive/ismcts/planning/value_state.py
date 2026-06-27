"""Belief-augmented state vectors for value-network training and inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from poke_env.battle.double_battle import DoubleBattle

from archive.rl.env.observation import FIELD_DIM, N_FEATURES, _encode_field, embed_battle

if TYPE_CHECKING:
    from src.planning.belief_state import BeliefState

# 2 actives × (hp, spe) each side + 6 preview slots × 4 belief features + field + full obs
BELIEF_ROSTER_DIM = 6 * 4
OUR_ACTIVE_DIM = 4
OPP_ACTIVE_DIM = 4
VALUE_EXTRA_DIM = OUR_ACTIVE_DIM + OPP_ACTIVE_DIM + BELIEF_ROSTER_DIM + FIELD_DIM
VALUE_STATE_DIM = VALUE_EXTRA_DIM + N_FEATURES


def _hp_ratio(mon) -> float:
    if mon is None or mon.fainted or not mon.max_hp:
        return 0.0
    return float(mon.current_hp or 0) / float(mon.max_hp)


def _spe_norm(mon) -> float:
    if mon is None or mon.fainted:
        return 0.0
    spe = (mon.stats or {}).get("spe") or (mon.base_stats or {}).get("spe") or 0
    return float(np.clip(spe / 300.0, 0.0, 1.0))


def embed_belief_roster(belief: BeliefState | None) -> np.ndarray:
    """Probabilistic opponent roster features (6 preview slots)."""
    out = np.zeros(BELIEF_ROSTER_DIM, dtype=np.float32)
    if belief is None:
        return out

    mons = sorted(
        belief.pokemon,
        key=lambda m: (m.slot or 99, -m.brought_prob),
    )[:6]
    for i, mon in enumerate(mons):
        base = i * 4
        out[base] = float(mon.brought_prob)
        out[base + 1] = 1.0 if mon.confirmed_brought else 0.0
        out[base + 2] = 1.0 if mon.confirmed_absent else 0.0
        floor = mon.speed_floor or 0
        out[base + 3] = float(np.clip(floor / 300.0, 0.0, 1.0))
    return out


def embed_value_state(
    battle: DoubleBattle,
    belief: BeliefState | None = None,
    *,
    include_full_obs: bool = True,
) -> np.ndarray:
    """
    State vector for value-network training:
    active vitals + belief roster + field + (optional) full battle embedding.
    """
    parts: list[np.ndarray] = []

    for mon in battle.active_pokemon:
        parts.append(np.array([_hp_ratio(mon), _spe_norm(mon)], dtype=np.float32))
    for mon in battle.opponent_active_pokemon:
        parts.append(np.array([_hp_ratio(mon), _spe_norm(mon)], dtype=np.float32))

    parts.append(embed_belief_roster(belief))
    parts.append(_encode_field(battle))

    if include_full_obs:
        parts.append(embed_battle(battle))

    return np.concatenate(parts).astype(np.float32)
