"""Legal action masks from parsed singles log view states (BC eval + training)."""

from __future__ import annotations

import numpy as np

from src.core.data.log_tracker import BattleLogState
from src.core.data.move_utils import canonical_move_list
from src.core.data.roster_profile import roster_species_key
from src.core.model.transformer_bot import SINGLES_ACTION_SIZE
from src.singles.action_space_spec import decode_singles_action_index
from src.singles.bench_slots import log_our_bench_slots
from src.singles.log_action_codec import MEGA_BASE, MOVE_BASE, SWITCH_BASE


def _active_mon(view: BattleLogState, side: str):
    return view.mons.get(f"{side}a")


def _active_species_on_field(view: BattleLogState, side: str) -> set[str]:
    mon = _active_mon(view, side)
    if mon is None or mon.fainted or mon.hp <= 0 or not mon.species:
        return set()
    return {roster_species_key(mon.species)}


def _legal_switch_indices(view: BattleLogState, side: str) -> set[int]:
    """Bench-aligned switch indices 0-1 (token 5 / token 6)."""
    on_field = _active_species_on_field(view, side)
    legal: set[int] = set()
    for bench_idx, slot in enumerate(log_our_bench_slots(view, side)):
        mon = view.mons.get(slot)
        if mon is None or mon.fainted or mon.hp <= 0 or not mon.species:
            continue
        if roster_species_key(mon.species) in on_field:
            continue
        legal.add(SWITCH_BASE + bench_idx)
    return legal


def _can_mega_for_view(view: BattleLogState, side: str) -> bool:
    mon = _active_mon(view, side)
    return bool(mon and mon.can_mega)


def _legal_move_indices_for_mon(mon) -> set[int]:
    if mon is None:
        return set()
    moves = canonical_move_list(list(mon.moves))
    if not moves:
        return set()
    legal: set[int] = set()
    for slot in range(min(4, len(moves))):
        legal.add(MOVE_BASE + slot)
        if mon.can_mega and mon.mega_capable:
            legal.add(MEGA_BASE + slot)
    return legal


def _apply_surgical_singles_mask(
    mask: np.ndarray,
    *,
    view: BattleLogState,
    side: str,
    force_switch: bool,
) -> None:
    can_mega = _can_mega_for_view(view, side)
    for idx in range(SINGLES_ACTION_SIZE):
        if not mask[idx]:
            continue
        spec = decode_singles_action_index(idx)
        if force_switch:
            if not spec.is_switch:
                mask[idx] = False
            continue
        if spec.illegal_gimmick:
            mask[idx] = False
            continue
        if spec.mega and not can_mega:
            mask[idx] = False


def _safety_valve(mask: np.ndarray, ground_truth: int) -> None:
    if 0 <= ground_truth < SINGLES_ACTION_SIZE:
        mask[ground_truth] = True


def _ensure_nonempty(mask: np.ndarray) -> None:
    if not mask.any():
        mask[SWITCH_BASE] = True


def singles_turn_mask(view: BattleLogState, side: str) -> np.ndarray:
    mask = np.zeros(SINGLES_ACTION_SIZE, dtype=bool)
    mon = _active_mon(view, side)
    if mon is None or mon.fainted or mon.hp <= 0:
        for switch_idx in _legal_switch_indices(view, side):
            if switch_idx < MOVE_BASE:
                mask[switch_idx] = True
        _apply_surgical_singles_mask(mask, view=view, side=side, force_switch=True)
        _ensure_nonempty(mask)
        return mask

    for switch_idx in _legal_switch_indices(view, side):
        if switch_idx < MOVE_BASE:
            mask[switch_idx] = True
    for move_idx in _legal_move_indices_for_mon(mon):
        if move_idx < SINGLES_ACTION_SIZE:
            mask[move_idx] = True

    _apply_surgical_singles_mask(mask, view=view, side=side, force_switch=False)
    _ensure_nonempty(mask)
    return mask


def singles_force_switch_mask(view: BattleLogState, side: str) -> np.ndarray:
    mask = np.zeros(SINGLES_ACTION_SIZE, dtype=bool)
    for switch_idx in _legal_switch_indices(view, side):
        if switch_idx < MOVE_BASE:
            mask[switch_idx] = True
    _apply_surgical_singles_mask(mask, view=view, side=side, force_switch=True)
    _ensure_nonempty(mask)
    return mask


def singles_mask_for_eval(
    view: BattleLogState | None,
    *,
    side: str,
    sample_kind: str,
) -> np.ndarray | None:
    if view is None:
        return None
    if sample_kind == "force_switch":
        return singles_force_switch_mask(view, side)
    return singles_turn_mask(view, side)


def training_singles_mask(
    view: BattleLogState,
    side: str,
    sample_kind: str,
    *,
    ground_truth: int,
) -> np.ndarray:
    force_switch = sample_kind == "force_switch"
    if force_switch:
        mask = singles_force_switch_mask(view, side)
    else:
        mask = singles_turn_mask(view, side)

    _apply_surgical_singles_mask(
        mask, view=view, side=side, force_switch=force_switch
    )
    _safety_valve(mask, ground_truth)
    _ensure_nonempty(mask)
    return mask
