"""Legal action masks reconstructed from parsed log view states (BC eval)."""

from __future__ import annotations

import numpy as np
import torch
from poke_env.battle.move import Move
from poke_env.battle.target import Target
from poke_env.data import to_id_str

from src.doubles.battle.move_order import (
    apply_joint_slot1_mask_numpy,
    canonical_move_list,
    decode_move_action_index,
    encode_move_action_index,
    is_mega_action,
)
from src.doubles.data.action_space_spec import (
    ACTION_SIZE,
    TARGET_ALLY_SLOT_A,
    TARGET_ALLY_SLOT_B,
    TARGET_DEFAULT,
    TARGET_OPP_SLOT_A,
    TARGET_OPP_SLOT_B,
    move_default_target_offset,
)
from src.core.data.log_tracker import BattleLogState
from src.core.data.roster_profile import roster_species_key

_ALLY_TARGETS = frozenset(
    {
        Target.ADJACENT_ALLY,
        Target.ADJACENT_ALLY_OR_SELF,
        Target.ALLIES,
        Target.ALLY_SIDE,
        Target.ALLY_TEAM,
    }
)


def _active_mon(view: BattleLogState, side: str, suffix: str):
    return view.mons.get(f"{side}{suffix}")


def _species_fainted_on_side(view: BattleLogState, side: str, species_id: str) -> bool:
    sid = roster_species_key(species_id)
    for slot, mon in view.mons.items():
        if not slot.startswith(side):
            continue
        if roster_species_key(mon.species) != sid:
            continue
        if mon.fainted or mon.hp <= 0:
            return True
    return False


def _active_species_on_field(view: BattleLogState, side: str) -> set[str]:
    out: set[str] = set()
    for suffix in ("a", "b"):
        mon = _active_mon(view, side, suffix)
        if mon is None or mon.fainted or mon.hp <= 0:
            continue
        if mon.species:
            out.add(roster_species_key(mon.species))
    return out


def _legal_switch_indices(view: BattleLogState, side: str) -> set[int]:
    roster = view.team_roster.get(side, [])
    if not roster:
        return set()
    on_field = _active_species_on_field(view, side)
    brought = view.brought_species.get(side)
    legal: set[int] = set()
    for i, species in enumerate(roster[:6]):
        sid = roster_species_key(species)
        if brought and sid not in brought:
            continue
        if sid in on_field:
            continue
        if _species_fainted_on_side(view, side, sid):
            continue
        legal.add(i + 1)
    return legal


def _legal_move_indices_for_mon(mon) -> set[int]:
    if mon is None:
        return set()
    moves = canonical_move_list(list(mon.moves))
    if not moves:
        return set()

    out: set[int] = set()
    gimmicks: list[tuple[bool, bool]] = [(False, False)]
    if mon.can_mega:
        gimmicks.append((True, False))
    if not mon.terastallized:
        gimmicks.append((False, True))

    for move_name in moves:
        default = move_default_target_offset(move_name)
        offsets: set[int]
        if default is not None:
            offsets = {TARGET_DEFAULT}
        else:
            offsets = {TARGET_OPP_SLOT_A, TARGET_OPP_SLOT_B}
            try:
                move = Move(to_id_str(move_name))
                if move.deduced_target in _ALLY_TARGETS:
                    offsets.update({TARGET_ALLY_SLOT_A, TARGET_ALLY_SLOT_B})
            except Exception:
                pass

        for mega, tera in gimmicks:
            if tera and mon.terastallized:
                continue
            for offset in offsets:
                out.add(
                    encode_move_action_index(
                        moves,
                        move_name,
                        offset,
                        mega=mega,
                        terastallize=tera,
                    )
                )
    return out


def _empty_mask() -> np.ndarray:
    return np.zeros(ACTION_SIZE, dtype=bool)


def _forced_switch_suffix(view: BattleLogState, side: str) -> str | None:
    for suffix in ("a", "b"):
        mon = _active_mon(view, side, suffix)
        if mon is not None and (mon.fainted or mon.hp <= 0):
            return suffix
    return None


def is_dual_force_view(view: BattleLogState, side: str) -> bool:
    """Both actives must switch in this view (simultaneous faint)."""
    fa, fb = _force_switch_slot_flags(view, side)
    return fa and fb


def forced_switch_battle_flags(view: BattleLogState, side: str) -> list[bool]:
    """poke-env-style force_switch flags matching a parser force_switch view."""
    fa, fb = _force_switch_slot_flags(view, side)
    return [fa, fb]


def _force_switch_slot_flags(view: BattleLogState, side: str) -> tuple[bool, bool]:
    """Return (slot_a_must_switch, slot_b_must_switch)."""
    forced = _forced_switch_suffix(view, side)
    if forced == "a":
        return True, False
    if forced == "b":
        return False, True
    ma = _active_mon(view, side, "a")
    mb = _active_mon(view, side, "b")
    fa = ma is not None and (ma.fainted or ma.hp <= 0)
    fb = mb is not None and (mb.fainted or mb.hp <= 0)
    if fa and fb:
        return True, True
    return False, False


def _apply_targeting_and_mega_rules(mask: np.ndarray, mon) -> None:
    """Mask illegal move targets and megas; leaves pass/switch indices untouched."""
    if mon is None:
        mask[7:] = False
        return

    moves = canonical_move_list(list(mon.moves))
    for idx in range(7, ACTION_SIZE):
        if not mask[idx]:
            continue

        move_slot, target_offset, mega, _tera = decode_move_action_index(idx)

        if mega and not mon.can_mega:
            mask[idx] = False
            continue

        if target_offset in (TARGET_ALLY_SLOT_A, TARGET_ALLY_SLOT_B):
            mask[idx] = False
            continue

        if move_slot < 1 or move_slot > len(moves):
            mask[idx] = False
            continue

        move_name = moves[move_slot - 1]
        default = move_default_target_offset(move_name)
        if default is not None and target_offset != TARGET_DEFAULT:
            mask[idx] = False
            continue


def _build_slot_mask(
    view: BattleLogState,
    side: str,
    suffix: str,
    *,
    slot_forced: bool,
    partner_forced: bool,
) -> np.ndarray:
    """
    Build a 107-length mask with surgical force-switch, targeting, and mega rules.

    Force-switch:
      - forced slot: switches only
      - partner slot while other is forced: pass only
    """
    mask = _empty_mask()
    mon = _active_mon(view, side, suffix)

    if slot_forced:
        for switch_idx in _legal_switch_indices(view, side):
            if 1 <= switch_idx <= 6:
                mask[switch_idx] = True
        if not mask.any():
            mask[0] = True
        return mask

    if partner_forced:
        mask[0] = True
        return mask

    if mon is None or mon.fainted or mon.hp <= 0:
        mask[0] = True
        return mask

    mask[0] = True
    for switch_idx in _legal_switch_indices(view, side):
        if 1 <= switch_idx <= 6:
            mask[switch_idx] = True
    for move_idx in _legal_move_indices_for_mon(mon):
        if 0 <= move_idx < ACTION_SIZE:
            mask[move_idx] = True

    _apply_targeting_and_mega_rules(mask, mon)

    if not mask.any():
        mask[0] = True
    return mask


def log_turn_slot_mask(view: BattleLogState, side: str, suffix: str) -> np.ndarray:
    """Per-slot legality mask for a normal turn-start decision."""
    return _build_slot_mask(
        view,
        side,
        suffix,
        slot_forced=False,
        partner_forced=False,
    )


def log_force_switch_slot_masks(
    view: BattleLogState, side: str
) -> tuple[np.ndarray, np.ndarray]:
    """Strict force-switch masks: forced slot = switches only; partner = pass only."""
    fa, fb = _force_switch_slot_flags(view, side)
    if not fa and not fb:
        pass_only = _empty_mask()
        pass_only[0] = True
        return pass_only.copy(), pass_only.copy()

    mask0 = _build_slot_mask(
        view, side, "a", slot_forced=fa, partner_forced=fb and not fa
    )
    mask1 = _build_slot_mask(
        view, side, "b", slot_forced=fb, partner_forced=fa and not fb
    )
    return mask0, mask1


def _apply_joint_turn_constraints(
    mask1: np.ndarray, *, a0: int, force_switch: bool = False, view: BattleLogState | None = None, side: str = "p1"
) -> np.ndarray:
    del view, side  # per-mon mega legality is applied in _build_slot_mask
    return apply_joint_slot1_mask_numpy(
        mask1, a0_canonical=a0, force_switch=force_switch
    )


def _safety_valve(mask: np.ndarray, ground_truth: int) -> None:
    """Ground-truth human action is always legal for CrossEntropy supervision."""
    if 0 <= ground_truth < ACTION_SIZE:
        mask[ground_truth] = True


def pick_masked_argmax(logits: torch.Tensor, mask: np.ndarray) -> int:
    """Return highest-logit action among legal indices."""
    legal = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
    if not bool(legal.any()):
        return 0
    masked = logits.clone()
    masked[~legal] = -float("inf")
    return int(masked.argmax().item())


def training_slot_masks(
    view: BattleLogState,
    side: str,
    sample_kind: str,
    *,
    ground_truth_a0: int,
    ground_truth_a1: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-slot legal masks for BC training (107 bool each).

    Slot 1 uses joint constraints from ground-truth slot 0 so the loss only
    considers actions legal given the logged partner choice. Ground-truth
    indices are always kept legal so CrossEntropy can supervise rare joint lines.
    """
    if sample_kind == "force_switch":
        mask0, mask1 = log_force_switch_slot_masks(view, side)
    else:
        mask0 = log_turn_slot_mask(view, side, "a")
        mask1 = log_turn_slot_mask(view, side, "b")

    joint_a0 = ground_truth_a0 if ground_truth_a0 >= 0 else 0
    mask1 = _apply_joint_turn_constraints(
        mask1,
        a0=joint_a0,
        force_switch=(sample_kind == "force_switch"),
        view=view,
        side=side,
    )

    _safety_valve(mask0, ground_truth_a0)
    _safety_valve(mask1, ground_truth_a1)
    return mask0, mask1


def pick_masked_log_actions(
    logits0: torch.Tensor,
    logits1: torch.Tensor,
    *,
    view: BattleLogState,
    side: str,
    sample_kind: str,
) -> tuple[int, int]:
    """
    Pick joint legal actions from a parser-style log view.

    For simultaneous dual force-switch (both slots faint), use
    ``pick_masked_dual_force_actions`` instead — training emits two single-slot
    samples and live inference runs two forward passes.
    """
    if sample_kind == "force_switch":
        mask0, mask1 = log_force_switch_slot_masks(view, side)
    else:
        mask0 = log_turn_slot_mask(view, side, "a")
        mask1 = log_turn_slot_mask(view, side, "b")

    a0 = pick_masked_argmax(logits0, mask0)
    mask1_final = _apply_joint_turn_constraints(
        mask1, a0=a0, force_switch=(sample_kind == "force_switch"), view=view, side=side
    )
    a1 = pick_masked_argmax(logits1, mask1_final)
    return a0, a1


def slot_mask_for_eval(
    view: BattleLogState | None,
    *,
    side: str,
    sample_kind: str,
    slot_suffix: str,
    slot0_pred: int | None = None,
) -> np.ndarray | None:
    """Return per-slot mask for top-k display, or None when view is unavailable."""
    if view is None:
        return None
    if sample_kind == "force_switch":
        mask0, mask1 = log_force_switch_slot_masks(view, side)
        if slot_suffix == "a":
            return mask0
        if slot0_pred is None:
            return mask1
        return _apply_joint_turn_constraints(mask1, a0=slot0_pred, force_switch=True, view=view, side=side)
    if slot_suffix == "a":
        return log_turn_slot_mask(view, side, "a")
    return _apply_joint_turn_constraints(
        log_turn_slot_mask(view, side, "b"),
        a0=slot0_pred,
        force_switch=(sample_kind == "force_switch"),
        view=view,
        side=side,
    )
