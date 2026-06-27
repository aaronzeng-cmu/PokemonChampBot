"""Canonical move-slot ordering shared by parser, encoder, and live player."""

from __future__ import annotations

import numpy as np
import torch
from poke_env.battle.double_battle import DoubleBattle
from poke_env.data import to_id_str
from poke_env.environment.doubles_env import DoublesEnv

from src.core.data.move_utils import canonical_move_list
from src.doubles.data.action_space_spec import (
    ACTION_PASS,
    ACTION_SIZE,
    TARGET_DEFAULT,
    move_default_target_offset,
    target_offset_label,
)
from src.doubles.data.mega_state import live_can_mega_for_pos


def pokeenv_move_list(battle: DoubleBattle, pos: int) -> list[str]:
    """Move ids in poke-env action-slot order (team paste / request order)."""
    mon = battle.active_pokemon[pos]
    if mon is None:
        return []
    return [to_id_str(m.id) for m in list(mon.moves.values())[:4]]


def pokeenv_available_move_list(battle: DoubleBattle, pos: int) -> list[str]:
    """Currently selectable moves from the server request."""
    return [to_id_str(m.id) for m in battle.available_moves[pos]]


def decode_move_action_index(action_idx: int) -> tuple[int, int, bool, bool]:
    """Return (move_slot 1-4, target_offset, mega, tera) for a move action."""
    mega = False
    tera = False
    idx = action_idx
    if idx >= 87:
        tera = True
        idx -= 80
    elif idx >= 27:
        gimmick = (idx - 7) // 20
        mega = gimmick == 1
        if mega:
            idx -= 20
    move_part = idx - 7
    move_slot = move_part // 5 + 1
    target_offset = (move_part % 5) - 2
    return move_slot, target_offset, mega, tera


def is_mega_action(action_idx: int) -> bool:
    if action_idx < 7 or action_idx >= 87:
        return False
    return (action_idx - 7) // 20 == 1


def is_tera_action(action_idx: int) -> bool:
    return action_idx >= 87


def encode_move_action_index(
    move_ids: list[str],
    move_name: str,
    target_offset: int,
    *,
    mega: bool = False,
    terastallize: bool = False,
) -> int:
    """Encode a move into canonical-slot action space."""
    mid = to_id_str(move_name)
    ordered = canonical_move_list(move_ids)
    if mid and mid not in ordered:
        ordered = canonical_move_list(ordered + [mid])
    move_slot = 1
    for i, m in enumerate(ordered):
        if m == mid:
            move_slot = i + 1
            break
    base = 7 + (move_slot - 1) * 5 + (target_offset + 2)
    if mega:
        base += 20
    elif terastallize:
        base += 80
    return base


def remap_pokeenv_action_to_canonical(
    action_idx: int,
    battle: DoubleBattle,
    pos: int,
) -> int:
    """Inverse of remap_canonical_action_to_pokeenv for legality masks."""
    if action_idx <= 6:
        return action_idx

    mon = battle.active_pokemon[pos]
    if mon is None:
        return action_idx

    canonical = canonical_move_list([m.id for m in mon.moves.values()])
    pokeenv = pokeenv_move_list(battle, pos)
    if canonical == pokeenv:
        return action_idx

    move_slot, target_offset, mega, tera = decode_move_action_index(action_idx)
    if move_slot < 1 or move_slot > len(pokeenv):
        return action_idx
    move_id = pokeenv[move_slot - 1]
    try:
        ca_slot = canonical.index(move_id) + 1
    except ValueError:
        return action_idx

    base = 7 + (ca_slot - 1) * 5 + (target_offset + 2)
    if mega:
        base += 20
    elif tera:
        base += 80
    return base


def mask_mega_actions(mask: list[bool] | np.ndarray) -> None:
    """In-place: forbid every mega-modifier action index."""
    for idx in range(len(mask)):
        if is_mega_action(idx):
            mask[idx] = False


def pokeenv_action_mask_to_canonical(
    battle: DoubleBattle,
    pos: int,
    pokeenv_mask: list[bool],
) -> list[bool]:
    """Map poke-env per-slot legality mask to canonical alphabetical indices."""
    from src.doubles.data.action_space_spec import ACTION_SIZE

    out = [False] * ACTION_SIZE
    for pe_idx, legal in enumerate(pokeenv_mask):
        if not legal:
            continue
        ca_idx = remap_pokeenv_action_to_canonical(pe_idx, battle, pos)
        if 0 <= ca_idx < ACTION_SIZE:
            out[ca_idx] = True
    if not live_can_mega_for_pos(battle, pos):
        mask_mega_actions(out)
    return out


def _pokeenv_slot_profile(
    battle: DoubleBattle, pos: int
) -> tuple[bool, bool, bool]:
    """Return (has_switch, has_move, has_pass) for a poke-env per-slot mask."""
    pe = DoublesEnv.get_action_mask_individual(battle, pos)
    return bool(any(pe[1:7])), bool(any(pe[7:])), bool(pe[0])


def _valid_orders_slot_profile(
    battle: DoubleBattle, pos: int
) -> tuple[bool, bool, bool]:
    """Return (has_switch, has_move, has_pass) from battle.valid_orders."""
    kinds: set[str] = set()
    for order in battle.valid_orders[pos]:
        text = str(order).lower()
        if "pass" in text:
            kinds.add("pass")
        elif "switch" in text:
            kinds.add("switch")
        elif "move" in text:
            kinds.add("move")
    return "switch" in kinds, "move" in kinds, "pass" in kinds


def _force_switch_from_valid_orders(battle: DoubleBattle) -> tuple[bool, bool]:
    """Infer per-slot force-switch from Showdown-valid joint orders."""
    if battle.teampreview:
        return (False, False)
    s0, m0, p0 = _valid_orders_slot_profile(battle, 0)
    s1, m1, p1 = _valid_orders_slot_profile(battle, 1)
    if m0 and m1:
        return (False, False)
    if s0 and s1 and not m0 and not m1:
        return (True, True)
    if s0 and not s1 and p1 and not m0 and not m1:
        return (True, False)
    if s1 and not s0 and p0 and not m0 and not m1:
        return (False, True)
    return (False, False)


def effective_force_switch_flags(battle: DoubleBattle) -> tuple[bool, bool]:
    """
    Per-slot forced-switch flags.

    Uses ``battle.force_switch`` when poke-env has set it; otherwise infers from
    valid_orders and poke-env legal action shapes.
    """
    if any(battle.force_switch):
        return tuple(bool(x) for x in battle.force_switch)

    if battle.teampreview:
        return (False, False)

    if not hasattr(battle, "valid_orders"):
        return (False, False)

    vo = _force_switch_from_valid_orders(battle)
    if any(vo):
        return vo

    s0, m0, p0 = _pokeenv_slot_profile(battle, 0)
    s1, m1, p1 = _pokeenv_slot_profile(battle, 1)

    if m0 and m1:
        return (False, False)

    if s0 and s1 and not m0 and not m1:
        return (True, True)

    if s0 and not s1 and p1 and not m0 and not m1:
        return (True, False)
    if s1 and not s0 and p0 and not m0 and not m1:
        return (False, True)

    return (False, False)


def constrain_mask_to_valid_orders(
    battle: DoubleBattle, pos: int, mask: np.ndarray
) -> np.ndarray:
    """Intersect a canonical mask with what battle.valid_orders actually allows."""
    orders = list(getattr(battle, "valid_orders", [[]])[pos])
    if not orders:
        return mask
    out = mask.copy()
    has_switch, has_move, has_pass = _valid_orders_slot_profile(battle, pos)
    if not has_move:
        out[7:] = False
    if not has_switch:
        out[1:7] = False
    if not has_pass:
        out[0] = False
    if not out.any():
        if has_pass:
            out[0] = True
        elif has_switch:
            pe = pokeenv_force_switch_mask(battle, pos)
            for pe_idx, legal in enumerate(pe):
                if legal and pe_idx <= 6:
                    ca = remap_pokeenv_action_to_canonical(pe_idx, battle, pos)
                    if 0 <= ca < ACTION_SIZE:
                        out[ca] = True
        if not out.any():
            out[0] = True
    return out


def pokeenv_force_switch_mask(
    battle: DoubleBattle,
    pos: int,
) -> list[bool]:
    """
    Strict force-switch mask: switching slot = switches only; other slot = pass only.
    """
    mask = [False] * ACTION_SIZE
    flags = effective_force_switch_flags(battle)
    if not any(flags):
        return mask
    if flags[pos]:
        pe_mask = DoublesEnv.get_action_mask_individual(battle, pos)
        for idx, legal in enumerate(pe_mask):
            if legal and idx <= 6:
                mask[idx] = True
        if not any(mask):
            mask[0] = True
    else:
        mask[0] = True
    return mask


def canonical_force_switch_mask(
    battle: DoubleBattle,
    pos: int,
) -> np.ndarray:
    """Map poke-env force-switch mask to canonical action indices."""
    pe = pokeenv_force_switch_mask(battle, pos)
    out = np.zeros(ACTION_SIZE, dtype=bool)
    for pe_idx, legal in enumerate(pe):
        if not legal:
            continue
        ca_idx = remap_pokeenv_action_to_canonical(pe_idx, battle, pos)
        if 0 <= ca_idx < ACTION_SIZE:
            out[ca_idx] = True
    if not out.any():
        out[0] = True
    return out


def remap_canonical_action_to_pokeenv(
    action_idx: int,
    battle: DoubleBattle,
    pos: int,
) -> int:
    """
    Convert an action index from canonical (alphabetical) move slots to poke-env slots.
    Switch/pass indices (0-6) are unchanged.
    """
    if action_idx <= 6:
        return action_idx

    mon = battle.active_pokemon[pos]
    if mon is None:
        return action_idx

    canonical = canonical_move_list([m.id for m in mon.moves.values()])
    pokeenv = pokeenv_move_list(battle, pos)
    if canonical == pokeenv:
        return action_idx

    move_slot, target_offset, mega, tera = decode_move_action_index(action_idx)
    if move_slot < 1 or move_slot > len(canonical):
        return action_idx
    move_id = canonical[move_slot - 1]
    try:
        pe_slot = pokeenv.index(move_id) + 1
    except ValueError:
        return action_idx

    base = 7 + (pe_slot - 1) * 5 + (target_offset + 2)
    if mega:
        base += 20
    elif tera:
        base += 80
    return base


def _mask_mega_actions_torch(mask: torch.Tensor) -> None:
    mask_mega_actions(mask)


def apply_joint_slot1_mask_torch(
    mask1: torch.Tensor,
    *,
    a0_canonical: int,
    force_switch: bool = False,
) -> torch.Tensor:
    """
    Slot-1 mask after slot-0 pick.

    Always forbid the same bench switch on both slots. On normal turns, also forbid
    double-pass when both actives must act. When slot-0 picks mega, forbid mega on
    slot-1 (one mega per turn). Per-mon mega eligibility is handled upstream.
    """
    out = mask1.clone()
    if 1 <= a0_canonical <= 6:
        out[a0_canonical] = False
    if is_mega_action(a0_canonical):
        _mask_mega_actions_torch(out)
    if not force_switch and a0_canonical == 0:
        out[0] = False
    if not bool(out.any()):
        out[0] = True
    return out


def apply_joint_slot1_mask_numpy(
    mask1: np.ndarray,
    *,
    a0_canonical: int,
    force_switch: bool = False,
) -> np.ndarray:
    """Numpy twin of apply_joint_slot1_mask_torch for log eval."""
    out = mask1.copy()
    if 1 <= a0_canonical <= 6:
        out[a0_canonical] = False
    if is_mega_action(a0_canonical):
        mask_mega_actions(out)
    if not force_switch and a0_canonical == 0:
        out[0] = False
    if not out.any():
        out[0] = True
    return out


def compare_move_orders(
    *,
    label: str,
    canonical: list[str],
    pokeenv: list[str],
    available: list[str] | None = None,
) -> dict:
    mismatch = canonical != pokeenv
    slot_map = []
    for i, move_id in enumerate(canonical, start=1):
        pe_slot = pokeenv.index(move_id) + 1 if move_id in pokeenv else None
        slot_map.append(
            {
                "canonical_slot": i,
                "move": move_id,
                "pokeenv_slot": pe_slot,
                "same_slot": pe_slot == i,
            }
        )
    return {
        "label": label,
        "canonical_order": canonical,
        "pokeenv_order": pokeenv,
        "available_moves": available or [],
        "orders_match": not mismatch,
        "slot_map": slot_map,
        "would_misclick_without_remap": mismatch,
    }


def pick_masked_canonical_actions(
    battle: DoubleBattle,
    logits0: torch.Tensor,
    logits1: torch.Tensor,
) -> tuple[int, int, int, int]:
    """
    Masked argmax in canonical (alphabetical) action space, then remap to poke-env.
    Returns (canonical0, canonical1, pokeenv0, pokeenv1).
    """
    force = any(battle.force_switch)
    masks: list[torch.Tensor] = []
    for pos in (0, 1):
        if force:
            ca = canonical_force_switch_mask(battle, pos)
        else:
            pe = DoublesEnv.get_action_mask_individual(battle, pos)
            ca = np.array(pokeenv_action_mask_to_canonical(battle, pos, pe), dtype=bool)
        masks.append(torch.as_tensor(ca, dtype=torch.bool, device=logits0.device))

    def _pick(row: torch.Tensor, mask: torch.Tensor) -> int:
        masked = row.clone()
        masked[~mask] = -float("inf")
        if not bool(mask.any()):
            return 0
        return int(masked.argmax().item())

    a0 = _pick(logits0, masks[0])
    mask1 = apply_joint_slot1_mask_torch(
        masks[1],
        a0_canonical=a0,
        force_switch=force,
    )
    a1 = _pick(logits1, mask1)
    pe0 = remap_canonical_action_to_pokeenv(a0, battle, 0)
    pe1 = remap_canonical_action_to_pokeenv(a1, battle, 1)
    return a0, a1, pe0, pe1


def format_live_canonical_action(
    battle: DoubleBattle,
    pos: int,
    action_idx: int,
) -> str:
    """Human-readable canonical action for replay / inference logging."""
    if action_idx == 0:
        return "pass"
    if 1 <= action_idx <= 6:
        team = list(battle.team.values()) if hasattr(battle.team, "values") else list(battle.team)
        idx = action_idx - 1
        if idx < len(team) and team[idx] is not None:
            return f"switch -> {to_id_str(team[idx].species)}"
        return f"switch -> bench-{action_idx}"

    mon = battle.active_pokemon[pos]
    move_slot, target_offset, mega, tera = decode_move_action_index(action_idx)
    moves = canonical_move_list([m.id for m in mon.moves.values()]) if mon else []
    move_name = moves[move_slot - 1] if 0 < move_slot <= len(moves) else f"move{move_slot}"
    actor_name = to_id_str(mon.species) if mon else f"slot{pos}"
    target = target_offset_label(target_offset)
    flags: list[str] = []
    if mega:
        flags.append("mega")
    if tera:
        flags.append("tera")
    flag_text = f" ({', '.join(flags)})" if flags else ""
    return f"{actor_name}: {move_name} -> {target}{flag_text}"


def _slot_move_name(battle: DoubleBattle, pos: int, move_slot: int) -> str:
    """Canonical move id for a 1-based move slot, or '' when out of range."""
    mon = battle.active_pokemon[pos] if 0 <= pos < len(battle.active_pokemon) else None
    if mon is None:
        return ""
    moves = canonical_move_list([m.id for m in mon.moves.values()])
    if 1 <= move_slot <= len(moves):
        return moves[move_slot - 1]
    return ""


def classify_action_correction(
    battle: DoubleBattle,
    pos: int,
    raw_idx: int,
    picked_idx: int,
    *,
    force_switch: bool,
) -> str:
    """
    Classify a raw-argmax vs masked-pick correction for fallback metrics.

    Returns one of:
    - "none": no correction (raw == picked).
    - "structural": cosmetic mask correction that should NOT count as a fallback,
      i.e. (a) normalizing the target of a spread/self/field move to Default, or
      (b) forcing an idle slot to Pass during a force_switch phase.
    - "semantic": a real model hallucination (different move, illegal ally target,
      illegal switch/mega) that SHOULD count as a true fallback.
    """
    if raw_idx == picked_idx:
        return "none"

    # (b) Idle slot forced to Pass while the partner is the one switching.
    if force_switch and picked_idx == ACTION_PASS:
        return "structural"

    # (a) Same move re-targeted to Default because it is a spread/self/field move.
    if raw_idx >= 7 and picked_idx >= 7:
        r_slot, r_target, r_mega, r_tera = decode_move_action_index(raw_idx)
        p_slot, p_target, p_mega, p_tera = decode_move_action_index(picked_idx)
        same_move = (
            r_slot == p_slot and r_mega == p_mega and r_tera == p_tera
        )
        if same_move and r_target != p_target and p_target == TARGET_DEFAULT:
            move_name = _slot_move_name(battle, pos, p_slot)
            if move_name and move_default_target_offset(move_name) == TARGET_DEFAULT:
                return "structural"

    return "semantic"


def is_true_fallback(
    battle: DoubleBattle,
    pos: int,
    raw_idx: int,
    picked_idx: int,
    *,
    force_switch: bool,
) -> bool:
    """True only for semantic corrections (real hallucinations)."""
    return (
        classify_action_correction(
            battle, pos, raw_idx, picked_idx, force_switch=force_switch
        )
        == "semantic"
    )
