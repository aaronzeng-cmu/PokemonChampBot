"""Offline semantic-head evaluation helpers (log view states)."""

from __future__ import annotations

import numpy as np
import torch
from poke_env.data import to_id_str

from src.core.data.move_utils import canonical_move_list
from src.doubles.battle.move_order import encode_move_action_index
from src.doubles.data.action_space_spec import (
    ACTION_PASS,
    ACTION_UNKNOWN,
    move_default_target_offset,
)
from src.doubles.data.log_action_mask import slot_mask_for_eval
from src.core.data.log_tracker import BattleLogState
from src.core.data.roster_profile import roster_species_key
from src.doubles.data.semantic_action import (
    MOVE_ACTION_START,
    SEMANTIC_MODIFIER_GIMMICK,
    ActionVocabulary,
    semantic_target_to_log_offsets,
)


def _active_mon(view: BattleLogState, side: str, suffix: str):
    return view.mons.get(f"{side}{suffix}")


def legal_semantic_action_ids(
    view: BattleLogState,
    side: str,
    *,
    slot_suffix: str,
    sample_kind: str,
    vocab: ActionVocabulary,
    slot0_flat_pred: int | None = None,
) -> set[int]:
    """Allowed head_action ids for a log eval slot."""
    flat_mask = slot_mask_for_eval(
        view,
        side=side,
        sample_kind=sample_kind,
        slot_suffix=slot_suffix,
        slot0_pred=slot0_flat_pred,
    )
    legal: set[int] = {ACTION_PASS}
    for flat_idx in range(1, len(flat_mask)):
        if not flat_mask[flat_idx]:
            continue
        if flat_idx <= 6:
            legal.add(flat_idx)
            continue
        composed = compose_log_action_from_flat(view, side, slot_suffix, flat_idx)
        if composed is None:
            continue
        action_id, _, _ = composed
        if action_id >= MOVE_ACTION_START:
            legal.add(action_id)
    return legal


def compose_log_action(
    view: BattleLogState,
    side: str,
    slot_suffix: str,
    *,
    action_id: int,
    target_id: int,
    modifier_id: int,
    vocab: ActionVocabulary,
    sample_kind: str = "turn",
    slot0_flat_pred: int | None = None,
) -> int | None:
    """Map semantic head picks to legacy flat 0-106 index for reporting."""
    if action_id == ACTION_PASS:
        return ACTION_PASS
    if vocab.is_switch_action(action_id):
        return action_id

    if not vocab.is_move_action(action_id):
        return None

    mon = _active_mon(view, side, slot_suffix)
    if mon is None:
        return None

    move_id = vocab.token_for_id(action_id)
    moves = canonical_move_list(list(mon.moves))
    if move_id not in moves:
        return None

    offsets = semantic_target_to_log_offsets(target_id)
    mega = modifier_id == SEMANTIC_MODIFIER_GIMMICK
    for offset in offsets:
        if offset == move_default_target_offset(move_id):
            return encode_move_action_index(moves, move_id, offset, mega=mega)
        flat = encode_move_action_index(moves, move_id, offset, mega=mega)
        mask = slot_mask_for_eval(
            view,
            side=side,
            sample_kind=sample_kind,
            slot_suffix=slot_suffix,
            slot0_pred=slot0_flat_pred,
        )
        if mask is not None and 0 <= flat < len(mask) and mask[flat]:
            return flat

    offset = offsets[0] if offsets else move_default_target_offset(move_id)
    return encode_move_action_index(moves, move_id, offset, mega=mega)


def compose_log_action_from_flat(
    view: BattleLogState,
    side: str,
    slot_suffix: str,
    flat_idx: int,
) -> tuple[int, int, int] | None:
    """Best-effort semantic triple from a flat label (for legal-id mapping)."""
    from src.doubles.battle.move_order import decode_move_action_index
    from src.doubles.data.semantic_action import flat_action_to_semantic

    if flat_idx == ACTION_UNKNOWN:
        return None
    mon = _active_mon(view, side, slot_suffix)
    moves = list(mon.moves) if mon and mon.moves else []
    if flat_idx <= 6:
        return flat_idx, ACTION_UNKNOWN, ACTION_UNKNOWN
    move_slot, target_offset, mega, tera = decode_move_action_index(flat_idx)
    move_name = moves[move_slot - 1] if 0 < move_slot <= len(moves) else f"move{move_slot}"
    vocab = ActionVocabulary.create()
    return flat_action_to_semantic(
        flat_idx,
        vocab=vocab,
        move_name=move_name,
        target_offset=target_offset,
        mega=mega,
        terastallize=tera,
    )


def pick_masked_semantic_actions(
    act0: torch.Tensor,
    tgt0: torch.Tensor,
    mod0: torch.Tensor,
    act1: torch.Tensor,
    tgt1: torch.Tensor,
    mod1: torch.Tensor,
    *,
    view: BattleLogState,
    side: str,
    sample_kind: str,
    vocab: ActionVocabulary,
) -> tuple[int, int, int, int, int, int, int, int]:
    """
    Masked argmax on action heads; target/modifier from raw argmax.
    Returns (a0, t0, m0, flat0, a1, t1, m1, flat1).
    """
    legal0 = legal_semantic_action_ids(view, side, slot_suffix="a", sample_kind=sample_kind, vocab=vocab)

    def _pick_action(row: torch.Tensor, legal: set[int]) -> int:
        logits = row.clone()
        for i in range(logits.shape[0]):
            if i not in legal:
                logits[i] = -float("inf")
        return int(logits.argmax().item())

    a0 = _pick_action(act0, legal0)
    t0 = int(tgt0.argmax().item())
    m0 = int(mod0.argmax().item())
    flat0 = compose_log_action(
        view, side, "a", action_id=a0, target_id=t0, modifier_id=m0, vocab=vocab, sample_kind=sample_kind,
    )
    flat0 = flat0 if flat0 is not None else ACTION_PASS

    legal1 = legal_semantic_action_ids(
        view, side, slot_suffix="b", sample_kind=sample_kind, vocab=vocab, slot0_flat_pred=flat0,
    )
    a1 = _pick_action(act1, legal1)
    t1 = int(tgt1.argmax().item())
    m1 = int(mod1.argmax().item())

    flat1 = compose_log_action(
        view, side, "b", action_id=a1, target_id=t1, modifier_id=m1, vocab=vocab,
        sample_kind=sample_kind, slot0_flat_pred=flat0,
    )
    flat1 = flat1 if flat1 is not None else ACTION_PASS
    return a0, t0, m0, flat0, a1, t1, m1, flat1
