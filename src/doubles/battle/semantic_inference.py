"""Semantic head outputs and poke-env action composition."""

from __future__ import annotations

import torch
from poke_env.battle.double_battle import DoubleBattle
from poke_env.data import to_id_str
from poke_env.environment.doubles_env import DoublesEnv

from src.doubles.battle.move_order import (
    is_mega_action,
    is_tera_action,
    pokeenv_available_move_list,
)
from src.doubles.data.action_space_spec import ACTION_SIZE
from src.doubles.data.action_space_spec import ACTION_PASS
from src.doubles.data.semantic_action import (
    NUM_SEMANTIC_MODIFIERS,
    NUM_SEMANTIC_TARGETS,
    SEMANTIC_MODIFIER_GIMMICK,
    SEMANTIC_MODIFIER_NORMAL,
    SEMANTIC_TARGET_ALLY,
    ActionVocabulary,
    semantic_target_to_log_offsets,
)


def _pokeenv_move_slot_for_id(battle: DoubleBattle, pos: int, move_id: str) -> int | None:
    move_id = to_id_str(move_id)
    for i, mid in enumerate(pokeenv_available_move_list(battle, pos)):
        if mid == move_id:
            return i + 1
    return None


def compose_pokeenv_action(
    battle: DoubleBattle,
    pos: int,
    *,
    action_id: int,
    target_id: int,
    modifier_id: int,
    vocab: ActionVocabulary,
) -> int | None:
    """Map semantic head predictions to a poke-env action index, or None if hallucinated."""
    if action_id == ACTION_PASS:
        return 0
    if vocab.is_switch_action(action_id):
        return action_id

    if not vocab.is_move_action(action_id):
        return None

    move_id = vocab.token_for_id(action_id)
    pe_slot = _pokeenv_move_slot_for_id(battle, pos, move_id)
    if pe_slot is None:
        return None

    offsets = semantic_target_to_log_offsets(target_id)
    candidates: list[int] = []
    for offset in offsets:
        base = 7 + (pe_slot - 1) * 5 + (offset + 2)
        if modifier_id == SEMANTIC_MODIFIER_GIMMICK:
            for bonus in (20, 80):
                candidates.append(base + bonus)
        candidates.append(base)

    pe_mask = DoublesEnv.get_action_mask_individual(battle, pos)
    for idx in candidates:
        if 0 <= idx < ACTION_SIZE and pe_mask[idx]:
            return idx
    return None


def pick_semantic_action(
    battle: DoubleBattle,
    pos: int,
    *,
    action_logits: torch.Tensor,
    target_logits: torch.Tensor,
    modifier_logits: torch.Tensor,
    vocab: ActionVocabulary,
    forbidden_action_ids: set[int] | None = None,
) -> tuple[int, int, int, int | None]:
    """
    Argmax semantic heads with hallucination masking on head_action.
    Returns (action_id, target_id, modifier_id, pokeenv_idx).
    """
    forbidden = forbidden_action_ids or set()
    logits = action_logits.clone()
    for bad in forbidden:
        if 0 <= bad < logits.shape[0]:
            logits[bad] = -float("inf")

    for _ in range(logits.shape[0]):
        action_id = int(logits.argmax().item())
        target_id = int(target_logits.argmax().item())
        modifier_id = int(modifier_logits.argmax().item())

        if action_id in forbidden:
            logits[action_id] = -float("inf")
            continue

        if action_id == ACTION_PASS or vocab.is_switch_action(action_id):
            pe = compose_pokeenv_action(
                battle,
                pos,
                action_id=action_id,
                target_id=target_id,
                modifier_id=modifier_id,
                vocab=vocab,
            )
            if pe is not None:
                return action_id, target_id, modifier_id, pe
            logits[action_id] = -float("inf")
            continue

        move_id = vocab.token_for_id(action_id)
        if _pokeenv_move_slot_for_id(battle, pos, move_id) is None:
            logits[action_id] = -float("inf")
            continue

        pe = compose_pokeenv_action(
            battle,
            pos,
            action_id=action_id,
            target_id=target_id,
            modifier_id=modifier_id,
            vocab=vocab,
        )
        if pe is not None:
            return action_id, target_id, modifier_id, pe
        logits[action_id] = -float("inf")

    return ACTION_PASS, 0, SEMANTIC_MODIFIER_NORMAL, 0


def apply_joint_semantic_slot1(
    *,
    slot0_action_id: int,
    slot0_pe: int,
    forbidden: set[int],
    force_switch: bool,
) -> set[int]:
    """Semantic joint constraints for slot B action head."""
    out = set(forbidden)
    if 1 <= slot0_action_id <= 6:
        out.add(slot0_action_id)
    if is_mega_action(slot0_pe):
        # Forbid gimmick modifier on slot1 when slot0 mega'd — handled at compose time.
        pass
    if not force_switch and slot0_action_id == ACTION_PASS:
        out.add(ACTION_PASS)
    return out
