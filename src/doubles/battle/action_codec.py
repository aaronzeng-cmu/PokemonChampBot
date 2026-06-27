"""Encode/decode poke-env doubles per-slot action indices (gen 9: 0-106)."""

from __future__ import annotations

from poke_env.battle.double_battle import DoubleBattle
from poke_env.data import to_id_str
from poke_env.environment.doubles_env import DoublesEnv

from src.doubles.data.action_space_spec import ACTION_SIZE, encode_combo

ACTION_SPACE_SIZE = ACTION_SIZE  # DoublesEnv.get_action_space_size(9)


def mask_legal_actions(battle: DoubleBattle, pos: int) -> list[int]:
    return [i for i, v in enumerate(DoublesEnv.get_action_mask_individual(battle, pos)) if v]


def legal_combo_indices(battle: DoubleBattle) -> list[int]:
    """Joint legal combo indices from per-slot individual masks."""
    legal0 = mask_legal_actions(battle, 0)
    legal1 = mask_legal_actions(battle, 1)
    return [encode_combo(a0, a1) for a0 in legal0 for a1 in legal1]


def pick_masked_combo(logits, battle: DoubleBattle) -> tuple[int, int]:
    """Return highest-probability legal (slot0, slot1) from combo logits."""
    import torch

    from src.doubles.data.action_space_spec import decode_combo

    legal = legal_combo_indices(battle)
    if not legal:
        return 0, 0
    masked = logits.clone()
    legal_mask = torch.zeros_like(masked, dtype=torch.bool)
    legal_mask[legal] = True
    masked[~legal_mask] = float("-inf")
    combo = int(masked.argmax().item())
    return decode_combo(combo)


def pick_masked_action(logits, battle: DoubleBattle, pos: int) -> int:
    """Return highest-probability legal action index."""
    import torch

    legal = mask_legal_actions(battle, pos)
    if not legal:
        return 0
    masked = logits.clone()
    legal_mask = torch.zeros_like(masked, dtype=torch.bool)
    legal_mask[legal] = True
    masked[~legal_mask] = float("-inf")
    return int(masked.argmax().item())


def encode_switch_action(team_index: int) -> int:
    """1-based team slot -> poke-env switch index (1-6)."""
    return team_index


def encode_move_action(
    move_slot: int,
    target_offset: int,
    *,
    mega: bool = False,
    terastallize: bool = False,
) -> int:
    """
    Encode move choice into poke-env individual action index.
    move_slot: 1-4
    target_offset: -2..2 (doubles targeting)
    """
    base = 7 + (move_slot - 1) * 5 + (target_offset + 2)
    if mega:
        base += 20
    elif terastallize:
        base += 80
    return base


def order_to_slot_indices(battle: DoubleBattle, order, pos: int) -> int:
    """Convert a SingleBattleOrder back to poke-env action index."""
    return int(DoublesEnv.order_to_action(order, battle, fake=True, strict=False)[pos])
