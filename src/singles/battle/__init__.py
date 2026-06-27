"""Live singles battle helpers (canonical inference, legality)."""

from src.singles.battle.canonical_inference import (
    canonical_index_to_battle_order,
    canonical_moves_for_active,
    decode_canonical_submission,
    pick_masked_canonical_index,
    submission_debug,
)

__all__ = [
    "canonical_index_to_battle_order",
    "canonical_moves_for_active",
    "decode_canonical_submission",
    "pick_masked_canonical_index",
    "submission_debug",
]
