"""Legal action masks for BSS / Champions Singles (canonical 0-21 indices)."""

from __future__ import annotations

import numpy as np
from poke_env.battle.battle import Battle

from src.core.model.transformer_bot import SINGLES_ACTION_SIZE
from src.singles.battle.live_legality import build_singles_action_mask


def singles_action_mask(battle: Battle) -> np.ndarray:
    """Boolean mask over canonical singles action indices (Bring-3 aware)."""
    mask_list = build_singles_action_mask(battle, size=SINGLES_ACTION_SIZE)
    return np.asarray(mask_list, dtype=bool)


def pick_masked_argmax(logits: np.ndarray, mask: np.ndarray) -> int:
    """Argmax over legal singles actions; falls back to first legal index."""
    from src.singles.battle.canonical_inference import pick_masked_canonical_index

    return pick_masked_canonical_index(logits, mask)
