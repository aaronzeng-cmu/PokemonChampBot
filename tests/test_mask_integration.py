"""Mask hardening: legal masked indices map to valid orders (mock-free static checks)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from config.settings import MAX_COMBOS
from src.doubles.battle.action_space import combo_action_mask


def test_get_action_mask_length():
    """combo_action_mask must always return MAX_COMBOS entries."""
    combos = [(1, 7)]
    m = combo_action_mask(combos)
    assert len(m) == MAX_COMBOS


def test_masked_random_index_in_range():
    combos = [(0, 7), (1, 8), (2, 9)]
    mask = combo_action_mask(combos)
    legal = np.where(mask)[0]
    for _ in range(20):
        idx = int(np.random.choice(legal))
        assert 0 <= idx < len(combos)


if __name__ == "__main__":
    test_get_action_mask_length()
    test_masked_random_index_in_range()
    print("Mask tests passed.")
