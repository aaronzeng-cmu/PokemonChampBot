"""BCPolicy action-shape, masking, and safe-fallback behavior (fast, no checkpoints)."""

from __future__ import annotations

import numpy as np

from src.cv_bridge.bc_policy import BCPolicy
from src.core.model.transformer_bot import (
    SINGLES_ACTION_SIZE,
    VGCBehaviorCloner,
    VGCBehaviorClonerConfig,
)
from src.doubles.data.action_space_spec import ACTION_SIZE


def _policy(action_space: str) -> BCPolicy:
    cfg = VGCBehaviorClonerConfig(
        action_space=action_space,
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
    )
    pol = object.__new__(BCPolicy)
    pol.model = VGCBehaviorCloner(cfg).eval()
    pol.device = "cpu"
    pol.is_singles = action_space == "singles"
    return pol


def _obs() -> np.ndarray:
    return np.zeros((39, 25), dtype=np.int64)


def test_singles_returns_single_int():
    act = _policy("singles")(_obs(), None)
    assert isinstance(act, int)
    assert 0 <= act < SINGLES_ACTION_SIZE


def test_doubles_returns_tuple_of_two():
    act = _policy("doubles")(_obs(), None)
    assert isinstance(act, tuple) and len(act) == 2
    assert all(0 <= a < ACTION_SIZE for a in act)


def test_singles_respects_mask():
    mask = np.zeros(SINGLES_ACTION_SIZE, dtype=bool)
    mask[7] = True
    act = _policy("singles")(_obs(), {"slot_a": mask})
    assert act == 7


def test_doubles_respects_slot_masks_sequentially():
    mask_a = np.zeros(ACTION_SIZE, dtype=bool)
    mask_a[5] = True
    mask_b = np.zeros(ACTION_SIZE, dtype=bool)
    mask_b[0] = True  # pass is the only legal slot-B action
    ca0, ca1 = _policy("doubles")(_obs(), {"slot_a": mask_a, "slot_b": mask_b})
    assert ca0 == 5
    assert ca1 == 0


def test_empty_mask_falls_back_to_pass():
    empty = np.zeros(ACTION_SIZE, dtype=bool)
    ca0, ca1 = _policy("doubles")(_obs(), {"slot_a": empty, "slot_b": empty})
    assert (ca0, ca1) == (0, 0)


def test_bad_obs_triggers_safe_fallback_doubles():
    # Wrong-rank obs makes the forward pass raise -> safe (0, 0).
    bad = np.zeros((5,), dtype=np.int64)
    assert _policy("doubles")(bad, None) == (0, 0)


def test_bad_obs_triggers_safe_fallback_singles():
    bad = np.zeros((5,), dtype=np.int64)
    assert _policy("singles")(bad, None) == 0
