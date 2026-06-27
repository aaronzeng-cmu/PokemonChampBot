"""Joint slot-1 masking for doubles (switch dedup + double-pass rule)."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.doubles.battle.move_order import apply_joint_slot1_mask_numpy, apply_joint_slot1_mask_torch


def test_double_force_switch_blocks_same_bench_pick():
    mask = np.zeros(107, dtype=bool)
    mask[1] = True  # kingambit
    mask[2] = True  # basculegion
    out = apply_joint_slot1_mask_numpy(mask, a0_canonical=1, force_switch=True)
    assert out[1] is np.False_
    assert out[2] is np.True_


def test_normal_turn_blocks_double_pass():
    mask = np.zeros(107, dtype=bool)
    mask[0] = True
    mask[7] = True
    out = apply_joint_slot1_mask_numpy(mask, a0_canonical=0, force_switch=False)
    assert out[0] is np.False_
    assert out[7] is np.True_


def test_force_switch_allows_slot1_pass_when_slot0_switches():
    mask = np.zeros(107, dtype=bool)
    mask[0] = True
    out = apply_joint_slot1_mask_numpy(mask, a0_canonical=1, force_switch=True)
    assert out[0] is np.True_


def test_torch_matches_numpy():
    mask_np = np.zeros(107, dtype=bool)
    mask_np[1:4] = True
    mask_t = torch.as_tensor(mask_np)
    out_np = apply_joint_slot1_mask_numpy(mask_np, a0_canonical=2, force_switch=True)
    out_t = apply_joint_slot1_mask_torch(mask_t, a0_canonical=2, force_switch=True)
    assert torch.equal(out_t, torch.as_tensor(out_np))


def test_slot1_mega_stays_legal_when_slot0_cannot_mega():
    """Partner without a mega stone must not block slot-1 mega actions."""
    mask = np.zeros(107, dtype=bool)
    mask[27] = True  # mega gimmick index
    mask[7] = True  # non-mega move
    out = apply_joint_slot1_mask_numpy(mask, a0_canonical=7, force_switch=False)
    assert out[27] is np.True_


def test_slot0_mega_pick_blocks_slot1_mega():
    mask = np.zeros(107, dtype=bool)
    mask[27] = True
    out = apply_joint_slot1_mask_numpy(mask, a0_canonical=27, force_switch=False)
    assert out[27] is np.False_


if __name__ == "__main__":
    test_double_force_switch_blocks_same_bench_pick()
    test_normal_turn_blocks_double_pass()
    test_force_switch_allows_slot1_pass_when_slot0_switches()
    test_torch_matches_numpy()
    print("joint_slot_mask tests passed.")
