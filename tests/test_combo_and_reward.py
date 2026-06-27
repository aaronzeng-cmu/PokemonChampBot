"""Unit tests for action masks and reward shaping (no Showdown required)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from config.settings import MAX_COMBOS
from archive.rl.env.observation import N_FEATURES, observation_size
from archive.rl.env.rewards import BattleSnapshot, calc_step_reward
from src.doubles.battle.action_space import combo_action_mask


def test_combo_action_mask_shape():
    combos = [(7, 12), (8, 15)]
    mask = combo_action_mask(combos)
    assert mask.shape == (MAX_COMBOS,)
    assert mask.sum() == 2
    assert mask[0] and mask[1]
    assert not mask[2]


def test_observation_feature_count():
    assert observation_size() == N_FEATURES


def test_reward_hp_delta():
    # Snapshots aligned with fake team HP totals (2 mon per side)
    last = BattleSnapshot(
        our_hp_sum=2.0,
        opp_hp_sum=2.0,
        our_fainted=0,
        opp_fainted=0,
        our_mega_active=0,
    )

    class FakeMon:
        def __init__(self, hp, fainted=False):
            self.current_hp_fraction = hp
            self.fainted = fainted

    class FakeBattle:
        won = False
        lost = False

        def __init__(self):
            self.team = {"a": FakeMon(1.0), "b": FakeMon(1.0)}
            self.opponent_team = {"c": FakeMon(1.0), "d": FakeMon(0.5)}
            self.active_pokemon = [None, None]

    reward, snap = calc_step_reward(last, FakeBattle())
    assert reward > 0  # opponent lost 0.5 hp, ours unchanged
    assert snap.opp_hp_sum == 1.5


if __name__ == "__main__":
    test_combo_action_mask_shape()
    test_observation_feature_count()
    test_reward_hp_delta()
    print("All tests passed.")
