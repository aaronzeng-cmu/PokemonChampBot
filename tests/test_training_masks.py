"""Training-time legal action masks include ground-truth labels."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.doubles.data.action_codec import ACTION_UNKNOWN
from src.doubles.data.replay_parser import parse_log_file, build_dataset
from src.doubles.planning.meta_database import MetaDatabase


@pytest.fixture(scope="module")
def samples():
    log_dir = Path("data/raw_logs")
    paths = sorted(log_dir.glob("*.log"))
    if not paths:
        pytest.skip("no raw logs")
    meta_db = MetaDatabase(live_fetch=False)
    out = []
    for path in paths[:5]:
        out.extend(parse_log_file(path, skip_rating=True, meta_db=meta_db))
    if not out:
        pytest.skip("no parsed samples")
    return out


def test_samples_carry_masks(samples):
    for s in samples[:50]:
        assert s.mask_slot0 is not None
        assert s.mask_slot1 is not None
        assert s.mask_slot0.shape == (107,)
        assert s.mask_slot1.shape == (107,)


def test_ground_truth_in_mask(samples):
    for s in samples:
        if s.action_slot0 != ACTION_UNKNOWN:
            assert s.mask_slot0[s.action_slot0], (
                f"slot0 gt {s.action_slot0} not legal turn={s.turn} kind={s.sample_kind}"
            )
        if s.action_slot1 != ACTION_UNKNOWN:
            assert s.mask_slot1[s.action_slot1], (
                f"slot1 gt {s.action_slot1} not legal turn={s.turn} kind={s.sample_kind}"
            )


def test_build_dataset_includes_masks(samples):
    ds = build_dataset(samples[:10])
    assert "mask_slot0" in ds
    assert "mask_slot1" in ds
    assert ds["mask_slot0"].shape == (10, 107)
    assert ds["mask_slot1"].dtype == bool
