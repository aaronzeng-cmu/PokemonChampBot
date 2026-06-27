"""Tests for semantic action vocabulary and inference bridge."""

from __future__ import annotations

import random

from src.doubles.data.meta_move_imputation import impute_moves_to_four
from src.doubles.data.semantic_action import (
    ActionVocabulary,
    flat_action_to_semantic,
    log_offset_to_semantic_target,
)
from src.doubles.planning.meta_database import MetaDatabase


def test_meta_imputation_fills_to_four():
    meta = MetaDatabase(live_fetch=False)
    moves = impute_moves_to_four("Garchomp", ["earthquake"], meta, random.Random(42))
    assert len(moves) == 4
    assert "earthquake" in moves


def test_semantic_target_mapping():
    assert log_offset_to_semantic_target(0) == 0
    assert log_offset_to_semantic_target(1) == 1
    assert log_offset_to_semantic_target(-1) == 3


def test_flat_to_semantic_move():
    vocab = ActionVocabulary.create()
    # move slot 1, default target, no gimmick -> flat 7
    aid, tid, mid = flat_action_to_semantic(
        7,
        vocab=vocab,
        move_name="earthquake",
        target_offset=0,
    )
    assert vocab.token_for_id(aid) == "earthquake"
    assert tid == 0
    assert mid == 0
