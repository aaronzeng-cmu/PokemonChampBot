"""Per-side trajectory stacking must match live play (one perspective per turn)."""

from __future__ import annotations

import numpy as np

from src.doubles.data.replay_parser import parse_log_file
from src.core.data.state_tokenizer import N_TOKENS, TRAJECTORY_DEPTH, trajectory_frame_fingerprints


def test_per_side_trajectory_not_interleaved():
    path = __import__("pathlib").Path("data/raw_logs/gen9championsvgc2026regma-2629508270.log")
    samples = parse_log_file(path, skip_rating=True)
    p1 = [s for s in samples if s.side == "p1" and s.sample_kind == "turn"]
    p1.sort(key=lambda s: s.turn)
    assert len(p1) >= 3

    # Turn 3 p1 stack should be [t1, t2, t3] same-side frames, not mixed with p2.
    s3 = p1[2]
    frames = s3.tokens.reshape(TRAJECTORY_DEPTH, N_TOKENS, -1)
    f1 = frames[0]
    f2 = frames[1]
    assert f1.any() and f2.any()
    # Oldest frame (turn 1) should differ from turn 3 current frame.
    assert not np.array_equal(f1, frames[2])
    labels = trajectory_frame_fingerprints(s3.tokens)
    assert labels[0] != "t-2:empty"
    assert labels[1] != "t-1:empty"
    assert "empty" not in labels[2]


def test_p1_p2_histories_independent_lengths():
    path = __import__("pathlib").Path("data/raw_logs/gen9championsvgc2026regma-2629508270.log")
    samples = parse_log_file(path, skip_rating=True)
    p1_turns = {s.turn for s in samples if s.side == "p1" and s.sample_kind == "turn"}
    p2_turns = {s.turn for s in samples if s.side == "p2" and s.sample_kind == "turn"}
    assert p1_turns == p2_turns
