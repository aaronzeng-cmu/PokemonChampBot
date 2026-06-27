#!/usr/bin/env python3
"""Audit live trajectory stacking vs ReplayParser for first two turns."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.doubles.data.replay_parser import parse_log_file
from src.core.data.state_tokenizer import N_TOKENS, TRAJECTORY_DEPTH, stack_trajectory
from src.doubles.players.transformer_player import TransformerPlayer


def _frame_label(frame: np.ndarray) -> str:
    role0 = int(frame[1, 1]) if frame.shape[0] > 1 else 0
    role1 = int(frame[2, 1]) if frame.shape[0] > 2 else 0
    return f"hash({role0},{role1})"


def audit_replay(path: Path, side: str = "p1") -> None:
    samples = parse_log_file(path, skip_rating=True)
    by_turn = [s for s in samples if s.side == side and s.sample_kind == "turn"]
    by_turn.sort(key=lambda s: s.turn)
    print(f"Replay {path.name} ({side})")
    for s in by_turn[:3]:
        t = s.tokens.reshape(TRAJECTORY_DEPTH, N_TOKENS, -1)
        labels = [_frame_label(t[i]) for i in range(TRAJECTORY_DEPTH)]
        distinct = len({lbl for lbl in labels if not np.all(t[labels.index(lbl)] == 0)})
        print(f"  turn {s.turn}: frames={labels} distinct_nonempty={distinct}")


def simulate_live_stack() -> None:
    """Simulate TransformerPlayer history updates for 2 turns."""
    p = TransformerPlayer.__new__(TransformerPlayer)
    p.device = "cpu"
    p._trajectory_history = {}

    class _FakeBattle:
        battle_tag = "sim"
        turn = 1
        force_switch = [False, False]
        wait = False

    def _snap(val: int) -> np.ndarray:
        arr = np.zeros((N_TOKENS, 16), dtype=np.int64)
        arr[1, 1] = val
        arr[2, 1] = val + 100
        return arr

    # Monkeypatch encode_battle for simulation
    import src.doubles.players.transformer_player as tp

    calls: list[int] = []

    def _fake_encode(_battle):
        calls.append(_battle.turn)
        return _snap(_battle.turn)

    orig = tp.encode_battle
    tp.encode_battle = _fake_encode
    try:
        b = _FakeBattle()
        x1 = p._stacked_input(b)
        b.turn = 2
        x2 = p._stacked_input(b)
        b.turn = 2
        b.force_switch = [False, True]
        x2fs = p._stacked_input(b)
        b.turn = 3
        b.force_switch = [False, False]
        x3 = p._stacked_input(b)

        def _top_frame_val(x):
            return int(x[0, 2 * N_TOKENS, 1].item())

        print("Live simulation:")
        print(f"  turn1 current frame marker: {_top_frame_val(x1)} (expect 1)")
        print(f"  turn2 current frame marker: {_top_frame_val(x2)} (expect 2)")
        print(f"  turn2 mid-frame marker:     {int(x2[0, N_TOKENS, 1].item())} (expect 1)")
        print(f"  turn2 force_switch history len: {len(p._trajectory_history['sim'])} (expect 2, no append on force_switch)")
        print(f"  turn3 prior frame marker:   {int(x3[0, N_TOKENS, 1].item())} (expect 2)")
        print(f"  turn3 current frame marker: {_top_frame_val(x3)} (expect 3)")
    finally:
        tp.encode_battle = orig


def main() -> None:
    audit_replay(Path("data/raw_logs/gen9championsvgc2026regma-2629508270.log"))
    simulate_live_stack()


if __name__ == "__main__":
    main()
