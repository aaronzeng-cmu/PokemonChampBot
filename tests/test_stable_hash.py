"""Stable hashing must be identical across Python processes."""

from __future__ import annotations

import subprocess
import sys

from src.core.data.perspective import HASH_MOD, hash_token, stable_hash, stable_seed_int


def test_stable_hash_empty_returns_zero():
    assert stable_hash("") == 0
    assert hash_token("") == 0


def test_stable_hash_deterministic():
    a = stable_hash("garchomp", vocab_size=HASH_MOD)
    b = stable_hash("garchomp", vocab_size=HASH_MOD)
    assert a == b
    assert 0 <= a < HASH_MOD


def test_hash_token_normalizes_showdown_ids():
    assert hash_token("Heat Wave") == hash_token("heatwave")


def test_stable_hash_in_range():
    for name in ("whimsicott", "protect", "earthquake", "charizard"):
        h = hash_token(name)
        assert 0 <= h < HASH_MOD


def test_stable_seed_int_deterministic():
    assert stable_seed_int("replay", 1, "p1", "turn") == stable_seed_int(
        "replay", 1, "p1", "turn"
    )


def test_stable_hash_same_in_subprocess():
    code = (
        "from src.core.data.perspective import hash_token; "
        "print(hash_token('garchomp'))"
    )
    out = subprocess.check_output(
        [sys.executable, "-c", code],
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
        text=True,
    ).strip()
    assert int(out) == hash_token("garchomp")
