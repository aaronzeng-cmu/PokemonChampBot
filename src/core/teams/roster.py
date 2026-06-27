"""Showdown team paste helpers (no env imports)."""

from __future__ import annotations

import random


def split_showdown_team(export: str) -> list[str]:
    return [b.strip() for b in export.strip().split("\n\n") if b.strip()]


def join_showdown_team(blocks: list[str]) -> str:
    return "\n\n".join(blocks) + "\n"


def shuffle_showdown_team(
    export: str,
    *,
    rng: random.Random | None = None,
) -> str:
    """Randomize paste block order so no mon is stuck in unbringable preview slots 1-2."""
    blocks = split_showdown_team(export)
    if len(blocks) < 2:
        return export
    r = rng or random
    r.shuffle(blocks)
    return join_showdown_team(blocks)
