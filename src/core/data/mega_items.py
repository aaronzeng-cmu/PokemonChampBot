"""Mega Stone item detection (no battle-state imports)."""

from __future__ import annotations

from poke_env.data import to_id_str


def is_mega_stone_item(item: str) -> bool:
    """True when item id is a Mega Stone (ite / itex / itey suffix)."""
    iid = to_id_str(item)
    if not iid:
        return False
    return iid.endswith("ite") or iid.endswith("itex") or iid.endswith("itey")
