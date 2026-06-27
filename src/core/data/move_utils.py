"""Format-agnostic move list helpers shared by core and doubles."""

from __future__ import annotations

from poke_env.data import to_id_str


def canonical_move_list(moves: list[str]) -> list[str]:
    """Stable alphabetical move slots 1-4 (Showdown ids)."""
    seen: list[str] = []
    for move in moves:
        mid = to_id_str(move)
        if mid and mid not in seen:
            seen.append(mid)
    return sorted(seen)[:4]


def moves_for_action_encoding(known: list[str], move_name: str) -> list[str]:
    """Four move slots that always include the ground-truth move (for log labeling)."""
    move_id = to_id_str(move_name)
    ids: list[str] = []
    for move in known:
        mid = to_id_str(move)
        if mid and mid not in ids:
            ids.append(mid)
    if move_id and move_id not in ids:
        ids.append(move_id)
    if not move_id:
        return canonical_move_list(known)
    sorted_ids = sorted(ids)
    if move_id in sorted_ids[:4]:
        return sorted_ids[:4]
    others = [mid for mid in sorted_ids if mid != move_id]
    return sorted(others[:3] + [move_id])
