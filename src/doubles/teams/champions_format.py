"""Normalize Showdown exports for Pokémon Champions (Reg M-A) stat points."""

from __future__ import annotations

import re

# Champions: up to 32 per stat, 66 total (sheet labels these "EVs")
CHAMPIONS_MAX_PER_STAT = 32
CHAMPIONS_MAX_TOTAL = 66
# Classic cartridge EV → Stat Point (see RotomPicks / Champions import tools)
EV_TO_STAT_POINT = 8

EV_LINE_RE = re.compile(r"^EVs:\s*(.+)$", re.MULTILINE)
EV_PART_RE = re.compile(r"(\d+)\s+(HP|Atk|Def|SpA|SpD|Spe)")


def _parse_ev_parts(ev_body: str) -> list[tuple[int, str]]:
    return [(int(amount), stat) for amount, stat in EV_PART_RE.findall(ev_body)]


def _format_ev_line(parts: list[tuple[int, str]]) -> str:
    order = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]
    by_stat = {stat: amt for amt, stat in parts}
    segments = [f"{by_stat[stat]} {stat}" for stat in order if stat in by_stat]
    return "EVs: " + " / ".join(segments)


def _needs_conversion(parts: list[tuple[int, str]]) -> bool:
    if not parts:
        return False
    amounts = [a for a, _ in parts]
    return max(amounts) > CHAMPIONS_MAX_PER_STAT or sum(amounts) > CHAMPIONS_MAX_TOTAL


def _legacy_evs_to_stat_points(parts: list[tuple[int, str]]) -> list[tuple[int, str]]:
    converted = [(max(0, min(CHAMPIONS_MAX_PER_STAT, round(amt / EV_TO_STAT_POINT))), stat) for amt, stat in parts]
    total = sum(a for a, _ in converted)
    if total <= CHAMPIONS_MAX_TOTAL:
        return converted
    # Scale down proportionally if still over 66
    scale = CHAMPIONS_MAX_TOTAL / total
    scaled = [(max(0, min(CHAMPIONS_MAX_PER_STAT, int(a * scale))), stat) for a, stat in converted]
    while sum(a for a, _ in scaled) > CHAMPIONS_MAX_TOTAL:
        idx = max(range(len(scaled)), key=lambda i: scaled[i][0])
        amt, stat = scaled[idx]
        if amt <= 0:
            break
        scaled[idx] = (amt - 1, stat)
    return scaled


def _convert_mon_block(block: str) -> str:
    match = EV_LINE_RE.search(block)
    if not match:
        return block
    parts = _parse_ev_parts(match.group(1))
    if not parts or not _needs_conversion(parts):
        return block
    new_line = _format_ev_line(_legacy_evs_to_stat_points(parts))
    return block[: match.start()] + new_line + block[match.end() :]


def convert_export_to_champions(export: str) -> str:
    """Convert classic 252/252-style EV spreads to Champions stat points when needed."""
    blocks = [b.strip() for b in export.strip().split("\n\n") if b.strip()]
    if not blocks:
        return export
    converted = [_convert_mon_block(block) for block in blocks]
    return "\n\n".join(converted) + "\n"
