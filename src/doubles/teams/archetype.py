"""Heuristic meta archetype tags for gauntlet opponent teams."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TRICK_ROOM = re.compile(r"\bTrick Room\b", re.I)
_TAILWIND = re.compile(r"\bTailwind\b", re.I)
_SUN = re.compile(r"\b(Torkoal|Ninetales|Groudon|Drought)\b", re.I)
_RAIN = re.compile(r"\b(Pelipper|Politoed|Kyogre|Rain Dance|Raindance)\b", re.I)
_HYPER = re.compile(r"\b(Flutter Mane|Iron Bundle|Chi-Yu|Chien-Pao|Roaring Moon)\b", re.I)
_BALANCE = re.compile(r"\b(Incineroar|Rillaboom|Amoonguss|Landorus)\b", re.I)


@dataclass(frozen=True)
class TeamArchetype:
    primary: str
    tags: tuple[str, ...]


def classify_team_export(text: str) -> TeamArchetype:
    """Tag a Showdown export with coarse Reg M-A archetype labels."""
    tags: list[str] = []
    if _TRICK_ROOM.search(text):
        tags.append("trick_room")
    if _TAILWIND.search(text):
        tags.append("tailwind")
    if _SUN.search(text):
        tags.append("sun")
    if _RAIN.search(text):
        tags.append("rain")
    if _HYPER.search(text) and _TAILWIND.search(text):
        tags.append("hyper_offense")
    if _BALANCE.search(text) and len(tags) <= 1:
        tags.append("balance")

    if not tags:
        tags.append("balance")
    primary = tags[0]
    if "hyper_offense" in tags:
        primary = "hyper_offense"
    elif "trick_room" in tags and "trick_room" != primary:
        primary = "trick_room"
    return TeamArchetype(primary=primary, tags=tuple(dict.fromkeys(tags)))


def classify_team_file(path: Path) -> TeamArchetype:
    return classify_team_export(path.read_text(encoding="utf-8"))
