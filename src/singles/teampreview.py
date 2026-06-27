"""Team preview helpers for BSS Bring-3 (select exactly 3 Pokémon)."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poke_env.battle.battle import Battle


def random_teampreview_command(
    battle: Battle,
    *,
    rng: random.Random | None = None,
) -> str:
    """Return a random legal /team command (3 mons for BSS)."""
    r = rng or random
    members = list(range(1, len(battle.team) + 1))
    r.shuffle(members)
    return "/team " + "".join(str(c) for c in members[:3])


def parse_team_command(command: str) -> list[int]:
    """Parse Showdown /team digits into 1-based roster slot indices."""
    body = command.strip()
    if body.startswith("/team"):
        body = body[5:].strip()
    return [int(ch) for ch in body if ch.isdigit()]


def parse_preview_selection(command: str, *, expected: int = 3) -> list[int]:
    """Validate and return exactly ``expected`` bring slots from a preview command."""
    slots = parse_team_command(command)
    if len(slots) != expected:
        raise ValueError(f"expected {expected} preview slots, got {len(slots)}: {command!r}")
    return slots


def slots_to_species(battle: Battle, slots: list[int]) -> list[str]:
    team_list = list(battle.team.values())
    return [team_list[i - 1].species for i in slots]


def battle_team_summary(battle: Battle) -> dict:
    """Species brought (3) and active lead after preview."""
    team_list = list(battle.team.values())
    brought = [
        p.species
        for p in team_list
        if getattr(p, "selected_in_teampreview", False)
    ]
    if not brought:
        brought = [p.species for p in team_list[:3]]
    active = battle.active_pokemon
    lead = active.species if active is not None else None
    return {"lead": lead, "brought": brought}


def opponent_team_summary(battle: Battle) -> dict:
    """Opponent's brought-3 lineup after preview."""
    team_list = list(battle.opponent_team.values())
    brought = [
        p.species
        for p in team_list
        if getattr(p, "selected_in_teampreview", False)
    ]
    if not brought:
        brought = [p.species for p in team_list[:3]]
    active = battle.opponent_active_pokemon
    lead = active.species if active is not None else None
    return {"lead": lead, "brought": brought}


def opponent_full_team_summary(battle: Battle) -> dict:
    """All six species on the opponent's roster."""
    return {"full_team": [p.species for p in battle.opponent_team.values()]}
