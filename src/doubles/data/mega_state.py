"""Mega Evolution legality helpers for training logs and live inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

from poke_env.battle.double_battle import DoubleBattle

from src.core.data.mega_items import is_mega_stone_item
from src.core.data.perspective import MonPerspective

if TYPE_CHECKING:
    from src.core.data.log_tracker import BattleLogState
    from src.core.data.roster_profile import MatchRosters, MonRosterEntry


def roster_entry_mega_capable(entry: MonRosterEntry) -> bool:
    if entry.mega:
        return True
    if entry.item and is_mega_stone_item(entry.item):
        return True
    return False


def mon_mega_capable(
    mon: MonPerspective, *, roster_entry: MonRosterEntry | None = None
) -> bool:
    if roster_entry is not None and roster_entry_mega_capable(roster_entry):
        return True
    if mon.item_revealed and mon.item and is_mega_stone_item(mon.item):
        return True
    return False


def compute_can_mega(mon: MonPerspective, *, team_mega_used: bool) -> bool:
    """Whether this mon may legally Mega Evolve on the current turn."""
    if not mon.active:
        return False
    if mon.mega:
        return False
    if team_mega_used:
        return False
    return bool(mon.mega_capable)


def apply_own_mega_capable(view: BattleLogState, side: str, rosters: MatchRosters) -> None:
    """Tag our mons with mega_capable from look-ahead roster + revealed items."""
    from src.core.data.roster_profile import roster_species_key

    roster = rosters.for_side(side)
    for slot, mon in view.mons.items():
        if not slot.startswith(side):
            continue
        entry = roster.get(roster_species_key(mon.species)) or roster.get(mon.species)
        if mon_mega_capable(mon, roster_entry=entry):
            mon.mega_capable = True


def apply_can_mega_flags(view: BattleLogState, side: str) -> None:
    """Set per-mon can_mega for our side from capability + team mega usage."""
    team_used = bool(view.team_mega_used.get(side, False))
    for slot, mon in view.mons.items():
        if not slot.startswith(side):
            mon.can_mega = False
            continue
        mon.can_mega = compute_can_mega(mon, team_mega_used=team_used)


def live_can_mega_for_pos(battle: DoubleBattle, pos: int) -> bool:
    """poke-env legality: active slot may Mega Evolve this turn."""
    flags = getattr(battle, "can_mega_evolve", None)
    if not flags or pos >= len(flags):
        return False
    return bool(flags[pos])


def live_can_mega_for_singles(battle) -> bool:
    """poke-env legality: singles active may Mega Evolve this turn."""
    return bool(getattr(battle, "can_mega_evolve", False))
