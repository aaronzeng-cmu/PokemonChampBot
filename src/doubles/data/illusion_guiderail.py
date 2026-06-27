"""Zoroark Illusion guiderails for log parsing and move encoding."""

from __future__ import annotations

from poke_env.data import to_id_str

from src.core.data.log_tracker import BattleLogState, _species_name, _slot_key
from src.core.data.perspective import MonPerspective
from src.core.data.roster_profile import MatchRosters, roster_species_key

# Moves that identify Hisuian Zoroark when used under an Illusion disguise.
_ZOROARK_SIGNATURE_MOVES = frozenset(
    {
        "bittermalice",
        "shadowball",
        "hypervoice",
        "uturn",
        "focusblast",
        "nastyplot",
    }
)


def is_illusion_end_line(parts: list[str]) -> bool:
    return len(parts) >= 4 and to_id_str(parts[3]) == "illusion"


def apply_illusion_replace(
    mon: MonPerspective,
    *,
    true_species: str,
) -> None:
    """Reveal true identity after |replace| (Illusion broken)."""
    true_id = to_id_str(_species_name(true_species))
    if mon.species and mon.species != true_id:
        mon.illusion_disguise = mon.species
    mon.species = true_id
    mon.illusion_broken = True
    mon.seen = True


def reconcile_illusion_roster(
    view: BattleLogState,
    side: str,
    rosters: MatchRosters,
) -> None:
    """
    After Illusion breaks, re-bind our slot to the true species roster entry.
    Strips disguise-species imputed moves so encoding uses Zoroark's set.
    """
    from src.core.data.move_utils import canonical_move_list

    roster = rosters.for_side(side)
    for slot, mon in view.mons.items():
        if not slot.startswith(side) or not mon.illusion_broken:
            continue
        entry = roster.get(mon.species)
        if entry is None:
            continue
        revealed = canonical_move_list(list(mon.moves))
        if entry.moves:
            true_moves = canonical_move_list(list(entry.moves))
            merged = [m for m in revealed if m in true_moves]
            mon.moves = merged or true_moves
        elif revealed:
            mon.moves = revealed


def true_species_for_slot_from_lines(
    lines: list[str],
    slot: str,
    *,
    before_idx: int | None = None,
) -> str | None:
    """Scan turn (or full battle) lines for |replace| revealing true species."""
    end = len(lines) if before_idx is None else before_idx
    true_species: str | None = None
    for line in lines[:end]:
        if not line.startswith("|replace|"):
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        if _slot_key(parts[2]) != slot:
            continue
        true_species = to_id_str(_species_name(parts[3]))
    return true_species


def guiderail_move_encoding_species(
    mon: MonPerspective | None,
    move_name: str,
    *,
    side: str,
    team_roster: list[str],
    turn_lines: list[str] | None,
    actor_slot: str,
) -> str | None:
    """
    Species key for move-slot encoding when Illusion may have mislabeled the slot.

    Prefers broken-illusion true species, then |replace| in turn lines, then
    roster lookup for which team member knows this move.
    """
    if mon is None:
        return None
    if mon.illusion_broken and mon.species:
        return mon.species

    if turn_lines:
        revealed = true_species_for_slot_from_lines(turn_lines, actor_slot)
        if revealed:
            return revealed

    move_id = to_id_str(move_name)
    if not move_id:
        return mon.species or None

    if mon.species and move_id in _ZOROARK_SIGNATURE_MOVES:
        disguise = roster_species_key(mon.species)
        if not disguise.startswith("zoroark"):
            for species in team_roster:
                sid = roster_species_key(species)
                if sid.startswith("zoroark"):
                    return sid

    return mon.species or None
