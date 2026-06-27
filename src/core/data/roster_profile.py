"""Look-ahead roster profiles: aggregate each side's revealed sets over a full log."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from poke_env.data import to_id_str

from src.core.data.log_tracker import _species_name, _slot_key
from src.core.data.mega_items import is_mega_stone_item

BROUGHT_BY_FORMAT = {"doubles": 4, "singles": 3}
BENCH_BY_FORMAT = {"doubles": 4, "singles": 2}
ACTIVE_SUFFIXES_BY_FORMAT = {"doubles": ("a", "b"), "singles": ("a",)}

# Showdown forme suffixes merged onto team-roster |poke| species (e.g. cameruptmega -> camerupt).
_MEGA_FORM_SUFFIXES = ("megax", "megay", "mega")
# Battle-only formes that share roster/mega state with the base species (e.g. Floette-Eternal).
_ETERNAL_FORM_SUFFIX = "eternal"
# Gender forme markers in battle ids (Meowstic-M-Mega -> Meowstic-Mega).
_GENDER_FORME_RE = re.compile(r"-[mf](?=-|$)", re.IGNORECASE)


def roster_species_key(species: str) -> str:
    """Map battle forme ids to the species key used in |poke| team rosters."""
    raw = _species_name(species) if "," in species else species
    raw = _GENDER_FORME_RE.sub("", raw)
    sid = to_id_str(raw)
    for suffix in _MEGA_FORM_SUFFIXES:
        if sid.endswith(suffix) and len(sid) > len(suffix):
            sid = sid[: -len(suffix)]
            break
    if sid.endswith(_ETERNAL_FORM_SUFFIX) and len(sid) > len(_ETERNAL_FORM_SUFFIX):
        sid = sid[: -len(_ETERNAL_FORM_SUFFIX)]
    if len(sid) > 5 and sid[-1] in "mf" and sid[-2].isalpha():
        sid = sid[:-1]
    return sid


@dataclass
class MonRosterEntry:
    species: str
    moves: set[str] = field(default_factory=set)
    item: str = ""
    item_revealed: bool = False
    ability: str = ""
    ability_revealed: bool = False
    mega: bool = False
    mega_capable: bool = False
    brought: bool = False

    def merge_moves(self, moves: list[str]) -> None:
        for m in moves:
            if m:
                self.moves.add(to_id_str(m))


@dataclass
class SideRoster:
    """Full-roster knowledge for one player (p1 or p2)."""

    side: str
    entries: dict[str, MonRosterEntry] = field(default_factory=dict)

    def get(self, species: str) -> MonRosterEntry | None:
        return self.entries.get(roster_species_key(species))

    def ensure(self, species: str) -> MonRosterEntry:
        sid = roster_species_key(species)
        if sid not in self.entries:
            self.entries[sid] = MonRosterEntry(species=sid)
        return self.entries[sid]


@dataclass
class MatchRosters:
    p1: SideRoster = field(default_factory=lambda: SideRoster(side="p1"))
    p2: SideRoster = field(default_factory=lambda: SideRoster(side="p2"))

    def for_side(self, side: str) -> SideRoster:
        return self.p1 if side == "p1" else self.p2


def _side_from_slot(slot: str) -> str:
    return slot[:2]


def _species_for_slot(slot_species: dict[str, str], actor: str, *, fallback: str = "") -> str:
    """Resolve species id from active slot (switch details), not actor nickname."""
    slot = _slot_key(actor)
    if slot in slot_species:
        return slot_species[slot]
    if fallback:
        return to_id_str(_species_name(fallback))
    return _species_from_actor(actor)


def build_match_rosters(lines: list[str]) -> MatchRosters:
    """
    Fast pre-pass: scan the entire log and build per-species move/item/ability
    profiles for each player. Used to populate OUR mons with full known sets on
    early turns (player knows their own team from team preview / past reveals).
    """
    rosters = MatchRosters()
    slot_species: dict[str, str] = {}
    team_order: dict[str, list[str]] = {"p1": [], "p2": []}

    for line in lines:
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        cmd = parts[1]

        if cmd == "poke" and len(parts) >= 4:
            side = parts[2][:2]
            species = _species_name(parts[3])
            rosters.for_side(side).ensure(species)
            team_order.setdefault(side, []).append(species)
        elif cmd == "useteam" and len(parts) >= 4:
            side = parts[2][:2]
            slots = [int(ch) for ch in parts[3] if ch.isdigit()]
            order = team_order.get(side, [])
            for idx in slots:
                if 1 <= idx <= len(order):
                    rosters.for_side(side).ensure(order[idx - 1]).brought = True
        elif cmd == "replace" and len(parts) >= 4:
            slot = _slot_key(parts[2])
            side = _side_from_slot(slot)
            disguise = slot_species.get(slot)
            species = _species_name(parts[3])
            true_entry = rosters.for_side(side).ensure(species)
            if disguise:
                disguise_entry = rosters.for_side(side).get(disguise)
                if disguise_entry and disguise_entry.moves:
                    true_entry.merge_moves(list(disguise_entry.moves))
            slot_species[slot] = species
            true_entry.brought = True
        elif cmd in ("switch", "drag") and len(parts) >= 4:
            slot = _slot_key(parts[2])
            side = _side_from_slot(slot)
            species = _species_name(parts[3])
            slot_species[slot] = species
            entry = rosters.for_side(side).ensure(species)
            entry.brought = True
        elif cmd == "move" and len(parts) >= 4:
            slot = _slot_key(parts[2])
            side = _side_from_slot(slot)
            species = _species_for_slot(slot_species, parts[2])
            mon = rosters.for_side(side).ensure(species)
            mon.merge_moves([parts[3]])
            mon.brought = True
        elif cmd == "-ability" and len(parts) >= 4:
            slot = _slot_key(parts[2])
            side = _side_from_slot(slot)
            species = _species_for_slot(slot_species, parts[2])
            entry = rosters.for_side(side).ensure(species)
            entry.ability = to_id_str(parts[3])
            entry.ability_revealed = True
        elif cmd == "-item" and len(parts) >= 4:
            slot = _slot_key(parts[2])
            side = _side_from_slot(slot)
            species = _species_for_slot(slot_species, parts[2])
            entry = rosters.for_side(side).ensure(species)
            entry.item = to_id_str(parts[3])
            entry.item_revealed = True
            if is_mega_stone_item(entry.item):
                entry.mega_capable = True
        elif cmd == "-enditem" and len(parts) >= 4:
            slot = _slot_key(parts[2])
            side = _side_from_slot(slot)
            species = _species_for_slot(slot_species, parts[2])
            entry = rosters.for_side(side).ensure(species)
            entry.item = to_id_str(parts[3])
            entry.item_revealed = True
            if is_mega_stone_item(entry.item):
                entry.mega_capable = True
        elif cmd in ("-damage", "-heal") and len(parts) >= 5:
            suffix = parts[-1]
            if "[from] item:" in suffix:
                slot = _slot_key(parts[2])
                side = _side_from_slot(slot)
                item = suffix.split("[from] item:")[-1].strip()
                species = _species_for_slot(slot_species, parts[2])
                entry = rosters.for_side(side).ensure(species)
                entry.item = to_id_str(item)
                entry.item_revealed = True
                if is_mega_stone_item(entry.item):
                    entry.mega_capable = True
        elif cmd == "detailschange" and len(parts) >= 4:
            slot = _slot_key(parts[2])
            side = _side_from_slot(slot)
            species = _species_name(parts[3])
            slot_species[slot] = species
            entry = rosters.for_side(side).ensure(species)
            if "mega" in species:
                entry.mega = True
                entry.mega_capable = True
        elif cmd == "-mega" and len(parts) >= 3:
            slot = _slot_key(parts[2])
            side = _side_from_slot(slot)
            species = _species_for_slot(slot_species, parts[2])
            entry = rosters.for_side(side).ensure(species)
            entry.mega = True
            entry.mega_capable = True

    return rosters


def _species_from_actor(actor: str) -> str:
    """Extract species label from 'p1a: Garchomp' style actor."""
    if ":" in actor:
        return to_id_str(actor.split(":", 1)[1].strip())
    return to_id_str(actor)


def brought_species_set(rosters: MatchRosters, side: str) -> set[str]:
    return {
        roster_species_key(entry.species)
        for entry in rosters.for_side(side).entries.values()
        if entry.brought
    }


def _mon_from_roster_entry(slot: str, entry: MonRosterEntry) -> "MonPerspective":
    from src.core.data.move_utils import canonical_move_list
    from src.core.data.perspective import MonPerspective

    mon = MonPerspective(
        slot=slot,
        species=entry.species,
        hp=100,
        max_hp=100,
        active=False,
        seen=True,
        moves=canonical_move_list(list(entry.moves)),
    )
    if entry.item_revealed:
        mon.item = entry.item
        mon.item_revealed = True
    if entry.ability_revealed:
        mon.ability = entry.ability
        mon.ability_revealed = True
    if entry.mega_capable:
        mon.mega_capable = True
    return mon


def _species_present_on_side(state, side: str, species_key: str) -> bool:
    for slot, mon in state.mons.items():
        if not slot.startswith(side) or not mon.species:
            continue
        if roster_species_key(mon.species) == species_key:
            return True
    return False


def materialize_our_bench_from_roster(
    state,
    side: str,
    rosters: MatchRosters,
    *,
    format: str = "doubles",
) -> None:
    """
    At turn 1 (and whenever bench mons are not yet on field), inject our inactive
    brought Pokémon into state.mons so bench tokens are populated.
    """
    roster = rosters.for_side(side)
    brought = [entry for entry in roster.entries.values() if entry.brought]
    expected = BROUGHT_BY_FORMAT.get(format, 4)
    if len(brought) != expected:
        return

    lead_keys: set[str] = set()
    for suffix in ACTIVE_SUFFIXES_BY_FORMAT.get(format, ("a", "b")):
        mon = state.mons.get(f"{side}{suffix}")
        if mon is not None and mon.active and mon.species:
            lead_keys.add(roster_species_key(mon.species))

    roster_order = state.team_roster.get(side, [])
    bench_species: list[str] = []
    for species in roster_order:
        key = roster_species_key(species)
        entry = roster.get(species)
        if entry is None or not entry.brought or key in lead_keys:
            continue
        if _species_present_on_side(state, side, key):
            continue
        bench_species.append(species)

    bench_slots = BENCH_BY_FORMAT.get(format, 4)
    free_indices: list[int] = []
    for i in range(bench_slots):
        bench_key = f"{side}_b{i}"
        mon = state.mons.get(bench_key)
        if mon is None or not mon.species or mon.fainted:
            free_indices.append(i)

    for species, bench_i in zip(bench_species, free_indices):
        bench_key = f"{side}_b{bench_i}"
        entry = roster.get(species)
        if entry is None:
            continue
        state.mons[bench_key] = _mon_from_roster_entry(bench_key, entry)


def _lead_species_keys(state, side: str, *, format: str = "doubles") -> set[str]:
    keys: set[str] = set()
    for suffix in ACTIVE_SUFFIXES_BY_FORMAT.get(format, ("a", "b")):
        mon = state.mons.get(f"{side}{suffix}")
        if mon is not None and mon.active and mon.species:
            keys.add(roster_species_key(mon.species))
    return keys


def materialize_opp_preview_bench(
    state,
    opp_side: str,
    rosters: MatchRosters,
    *,
    format: str = "doubles",
) -> None:
    """
    Populate opp bench tokens with non-lead species from team preview (species only).
    Fog of war: unseen until they switch in.
    """
    roster_order = state.team_roster.get(opp_side, [])
    if len(roster_order) < 6:
        return

    lead_keys = _lead_species_keys(state, opp_side, format=format)
    preview_species: list[str] = []
    for species in roster_order:
        key = roster_species_key(species)
        if key in lead_keys:
            continue
        if _species_present_on_side(state, opp_side, key):
            continue
        preview_species.append(species)

    from src.core.data.perspective import MonPerspective

    bench_slots = BENCH_BY_FORMAT.get(format, 4)
    # Place preview species into FREE bench slots only. Revealed mons that
    # switched out occupy their own bench keys; indexing preview species
    # positionally would collide with those and silently drop backline
    # threats, so we skip occupied keys instead of overwriting/skipping species.
    free_keys = [
        f"{opp_side}_b{i}"
        for i in range(bench_slots)
        if not (state.mons.get(f"{opp_side}_b{i}") and state.mons[f"{opp_side}_b{i}"].species)
    ]
    for bench_key, species in zip(free_keys, preview_species):
        state.mons[bench_key] = MonPerspective(
            slot=bench_key,
            species=roster_species_key(species),
            hp=100,
            max_hp=100,
            active=False,
            seen=False,
            moves=[],
        )


def apply_own_roster(
    state,
    side: str,
    rosters: MatchRosters,
) -> None:
    """
    Enrich our mon tokens with look-ahead roster data (moves/item/ability).
    Opponent slots are untouched — fog of war preserved.
    """
    roster = rosters.for_side(side)
    for slot, mon in state.mons.items():
        if not slot.startswith(side):
            continue
        entry = roster.get(roster_species_key(mon.species)) or roster.get(mon.species)
        if entry is None:
            continue
        if entry.moves:
            from src.core.data.move_utils import canonical_move_list

            mon.moves = canonical_move_list(list(entry.moves))
        if entry.item_revealed:
            mon.item = entry.item
            mon.item_revealed = True
        if entry.ability_revealed:
            mon.ability = entry.ability
            mon.ability_revealed = True
        # Do not set mon.mega from look-ahead roster (entry.mega); that leaks future
        # |-mega|/detailschange and breaks can_mega on pre-evolution decision samples.
        if entry.mega_capable:
            mon.mega_capable = True
