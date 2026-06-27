"""Bring-3 alignment and legal switch/move sets for live singles inference."""

from __future__ import annotations

from poke_env.battle.battle import Battle
from poke_env.data import to_id_str

from src.core.data.move_utils import canonical_move_list
from src.core.data.roster_profile import roster_species_key
from src.singles.action_space_spec import decode_singles_action_index
from src.singles.bench_slots import live_our_bench_mons, live_our_bench_species
from src.singles.log_action_codec import MEGA_BASE, MOVE_BASE, SWITCH_BASE

SINGLES_BRING_COUNT = 3


def live_team_list(battle: Battle) -> list:
    return list(battle.team.values())


def live_brought_species(battle: Battle) -> set[str]:
    from src.singles.teampreview import battle_team_summary

    brought = battle_team_summary(battle).get("brought") or []
    if brought:
        return {roster_species_key(sp) for sp in brought}
    team_list = live_team_list(battle)
    selected = [p for p in team_list if getattr(p, "selected_in_teampreview", False)]
    if selected:
        selected.sort(key=lambda p: team_list.index(p))
        return {roster_species_key(p.species) for p in selected[:SINGLES_BRING_COUNT]}
    return set()


def _active_species_on_field(battle: Battle) -> set[str]:
    mon = battle.active_pokemon
    if mon is None or bool(getattr(mon, "fainted", False)):
        return set()
    return {roster_species_key(mon.species)}


def legal_switch_indices(battle: Battle) -> set[int]:
    """Bench-aligned switch indices 0-1 matching state tokens 5-6."""
    on_field = _active_species_on_field(battle)
    avail = {roster_species_key(m.species) for m in battle.available_switches}
    legal: set[int] = set()

    for bench_idx, mon in enumerate(live_our_bench_mons(battle)):
        sid = roster_species_key(mon.species)
        if sid in on_field:
            continue
        if bool(getattr(mon, "fainted", False)):
            continue
        if avail and sid not in avail:
            continue
        legal.add(SWITCH_BASE + bench_idx)
    return legal


def legal_move_indices(battle: Battle) -> set[int]:
    mon = battle.active_pokemon
    if mon is None or bool(getattr(mon, "fainted", False)):
        return set()

    move_ids: list[str] = []
    try:
        move_ids = [to_id_str(m.id) for m in mon.moves.values() if m]
    except (AttributeError, TypeError):
        move_ids = []
    if not move_ids:
        move_ids = [to_id_str(m.id) for m in battle.available_moves]
    move_ids = canonical_move_list(move_ids)
    if not move_ids:
        return set()

    avail_ids = {to_id_str(m.id) for m in battle.available_moves}
    can_mega = bool(getattr(battle, "can_mega_evolve", False))

    legal: set[int] = set()
    for slot in range(min(4, len(move_ids))):
        mid = move_ids[slot]
        if avail_ids and mid not in avail_ids:
            continue
        legal.add(MOVE_BASE + slot)
        if can_mega:
            legal.add(MEGA_BASE + slot)
    return legal


def build_singles_action_mask(battle: Battle, *, size: int) -> list[bool]:
    mask = [False] * size
    force_switch = bool(getattr(battle, "force_switch", False))
    active = battle.active_pokemon
    must_switch = force_switch or active is None or bool(getattr(active, "fainted", False))

    if must_switch:
        for idx in legal_switch_indices(battle):
            if idx < MOVE_BASE:
                mask[idx] = True
    else:
        for idx in legal_switch_indices(battle):
            if idx < MOVE_BASE:
                mask[idx] = True
        for idx in legal_move_indices(battle):
            if idx < size:
                mask[idx] = True

    can_mega = bool(getattr(battle, "can_mega_evolve", False))
    for idx in range(size):
        if not mask[idx]:
            continue
        spec = decode_singles_action_index(idx)
        if force_switch and not spec.is_switch:
            mask[idx] = False
            continue
        if spec.illegal_gimmick:
            mask[idx] = False
            continue
        if spec.mega and not can_mega:
            mask[idx] = False

    if not any(mask):
        for idx in sorted(legal_switch_indices(battle)):
            if idx < MOVE_BASE:
                mask[idx] = True
                break
    return mask
