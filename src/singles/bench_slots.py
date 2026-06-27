"""Bring-3 bench slot ordering shared by state tokens, labels, and live inference."""

from __future__ import annotations

from poke_env.battle.battle import Battle
from poke_env.data import to_id_str

from src.core.data.log_tracker import BattleLogState, _species_name
from src.core.data.roster_profile import roster_species_key
from src.core.data.state_tokenizer import SINGLES_BENCH_SLOTS, _bench_slots_ordered, _live_singles_bench


def log_our_bench_slots(state: BattleLogState, side: str) -> list[str]:
    """Log slot ids for our bench tokens 5-6 (inactive, stable roster order)."""
    return _bench_slots_ordered(state, side)[:SINGLES_BENCH_SLOTS]


def log_our_bench_species(state: BattleLogState, side: str) -> list[str]:
    out: list[str] = []
    for slot in log_our_bench_slots(state, side):
        mon = state.mons.get(slot)
        if mon and mon.species:
            out.append(roster_species_key(mon.species))
    return out


def live_our_bench_mons(battle: Battle) -> list:
    """Live bench Pokémon in token 5-6 order (Bring-3, off-field)."""
    return _live_singles_bench(battle.team.values(), is_ours=True, battle=battle)


def live_our_bench_species(battle: Battle) -> list[str]:
    return [roster_species_key(p.species) for p in live_our_bench_mons(battle)]


def species_to_bench_switch_index(
    state: BattleLogState,
    side: str,
    species_details: str,
) -> int | None:
    """Map a switch target to bench slot 0 or 1, or None if not on our bench tokens."""
    target = roster_species_key(
        _species_name(species_details) if "," in species_details else species_details
    )
    for i, slot in enumerate(log_our_bench_slots(state, side)):
        mon = state.mons.get(slot)
        if mon and roster_species_key(mon.species) == target:
            return i
    return None


def bench_switch_index_to_species_log(
    state: BattleLogState,
    side: str,
    bench_idx: int,
) -> str:
    slots = log_our_bench_slots(state, side)
    if bench_idx < 0 or bench_idx >= len(slots):
        return f"bench-{bench_idx + 1}"
    mon = state.mons.get(slots[bench_idx])
    return mon.species if mon and mon.species else f"bench-{bench_idx + 1}"


def bench_switch_index_to_species_live(battle: Battle, bench_idx: int) -> str:
    bench = live_our_bench_mons(battle)
    if bench_idx < 0 or bench_idx >= len(bench):
        return f"bench-{bench_idx + 1}"
    return to_id_str(bench[bench_idx].species)
