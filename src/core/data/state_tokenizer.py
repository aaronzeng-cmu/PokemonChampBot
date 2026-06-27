"""13-token state encoder for BC Transformer (logs + live poke-env battles)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from poke_env.battle import Field, SideCondition, Status, Weather
from poke_env.battle.battle import Battle
from poke_env.battle.double_battle import DoubleBattle
from poke_env.data import to_id_str

from src.core.data.move_utils import canonical_move_list
from src.core.data.log_tracker import BattleLogState
from src.core.data.roster_profile import roster_species_key
from src.doubles.data.mega_state import live_can_mega_for_pos, live_can_mega_for_singles
from src.core.data.perspective import MonPerspective, boost_id, hash_token, move_vocab_id, status_id

_UNKNOWN_ITEM = "unknown_item"

N_TOKENS = 13
N_FIELDS = 25
TRAJECTORY_DEPTH = 3
STACKED_N_TOKENS = N_TOKENS * TRAJECTORY_DEPTH

BattleFormat = Literal["doubles", "singles"]
SINGLES_BENCH_SLOTS = 2
SINGLES_BRING_COUNT = 3

# Pokémon token field indices (non-overlapping layout).
FIELD_ROLE = 0
FIELD_SPECIES = 1
FIELD_ABILITY = 2
FIELD_ITEM = 3
FIELD_HP = 4
FIELD_STATUS = 5
FIELD_BOOST_START = 6
FIELD_MOVE_START = 13
FIELD_MOVE_DISABLED_START = 17
FIELD_FLAGS = 21
FIELD_TURNS_ACTIVE = 22
FIELD_PROTECT_COUNTER = 23
FIELD_LAST_MOVE_ID = 24

# Field token (environment) indices.
FIELD_WEATHER = 1
FIELD_TERRAIN = 2
FIELD_TRICK_ROOM = 3
FIELD_TAILWIND_OURS = 4
FIELD_TAILWIND_OPP = 5
FIELD_REFLECT_OURS = 6
FIELD_LIGHT_SCREEN_OURS = 7
FIELD_AURORA_VEIL_OURS = 8
FIELD_REFLECT_OPP = 9
FIELD_LIGHT_SCREEN_OPP = 10
FIELD_AURORA_VEIL_OPP = 11
FIELD_DECISION_FLAGS = 12

FLAG_MEGA = 1
FLAG_SEEN = 2
FLAG_FAINTED = 4
FLAG_ACTIVE = 8
FLAG_ITEM_REVEALED = 16
FLAG_CAN_MEGA = 32

DECISION_FORCE_SWITCH = 1

STAT_NAMES = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

TOKEN_FIELD = 0
TOKEN_OUR_ACTIVE = 1
TOKEN_OPP_ACTIVE = 3
TOKEN_OUR_BENCH = 5
TOKEN_OPP_BENCH = 9

TOKEN_ROLE_NAMES = {
    0: "field",
    1: "our_active_1",
    2: "our_active_2",
    3: "opp_active_1",
    4: "opp_active_2",
    5: "our_bench_1",
    6: "our_bench_2",
    7: "our_bench_3",
    8: "our_bench_4",
    9: "opp_bench_1",
    10: "opp_bench_2",
    11: "opp_bench_3",
    12: "opp_bench_4",
}


def _weather_id(weather: str) -> int:
    mapping = {
        "": 0,
        "sunnyday": 1,
        "raindance": 2,
        "sandstorm": 3,
        "snowscape": 4,
        "hail": 4,
    }
    return mapping.get(to_id_str(weather), 0)


def _terrain_id(terrain: str) -> int:
    mapping = {
        "": 0,
        "electricterrain": 1,
        "grassyterrain": 2,
        "mistyterrain": 3,
        "psychicterrain": 4,
    }
    return mapping.get(to_id_str(terrain), 0)


def _move_disabled_flags(mon: MonPerspective, moves: list[str]) -> list[int]:
    """Per-move-slot disabled/empty flags (1 = unavailable)."""
    flags = [0, 0, 0, 0]
    disabled = list(mon.move_disabled) if mon.move_disabled else []
    for i in range(4):
        if i >= len(moves) or not moves[i]:
            flags[i] = 1
        elif i < len(disabled) and disabled[i]:
            flags[i] = 1
    return flags


def _encode_mon_token(
    mon: MonPerspective | None,
    *,
    role: int,
    is_ours: bool,
    include_temporal: bool = False,
) -> np.ndarray:
    out = np.zeros(N_FIELDS, dtype=np.int64)
    out[FIELD_ROLE] = role
    if mon is None:
        return out

    out[FIELD_SPECIES] = hash_token(mon.species)

    if is_ours:
        ability = mon.ability if mon.ability_revealed else ""
        item = mon.item if mon.item_revealed else ""
        moves = canonical_move_list(mon.moves)
    else:
        ability = mon.visible_ability()
        item = mon.visible_item()
        moves = canonical_move_list(mon.visible_moves())

    out[FIELD_ABILITY] = hash_token(ability)
    out[FIELD_ITEM] = hash_token(item)
    out[FIELD_HP] = int(mon.hp_fraction * 20)
    out[FIELD_STATUS] = status_id(mon.status)
    for i, stat in enumerate(STAT_NAMES):
        out[FIELD_BOOST_START + i] = boost_id(mon.boosts.get(stat, 0))

    for j, move in enumerate(moves[:4]):
        out[FIELD_MOVE_START + j] = hash_token(move)
    for j, flag in enumerate(_move_disabled_flags(mon, moves)):
        out[FIELD_MOVE_DISABLED_START + j] = flag

    flags = 0
    if mon.mega:
        flags |= FLAG_MEGA
    if mon.seen or is_ours:
        flags |= FLAG_SEEN
    if mon.fainted:
        flags |= FLAG_FAINTED
    if mon.active:
        flags |= FLAG_ACTIVE
    if mon.item_revealed or is_ours:
        flags |= FLAG_ITEM_REVEALED
    if is_ours and mon.can_mega:
        flags |= FLAG_CAN_MEGA
    out[FIELD_FLAGS] = flags

    if include_temporal:
        out[FIELD_TURNS_ACTIVE] = min(int(mon.turns_active), 4095)
        out[FIELD_PROTECT_COUNTER] = min(int(mon.protect_counter), 4095)
        out[FIELD_LAST_MOVE_ID] = min(int(mon.last_move_id), 4095)
    return out


def _encode_field_token(
    field,
    *,
    perspective: str,
    force_switch: bool = False,
) -> np.ndarray:
    out = np.zeros(N_FIELDS, dtype=np.int64)
    out[FIELD_ROLE] = TOKEN_FIELD
    out[FIELD_WEATHER] = _weather_id(getattr(field, "weather", ""))
    out[FIELD_TERRAIN] = _terrain_id(getattr(field, "terrain", ""))
    out[FIELD_TRICK_ROOM] = 1 if getattr(field, "trick_room", False) else 0
    if perspective == "p1":
        out[FIELD_TAILWIND_OURS] = min(4, int(getattr(field, "tailwind_p1", 0)))
        out[FIELD_TAILWIND_OPP] = min(4, int(getattr(field, "tailwind_p2", 0)))
        out[FIELD_REFLECT_OURS] = min(5, int(getattr(field, "reflect_p1", 0)))
        out[FIELD_LIGHT_SCREEN_OURS] = min(5, int(getattr(field, "light_screen_p1", 0)))
        out[FIELD_AURORA_VEIL_OURS] = min(5, int(getattr(field, "aurora_veil_p1", 0)))
        out[FIELD_REFLECT_OPP] = min(5, int(getattr(field, "reflect_p2", 0)))
        out[FIELD_LIGHT_SCREEN_OPP] = min(5, int(getattr(field, "light_screen_p2", 0)))
        out[FIELD_AURORA_VEIL_OPP] = min(5, int(getattr(field, "aurora_veil_p2", 0)))
    else:
        out[FIELD_TAILWIND_OURS] = min(4, int(getattr(field, "tailwind_p2", 0)))
        out[FIELD_TAILWIND_OPP] = min(4, int(getattr(field, "tailwind_p1", 0)))
        out[FIELD_REFLECT_OURS] = min(5, int(getattr(field, "reflect_p2", 0)))
        out[FIELD_LIGHT_SCREEN_OURS] = min(5, int(getattr(field, "light_screen_p2", 0)))
        out[FIELD_AURORA_VEIL_OURS] = min(5, int(getattr(field, "aurora_veil_p2", 0)))
        out[FIELD_REFLECT_OPP] = min(5, int(getattr(field, "reflect_p1", 0)))
        out[FIELD_LIGHT_SCREEN_OPP] = min(5, int(getattr(field, "light_screen_p1", 0)))
        out[FIELD_AURORA_VEIL_OPP] = min(5, int(getattr(field, "aurora_veil_p1", 0)))
    if force_switch:
        out[FIELD_DECISION_FLAGS] |= DECISION_FORCE_SWITCH
    return out


def _bench_slots_ordered(state: BattleLogState, side: str) -> list[str]:
    roster = state.team_roster.get(side, [])

    def _rank(mon: MonPerspective) -> int:
        key = roster_species_key(mon.species)
        for i, species in enumerate(roster):
            if roster_species_key(species) == key:
                return i
        return 99

    inactive = [
        (slot, mon)
        for slot, mon in state.mons.items()
        if slot.startswith(side) and not mon.active and mon.species and not mon.fainted
    ]
    inactive.sort(key=lambda item: (_rank(item[1]), item[0]))
    return [slot for slot, _ in inactive[:4]]


def empty_slot_token(role: int) -> np.ndarray:
    """EMPTY_SLOT baseline: FIELD_ROLE set, all other fields absolute zero."""
    out = np.zeros(N_FIELDS, dtype=np.int64)
    out[FIELD_ROLE] = role
    return out


def encode_log_state(
    state: BattleLogState,
    side: str,
    *,
    format: BattleFormat = "doubles",
    force_switch: bool = False,
) -> np.ndarray:
    opp = "p2" if side == "p1" else "p1"
    tokens = np.zeros((N_TOKENS, N_FIELDS), dtype=np.int64)
    tokens[0] = _encode_field_token(
        state.field,
        perspective=side,
        force_switch=force_switch,
    )

    if format == "singles":
        tokens[TOKEN_OUR_ACTIVE] = _encode_mon_token(
            state.mons.get(f"{side}a"),
            role=TOKEN_OUR_ACTIVE,
            is_ours=True,
            include_temporal=True,
        )
        tokens[TOKEN_OUR_ACTIVE + 1] = empty_slot_token(TOKEN_OUR_ACTIVE)

        tokens[TOKEN_OPP_ACTIVE] = _encode_mon_token(
            state.mons.get(f"{opp}a"),
            role=TOKEN_OPP_ACTIVE,
            is_ours=False,
            include_temporal=True,
        )
        tokens[TOKEN_OPP_ACTIVE + 1] = empty_slot_token(TOKEN_OPP_ACTIVE)

        our_bench = _bench_slots_ordered(state, side)[:SINGLES_BENCH_SLOTS]
        opp_bench = _bench_slots_ordered(state, opp)[:SINGLES_BENCH_SLOTS]
        for i in range(SINGLES_BENCH_SLOTS):
            tokens[TOKEN_OUR_BENCH + i] = _encode_mon_token(
                state.mons.get(our_bench[i]) if i < len(our_bench) else None,
                role=TOKEN_OUR_BENCH,
                is_ours=True,
            )
            tokens[TOKEN_OPP_BENCH + i] = _encode_mon_token(
                state.mons.get(opp_bench[i]) if i < len(opp_bench) else None,
                role=TOKEN_OPP_BENCH,
                is_ours=False,
            )
        for i in range(SINGLES_BENCH_SLOTS, 4):
            tokens[TOKEN_OUR_BENCH + i] = empty_slot_token(TOKEN_OUR_BENCH)
            tokens[TOKEN_OPP_BENCH + i] = empty_slot_token(TOKEN_OPP_BENCH)
        return tokens

    for i, slot in enumerate([f"{side}a", f"{side}b"]):
        tokens[TOKEN_OUR_ACTIVE + i] = _encode_mon_token(
            state.mons.get(slot),
            role=TOKEN_OUR_ACTIVE,
            is_ours=True,
            include_temporal=True,
        )
    for i, slot in enumerate([f"{opp}a", f"{opp}b"]):
        tokens[TOKEN_OPP_ACTIVE + i] = _encode_mon_token(
            state.mons.get(slot),
            role=TOKEN_OPP_ACTIVE,
            is_ours=False,
            include_temporal=True,
        )

    our_bench = _bench_slots_ordered(state, side)
    opp_bench = _bench_slots_ordered(state, opp)
    for i in range(4):
        tokens[TOKEN_OUR_BENCH + i] = _encode_mon_token(
            state.mons.get(our_bench[i]) if i < len(our_bench) else None,
            role=TOKEN_OUR_BENCH,
            is_ours=True,
        )
        tokens[TOKEN_OPP_BENCH + i] = _encode_mon_token(
            state.mons.get(opp_bench[i]) if i < len(opp_bench) else None,
            role=TOKEN_OPP_BENCH,
            is_ours=False,
        )
    return tokens


def empty_snapshot() -> np.ndarray:
    """Single-turn 13×N_FIELDS zero tensor for trajectory padding."""
    return np.zeros((N_TOKENS, N_FIELDS), dtype=np.int64)


def stack_trajectory(
    prior_snapshots: list[np.ndarray],
    current: np.ndarray,
    *,
    depth: int = TRAJECTORY_DEPTH,
) -> np.ndarray:
    """
    Stack T turn snapshots into (depth * 13, N_FIELDS).
    Order: [t-(depth-1), …, t-1, t0 current]. Missing history padded with zeros.
    """
    current = np.asarray(current, dtype=np.int64).copy()
    frames = [np.asarray(f, dtype=np.int64).copy() for f in prior_snapshots] + [current]
    pad = empty_snapshot()
    while len(frames) < depth:
        frames.insert(0, pad)
    frames = frames[-depth:]
    return np.concatenate(frames, axis=0)


def push_trajectory(
    history: list[np.ndarray],
    snapshot: np.ndarray,
    *,
    depth: int = TRAJECTORY_DEPTH,
    maxlen: int | None = None,
    history_snapshot: np.ndarray | None = None,
) -> np.ndarray:
    """Append snapshot to rolling history and return stacked (39, N_FIELDS) tensor.

    When ``history_snapshot`` is set, it is stored in ``history`` while
    ``snapshot`` is used only for the stacked t-0 frame (turn flush parity).
    """
    snapshot = np.asarray(snapshot, dtype=np.int64).copy()
    stacked = stack_trajectory(history, snapshot, depth=depth)
    stored = (
        np.asarray(history_snapshot, dtype=np.int64).copy()
        if history_snapshot is not None
        else snapshot
    )
    history.append(stored)
    if maxlen is not None:
        while len(history) > maxlen:
            history.pop(0)
    return stacked


def trajectory_frame_fingerprints(stacked: np.ndarray) -> list[str]:
    """Compact per-frame labels for trajectory audit logs (oldest → newest)."""
    depth = stacked.shape[0] // N_TOKENS
    frames = stacked.reshape(depth, N_TOKENS, N_FIELDS)
    labels: list[str] = []
    for i in range(depth):
        f = frames[i]
        if not f.any():
            labels.append(f"t-{depth - 1 - i}:empty")
            continue
        a0 = int(f[TOKEN_OUR_ACTIVE, FIELD_SPECIES])
        a1 = int(f[TOKEN_OUR_ACTIVE + 1, FIELD_SPECIES])
        labels.append(f"t-{depth - 1 - i}:our=({a0},{a1})")
    return labels


def human_readable_state(state: BattleLogState, side: str) -> dict:
    """Decode first-person state for sanity-check printing (no hashes)."""
    opp = "p2" if side == "p1" else "p1"

    def _mon(slot: str, is_ours: bool) -> dict:
        mon = state.mons.get(slot)
        if mon is None:
            return {"slot": slot, "present": False}
        return {
            "slot": slot,
            "present": True,
            "species": mon.species or None,
            "hp": f"{mon.hp}/{mon.max_hp}",
            "status": mon.status or None,
            "boosts": dict(mon.boosts) if mon.boosts else {},
            "ability": (
                mon.ability
                if is_ours and mon.ability_revealed
                else (mon.visible_ability() if not is_ours else None)
            )
            or None,
            "item": (
                mon.item
                if is_ours and mon.item_revealed
                else (mon.visible_item() if not is_ours else None)
            )
            or None,
            "moves": list(mon.moves) if is_ours else mon.visible_moves(),
            "mega": mon.mega,
            "mega_capable": mon.mega_capable,
            "can_mega": mon.can_mega,
            "illusion_disguise": mon.illusion_disguise or None,
            "illusion_broken": mon.illusion_broken,
            "tera": mon.terastallized,
            "tera_type": mon.tera_type or None,
            "seen": mon.seen,
            "item_revealed": mon.item_revealed,
            "ability_revealed": mon.ability_revealed,
            "active": mon.active,
            "fainted": mon.fainted,
            "turns_active": mon.turns_active,
            "protect_counter": mon.protect_counter,
            "last_move_id": mon.last_move_id,
        }

    return {
        "perspective": side,
        "turn": state.turn,
        "field": {
            "weather": state.field.weather or None,
            "terrain": state.field.terrain or None,
            "trick_room": state.field.trick_room,
            "tailwind_ours": (
                state.field.tailwind_p1 if side == "p1" else state.field.tailwind_p2
            ),
            "tailwind_opp": (
                state.field.tailwind_p2 if side == "p1" else state.field.tailwind_p1
            ),
            "reflect_ours": (
                state.field.reflect_p1 if side == "p1" else state.field.reflect_p2
            ),
            "light_screen_ours": (
                state.field.light_screen_p1 if side == "p1" else state.field.light_screen_p2
            ),
            "aurora_veil_ours": (
                state.field.aurora_veil_p1 if side == "p1" else state.field.aurora_veil_p2
            ),
        },
        "our_actives": [_mon(f"{side}a", True), _mon(f"{side}b", True)],
        "opp_actives": [_mon(f"{opp}a", False), _mon(f"{opp}b", False)],
        "our_bench": [
            _mon(s, True)
            for s, m in state.mons.items()
            if s.startswith(side) and not m.active
        ][:4],
        "opp_bench": [
            _mon(s, False)
            for s, m in state.mons.items()
            if s.startswith(opp) and not m.active
        ][:4],
    }


def _brought_team_members(
    team_values, battle: DoubleBattle | Battle | None = None, *, max_bring: int | None = None
) -> list:
    """Pokémon brought to this match; stable team-list order."""
    team_list = list(team_values)
    brought = [p for p in team_list if getattr(p, "selected_in_teampreview", False)]
    if brought:
        brought.sort(key=lambda p: team_list.index(p))
        if max_bring is not None:
            return brought[:max_bring]
        return brought

    if battle is not None:
        pool: list = []
        seen: set[int] = set()
        for p in team_list:
            if p.active:
                pid = id(p)
                if pid not in seen:
                    pool.append(p)
                    seen.add(pid)
        for p in battle.available_switches:
            pid = id(p)
            if pid not in seen:
                pool.append(p)
                seen.add(pid)
        if pool:
            pool.sort(key=lambda p: team_list.index(p))
            if max_bring is not None:
                return pool[:max_bring]
            return pool

    active = [p for p in team_list if p.active]
    if active:
        return active[:max_bring] if max_bring is not None else active
    revealed = [p for p in team_list if bool(getattr(p, "revealed", False))]
    if revealed:
        revealed.sort(key=lambda p: team_list.index(p))
        if max_bring is not None:
            return revealed[:max_bring]
        return revealed
    cap = max_bring if max_bring is not None else 4
    return team_list[:cap]


def _live_move_disabled(mon, moves: list[str], *, is_ours: bool) -> list[bool]:
    flags = [False, False, False, False]
    move_objs = list(mon.moves.values()) if hasattr(mon, "moves") else []
    ordered: list[str] = []
    for m in move_objs:
        if m is None:
            continue
        ordered.append(to_id_str(getattr(m, "id", "") or ""))
    ordered = canonical_move_list(ordered)
    for i, name in enumerate(moves[:4]):
        if not name:
            flags[i] = True
            continue
        if not is_ours:
            continue
        for m in move_objs:
            if to_id_str(getattr(m, "id", "") or "") == name:
                if getattr(m, "disabled", False):
                    flags[i] = True
                pp = getattr(m, "current_pp", None)
                if pp is not None and int(pp) <= 0:
                    flags[i] = True
                break
    for i in range(len(moves), 4):
        flags[i] = True
    return flags


def _opponent_lead_species_ids(battle: Battle | DoubleBattle) -> set[str]:
    """Species currently active on the opponent's side (singles or doubles)."""
    if isinstance(battle, DoubleBattle):
        return {
            to_id_str(p.species)
            for p in battle.opponent_active_pokemon
            if p is not None
        }
    opp = battle.opponent_active_pokemon
    if opp is None:
        return set()
    return {to_id_str(opp.species)}


def _opp_preview_bench_from_team(
    team_list: list, *, lead_species: set[str], limit: int
) -> list:
    """Non-lead species from team preview (species-only fog-of-war belief)."""
    pool: list = []
    for p in team_list:
        sp = to_id_str(p.species)
        if sp in lead_species or p.active:
            continue
        if p in pool:
            continue
        pool.append(p)
    if len(pool) < limit:
        for p in team_list:
            sp = to_id_str(p.species)
            if sp in lead_species:
                continue
            if p not in pool:
                pool.append(p)
    pool.sort(key=lambda p: team_list.index(p))
    return pool[:limit]


def _opp_preview_bench_members(team_values, *, battle: Battle | DoubleBattle) -> list:
    """Non-lead species from opponent's 6-mon paste (species-only preview belief)."""
    team_list = list(team_values)
    leads = _opponent_lead_species_ids(battle)
    return _opp_preview_bench_from_team(team_list, lead_species=leads, limit=4)


def _live_bench_members(
    team_values, *, is_ours: bool, battle: DoubleBattle | None = None
) -> list:
    """
    Bench tokens mirror encode_log_state: only brought/seen mons not on field.
    Opponent: include 4 preview roster species (unrevealed) when battle is early.
    """
    team_list = list(team_values)
    if is_ours:
        pool = _brought_team_members(team_list, battle=battle)
        bench = []
        for p in pool:
            if p.active:
                continue
            bench.append(p)
        bench.sort(key=lambda p: team_list.index(p))
        return bench[:4]

    if battle is not None:
        revealed_bench = []
        for p in team_list:
            if p.active:
                continue
            if bool(getattr(p, "revealed", False)):
                revealed_bench.append(p)
        if revealed_bench:
            revealed_bench.sort(key=lambda p: team_list.index(p))
            return revealed_bench[:4]
        return _opp_preview_bench_members(team_list, battle=battle)
    bench = []
    for p in team_list:
        if p.active:
            continue
        if not bool(getattr(p, "revealed", False)):
            continue
        bench.append(p)
    bench.sort(key=lambda p: team_list.index(p))
    return bench[:4]


def _live_hp_pair(mon) -> tuple[int, int]:
    max_hp = int(getattr(mon, "max_hp", 0) or 0)
    current_hp = int(getattr(mon, "current_hp", 0) or 0)
    if max_hp <= 0 and current_hp > 0:
        max_hp = current_hp
    return current_hp, max_hp


def _live_item_revealed(mon, *, is_ours: bool) -> bool:
    """First-person knowledge: we always see our items; opponent items need reveal."""
    item = getattr(mon, "item", None)
    if item is None:
        return False
    if to_id_str(str(item)) == _UNKNOWN_ITEM:
        return False
    if is_ours:
        return True
    return bool(getattr(mon, "revealed", False))


def _live_ability_revealed(mon, *, is_ours: bool) -> bool:
    """First-person knowledge: we always see our ability; opponent needs reveal."""
    if not getattr(mon, "ability", None):
        return False
    if is_ours:
        return True
    return bool(getattr(mon, "revealed", False))


def _live_moves(mon, *, is_ours: bool) -> list[str]:
    """Stable alphabetical move slots — must match encode_log_state / canonical_move_list."""
    if mon is None:
        return []
    raw: list[str] = []
    for m in mon.moves.values() if hasattr(mon, "moves") else []:
        if m is None:
            continue
        raw.append(to_id_str(getattr(m, "id", "") or ""))
    if not is_ours and not bool(getattr(mon, "revealed", False)):
        return []
    if not raw:
        return []
    return canonical_move_list(raw)


def encode_battle(battle: DoubleBattle) -> np.ndarray:
    """Encode live poke-env DoubleBattle into 13 tokens."""
    tokens = np.zeros((N_TOKENS, N_FIELDS), dtype=np.int64)

    class _LiveField:
        weather = ""
        terrain = ""
        trick_room = False
        tailwind_p1 = 0
        tailwind_p2 = 0
        reflect_p1 = 0
        reflect_p2 = 0
        light_screen_p1 = 0
        light_screen_p2 = 0
        aurora_veil_p1 = 0
        aurora_veil_p2 = 0

    live_field = _LiveField()
    weather_map = {
        Weather.SUNNYDAY: "sunnyday",
        Weather.RAINDANCE: "raindance",
        Weather.SANDSTORM: "sandstorm",
        Weather.SNOW: "snowscape",
        Weather.SNOWSCAPE: "snowscape",
        Weather.HAIL: "hail",
    }
    for w in battle.weather:
        live_field.weather = weather_map.get(w, "")
        break
    terrain_map = {
        Field.ELECTRIC_TERRAIN: "electricterrain",
        Field.GRASSY_TERRAIN: "grassyterrain",
        Field.MISTY_TERRAIN: "mistyterrain",
        Field.PSYCHIC_TERRAIN: "psychicterrain",
    }
    for t in battle.fields:
        if t in terrain_map:
            live_field.terrain = terrain_map[t]
        if t == Field.TRICK_ROOM:
            live_field.trick_room = True

    def _pull_side_conds(side_conds, *, prefix: str) -> None:
        tw = int(side_conds.get(SideCondition.TAILWIND, 0) or 0)
        refl = int(side_conds.get(SideCondition.REFLECT, 0) or 0)
        ls = int(side_conds.get(SideCondition.LIGHT_SCREEN, 0) or 0)
        av = int(side_conds.get(SideCondition.AURORA_VEIL, 0) or 0)
        setattr(live_field, f"tailwind_{prefix}", min(4, tw))
        setattr(live_field, f"reflect_{prefix}", min(5, refl))
        setattr(live_field, f"light_screen_{prefix}", min(5, ls))
        setattr(live_field, f"aurora_veil_{prefix}", min(5, av))

    _pull_side_conds(battle.side_conditions, prefix="p1")
    _pull_side_conds(battle.opponent_side_conditions, prefix="p2")

    tokens[0] = _encode_field_token(live_field, perspective="p1")

    def _live_temporal(mon) -> tuple[int, int, int]:
        if mon is None:
            return 0, 0, 0
        turns_active = int(getattr(mon, "_active_turns", 0) or 0)
        # poke-env increments on |turn|N| before choose_move; parser pre-turn counts
        # completed field turns only (0 on turn 1, 1 on turn 2, …).
        turns_active = max(0, turns_active - 1)
        protect_counter = int(getattr(mon, "protect_counter", 0) or 0)
        last_move = getattr(mon, "last_move", None)
        last_move_id = 0
        if last_move is not None:
            last_move_id = move_vocab_id(to_id_str(getattr(last_move, "id", "") or ""))
        return turns_active, protect_counter, last_move_id

    def _live_mon(
        mon,
        *,
        role: int,
        is_ours: bool,
        active: bool,
        include_temporal: bool = False,
        can_mega: bool = False,
        seen_override: bool | None = None,
    ) -> np.ndarray:
        if mon is None:
            return _encode_mon_token(None, role=role, is_ours=is_ours)
        turns_active, protect_counter, last_move_id = _live_temporal(mon)
        current_hp, max_hp = _live_hp_pair(mon)
        item_revealed = _live_item_revealed(mon, is_ours=is_ours)
        ability_revealed = _live_ability_revealed(mon, is_ours=is_ours)
        item_str = to_id_str(str(mon.item or "")) if item_revealed else ""
        ability_str = to_id_str(str(mon.ability or "")) if ability_revealed else ""
        move_names = _live_moves(mon, is_ours=is_ours)
        if seen_override is not None:
            seen = seen_override
        else:
            seen = True if is_ours else bool(getattr(mon, "revealed", False))
        fake = MonPerspective(
            species=to_id_str(getattr(mon, "species", "") or ""),
            hp=current_hp,
            max_hp=max_hp,
            status=to_id_str(getattr(mon.status, "name", "") if mon.status else ""),
            boosts={k: int(v) for k, v in mon.boosts.items()},
            ability=ability_str,
            item=item_str,
            moves=move_names,
            move_disabled=_live_move_disabled(mon, move_names, is_ours=is_ours),
            mega="mega" in to_id_str(str(mon.species)),
            fainted=bool(mon.fainted),
            active=active,
            seen=seen,
            item_revealed=item_revealed,
            ability_revealed=ability_revealed,
            turns_active=turns_active,
            protect_counter=protect_counter,
            last_move_id=last_move_id,
            can_mega=can_mega if is_ours else False,
        )
        return _encode_mon_token(
            fake, role=role, is_ours=is_ours, include_temporal=include_temporal
        )

    actives = battle.active_pokemon
    opps = battle.opponent_active_pokemon
    tokens[1] = _live_mon(
        actives[0] if len(actives) > 0 else None,
        role=TOKEN_OUR_ACTIVE,
        is_ours=True,
        active=True,
        include_temporal=True,
        can_mega=live_can_mega_for_pos(battle, 0),
    )
    tokens[2] = _live_mon(
        actives[1] if len(actives) > 1 else None,
        role=TOKEN_OUR_ACTIVE,
        is_ours=True,
        active=True,
        include_temporal=True,
        can_mega=live_can_mega_for_pos(battle, 1),
    )
    tokens[3] = _live_mon(
        opps[0] if len(opps) > 0 else None,
        role=TOKEN_OPP_ACTIVE,
        is_ours=False,
        active=True,
        include_temporal=True,
    )
    tokens[4] = _live_mon(
        opps[1] if len(opps) > 1 else None,
        role=TOKEN_OPP_ACTIVE,
        is_ours=False,
        active=True,
        include_temporal=True,
    )

    bench = _live_bench_members(battle.team.values(), is_ours=True, battle=battle)
    opp_bench = _live_bench_members(
        battle.opponent_team.values(), is_ours=False, battle=battle
    )
    for i in range(4):
        tokens[5 + i] = _live_mon(
            bench[i] if i < len(bench) else None,
            role=TOKEN_OUR_BENCH,
            is_ours=True,
            active=False,
        )
        tokens[9 + i] = _live_mon(
            opp_bench[i] if i < len(opp_bench) else None,
            role=TOKEN_OPP_BENCH,
            is_ours=False,
            active=False,
            seen_override=(
                False
                if i < len(opp_bench) and not bool(getattr(opp_bench[i], "revealed", False))
                else None
            ),
        )
    return tokens


def _live_field_from_battle(battle: Battle | DoubleBattle) -> object:
    class _LiveField:
        weather = ""
        terrain = ""
        trick_room = False
        tailwind_p1 = 0
        tailwind_p2 = 0
        reflect_p1 = 0
        reflect_p2 = 0
        light_screen_p1 = 0
        light_screen_p2 = 0
        aurora_veil_p1 = 0
        aurora_veil_p2 = 0

    live_field = _LiveField()
    weather_map = {
        Weather.SUNNYDAY: "sunnyday",
        Weather.RAINDANCE: "raindance",
        Weather.SANDSTORM: "sandstorm",
        Weather.SNOW: "snowscape",
        Weather.SNOWSCAPE: "snowscape",
        Weather.HAIL: "hail",
    }
    for w in battle.weather:
        live_field.weather = weather_map.get(w, "")
        break
    terrain_map = {
        Field.ELECTRIC_TERRAIN: "electricterrain",
        Field.GRASSY_TERRAIN: "grassyterrain",
        Field.MISTY_TERRAIN: "mistyterrain",
        Field.PSYCHIC_TERRAIN: "psychicterrain",
    }
    for t in battle.fields:
        if t in terrain_map:
            live_field.terrain = terrain_map[t]
        if t == Field.TRICK_ROOM:
            live_field.trick_room = True

    def _pull_side_conds(side_conds, *, prefix: str) -> None:
        tw = int(side_conds.get(SideCondition.TAILWIND, 0) or 0)
        refl = int(side_conds.get(SideCondition.REFLECT, 0) or 0)
        ls = int(side_conds.get(SideCondition.LIGHT_SCREEN, 0) or 0)
        av = int(side_conds.get(SideCondition.AURORA_VEIL, 0) or 0)
        setattr(live_field, f"tailwind_{prefix}", min(4, tw))
        setattr(live_field, f"reflect_{prefix}", min(5, refl))
        setattr(live_field, f"light_screen_{prefix}", min(5, ls))
        setattr(live_field, f"aurora_veil_{prefix}", min(5, av))

    _pull_side_conds(battle.side_conditions, prefix="p1")
    _pull_side_conds(battle.opponent_side_conditions, prefix="p2")
    return live_field


def _live_singles_bench(
    team_values, *, is_ours: bool, battle: Battle | None = None
) -> list:
    """
    Singles bench tokens: our brought mons off-field; opponent preview or revealed bench.
    Order matches encode_log_state _bench_slots_ordered (roster / team-list index).
    """
    team_list = list(team_values)
    if is_ours:
        pool = _brought_team_members(
            team_list, battle=battle, max_bring=SINGLES_BRING_COUNT
        )
        roster_keys = [roster_species_key(to_id_str(p.species)) for p in team_list]

        def _roster_rank(p) -> int:
            key = roster_species_key(to_id_str(p.species))
            for i, rk in enumerate(roster_keys):
                if rk == key:
                    return i
            return 99

        bench = [p for p in pool if not p.active and not bool(getattr(p, "fainted", False))]
        bench.sort(key=lambda p: (_roster_rank(p), team_list.index(p)))
        return bench[:SINGLES_BENCH_SLOTS]

    if battle is not None:
        revealed_bench = []
        for p in team_list:
            if p.active:
                continue
            if bool(getattr(p, "revealed", False)):
                revealed_bench.append(p)
        if revealed_bench:
            revealed_bench.sort(key=lambda p: team_list.index(p))
            return revealed_bench[:SINGLES_BENCH_SLOTS]
        leads = _opponent_lead_species_ids(battle)
        return _opp_preview_bench_from_team(
            team_list, lead_species=leads, limit=SINGLES_BENCH_SLOTS
        )

    bench = []
    for p in team_list:
        if p.active:
            continue
        if not bool(getattr(p, "revealed", False)):
            continue
        bench.append(p)
    bench.sort(key=lambda p: team_list.index(p))
    return bench[:SINGLES_BENCH_SLOTS]


def encode_singles_battle(battle: Battle) -> np.ndarray:
    """Encode live poke-env singles Battle into 13 tokens (singles slot layout)."""
    tokens = np.zeros((N_TOKENS, N_FIELDS), dtype=np.int64)
    live_field = _live_field_from_battle(battle)
    tokens[0] = _encode_field_token(
        live_field,
        perspective="p1",
        force_switch=bool(battle.force_switch),
    )

    def _live_temporal(mon) -> tuple[int, int, int]:
        if mon is None:
            return 0, 0, 0
        turns_active = int(getattr(mon, "_active_turns", 0) or 0)
        turns_active = max(0, turns_active - 1)
        protect_counter = int(getattr(mon, "protect_counter", 0) or 0)
        last_move = getattr(mon, "last_move", None)
        last_move_id = 0
        if last_move is not None:
            last_move_id = move_vocab_id(to_id_str(getattr(last_move, "id", "") or ""))
        return turns_active, protect_counter, last_move_id

    def _live_mon_singles(
        mon,
        *,
        role: int,
        is_ours: bool,
        active: bool,
        include_temporal: bool = False,
        seen_override: bool | None = None,
        can_mega: bool = False,
    ) -> np.ndarray:
        if mon is None:
            return empty_slot_token(role)
        turns_active, protect_counter, last_move_id = _live_temporal(mon)
        current_hp, max_hp = _live_hp_pair(mon)
        item_revealed = _live_item_revealed(mon, is_ours=is_ours)
        ability_revealed = _live_ability_revealed(mon, is_ours=is_ours)
        item_str = to_id_str(str(mon.item or "")) if item_revealed else ""
        ability_str = to_id_str(str(mon.ability or "")) if ability_revealed else ""
        move_names = _live_moves(mon, is_ours=is_ours)
        if seen_override is not None:
            seen = seen_override
        else:
            seen = True if is_ours else bool(getattr(mon, "revealed", False))
        fake = MonPerspective(
            species=to_id_str(getattr(mon, "species", "") or ""),
            hp=current_hp,
            max_hp=max_hp,
            status=to_id_str(getattr(mon.status, "name", "") if mon.status else ""),
            boosts={k: int(v) for k, v in mon.boosts.items()},
            ability=ability_str,
            item=item_str,
            moves=move_names,
            move_disabled=_live_move_disabled(mon, move_names, is_ours=is_ours),
            mega="mega" in to_id_str(str(mon.species)),
            fainted=bool(mon.fainted),
            active=active,
            seen=seen,
            item_revealed=item_revealed,
            ability_revealed=ability_revealed,
            turns_active=turns_active,
            protect_counter=protect_counter,
            last_move_id=last_move_id,
            can_mega=can_mega if is_ours else False,
        )
        return _encode_mon_token(
            fake, role=role, is_ours=is_ours, include_temporal=include_temporal
        )

    our_active = battle.active_pokemon
    opp_active = battle.opponent_active_pokemon
    tokens[TOKEN_OUR_ACTIVE] = _live_mon_singles(
        our_active,
        role=TOKEN_OUR_ACTIVE,
        is_ours=True,
        active=True,
        include_temporal=True,
        can_mega=live_can_mega_for_singles(battle),
    )
    tokens[TOKEN_OUR_ACTIVE + 1] = empty_slot_token(TOKEN_OUR_ACTIVE)
    tokens[TOKEN_OPP_ACTIVE] = _live_mon_singles(
        opp_active, role=TOKEN_OPP_ACTIVE, is_ours=False, active=True, include_temporal=True
    )
    tokens[TOKEN_OPP_ACTIVE + 1] = empty_slot_token(TOKEN_OPP_ACTIVE)

    bench = _live_singles_bench(
        battle.team.values(), is_ours=True, battle=battle
    )
    opp_bench = _live_singles_bench(
        battle.opponent_team.values(), is_ours=False, battle=battle
    )
    for i in range(SINGLES_BENCH_SLOTS):
        tokens[TOKEN_OUR_BENCH + i] = _live_mon_singles(
            bench[i] if i < len(bench) else None,
            role=TOKEN_OUR_BENCH,
            is_ours=True,
            active=False,
        )
        tokens[TOKEN_OPP_BENCH + i] = _live_mon_singles(
            opp_bench[i] if i < len(opp_bench) else None,
            role=TOKEN_OPP_BENCH,
            is_ours=False,
            active=False,
            seen_override=(
                False
                if i < len(opp_bench) and not bool(getattr(opp_bench[i], "revealed", False))
                else None
            ),
        )
    for i in range(SINGLES_BENCH_SLOTS, 4):
        tokens[TOKEN_OUR_BENCH + i] = empty_slot_token(TOKEN_OUR_BENCH)
        tokens[TOKEN_OPP_BENCH + i] = empty_slot_token(TOKEN_OPP_BENCH)
    return tokens
