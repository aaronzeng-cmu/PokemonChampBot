"""Map Showdown singles log lines to poke-env action indices (0-21)."""

from __future__ import annotations

import json

from poke_env.data import to_id_str

from src.core.data.log_tracker import BattleLogState
from src.core.data.move_utils import canonical_move_list, moves_for_action_encoding
from src.core.data.roster_profile import roster_species_key
from src.singles.bench_slots import (
    bench_switch_index_to_species_log,
    log_our_bench_slots,
    species_to_bench_switch_index,
)
from src.core.model.transformer_bot import SINGLES_ACTION_SIZE

ACTION_UNKNOWN = -100
SINGLES_SWITCH_SLOTS = 2
SWITCH_BASE = 0
MOVE_BASE = 2
MEGA_BASE = 6
DMAX_TERA_BASE = 14

_CANT_HIDDEN_SELECTION = frozenset({"flinch", "slp", "par", "frz"})
_CANT_LEGITIMATE_PASS = frozenset({"recharge"})

PIVOT_MOVE_IDS = frozenset({
    "uturn",
    "voltswitch",
    "flipturn",
    "partingshot",
    "batonpass",
    "shedtail",
    "chillyreception",
})


def _slot_key(ident: str) -> str:
    return ident.split(":")[0].strip()


def _roster_species_id(species_details: str) -> str:
    from src.core.data.log_tracker import _species_name

    raw = _species_name(species_details)
    base = raw.split("-")[0]
    return to_id_str(base)


def encode_singles_log_switch(state: BattleLogState, actor: str, species: str) -> int:
    side = _slot_key(actor)[:2]
    bench_idx = species_to_bench_switch_index(state, side, species)
    if bench_idx is None:
        return ACTION_UNKNOWN
    return SWITCH_BASE + bench_idx


def encode_singles_log_move(
    state: BattleLogState,
    actor: str,
    move_name: str,
    *,
    mega: bool = False,
    terastallize: bool = False,
) -> int:
    slot = _slot_key(actor)
    mon = state.mons.get(slot)
    known = list(mon.moves) if mon and mon.moves else []
    moves = moves_for_action_encoding(known, move_name)
    move_id = to_id_str(move_name)
    try:
        move_slot = moves.index(move_id)
    except ValueError:
        return ACTION_UNKNOWN
    if terastallize:
        return DMAX_TERA_BASE + move_slot
    if mega:
        return MEGA_BASE + move_slot
    return MOVE_BASE + move_slot


def _mega_for_actor(lines: list[str], actor_slot: str) -> bool:
    for line in lines:
        if line.startswith("|-mega|") and actor_slot in line:
            return True
        if line.startswith("|detailschange|") and actor_slot in line and "mega" in line.lower():
            return True
    return False


def _tera_for_actor(lines: list[str], actor_slot: str) -> bool:
    for line in lines:
        if line.startswith("|-terastallize|") and actor_slot in line:
            return True
    return False


def _is_pivot_switch(parts: list[str]) -> bool:
    return any(p.strip().startswith("[from]") for p in parts[4:])


def _prior_force_switch_request(lines: list[str], before_idx: int, *, side: str = "p1") -> bool:
    for j in range(before_idx):
        if is_force_switch_request_line(lines[j], lines, j, side=side):
            return True
    return False


def next_force_switch_switch_parts(
    lines: list[str],
    after_idx: int,
    *,
    side: str,
) -> list[str] | None:
    """Next pivot or faint replacement ``|switch|`` for ``side`` after ``after_idx``."""
    for j in range(after_idx + 1, len(lines)):
        parts = lines[j].split("|")
        if len(parts) < 4 or parts[1] != "switch":
            continue
        if not _slot_key(parts[2]).startswith(side):
            continue
        if is_force_switch_decision(parts, lines, j):
            return parts
    return None


def _slot_fainted_before_line(lines: list[str], slot: str, before_idx: int) -> bool:
    for line in lines[:before_idx]:
        if not line.startswith("|faint|"):
            continue
        parts = line.split("|")
        if len(parts) >= 3 and _slot_key(parts[2]) == slot:
            return True
    return False


def _active_at_turn_start(view: BattleLogState, side: str) -> bool:
    mon = view.mons.get(f"{side}a")
    return mon is not None and not mon.fainted


def _cant_lines_for_slot(lines: list[str], slot: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        if not line.startswith("|cant|"):
            continue
        parts = line.split("|")
        if len(parts) >= 4 and _slot_key(parts[2]) == slot:
            rows.append(parts)
    return rows


def _slot_fainted_before_own_action(lines: list[str], slot: str) -> bool:
    """True when the slot fainted before recording a |move| or voluntary |switch|."""
    for line_idx, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        cmd = parts[1]
        if cmd == "move" and len(parts) >= 3 and _slot_key(parts[2]) == slot:
            return False
        if cmd == "switch" and len(parts) >= 3 and _slot_key(parts[2]) == slot:
            if not _is_pivot_switch(parts) and not _slot_fainted_before_line(
                lines, slot, line_idx
            ):
                return False
        if cmd == "faint" and len(parts) >= 3 and _slot_key(parts[2]) == slot:
            return True
    return False


def _parse_champchoice_line(
    line: str,
    view: BattleLogState,
    side: str,
) -> int | None:
    """Parse ``|champchoice|`` lines logged by the live bot (submitted but not executed)."""
    if not line.startswith("|champchoice|"):
        return None
    parts = line.split("|")
    if len(parts) < 5 or parts[2] != side:
        return None
    kind, payload = parts[3], parts[4]
    actor = f"{side}a: x"
    if kind == "move":
        encoded = encode_singles_log_move(view, actor, payload)
        return encoded if encoded != ACTION_UNKNOWN else None
    if kind == "switch":
        bench_idx = species_to_bench_switch_index(view, side, payload)
        if bench_idx is None:
            return None
        return SWITCH_BASE + bench_idx
    return None


def _resolve_missing_action(slot: str, lines: list[str], view: BattleLogState, side: str) -> int:
    if not _active_at_turn_start(view, side):
        return ACTION_UNKNOWN

    for line in lines:
        choice = _parse_champchoice_line(line, view, side)
        if choice is not None:
            return choice

    for parts in _cant_lines_for_slot(lines, slot):
        reason = parts[3]
        if reason in _CANT_LEGITIMATE_PASS:
            return ACTION_UNKNOWN
        if reason in _CANT_HIDDEN_SELECTION:
            return ACTION_UNKNOWN
        if reason.startswith("move:"):
            return ACTION_UNKNOWN

    if _slot_fainted_before_own_action(lines, slot):
        return ACTION_UNKNOWN

    return ACTION_UNKNOWN


def _should_record_turn_switch(
    lines: list[str],
    *,
    slot: str,
    parts: list[str],
    line_idx: int,
    actions: dict[str, int],
) -> bool:
    if _is_pivot_switch(parts):
        return False
    if slot in actions:
        return False
    if _slot_fainted_before_line(lines, slot, line_idx):
        return False
    return True


def _force_switch_request_for_encoding(
    data: dict,
    lines: list[str],
    line_idx: int,
    *,
    side: str = "p1",
) -> bool:
    """Post-faint or pivot (U-turn) switch prompt — not a voluntary move request."""
    if not data.get("forceSwitch"):
        return False
    return not data.get("active")


def _true_force_switch_request(
    data: dict,
    lines: list[str],
    line_idx: int,
    *,
    side: str = "p1",
) -> bool:
    """Forced switch after a faint — not a pivot (U-turn) replacement prompt."""
    if not _force_switch_request_for_encoding(data, lines, line_idx, side=side):
        return False
    slot = f"{side}a"
    return _slot_fainted_before_line(lines, slot, line_idx)


def is_force_switch_request_line(
    line: str,
    turn_lines: list[str] | None = None,
    line_idx: int | None = None,
    *,
    side: str = "p1",
) -> bool:
    """True when a ``|request|`` JSON payload is a post-faint forced-switch prompt."""
    if not line.startswith("|request|"):
        return False
    try:
        payload = line.split("|", 2)[2]
        data = json.loads(payload)
    except (IndexError, json.JSONDecodeError):
        return False
    if turn_lines is None or line_idx is None:
        return _force_switch_request_for_encoding(data, [], 0, side=side)
    return _force_switch_request_for_encoding(data, turn_lines, line_idx, side=side)


def is_force_switch_decision(parts: list[str], lines: list[str], line_idx: int) -> bool:
    if len(parts) >= 3 and parts[1] == "request":
        try:
            data = json.loads(parts[2])
        except json.JSONDecodeError:
            return False
        return _force_switch_request_for_encoding(data, lines, line_idx)
    if len(parts) < 4 or parts[1] != "switch":
        return False
    if _is_pivot_switch(parts):
        return True
    slot = _slot_key(parts[2])
    return _slot_fainted_before_line(lines, slot, line_idx)


def parse_singles_turn_action(
    lines: list[str],
    view: BattleLogState,
    side: str,
) -> int:
    slot = f"{side}a"
    actions: dict[str, int] = {}
    for line_idx, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        cmd = parts[1]
        if cmd == "champchoice":
            choice = _parse_champchoice_line(line, view, side)
            if choice is not None and choice != ACTION_UNKNOWN:
                actions[slot] = choice
        elif cmd == "move" and len(parts) >= 5:
            actor, move = parts[2], parts[3]
            if not _slot_key(actor).startswith(side):
                continue
            encoded = encode_singles_log_move(
                view,
                actor,
                move,
                mega=_mega_for_actor(lines, _slot_key(actor)),
                terastallize=_tera_for_actor(lines, _slot_key(actor)),
            )
            if encoded != ACTION_UNKNOWN:
                actions[_slot_key(actor)] = encoded
        elif cmd == "switch" and len(parts) >= 4:
            actor = parts[2]
            if not _slot_key(actor).startswith(side):
                continue
            if _should_record_turn_switch(
                lines,
                slot=_slot_key(actor),
                parts=parts,
                line_idx=line_idx,
                actions=actions,
            ):
                encoded = encode_singles_log_switch(view, actor, parts[3])
                if encoded != ACTION_UNKNOWN:
                    actions[_slot_key(actor)] = encoded

    if slot in actions:
        return actions[slot]
    return _resolve_missing_action(slot, lines, view, side)


def parse_singles_force_switch(
    parts: list[str],
    view: BattleLogState,
    side: str,
) -> int | None:
    actor, species = parts[2], parts[3]
    slot = _slot_key(actor)
    if not slot.startswith(side):
        return None
    return encode_singles_log_switch(view, actor, species)


def decode_singles_log_action(
    state: BattleLogState,
    side: str,
    action_idx: int,
) -> str:
    """Human-readable singles action from a log view state."""
    if action_idx == ACTION_UNKNOWN:
        return "UNKNOWN (erased selection)"

    slot = f"{side}a"
    mon = state.mons.get(slot)

    if 0 <= action_idx < MOVE_BASE:
        bench_idx = action_idx - SWITCH_BASE
        species = bench_switch_index_to_species_log(state, side, bench_idx)
        return f"switch -> {species} (bench {bench_idx + 1})"

    move_slot = None
    flags: list[str] = []
    if MOVE_BASE <= action_idx < MEGA_BASE:
        move_slot = action_idx - MOVE_BASE
    elif MEGA_BASE <= action_idx < DMAX_TERA_BASE:
        move_slot = action_idx - MEGA_BASE
        flags.append("mega")
    elif DMAX_TERA_BASE <= action_idx < SINGLES_ACTION_SIZE:
        move_slot = action_idx - DMAX_TERA_BASE
        flags.append("tera")

    if move_slot is None:
        return f"unknown action {action_idx}"

    moves = canonical_move_list(mon.moves if mon and mon.moves else [])
    move_name = moves[move_slot] if move_slot < len(moves) else f"move{move_slot}"
    actor_name = mon.species if mon and mon.species else slot
    flag_text = f" ({', '.join(flags)})" if flags else ""
    return f"{actor_name}: {move_name}{flag_text}"


def format_singles_log_action(
    state: BattleLogState,
    side: str,
    action_idx: int,
) -> str:
    return decode_singles_log_action(state, side, action_idx)


def training_action_mask(
    view: BattleLogState,
    side: str,
    *,
    ground_truth: int,
    sample_kind: str = "turn",
) -> list[bool]:
    """Legal-action mask for BC training (full log mask, not GT-only)."""
    from src.singles.log_action_mask import training_singles_mask

    mask = training_singles_mask(
        view,
        side,
        sample_kind,
        ground_truth=ground_truth,
    )
    return mask.tolist()
