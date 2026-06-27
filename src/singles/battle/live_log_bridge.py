"""Build BC-compatible singles log views from live poke-env battles + protocol."""

from __future__ import annotations

import json
import random

import numpy as np
from poke_env.battle.battle import Battle
from poke_env.data import to_id_str

from src.core.data.log_tracker import BattleLogState, LogStateTracker, project_first_person
from src.core.data.move_utils import canonical_move_list
from src.core.data.perspective import stable_seed_int
from src.core.data.roster_profile import build_match_rosters, roster_species_key
from src.core.data.state_tokenizer import encode_log_state
from src.singles.log_action_codec import (
    ACTION_UNKNOWN,
    is_force_switch_decision,
    is_force_switch_request_line,
    parse_singles_turn_action,
)
from src.singles.meta_database import load_meta_database
from src.singles.teampreview import parse_team_command

_META_DB_CACHE: object | None = None


def _meta_db(meta_db=None):
    global _META_DB_CACHE
    if meta_db is not None:
        return meta_db
    if _META_DB_CACHE is None:
        _META_DB_CACHE = load_meta_database(format="singles", live_fetch=False)
    return _META_DB_CACHE


def teampreview_protocol_line(side: str, command: str) -> str:
    """Synthetic protocol line so build_match_rosters knows our Bring-3."""
    slots = parse_team_command(command)
    return f"|useteam|{side}|{''.join(str(s) for s in slots)}"


def rosters_through_turn(lines: list[str], turn: int) -> "MatchRosters":
    """
    Causal rosters for a turn-start decision: only log lines through |turn|N|,
    plus any |useteam| line (live teampreview) when present.
    """
    useteam = next((ln for ln in lines if ln.startswith("|useteam|")), None)
    end = len(lines)
    for i, line in enumerate(lines):
        if line.startswith(f"|turn|{turn}"):
            end = i + 1
            break
    subset = list(lines[:end])
    if useteam is not None and useteam not in subset:
        subset.append(useteam)
    return build_match_rosters(subset)


def _project_view(
    state: BattleLogState,
    *,
    side: str,
    turn: int,
    sample_kind: str,
    replay_id: str,
    rosters,
    meta_db,
    deterministic_moves: bool = False,
) -> BattleLogState:
    rng = None if deterministic_moves else random.Random(
        stable_seed_int(replay_id, turn, side, sample_kind, "singles")
    )
    return project_first_person(
        state,
        side,
        rosters=rosters,
        meta_db=meta_db,
        rng=rng,
        format="singles",
        deterministic_moves=deterministic_moves,
    )


def protocol_through_flush_turn(
    protocol: list[str],
    flushed_turn: int,
    protocol_len: int | None = None,
) -> list[str]:
    """Protocol prefix through the last line of ``flushed_turn`` (before ``|turn|N+1|``)."""
    end = min(int(protocol_len if protocol_len is not None else len(protocol)), len(protocol))
    subset = protocol[:end]
    cap = len(subset)
    for i, line in enumerate(subset):
        if line.startswith(f"|turn|{flushed_turn + 1}"):
            cap = i
            break
    return subset[:cap]


def view_at_turn_flush(
    protocol: list[str],
    *,
    turn: int,
    side: str,
    replay_id: str,
    meta_db=None,
    protocol_len: int | None = None,
) -> BattleLogState | None:
    """Turn-start view encoded into trajectory history on ``|turn|N+1|`` flush."""
    if turn < 1:
        return None
    meta_db = _meta_db(meta_db)
    capped = protocol_through_flush_turn(protocol, turn, protocol_len)
    return replay_view_at_turn_start(
        capped,
        side=side,
        turn=turn,
        replay_id=replay_id,
        meta_db=meta_db,
        deterministic_moves=True,
    )


def snapshot_for_turn_flush(
    protocol: list[str],
    *,
    turn: int,
    side: str,
    replay_id: str,
    meta_db=None,
    protocol_len: int | None = None,
) -> np.ndarray | None:
    """
    Trajectory frame pushed when ``|turn|N+1|`` arrives (turn-start snapshot for turn N).

    Single source of truth for parser ``_flush_turn`` and live trajectory flush.
    """
    view = view_at_turn_flush(
        protocol,
        turn=turn,
        side=side,
        replay_id=replay_id,
        meta_db=meta_db,
        protocol_len=protocol_len,
    )
    if view is None:
        return None
    return encode_log_state(view, side, format="singles", force_switch=False)


def apply_active_request_overlay(
    view: BattleLogState,
    turn_lines: list[str],
    side: str,
) -> None:
    """Overlay decision-time active slot from the turn's voluntary ``|request|``."""
    mon = view.mons.get(f"{side}a")
    if mon is None:
        return
    for line in turn_lines:
        if not line.startswith("|request|"):
            continue
        try:
            data = json.loads(line.split("|", 2)[2])
        except (IndexError, json.JSONDecodeError):
            continue
        if data.get("forceSwitch") or data.get("wait"):
            continue
        active = data.get("active")
        if not active:
            continue
        slot = active[0]
        moves = [
            to_id_str(m.get("id") or m.get("move", ""))
            for m in slot.get("moves", [])
        ]
        moves = [m for m in moves if m]
        if moves:
            mon.moves = canonical_move_list(moves)
        if bool(slot.get("canMegaEvo", False)):
            mon.can_mega = True
            mon.mega_capable = True
        return


# Back-compat alias
apply_active_request_moves = apply_active_request_overlay


def encode_champchoice_protocol_line(
    battle: Battle,
    action_idx: int,
    *,
    side: str = "p1",
) -> str | None:
    """Synthetic line recording the bot's submitted choice (including unexecuted picks)."""
    from src.singles.action_space_spec import decode_singles_action_index
    from src.singles.battle.canonical_inference import canonical_moves_for_active
    from src.singles.bench_slots import live_our_bench_mons
    from src.singles.log_action_codec import SWITCH_BASE

    spec = decode_singles_action_index(action_idx)
    if spec.is_switch:
        bench_idx = spec.index - SWITCH_BASE
        bench = live_our_bench_mons(battle)
        if bench_idx < 0 or bench_idx >= len(bench):
            return None
        species = to_id_str(bench[bench_idx].species)
        return f"|champchoice|{side}|switch|{species}"
    if spec.move_slot is None:
        return None
    moves = canonical_moves_for_active(battle)
    if spec.move_slot < 0 or spec.move_slot >= len(moves):
        return None
    return f"|champchoice|{side}|move|{moves[spec.move_slot]}"


def replay_view_at_turn_start(
    lines: list[str],
    *,
    side: str,
    turn: int,
    replay_id: str,
    meta_db,
    deterministic_moves: bool = False,
) -> BattleLogState | None:
    """Mirror singles replay_parser pre_turn_state at |turn|N| (before turn lines)."""
    rosters = rosters_through_turn(lines, turn)
    tracker = LogStateTracker()
    current_turn = 0
    view_state: BattleLogState | None = None

    for line in lines:
        if line.startswith("|turn|"):
            parts = line.split("|")
            if len(parts) >= 3:
                current_turn = int(parts[2])
            tracker.process_line(line)
            if current_turn == turn:
                view_state = tracker.state.clone()
                break
        else:
            tracker.process_line(line)

    if view_state is None:
        return None
    view = _project_view(
        view_state,
        side=side,
        turn=turn,
        sample_kind="turn",
        replay_id=replay_id,
        rosters=rosters,
        meta_db=meta_db,
        deterministic_moves=deterministic_moves,
    )
    apply_active_request_overlay(view, lines_for_turn(lines, turn), side)
    return view


def singles_mask_for_live_decision(
    battle: Battle,
    protocol_lines: list[str],
    *,
    side: str = "p1",
    meta_db=None,
) -> np.ndarray:
    """BC-parser-aligned action mask for live inference (same path as audit)."""
    from src.core.model.transformer_bot import SINGLES_ACTION_SIZE
    from src.singles.battle.live_legality import build_singles_action_mask
    from src.singles.log_action_mask import singles_mask_for_eval

    meta_db = _meta_db(meta_db)
    turn = int(battle.turn)
    replay_id = battle.battle_tag
    force = bool(getattr(battle, "force_switch", False))
    sample_kind = "force_switch" if force else "turn"

    if force:
        view = replay_view_at_force_switch(
            protocol_lines,
            side=side,
            turn=turn,
            replay_id=replay_id,
            meta_db=meta_db,
        )
    else:
        view = replay_view_at_turn_start(
            protocol_lines,
            side=side,
            turn=turn,
            replay_id=replay_id,
            meta_db=meta_db,
            deterministic_moves=False,
        )

    mask = singles_mask_for_eval(view, side=side, sample_kind=sample_kind)
    if mask is not None:
        return mask
    return np.array(build_singles_action_mask(battle, size=SINGLES_ACTION_SIZE), dtype=bool)


def lines_for_turn(protocol: list[str], turn: int) -> list[str]:
    """Protocol lines after ``|turn|N|`` until the next ``|turn|``."""
    in_turn = False
    turn_lines: list[str] = []
    for line in protocol:
        if line.startswith(f"|turn|{turn}"):
            in_turn = True
            continue
        if in_turn:
            if line.startswith("|turn|"):
                break
            turn_lines.append(line)
    return turn_lines


def turn_trajectory_should_push(
    protocol: list[str],
    *,
    turn: int,
    side: str,
    replay_id: str,
    meta_db=None,
    protocol_len: int | None = None,
) -> bool:
    """True when replay_parser would push trajectory on flush for this turn."""
    if turn < 1:
        return False
    meta_db = _meta_db(meta_db)
    capped = protocol_through_flush_turn(
        protocol,
        turn,
        protocol_len if protocol_len is not None else len(protocol),
    )
    view = replay_view_at_turn_start(
        capped,
        side=side,
        turn=turn,
        replay_id=replay_id,
        meta_db=meta_db,
        deterministic_moves=False,
    )
    if view is None:
        return False
    turn_lines = lines_for_turn(capped, turn)
    if not turn_lines:
        return False
    action = parse_singles_turn_action(turn_lines, view, side)
    return action != ACTION_UNKNOWN


def replay_view_at_force_switch(
    lines: list[str],
    *,
    side: str,
    turn: int,
    replay_id: str,
    meta_db,
) -> BattleLogState | None:
    """Pre-switch view for the next forced switch on this turn (singles parser)."""
    tracker = LogStateTracker()
    current_turn = 0
    turn_lines: list[str] = []
    view_state: BattleLogState | None = None
    lines_so_far: list[str] = []

    for line in lines:
        lines_so_far.append(line)
        if line.startswith("|turn|"):
            parts = line.split("|")
            if len(parts) >= 3:
                current_turn = int(parts[2])
            turn_lines = []
            tracker.process_line(line)
            continue

        if current_turn >= 1:
            turn_lines.append(line)
            if current_turn == turn and is_force_switch_request_line(
                line, turn_lines, len(turn_lines) - 1, side=side
            ):
                tracker.process_line(line)
                pre_switch = tracker.state.clone()
                rosters = rosters_through_turn(lines_so_far, turn)
                view_state = _project_view(
                    pre_switch,
                    side=side,
                    turn=turn,
                    sample_kind="force_switch",
                    replay_id=replay_id,
                    rosters=rosters,
                    meta_db=meta_db,
                )
                break

            parts = line.split("|")
            if len(parts) >= 2 and current_turn == turn:
                if is_force_switch_decision(parts, turn_lines, len(turn_lines) - 1):
                    pre_switch = tracker.state.clone()
                    rosters = rosters_through_turn(lines_so_far, turn)
                    view_state = _project_view(
                        pre_switch,
                        side=side,
                        turn=turn,
                        sample_kind="force_switch",
                        replay_id=replay_id,
                        rosters=rosters,
                        meta_db=meta_db,
                    )
                    break
        tracker.process_line(line)

    return view_state


def encode_live_as_log(
    battle: Battle,
    *,
    protocol_lines: list[str],
    side: str = "p1",
    meta_db=None,
) -> tuple[np.ndarray, BattleLogState, str] | None:
    """
    Encode the current live decision with encode_log_state (BC / parser path).

    Returns (snapshot, view, sample_kind) or None when the view cannot be built.
    """
    if not protocol_lines:
        return None
    if meta_db is None:
        meta_db = _meta_db()

    turn = int(battle.turn)
    replay_id = battle.battle_tag
    force = bool(getattr(battle, "force_switch", False))
    sample_kind = "force_switch" if force else "turn"

    if force:
        view = replay_view_at_force_switch(
            protocol_lines,
            side=side,
            turn=turn,
            replay_id=replay_id,
            meta_db=meta_db,
        )
    else:
        view = replay_view_at_turn_start(
            protocol_lines,
            side=side,
            turn=turn,
            replay_id=replay_id,
            meta_db=meta_db,
        )

    if view is None:
        return None

    snap = encode_log_state(
        view,
        side,
        format="singles",
        force_switch=force,
    )
    return snap, view, sample_kind


def canonical_moves_for_battle_team(battle: Battle) -> list[str]:
    """Alphabetized move ids for the active Pokémon (paste order independent)."""
    mon = battle.active_pokemon
    if mon is None:
        return []
    raw = [to_id_str(m.id) for m in mon.moves.values() if m]
    if not raw:
        raw = [to_id_str(m.id) for m in battle.available_moves]
    return canonical_move_list(raw)
