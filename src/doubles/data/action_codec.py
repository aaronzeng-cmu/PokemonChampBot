"""Map Showdown log actions to poke-env per-slot and combo indices."""

from __future__ import annotations

from poke_env.data import to_id_str

from src.doubles.battle.move_order import (
    canonical_move_list,
    decode_move_action_index,
    encode_move_action_index,
)
from src.doubles.data.action_space_spec import (
    ACTION_PASS,
    ACTION_UNKNOWN,
    decode_combo,
    encode_combo,
    resolve_log_move_target,
    target_offset_label,
)
from src.doubles.data.illusion_guiderail import guiderail_move_encoding_species
from src.core.data.log_tracker import BattleLogState, _species_name

# poke-env doubles move target offsets: -2..2 encoded in slots 7-106

# Cant reasons where the slot could not act and the selection is hidden -> UNKNOWN.
_CANT_HIDDEN_SELECTION = frozenset({"flinch", "slp", "par", "frz"})

# Cant reasons where no new selection was required this turn -> legal pass.
_CANT_LEGITIMATE_PASS = frozenset({"recharge"})


def _roster_species_id(species_details: str) -> str:
    """Normalize a switch |details| string to a roster species id."""
    raw = _species_name(species_details)
    # Meganium-Mega -> meganium so we match |poke| roster entries.
    base = raw.split("-")[0]
    return to_id_str(base)


def _team_index(state: BattleLogState, side: str, species_details: str) -> int:
    roster = state.team_roster.get(side, [])
    sid = _roster_species_id(species_details)
    for i, sp in enumerate(roster):
        if to_id_str(sp) == sid:
            return i + 1
    return 1


def encode_log_move(
    state: BattleLogState,
    actor: str,
    move_name: str,
    target: str,
    *,
    mega: bool = False,
    terastallize: bool = False,
    turn_lines: list[str] | None = None,
) -> int:
    slot = actor.split(":")[0]
    mon = state.mons.get(slot)
    side = slot[:2]
    _ = guiderail_move_encoding_species(
        mon,
        move_name,
        side=side,
        team_roster=state.team_roster.get(side, []),
        turn_lines=turn_lines,
        actor_slot=slot,
    )
    known = list(mon.moves) if mon and mon.moves else []
    moves = canonical_move_list(known + [move_name])
    offset = resolve_log_move_target(
        actor, move_name, target, turn_lines=turn_lines
    )
    if offset is None:
        return ACTION_UNKNOWN
    return encode_move_action_index(
        moves,
        move_name,
        offset,
        mega=mega,
        terastallize=terastallize,
    )


def encode_log_switch(state: BattleLogState, actor: str, species: str) -> int:
    slot = actor.split(":")[0]
    side = slot[:2]
    return _team_index(state, side, species)


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


def _slot_key(ident: str) -> str:
    return ident.split(":")[0].strip()


def _is_active_slot_at_turn_start(view: BattleLogState, side: str, suffix: str) -> bool:
    """True if this doubles slot had a living Pokémon when the turn began."""
    mon = view.mons.get(f"{side}{suffix}")
    return mon is not None and not mon.fainted


def _cant_lines_for_slot(lines: list[str], slot: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        if not line.startswith("|cant|"):
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        if _slot_key(parts[2]) != slot:
            continue
        rows.append(parts)
    return rows


def _cant_is_opponent_side_effect(parts: list[str]) -> bool:
    """Armor Tail etc.: cant names our mon but documents a foe move failing."""
    from src.doubles.data.action_space_spec import _parse_cant_tail

    reason_parts, of_actor = _parse_cant_tail(parts)
    if of_actor is not None and reason_parts and reason_parts[0].startswith("ability:"):
        return True
    return False


def _try_encode_cant_revealed_move(
    parts: list[str],
    view: BattleLogState,
    *,
    lines: list[str],
    slot: str,
) -> int | None:
    if len(parts) < 4:
        return None
    reason = parts[3]
    move_name: str | None = None
    if reason == "Disable" and len(parts) >= 5:
        move_name = parts[4]
    elif reason.startswith("move:"):
        # move: Taunt | Yawn  OR  move: Throat Chop (silenced — no revealed selection)
        if len(parts) >= 5 and not parts[4].startswith("["):
            move_name = parts[4]
    if not move_name:
        return None
    actor = parts[2]
    return encode_log_move(
        view,
        actor,
        move_name,
        "",
        mega=_mega_for_actor(lines, slot),
        terastallize=_tera_for_actor(lines, slot),
        turn_lines=lines,
    )


def _slot_fainted_before_own_action(lines: list[str], slot: str) -> bool:
    acted = False
    for line in lines:
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        cmd = parts[1]
        if cmd in ("move", "switch") and len(parts) >= 3:
            if _slot_key(parts[2]) == slot:
                acted = True
                break
        if cmd == "faint" and len(parts) >= 3:
            if _slot_key(parts[2]) == slot:
                return not acted
    return False


def _resolve_missing_slot_action(
    slot: str,
    lines: list[str],
    view: BattleLogState,
    side: str,
    suffix: str,
) -> int:
    """
    Label a slot with no |move|/|switch| line.

    - Empty / fainted at turn start -> legal pass (0)
    - Recharge cant -> pass
    - Cant naming the attempted move (Disable/Taunt) -> encode that move
    - Flinch, sleep, para, freeze, OHKO-before-move -> UNKNOWN (-100)
    """
    if not _is_active_slot_at_turn_start(view, side, suffix):
        return ACTION_PASS

    for parts in _cant_lines_for_slot(lines, slot):
        if _cant_is_opponent_side_effect(parts):
            continue
        reason = parts[3]
        if reason in _CANT_LEGITIMATE_PASS:
            return ACTION_PASS
        revealed = _try_encode_cant_revealed_move(parts, view, lines=lines, slot=slot)
        if revealed is not None:
            return revealed
        if reason in _CANT_HIDDEN_SELECTION:
            return ACTION_UNKNOWN
        if reason.startswith("move:"):
            # Encore/Taunt/Throat Chop silence without naming the player's pick.
            return ACTION_UNKNOWN

    if _slot_fainted_before_own_action(lines, slot):
        return ACTION_UNKNOWN

    return ACTION_UNKNOWN


def _is_pivot_switch(parts: list[str]) -> bool:
    """Switch caused by U-turn / Parting Shot etc., not a turn decision."""
    return any(p.strip().startswith("[from]") for p in parts[4:])


def _slot_fainted_before_line(lines: list[str], slot: str, before_idx: int) -> bool:
    for line in lines[:before_idx]:
        if not line.startswith("|faint|"):
            continue
        faint_parts = line.split("|")
        if len(faint_parts) >= 3 and _slot_key(faint_parts[2]) == slot:
            return True
    return False


def _should_record_turn_switch(
    lines: list[str],
    *,
    slot: str,
    parts: list[str],
    line_idx: int,
    actions: dict[str, int],
) -> bool:
    """True only for voluntary switch-ins selected at turn start (not pivot/faint)."""
    if _is_pivot_switch(parts):
        return False
    # Move already recorded — any later switch is a faint replacement.
    if slot in actions and actions[slot] >= 7:
        return False
    if slot in actions and actions[slot] <= 6:
        return False
    # OHKO / faint replacement before the switch line — not a human pick.
    if _slot_fainted_before_line(lines, slot, line_idx):
        return False
    return True


def _record_turn_switch(
    lines: list[str],
    *,
    state: BattleLogState,
    actions: dict[str, int],
    line_idx: int,
    parts: list[str],
) -> None:
    actor, species = parts[2], parts[3]
    slot = _slot_key(actor)
    if not _should_record_turn_switch(
        lines, slot=slot, parts=parts, line_idx=line_idx, actions=actions
    ):
        return
    actions[slot] = encode_log_switch(state, actor, species)


def is_force_switch_decision(
    parts: list[str],
    lines: list[str],
    line_idx: int,
) -> bool:
    """
    True when a |switch| line is a mid-turn replacement decision (pivot or faint).
    Distinct from voluntary turn-start switches in parse_side_turn_actions.
    """
    if len(parts) < 4 or parts[1] != "switch":
        return False
    if _is_pivot_switch(parts):
        return True
    slot = _slot_key(parts[2])
    return _slot_fainted_before_line(lines, slot, line_idx)


def encode_force_switch_pair(
    view: BattleLogState,
    side: str,
    switching_suffix: str,
    switch_idx: int,
) -> tuple[int, int]:
    """One slot switches; the other is ACTION_UNKNOWN (-100)."""
    if switching_suffix == "a":
        return switch_idx, ACTION_UNKNOWN
    return ACTION_UNKNOWN, switch_idx


def parse_force_switch_actions(
    parts: list[str],
    view: BattleLogState,
    side: str,
) -> tuple[int, int] | None:
    """Label a single forced |switch| line for one side's perspective."""
    actor, species = parts[2], parts[3]
    slot = _slot_key(actor)
    if not slot.startswith(side):
        return None
    suffix = slot[-1]
    if suffix not in ("a", "b"):
        return None
    switch_idx = encode_log_switch(view, actor, species)
    return encode_force_switch_pair(view, side, suffix, switch_idx)


def parse_turn_actions(
    lines: list[str],
    state: BattleLogState,
) -> dict[str, int]:
    """Extract per-slot actions using the first-person view state for encoding."""
    actions: dict[str, int] = {}
    for line_idx, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        cmd = parts[1]
        if cmd == "move" and len(parts) >= 5:
            actor, move, target = parts[2], parts[3], parts[4]
            slot = actor.split(":")[0]
            actions[slot] = encode_log_move(
                state,
                actor,
                move,
                target,
                mega=_mega_for_actor(lines, slot),
                terastallize=_tera_for_actor(lines, slot),
                turn_lines=lines,
            )
        elif cmd == "switch" and len(parts) >= 4:
            _record_turn_switch(
                lines, state=state, actions=actions, line_idx=line_idx, parts=parts
            )
    return actions


def parse_side_turn_actions(
    lines: list[str],
    view: BattleLogState,
    side: str,
) -> tuple[int, int]:
    """Parse one side's slot actions; missing erased selections -> ACTION_UNKNOWN."""
    actions: dict[str, int] = {}
    for line_idx, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        cmd = parts[1]
        if cmd == "move" and len(parts) >= 5:
            actor, move, target = parts[2], parts[3], parts[4]
            slot = actor.split(":")[0]
            if not slot.startswith(side):
                continue
            actions[slot] = encode_log_move(
                view,
                actor,
                move,
                target,
                mega=_mega_for_actor(lines, slot),
                terastallize=_tera_for_actor(lines, slot),
                turn_lines=lines,
            )
        elif cmd == "switch" and len(parts) >= 4:
            actor = parts[2]
            slot = actor.split(":")[0]
            if not slot.startswith(side):
                continue
            _record_turn_switch(
                lines, state=view, actions=actions, line_idx=line_idx, parts=parts
            )

    resolved: dict[str, int] = {}
    for suffix in ("a", "b"):
        slot = f"{side}{suffix}"
        if slot in actions:
            resolved[slot] = actions[slot]
        else:
            resolved[slot] = _resolve_missing_slot_action(slot, lines, view, side, suffix)
    return side_slot_actions(resolved, side)


def side_slot_actions(actions: dict[str, int], side: str) -> tuple[int, int]:
    return actions.get(f"{side}a", ACTION_PASS), actions.get(f"{side}b", ACTION_PASS)


def side_combo_action(actions: dict[str, int], side: str) -> int:
    a0, a1 = side_slot_actions(actions, side)
    return encode_combo(a0, a1)


def describe_action(slot0: int, slot1: int) -> dict:
    return {
        "slot0_index": slot0,
        "slot1_index": slot1,
        "combo_index": encode_combo(slot0, slot1),
        "combo_decode": decode_combo(encode_combo(slot0, slot1)),
    }


def decode_log_slot_action(
    state: BattleLogState,
    side: str,
    slot_suffix: str,
    action_idx: int,
) -> str:
    """Human-readable action for one active slot (from log state)."""
    if action_idx == ACTION_UNKNOWN:
        return "UNKNOWN (erased selection)"
    if action_idx == ACTION_PASS:
        return "pass"

    actor = f"{side}{slot_suffix}"
    mon = state.mons.get(actor)

    if 1 <= action_idx <= 6:
        roster = state.team_roster.get(side, [])
        idx = action_idx - 1
        species = roster[idx] if idx < len(roster) else f"bench-{action_idx}"
        return f"switch -> {species}"

    if action_idx < 7:
        return f"unknown action {action_idx}"

    move_slot, target_offset, mega, tera = decode_move_action_index(action_idx)
    moves = canonical_move_list(mon.moves if mon and mon.moves else [])
    move_name = moves[move_slot - 1] if move_slot - 1 < len(moves) else f"move{move_slot}"
    actor_name = mon.species if mon and mon.species else actor
    target = target_offset_label(target_offset)
    flags = []
    if mega:
        flags.append("mega")
    if tera:
        flags.append("tera")
    flag_text = f" ({', '.join(flags)})" if flags else ""
    return f"{actor_name}: {move_name} -> {target}{flag_text}"


def format_log_action_pair(
    state: BattleLogState,
    side: str,
    slot0: int,
    slot1: int,
) -> str:
    """Format both active-slot actions as one line."""
    a = decode_log_slot_action(state, side, "a", slot0)
    b = decode_log_slot_action(state, side, "b", slot1)
    return f"[{a}] | [{b}]"
