"""
Action output space for BC training (logged decision).

Dual-head training uses per-slot indices 0-106 (poke-env doubles gen 9).
Combo flattening helpers remain for legacy tooling.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from poke_env.battle.move import Move
from poke_env.battle.target import Target
from poke_env.data import to_id_str

ACTION_SIZE = 107  # DoublesEnv.get_action_space_size(9)
COMBO_VOCAB_SIZE = ACTION_SIZE * ACTION_SIZE
ACTION_PASS = 0
# PyTorch CrossEntropyLoss default ignore_index; human selection erased by flinch/OHKO/etc.
ACTION_UNKNOWN = -100

# poke-env DoubleBattle showdown target positions (move_target field).
TARGET_ALLY_SLOT_A = -1  # POKEMON_1_POSITION
TARGET_ALLY_SLOT_B = -2  # POKEMON_2_POSITION
TARGET_DEFAULT = 0  # EMPTY_TARGET_POSITION — self, field, spread
TARGET_OPP_SLOT_A = 1  # OPPONENT_1_POSITION
TARGET_OPP_SLOT_B = 2  # OPPONENT_2_POSITION

VALID_TARGET_OFFSETS = frozenset(
    {
        TARGET_ALLY_SLOT_B,
        TARGET_ALLY_SLOT_A,
        TARGET_DEFAULT,
        TARGET_OPP_SLOT_A,
        TARGET_OPP_SLOT_B,
    }
)

# Moves that never take an explicit foe target in doubles action space.
_SELF_FIELD_TARGETS = frozenset(
    {
        Target.SELF,
        Target.ALL,
        Target.ALL_ADJACENT,
        Target.ALL_ADJACENT_FOES,
        Target.ALLIES,
        Target.ALLY_SIDE,
        Target.ALLY_TEAM,
        Target.FOE_SIDE,
        Target.RANDOM_NORMAL,
        Target.SCRIPTED,
    }
)


def target_offset_label(offset: int) -> str:
    return {
        TARGET_ALLY_SLOT_B: "ally slot B",
        TARGET_ALLY_SLOT_A: "ally slot A",
        TARGET_DEFAULT: "default",
        TARGET_OPP_SLOT_A: "opp slot A",
        TARGET_OPP_SLOT_B: "opp slot B",
    }.get(offset, f"target {offset}")


def _actor_slot(actor_ident: str) -> str:
    return actor_ident.split(":")[0].strip()


def _target_slot(target: str) -> str:
    body = (target or "").strip()
    if not body or body.lower().startswith("[spread"):
        return ""
    return body.split(":")[0].strip()


def _opponent_side(player_side: str) -> str:
    return "p2" if player_side == "p1" else "p1"


def _slot_offset_map(player_side: str) -> dict[str, int]:
    opp = _opponent_side(player_side)
    return {
        f"{player_side}a": TARGET_ALLY_SLOT_A,
        f"{player_side}b": TARGET_ALLY_SLOT_B,
        f"{opp}a": TARGET_OPP_SLOT_A,
        f"{opp}b": TARGET_OPP_SLOT_B,
    }


def move_default_target_offset(move_name: str, *, gen: int = 9) -> int | None:
    """Return DEFAULT offset for self/field/spread moves, else None."""
    try:
        move = Move(to_id_str(move_name), gen=gen)
    except Exception:
        return None
    if move.deduced_target in _SELF_FIELD_TARGETS:
        return TARGET_DEFAULT
    return None


def _parse_cant_tail(parts: list[str]) -> tuple[list[str], str | None]:
    """Split a |cant| line into reason tokens and optional |[of] actor| suffix."""
    if len(parts) < 4:
        return [], None
    rest = parts[3:]
    of_actor: str | None = None
    if rest and rest[-1].startswith("[of] "):
        of_actor = rest[-1][5:].strip()
        rest = rest[:-1]
    return rest, of_actor


def _infer_target_from_failure_cant(
    turn_lines: list[str] | None,
    actor_ident: str,
    move_name: str,
) -> str | None:
    """
    Recover intended target when a move failed before resolving (e.g. Armor Tail).

    Example: |move|p2b: Sableye|Taunt||[still]
             |cant|p1a: Farigiraf|ability: Armor Tail|Taunt|[of] p2b: Sableye
    """
    if not turn_lines:
        return None
    actor_slot = _actor_slot(actor_ident)
    move_id = to_id_str(move_name)
    for line in turn_lines:
        if not line.startswith("|cant|"):
            continue
        parts = line.split("|")
        reason_parts, of_actor = _parse_cant_tail(parts)
        if of_actor is None or _actor_slot(of_actor) != actor_slot:
            continue
        if not reason_parts:
            continue
        cited_move = reason_parts[-1]
        if to_id_str(cited_move) != move_id:
            continue
        candidate = parts[2]
        if _actor_slot(candidate) != actor_slot:
            return candidate
    return None


def _offset_for_target_ident(player_side: str, target_ident: str) -> int | None:
    tslot = _target_slot(target_ident)
    if not tslot:
        return None
    return _slot_offset_map(player_side).get(tslot)


def resolve_log_move_target(
    actor_ident: str,
    move_name: str,
    target: str,
    *,
    gen: int = 9,
    turn_lines: list[str] | None = None,
) -> int | None:
    """
    Map a Showdown log move target to poke-env move_target offset (-2..2).

    Rules (poke-env doubles):
    - Self/field/spread (Protect, Trick Room, Earthquake) -> 0 (label: default)
    - Ally slot a/b -> -1 / -2 (only when log explicitly targets partner)
    - Opponent slot a/b -> 1 / 2
    - Missing target on a foe-selected move -> infer from failure cant, else None
    """
    actor = _actor_slot(actor_ident)
    player_side = actor[:2]
    target_body = (target or "").strip()

    if target_body.lower().startswith("[spread"):
        return TARGET_DEFAULT

    if not target_body:
        forced = move_default_target_offset(move_name, gen=gen)
        if forced is not None:
            return forced
        inferred = _infer_target_from_failure_cant(turn_lines, actor_ident, move_name)
        if inferred is not None:
            return _offset_for_target_ident(player_side, inferred)
        return None

    forced = move_default_target_offset(move_name, gen=gen)
    if forced is not None:
        return forced

    tslot = _target_slot(target_body)
    if not tslot:
        return TARGET_DEFAULT

    # Showdown echoes the user as target for self/field/spread moves only.
    if tslot == actor:
        if move_default_target_offset(move_name, gen=gen) is not None:
            return TARGET_DEFAULT
        return None

    offset = _slot_offset_map(player_side).get(tslot)
    if offset is not None:
        return offset

    return TARGET_DEFAULT


def encode_move_target_offset(target_offset: int) -> int:
    """Embed poke-env target offset into the 5-wide move target band."""
    if target_offset not in VALID_TARGET_OFFSETS:
        raise ValueError(f"Invalid target offset {target_offset}")
    return target_offset + 2


def decode_move_target_offset(encoded: int) -> int:
    return int(encoded) - 2


@dataclass(frozen=True)
class ActionSpaceDecision:
    encoding: str = "dual_slot_heads"
    action_size_per_slot: int = ACTION_SIZE
    combo_vocab_size: int = COMBO_VOCAB_SIZE
    formula: str = "per-slot index 0-106; move base = 7 + 5*(slot-1) + (target_offset+2)"
    decode_slot0: str = "combo_index // ACTION_SIZE"
    decode_slot1: str = "combo_index % ACTION_SIZE"
    inference_masking: str = "enumerate_legal_combos + per-slot DoublesEnv masks"
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def encode_combo(slot0: int, slot1: int) -> int:
    """Flatten slot pair for legacy metadata; UNKNOWN slots count as pass in combo."""
    s0 = ACTION_PASS if int(slot0) == ACTION_UNKNOWN else max(ACTION_PASS, int(slot0))
    s1 = ACTION_PASS if int(slot1) == ACTION_UNKNOWN else max(ACTION_PASS, int(slot1))
    return s0 * ACTION_SIZE + s1


def decode_combo(combo_index: int) -> tuple[int, int]:
    return combo_index // ACTION_SIZE, combo_index % ACTION_SIZE


def log_action_space_decision(path: Path | None = None) -> Path:
    """Write the action-space decision JSON for reproducibility."""
    if path is None:
        path = (
            Path(__file__).resolve().parents[2]
            / "logs"
            / "parser_sanity"
            / "action_space_decision.json"
        )
    decision = ActionSpaceDecision(
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision.to_dict(), indent=2), encoding="utf-8")
    return path
