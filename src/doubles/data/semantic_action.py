"""Semantic action heads: move-id / switch / pass + target + modifier."""

from __future__ import annotations

import json
from dataclasses import dataclass

from poke_env.data import to_id_str

from src.doubles.battle.move_order import decode_move_action_index
from src.doubles.data.action_space_spec import (
    ACTION_PASS,
    ACTION_UNKNOWN,
    TARGET_ALLY_SLOT_A,
    TARGET_ALLY_SLOT_B,
    TARGET_DEFAULT,
    TARGET_OPP_SLOT_A,
    TARGET_OPP_SLOT_B,
)

SEMANTIC_TARGET_DEFAULT = 0
SEMANTIC_TARGET_OPP_A = 1
SEMANTIC_TARGET_OPP_B = 2
SEMANTIC_TARGET_ALLY = 3
NUM_SEMANTIC_TARGETS = 4

SEMANTIC_MODIFIER_NORMAL = 0
SEMANTIC_MODIFIER_GIMMICK = 1
NUM_SEMANTIC_MODIFIERS = 2

PASS_TOKEN = "pass"
SWITCH_PREFIX = "switch_"
MOVE_ACTION_START = 7


@dataclass
class ActionVocabulary:
    """Maps pass, switch_1..switch_6, and Showdown move ids to head_action indices."""

    token_to_id: dict[str, int]
    id_to_token: dict[int, str]

    @classmethod
    def create(cls) -> ActionVocabulary:
        token_to_id = {PASS_TOKEN: ACTION_PASS}
        for i in range(1, 7):
            token_to_id[f"{SWITCH_PREFIX}{i}"] = i
        id_to_token = {v: k for k, v in token_to_id.items()}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    @property
    def vocab_size(self) -> int:
        return max(self.id_to_token) + 1 if self.id_to_token else MOVE_ACTION_START

    def ensure_move(self, move_id: str) -> int:
        mid = to_id_str(move_id)
        if not mid:
            return ACTION_PASS
        if mid in self.token_to_id:
            return self.token_to_id[mid]
        idx = max(self.id_to_token) + 1 if self.id_to_token else MOVE_ACTION_START
        if idx < MOVE_ACTION_START:
            idx = MOVE_ACTION_START
        self.token_to_id[mid] = idx
        self.id_to_token[idx] = mid
        return idx

    def token_for_id(self, action_id: int) -> str:
        return self.id_to_token.get(action_id, "")

    def is_move_action(self, action_id: int) -> bool:
        return action_id >= MOVE_ACTION_START

    def is_switch_action(self, action_id: int) -> bool:
        return 1 <= action_id <= 6

    def switch_index(self, action_id: int) -> int:
        return action_id

    def to_dict(self) -> dict:
        return {"token_to_id": self.token_to_id, "id_to_token": {str(k): v for k, v in self.id_to_token.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> ActionVocabulary:
        id_to_token = {int(k): v for k, v in data.get("id_to_token", {}).items()}
        token_to_id = dict(data.get("token_to_id", {}))
        if not token_to_id and id_to_token:
            token_to_id = {v: k for k, v in id_to_token.items()}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    def save_json(self, path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path) -> ActionVocabulary:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def log_offset_to_semantic_target(offset: int) -> int:
    if offset == TARGET_DEFAULT:
        return SEMANTIC_TARGET_DEFAULT
    if offset == TARGET_OPP_SLOT_A:
        return SEMANTIC_TARGET_OPP_A
    if offset == TARGET_OPP_SLOT_B:
        return SEMANTIC_TARGET_OPP_B
    if offset in (TARGET_ALLY_SLOT_A, TARGET_ALLY_SLOT_B):
        return SEMANTIC_TARGET_ALLY
    return SEMANTIC_TARGET_DEFAULT


def semantic_target_to_log_offsets(target_id: int) -> list[int]:
    if target_id == SEMANTIC_TARGET_DEFAULT:
        return [TARGET_DEFAULT]
    if target_id == SEMANTIC_TARGET_OPP_A:
        return [TARGET_OPP_SLOT_A]
    if target_id == SEMANTIC_TARGET_OPP_B:
        return [TARGET_OPP_SLOT_B]
    if target_id == SEMANTIC_TARGET_ALLY:
        return [TARGET_ALLY_SLOT_A, TARGET_ALLY_SLOT_B]
    return [TARGET_DEFAULT]


def flat_action_to_semantic(
    flat_idx: int,
    *,
    vocab: ActionVocabulary,
    move_name: str | None = None,
    target_offset: int | None = None,
    mega: bool = False,
    terastallize: bool = False,
) -> tuple[int, int, int]:
    """Convert legacy flat 0-106 label to (action_id, target_id, modifier_id)."""
    if flat_idx == ACTION_UNKNOWN:
        return ACTION_UNKNOWN, ACTION_UNKNOWN, ACTION_UNKNOWN
    if flat_idx == ACTION_PASS:
        return ACTION_PASS, ACTION_UNKNOWN, ACTION_UNKNOWN
    if 1 <= flat_idx <= 6:
        return flat_idx, ACTION_UNKNOWN, ACTION_UNKNOWN

    if move_name is None or target_offset is None:
        move_slot, target_offset, mega, terastallize = decode_move_action_index(flat_idx)
        if move_name is None:
            move_name = f"move{move_slot}"

    action_id = vocab.ensure_move(move_name)
    target_id = log_offset_to_semantic_target(target_offset)
    modifier_id = SEMANTIC_MODIFIER_GIMMICK if (mega or terastallize) else SEMANTIC_MODIFIER_NORMAL
    return action_id, target_id, modifier_id


def semantic_labels_for_slot(
    flat_idx: int,
    *,
    vocab: ActionVocabulary,
    view,
    side: str,
    suffix: str,
) -> tuple[int, int, int]:
    if flat_idx == ACTION_UNKNOWN:
        return ACTION_UNKNOWN, ACTION_UNKNOWN, ACTION_UNKNOWN
    if flat_idx == ACTION_PASS:
        return ACTION_PASS, ACTION_UNKNOWN, ACTION_UNKNOWN
    if 1 <= flat_idx <= 6:
        return flat_idx, ACTION_UNKNOWN, ACTION_UNKNOWN

    move_slot, target_offset, mega, tera = decode_move_action_index(flat_idx)
    mon = view.mons.get(f"{side}{suffix}") if view is not None else None
    moves = list(mon.moves) if mon and mon.moves else []
    move_name = moves[move_slot - 1] if 0 < move_slot <= len(moves) else f"move{move_slot}"
    return flat_action_to_semantic(
        flat_idx,
        vocab=vocab,
        move_name=move_name,
        target_offset=target_offset,
        mega=mega,
        terastallize=tera,
    )
