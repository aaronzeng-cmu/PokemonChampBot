"""Shared first-person knowledge helpers for log and live battle tracking."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from poke_env.data import to_id_str

UNK = "<UNK>"

# Must match VGCBehaviorClonerConfig.vocab_size / Embedding num_embeddings.
HASH_MOD = 4096


def stable_hash(text: str, vocab_size: int = HASH_MOD) -> int:
    """Deterministic string -> [0, vocab_size) for embedding lookup (MD5)."""
    if not text:
        return 0
    hex_hash = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(hex_hash, 16) % vocab_size


def stable_seed_int(*parts: object) -> int:
    """Deterministic 32-bit seed for RNG (meta-imputation, etc.)."""
    text = "|".join(str(p) for p in parts)
    hex_hash = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(hex_hash[:8], 16)


@dataclass
class MonPerspective:
    slot: str = ""
    species: str = ""
    hp: int = 0
    max_hp: int = 0
    status: str = ""
    boosts: dict[str, int] = field(default_factory=dict)
    ability: str = ""
    item: str = ""
    moves: list[str] = field(default_factory=list)
    move_disabled: list[bool] = field(default_factory=list)
    tera_type: str = ""
    terastallized: bool = False
    mega: bool = False
    mega_capable: bool = False
    can_mega: bool = False
    illusion_disguise: str = ""
    illusion_broken: bool = False
    fainted: bool = False
    active: bool = False
    # Species visible on field (switch-in / active), not full knowledge.
    seen: bool = False
    item_revealed: bool = False
    ability_revealed: bool = False
    team_index: int = 0
    # Temporal memory (active slots only in tensor; tracked for all mons in logs).
    turns_active: int = 0
    protect_counter: int = 0
    last_move_id: int = 0
    # Per-turn scratch (committed on turn boundary).
    _turn_move_id: int = 0
    _turn_protect_success: bool = False

    @property
    def hp_fraction(self) -> float:
        if self.max_hp <= 0:
            return 0.0
        return max(0.0, min(1.0, self.hp / self.max_hp))

    def visible_item(self) -> str:
        return self.item if self.item_revealed else ""

    def visible_ability(self) -> str:
        return self.ability if self.ability_revealed else ""

    def visible_moves(self) -> list[str]:
        return list(self.moves)


def move_vocab_id(move_name: str) -> int:
    """Vocabulary index for last-move encoding; 0 = none / unknown."""
    return hash_token(move_name)


def hash_token(value: str, mod: int = HASH_MOD) -> int:
    """Normalize Showdown id then map to embedding index (stable across processes)."""
    return stable_hash(to_id_str(value), vocab_size=mod)


def status_id(status: str) -> int:
    mapping = {"": 0, "brn": 1, "par": 2, "slp": 3, "frz": 4, "psn": 5, "tox": 6}
    return mapping.get(to_id_str(status), 0)


def boost_id(stage: int) -> int:
    return int(max(-6, min(6, stage)) + 6)


def apply_reveal_move(mon: MonPerspective, move_id: str) -> None:
    """Record a revealed move; keep slots in canonical alphabetical order."""
    from src.core.data.move_utils import canonical_move_list

    move_id = to_id_str(move_id)
    if move_id and move_id not in mon.moves:
        mon.moves.append(move_id)
    mon.moves = canonical_move_list(mon.moves)


def apply_first_person_view(mon: MonPerspective, *, is_ours: bool) -> MonPerspective:
    """Return a copy with hidden opponent fields stripped."""
    import copy

    m = copy.deepcopy(mon)
    if is_ours:
        return m
    if not m.seen:
        m.ability = ""
        m.item = ""
        m.moves = []
        return m
    if not m.ability_revealed:
        m.ability = ""
    if not m.item_revealed:
        m.item = ""
    return m
