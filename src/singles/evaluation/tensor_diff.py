"""Diff stacked BC tensors (live vs parser) with field-level diagnostics."""

from __future__ import annotations

import numpy as np

from src.core.data.state_tokenizer import (
    FIELD_ABILITY,
    FIELD_BOOST_START,
    FIELD_DECISION_FLAGS,
    FIELD_FLAGS,
    FIELD_HP,
    FIELD_ITEM,
    FIELD_LAST_MOVE_ID,
    FIELD_MOVE_DISABLED_START,
    FIELD_MOVE_START,
    FIELD_PROTECT_COUNTER,
    FIELD_ROLE,
    FIELD_SPECIES,
    FIELD_STATUS,
    FIELD_TURNS_ACTIVE,
    N_FIELDS,
    N_TOKENS,
    STACKED_N_TOKENS,
    TOKEN_ROLE_NAMES,
    TRAJECTORY_DEPTH,
)

_FIELD_LABELS: dict[int, str] = {
    FIELD_ROLE: "role",
    FIELD_SPECIES: "species",
    FIELD_ABILITY: "ability",
    FIELD_ITEM: "item",
    FIELD_HP: "hp",
    FIELD_STATUS: "status",
    **{FIELD_BOOST_START + i: f"boost_{s}" for i, s in enumerate(
        ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
    )},
    FIELD_DECISION_FLAGS: "decision_flags",
    **{FIELD_MOVE_START + i: f"move_slot_{i}" for i in range(4)},
    **{FIELD_MOVE_DISABLED_START + i: f"move_disabled_{i}" for i in range(4)},
    FIELD_FLAGS: "flags",
    FIELD_TURNS_ACTIVE: "turns_active",
    FIELD_PROTECT_COUNTER: "protect_counter",
    FIELD_LAST_MOVE_ID: "last_move_id",
}


def _field_label(field_idx: int) -> str:
    if field_idx in _FIELD_LABELS:
        return _FIELD_LABELS[field_idx]
    if 1 <= field_idx <= 12:
        return f"field_env_{field_idx}"
    return f"field_{field_idx}"


def _token_label(token_idx: int) -> str:
    role = token_idx % N_TOKENS
    return TOKEN_ROLE_NAMES.get(role, f"token_{role}")


def format_tensor_diff(
    live: np.ndarray,
    bc: np.ndarray,
    *,
    max_lines: int = 40,
) -> str:
    """
    Subtract bc - live and list non-zero (frame, token, field) cells.

    Stacked layout: (TRAJECTORY_DEPTH * N_TOKENS, N_FIELDS) with frames
    oldest-first (t-2, t-1, t-0).
    """
    live_a = np.asarray(live, dtype=np.int64).reshape(-1, N_FIELDS)
    bc_a = np.asarray(bc, dtype=np.int64).reshape(-1, N_FIELDS)
    if live_a.shape != bc_a.shape:
        return (
            f"shape mismatch: live={live_a.shape} bc={bc_a.shape} "
            f"(expected ({STACKED_N_TOKENS}, {N_FIELDS}))"
        )

    diff = bc_a - live_a
    nz = np.argwhere(diff != 0)
    if nz.size == 0:
        return "tensors identical (all fields match)"

    lines = [f"non_zero_cells={len(nz)} (showing up to {max_lines}):"]
    move_hits = 0
    bench_hits = 0
    species_hits = 0
    item_ability_hits = 0

    for row_i, field_i in nz[:max_lines]:
        frame = row_i // N_TOKENS
        token = row_i % N_TOKENS
        t_label = f"t-{TRAJECTORY_DEPTH - 1 - frame}"
        tok_label = _token_label(token)
        fld_label = _field_label(int(field_i))
        lines.append(
            f"  {t_label} token={token} ({tok_label}) field={field_i} ({fld_label}): "
            f"live={int(live_a[row_i, field_i])} bc={int(bc_a[row_i, field_i])} "
            f"delta={int(diff[row_i, field_i])}"
        )
        fi = int(field_i)
        if FIELD_MOVE_START <= fi < FIELD_MOVE_START + 4:
            move_hits += 1
        if token in (5, 6, 9, 10):
            bench_hits += 1
        if fi == FIELD_SPECIES:
            species_hits += 1
        if fi in (FIELD_ABILITY, FIELD_ITEM):
            item_ability_hits += 1

    hints: list[str] = []
    if move_hits:
        hints.append("fields 13-16 (move hashes) differ — check canonical_move_list ordering")
    if item_ability_hits:
        hints.append("ability/item (fields 2-3) differ — reveal flags or normalization")
    if bench_hits:
        hints.append("tokens 5-6 or 9-10 (bench) differ — check Bring-3 bench ordering")
    if species_hits:
        hints.append("species (field 1) differs — roster_species_key / forme normalization")
    if hints:
        lines.append("hints: " + "; ".join(dict.fromkeys(hints)))

    if len(nz) > max_lines:
        lines.append(f"  ... +{len(nz) - max_lines} more")
    return "\n".join(lines)
