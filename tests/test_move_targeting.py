"""Regression tests for move target resolution (Liquidation, spread moves, etc.)."""

from __future__ import annotations

from src.doubles.battle.move_order import encode_move_action_index
from src.doubles.data.action_space_spec import (
    TARGET_DEFAULT,
    TARGET_OPP_SLOT_A,
    TARGET_OPP_SLOT_B,
    move_default_target_offset,
    resolve_log_move_target,
)
from src.doubles.data.log_action_mask import log_turn_slot_mask
from src.core.data.log_tracker import BattleLogState
from src.core.data.perspective import MonPerspective
from src.doubles.data.action_codec import decode_log_slot_action


def _mon(slot: str, species: str, moves: list[str]) -> MonPerspective:
    return MonPerspective(
        slot=slot,
        species=species,
        hp=100,
        max_hp=100,
        active=True,
        seen=True,
        moves=list(moves),
    )


def test_liquidation_is_not_forced_default():
    assert move_default_target_offset("liquidation") is None


def test_liquidation_resolves_foe_target():
    off = resolve_log_move_target(
        "p2a: Araquanid",
        "Liquidation",
        "p1b: Kingambit",
    )
    assert off == TARGET_OPP_SLOT_B


def test_liquidation_empty_target_returns_none_not_default():
    off = resolve_log_move_target("p2a: Araquanid", "Liquidation", "")
    assert off is None


def test_liquidation_self_echo_returns_none_not_default():
    """Foe-targeting moves must not be labeled default when log echoes self."""
    off = resolve_log_move_target(
        "p2a: Araquanid",
        "Liquidation",
        "p2a: Araquanid",
    )
    assert off is None


def test_protect_self_echo_returns_default():
    off = resolve_log_move_target(
        "p1a: Basculegion",
        "Protect",
        "p1a: Basculegion",
    )
    assert off == TARGET_DEFAULT


def test_earthquake_spread_is_default():
    assert move_default_target_offset("earthquake") == TARGET_DEFAULT


def test_encode_unknown_move_uses_correct_slot_not_last_slot_fallback():
    """Avoid mapping Liquidation to Protect's slot when move was missing from list."""
    moves = ["aquajet", "flipturn", "lastrespects", "protect"]
    idx = encode_move_action_index(moves, "liquidation", TARGET_OPP_SLOT_B)
    assert idx == encode_move_action_index(
        moves + ["liquidation"], "liquidation", TARGET_OPP_SLOT_B
    )


def test_liquidation_default_not_legal_in_log_mask():
    state = BattleLogState()
    state.mons["p2a"] = _mon("p2a", "araquanid", ["liquidation", "protect", "wideguard"])
    mask = log_turn_slot_mask(state, "p2", "a")
    default_idx = encode_move_action_index(
        ["liquidation", "protect", "wideguard"],
        "liquidation",
        TARGET_DEFAULT,
    )
    opp_b_idx = encode_move_action_index(
        ["liquidation", "protect", "wideguard"],
        "liquidation",
        TARGET_OPP_SLOT_B,
    )
    assert not mask[default_idx]
    assert mask[opp_b_idx]


def test_basculegion_turn3_replay_labels_protect_not_liquidation():
    from pathlib import Path
    from src.doubles.data.replay_parser import parse_log_file, find_sample_view_state

    rid = "gen9championsvgc2026regma-2619782791"
    sample = next(
        s
        for s in parse_log_file(Path("data/raw_logs") / f"{rid}.log")
        if s.turn == 3 and s.side == "p1"
    )
    view = find_sample_view_state(
        Path("data/raw_logs"),
        replay_id=rid,
        turn=3,
        side="p1",
    )
    label = decode_log_slot_action(view, "p1", "a", sample.action_slot0)
    assert label == "basculegion: protect -> default"
    assert "liquidation" not in label
