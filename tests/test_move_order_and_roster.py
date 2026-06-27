"""Roster species tracking and canonical/poke-env action alignment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.doubles.data.action_codec import format_log_action_pair
from src.doubles.data.replay_parser import parse_log_file
from src.core.data.roster_profile import build_match_rosters

def test_rotomwash_roster_and_action_labels():
    path = ROOT / "data/raw_logs/gen9championsvgc2026regma-2622290332.log"
    lines = path.read_text(encoding="utf-8").splitlines()
    rosters = build_match_rosters(lines)
    entry = rosters.p2.get("rotomwash")
    assert entry is not None, "rotomwash roster entry missing"
    assert "hydropump" in entry.moves
    assert "willowisp" in entry.moves

    samples = parse_log_file(path, skip_rating=True, keep_view_state=True)
    sample = next(s for s in samples if s.turn == 2 and s.side == "p2")
    view = sample.view_state
    assert "hydropump" in view.mons["p2b"].moves

    text = format_log_action_pair(view, "p2", sample.action_slot0, sample.action_slot1)
    assert "move2" not in text.lower()
    assert "hydropump" in text.lower()
    assert "opp slot b" in text.lower()


def test_mega_roster_moves_merge_to_base_species():
    path = ROOT / "data/raw_logs/gen9championsvgc2026regma-2623243747.log"
    lines = path.read_text(encoding="utf-8").splitlines()
    rosters = build_match_rosters(lines)
    entry = rosters.p2.get("camerupt")
    assert entry is not None
    assert "eruption" in entry.moves
    assert "protect" in entry.moves

    samples = parse_log_file(path, skip_rating=True, keep_view_state=True)
    sample = next(s for s in samples if s.turn == 4 and s.side == "p2")
    mon = sample.view_state.mons["p2b"]
    assert "eruption" in mon.moves
    text = format_log_action_pair(
        sample.view_state, "p2", sample.action_slot0, sample.action_slot1
    )
    assert "move1" not in text.lower()
    assert "eruption" in text.lower()


def test_faint_replacement_switch_not_labeled_as_turn_decision():
    path = ROOT / "data/raw_logs/gen9championsvgc2026regma-2621223148.log"
    samples = parse_log_file(path, skip_rating=True, keep_view_state=True)
    turn_sample = next(
        s for s in samples if s.turn == 3 and s.side == "p2" and s.sample_kind == "turn"
    )
    text = format_log_action_pair(
        turn_sample.view_state, "p2", turn_sample.action_slot0, turn_sample.action_slot1
    )
    assert "switch -> tyranitar" not in text.lower()
    assert "machpunch" in text.lower() or "mach punch" in text.lower()
    assert "switch -> starmie" in text.lower()

    fs = next(
        s for s in samples if s.turn == 3 and s.side == "p2" and s.sample_kind == "force_switch"
    )
    assert fs.tokens.shape == (39, 24)
    fs_text = format_log_action_pair(
        fs.view_state, "p2", fs.action_slot0, fs.action_slot1
    )
    assert "switch -> tyranitar" in fs_text.lower()
    assert "unknown" in fs_text.lower()


def test_trajectory_stacking_shape():
    path = ROOT / "data/raw_logs/gen9championsvgc2026regma-2622290332.log"
    samples = parse_log_file(path, skip_rating=True)
    assert samples
    for sample in samples:
        assert sample.tokens.shape == (39, 24), sample.tokens.shape


if __name__ == "__main__":
    test_rotomwash_roster_and_action_labels()
    test_faint_replacement_switch_not_labeled_as_turn_decision()
    test_trajectory_stacking_shape()
    test_mega_roster_moves_merge_to_base_species()
    print("move_order_and_roster tests passed.")
