"""Bench token population and brought-only switch masks."""

from __future__ import annotations

from src.doubles.data.log_action_mask import _legal_switch_indices
from src.core.data.log_tracker import LogStateTracker, project_first_person
from src.doubles.data.replay_parser import parse_log_lines
from src.core.data.roster_profile import build_match_rosters
from src.core.data.state_tokenizer import TOKEN_OUR_BENCH, FIELD_SPECIES, encode_log_state


SAMPLE_LOG = """|poke|p1|Starmie, L50|
|poke|p1|Pelipper, L50, F|
|poke|p1|Farigiraf, L50, F|
|poke|p1|Sneasler, L50, M|
|poke|p1|Mamoswine, L50, F|
|poke|p1|Milotic, L50, F|
|poke|p2|Feraligatr, L50, M|
|poke|p2|Ninetales-Alola, L50, M|
|poke|p2|Scizor, L50, M|
|poke|p2|Sinistcha, L50|
|poke|p2|Hydreigon, L50, F|
|poke|p2|Rotom-Heat, L50|
|teampreview|4
|teamsize|p1|4
|teamsize|p2|4
|start
|switch|p1a: Sneasler|Sneasler, L50, M|100/100
|switch|p1b: Starmie|Starmie, L50|100/100
|switch|p2a: Hydreigon|Hydreigon, L50, F|100/100
|switch|p2b: Ninetales|Ninetales-Alola, L50, M|100/100
|turn|1
|switch|p1a: Pelipper|Pelipper, L50, F|100/100
|turn|2
|switch|p1b: Milotic|Milotic, L50, F|100/100
|turn|3
"""


def _tracker_at_turn(lines: list[str], turn: int):
    tracker = LogStateTracker()
    for line in lines:
        if line.startswith("|turn|"):
            parts = line.split("|")
            current = int(parts[2])
            if current == turn:
                return tracker.state.clone()
        tracker.process_line(line)
    return tracker.state


def test_turn1_our_bench_populated_from_roster():
    lines = parse_log_lines(SAMPLE_LOG)
    rosters = build_match_rosters(lines)
    pre = _tracker_at_turn(lines, 1)
    view = project_first_person(pre, "p1", rosters=rosters)
    tokens = encode_log_state(view, "p1")
    assert tokens[TOKEN_OUR_BENCH, FIELD_SPECIES] != 0
    assert tokens[TOKEN_OUR_BENCH + 1, FIELD_SPECIES] != 0
    assert tokens[TOKEN_OUR_BENCH + 2, 1] == 0


def test_midgame_switch_moves_outgoing_to_bench():
    lines = parse_log_lines(SAMPLE_LOG)
    rosters = build_match_rosters(lines)
    tracker = LogStateTracker()
    for line in lines:
        tracker.process_line(line)
    view = project_first_person(tracker.state, "p1", rosters=rosters)
    bench_species = {
        view.mons[s].species
        for s in view.mons
        if s.startswith("p1") and not view.mons[s].active and view.mons[s].species
    }
    assert "sneasler" in bench_species
    assert "pelipper" in {view.mons["p1a"].species, view.mons["p1b"].species}


def test_switch_mask_excludes_unbrought_roster():
    lines = parse_log_lines(SAMPLE_LOG)
    rosters = build_match_rosters(lines)
    pre = _tracker_at_turn(lines, 1)
    view = project_first_person(pre, "p1", rosters=rosters)
    legal = _legal_switch_indices(view, "p1")
    roster = view.team_roster["p1"]
    brought = view.brought_species["p1"]
    for idx in legal:
        assert roster_species_key(roster[idx - 1]) in brought


from src.core.data.roster_profile import roster_species_key
