"""N_FIELDS layout: status/boosts must not be overwritten by moves."""

from __future__ import annotations

import numpy as np

from src.core.data.log_tracker import BattleLogState, LogStateTracker, project_first_person
from src.core.data.perspective import MonPerspective, boost_id, status_id
from src.doubles.data.replay_parser import parse_log_lines
from src.core.data.roster_profile import build_match_rosters
from src.core.data.state_tokenizer import (
    FIELD_ABILITY,
    FIELD_BOOST_START,
    FIELD_FLAGS,
    FIELD_HP,
    FIELD_ITEM,
    FIELD_MOVE_DISABLED_START,
    FIELD_MOVE_START,
    FIELD_SPECIES,
    FIELD_STATUS,
    FIELD_LAST_MOVE_ID,
    FIELD_REFLECT_OURS,
    FIELD_LIGHT_SCREEN_OPP,
    FIELD_TAILWIND_OURS,
    FIELD_TAILWIND_OPP,
    FLAG_CAN_MEGA,
    N_FIELDS,
    N_TOKENS,
    TOKEN_OPP_BENCH,
    encode_log_state,
)


def test_mon_token_preserves_status_and_boosts_with_four_moves():
    from src.core.data.state_tokenizer import _encode_mon_token

    mon = MonPerspective(
        species="garchomp",
        hp=80,
        max_hp=100,
        status="brn",
        boosts={"atk": -1, "def": 2, "spe": 1},
        ability="roughskin",
        ability_revealed=True,
        item="sitrusberry",
        item_revealed=True,
        moves=["earthquake", "rockslide", "protect", "dragonclaw"],
        active=True,
        seen=True,
    )
    tok = _encode_mon_token(mon, role=1, is_ours=True, include_temporal=True)

    assert tok[FIELD_STATUS] == status_id("brn")
    assert tok[FIELD_BOOST_START] == boost_id(-1)
    assert tok[FIELD_BOOST_START + 1] == boost_id(2)
    assert tok[FIELD_BOOST_START + 4] == boost_id(1)
    assert tok[FIELD_MOVE_START] != 0
    assert tok[FIELD_MOVE_START + 1] != 0
    assert tok[FIELD_MOVE_START + 2] != 0
    assert tok[FIELD_MOVE_START + 3] != 0
    assert tok[FIELD_SPECIES] != tok[FIELD_ABILITY]
    assert tok[FIELD_HP] == int(mon.hp_fraction * 20)
    assert tok[FIELD_ITEM] == tok[FIELD_ITEM]  # populated


def test_tensor_shape_is_13_by_n_fields():
    state = BattleLogState()
    state.mons["p1a"] = MonPerspective(species="pikachu", hp=100, max_hp=100, active=True, seen=True)
    tokens = encode_log_state(state, "p1")
    assert tokens.shape == (N_TOKENS, N_FIELDS)


def test_opp_preview_bench_species_only_unseen():
    log = """|poke|p1|A| 
|poke|p1|B| 
|poke|p1|C| 
|poke|p1|D| 
|poke|p1|E| 
|poke|p1|F| 
|poke|p2|G| 
|poke|p2|H| 
|poke|p2|I| 
|poke|p2|J| 
|poke|p2|K| 
|poke|p2|L| 
|start|
|switch|p1a: A|A|100/100
|switch|p1b: B|B|100/100
|switch|p2a: G|G|100/100
|switch|p2b: H|H|100/100
|turn|1
"""
    lines = parse_log_lines(log)
    rosters = build_match_rosters(lines)
    tracker = LogStateTracker()
    for line in lines:
        if line.startswith("|turn|1"):
            pre = tracker.state.clone()
            break
        tracker.process_line(line)
    view = project_first_person(pre, "p1", rosters=rosters)
    tokens = encode_log_state(view, "p1")
    assert tokens[TOKEN_OPP_BENCH, FIELD_SPECIES] != 0
    assert tokens[TOKEN_OPP_BENCH + 1, FIELD_SPECIES] != 0
    assert tokens[TOKEN_OPP_BENCH, FIELD_FLAGS] == 0
    assert tokens[TOKEN_OPP_BENCH, FIELD_MOVE_START] == 0


def test_field_token_side_conditions():
    state = BattleLogState()
    state.field.tailwind_p1 = 3
    state.field.tailwind_p2 = 2
    state.field.reflect_p1 = 4
    state.field.light_screen_p2 = 5
    tokens = encode_log_state(state, "p1")
    field = tokens[0]
    assert field[FIELD_TAILWIND_OURS] == 3
    assert field[FIELD_TAILWIND_OPP] == 2
    assert field[FIELD_REFLECT_OURS] == 4
    assert field[FIELD_LIGHT_SCREEN_OPP] == 5


def test_last_move_id_written_when_temporal():
    from src.core.data.perspective import move_vocab_id
    from src.core.data.state_tokenizer import _encode_mon_token

    mon = MonPerspective(
        species="garchomp",
        hp=100,
        max_hp=100,
        active=True,
        seen=True,
        moves=["earthquake", "protect", "rockslide", "dragonclaw"],
        last_move_id=move_vocab_id("earthquake"),
    )
    tok = _encode_mon_token(mon, role=1, is_ours=True, include_temporal=True)
    assert tok[FIELD_LAST_MOVE_ID] == move_vocab_id("earthquake")
    tok_no = _encode_mon_token(mon, role=1, is_ours=True, include_temporal=False)
    assert tok_no[FIELD_LAST_MOVE_ID] == 0


def test_can_mega_in_flags_not_separate_field():
    mon = MonPerspective(
        species="garchomp",
        hp=100,
        max_hp=100,
        active=True,
        seen=True,
        can_mega=True,
        moves=["earthquake", "protect", "rockslide", "dragonclaw"],
    )
    state = BattleLogState(mons={"p1a": mon})
    tokens = encode_log_state(state, "p1")
    assert tokens[1, FIELD_FLAGS] & FLAG_CAN_MEGA
