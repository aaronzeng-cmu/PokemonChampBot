"""Tests for can_mega feature extraction and mega action masking."""

from __future__ import annotations

import numpy as np

from src.doubles.battle.move_order import is_mega_action, mask_mega_actions, pokeenv_action_mask_to_canonical
from src.doubles.data.log_action_mask import log_turn_slot_mask
from src.core.data.log_tracker import BattleLogState, LogStateTracker, project_first_person
from src.core.data.mega_items import is_mega_stone_item
from src.core.data.perspective import MonPerspective
from src.core.data.roster_profile import build_match_rosters
from src.core.data.state_tokenizer import FIELD_BOOST_START, FIELD_FLAGS, FIELD_HP, FLAG_CAN_MEGA, N_FIELDS, encode_log_state


def test_is_mega_stone_item_suffixes():
    assert is_mega_stone_item("garchompite")
    assert is_mega_stone_item("mawilite")
    assert is_mega_stone_item("abomasnowite")
    assert not is_mega_stone_item("choiceband")
    assert not is_mega_stone_item("")


def test_roster_species_key_normalizes_eternal_and_mega_formes():
    from src.core.data.roster_profile import roster_species_key

    assert roster_species_key("floetteeternal") == "floette"
    assert roster_species_key("Floette-Eternal, L50, F") == "floette"
    assert roster_species_key("floettemega") == "floette"
    assert roster_species_key("garchompmega") == "garchomp"
    assert roster_species_key("rotomwash") == "rotomwash"
    assert roster_species_key("Meowstic-M-Mega, L50, M") == "meowstic"
    assert roster_species_key("meowsticm") == "meowstic"
    assert roster_species_key("meowsticf") == "meowstic"


def test_meowstic_gender_mega_capable_aligns_with_roster():
    lines = [
        "|poke|p1|Meowstic, L50, M|",
        "|poke|p2|Aegislash, L50, M|",
        "|turn|1",
        "|switch|p1a: Meowstic|Meowstic, L50, M|100/100",
        "|switch|p2a: Aegislash|Aegislash, L50, M|100/100",
        "|-mega|p1a: Meowstic|Meowstic|Meowsticite",
        "|detailschange|p1a: Meowstic|Meowstic-M-Mega, L50, M|100/100",
    ]
    tracker = LogStateTracker()
    for line in lines:
        tracker.process_line(line)
    rosters = build_match_rosters(lines)
    view = project_first_person(tracker.state, "p1", rosters=rosters, format="singles")
    mon = view.mons["p1a"]
    assert mon.mega_capable
    assert mon.mega


def test_floette_eternal_mega_capable_aligns_with_roster():
    lines = [
        "|poke|p1|Floette-Eternal, L50, F|",
        "|poke|p2|Aegislash, L50, M|",
        "|turn|1",
        "|switch|p1a: Floette|Floette-Eternal, L50, F|100/100",
        "|switch|p2a: Aegislash|Aegislash, L50, M|100/100",
        "|-mega|p1a: Floette|Floette|Floettite",
    ]
    tracker = LogStateTracker()
    for line in lines:
        tracker.process_line(line)
    rosters = build_match_rosters(lines)
    view = project_first_person(tracker.state, "p1", rosters=rosters, format="singles")
    mon = view.mons["p1a"]
    assert mon.mega_capable
    assert mon.mega


def test_team_mega_used_blocks_second_mega():
    tracker = LogStateTracker()
    lines = [
        "|poke|p1|Garchomp, M|",
        "|poke|p1|Charizard, M|",
        "|poke|p2|Rillaboom, M|",
        "|poke|p2|Incineroar, M|",
        "|turn|1",
        "|switch|p1a: Garchomp|Garchomp, M|100/100",
        "|switch|p1b: Charizard|Charizard, M|100/100",
        "|switch|p2a: Rillaboom|Rillaboom, M|100/100",
        "|switch|p2b: Incineroar|Incineroar, M|100/100",
        "|-mega|p1a: Garchomp",
        "|detailschange|p1a: Garchomp|Garchomp-Mega, M|100/100",
    ]
    for line in lines:
        tracker.process_line(line)

    rosters = build_match_rosters(lines)
    view = project_first_person(tracker.state, "p1", rosters=rosters)
    garchomp = view.mons["p1a"]
    charizard = view.mons["p1b"]
    assert garchomp.mega
    assert not garchomp.can_mega
    assert charizard.mega_capable is False
    assert not charizard.can_mega
    assert view.team_mega_used["p1"] is True


def test_can_mega_from_roster_mega_stone():
    lines = [
        "|poke|p1|Garchomp, M|",
        "|poke|p2|Rillaboom, M|",
        "|turn|1",
        "|switch|p1a: Garchomp|Garchomp, M|100/100",
        "|switch|p2a: Rillaboom|Rillaboom, M|100/100",
        "|-item|p1a: Garchomp|Garchompite",
    ]
    tracker = LogStateTracker()
    for line in lines:
        tracker.process_line(line)
    rosters = build_match_rosters(lines)
    view = project_first_person(tracker.state, "p1", rosters=rosters)
    mon = view.mons["p1a"]
    assert mon.mega_capable
    assert mon.can_mega


def test_encode_log_state_can_mega_field():
    mon = MonPerspective(
        species="garchomp",
        hp=100,
        max_hp=100,
        active=True,
        mega_capable=True,
        can_mega=True,
        item="garchompite",
        item_revealed=True,
        moves=["earthquake", "protect", "rockslide", "dragonclaw"],
    )
    state = BattleLogState(
        mons={"p1a": mon},
        team_mega_used={"p1": False, "p2": False},
    )
    tokens = encode_log_state(state, "p1")
    assert tokens[1, FIELD_FLAGS] & FLAG_CAN_MEGA
    assert tokens[0, FIELD_FLAGS] == 0


def test_log_mask_strips_mega_when_cannot():
    mon = MonPerspective(
        species="garchomp",
        hp=100,
        max_hp=100,
        active=True,
        mega_capable=False,
        can_mega=False,
        moves=["earthquake", "protect", "rockslide", "dragonclaw"],
    )
    view = BattleLogState(mons={"p1a": mon})
    mask = log_turn_slot_mask(view, "p1", "a")
    mega_indices = [i for i in range(mask.shape[0]) if is_mega_action(i)]
    assert mega_indices
    assert not any(mask[i] for i in mega_indices)


def test_mask_mega_actions_helper():
    mask = np.ones(107, dtype=bool)
    mask_mega_actions(mask)
    assert not any(mask[i] for i in range(107) if is_mega_action(i))
