"""LiveBattleTracker.apply_log_event: entity resolution + tensor encoding."""

from __future__ import annotations

import numpy as np

from src.core.data.perspective import boost_id
from src.core.data.state_tokenizer import (
    FIELD_BOOST_START,
    STAT_NAMES,
    TOKEN_OUR_ACTIVE,
    TOKEN_OPP_ACTIVE,
    N_FIELDS,
    TRAJECTORY_DEPTH,
    N_TOKENS,
)
from src.cv_bridge.battle_log_parser import parse_string
from src.cv_bridge.state_tracker import LiveBattleTracker


def _seed() -> LiveBattleTracker:
    t = LiveBattleTracker(battle_format="doubles", player_side="p1")
    t.update_from_perception(
        {
            "state": "TURN_DECISION",
            "battle_format": "doubles",
            "ocr": {
                "player_slot_a": {"species_id": "gyarados", "hp": 172, "max_hp": 172},
                "player_slot_b": {"species_id": "raichu", "hp": 137, "max_hp": 137},
                "opp_slot_a": {"species_id": "azumarill", "hp_percent": 100.0},
                "opp_slot_b": {"species_id": "heracross", "hp_percent": 100.0},
            },
        }
    )
    return t


def _boost_in_token(tensor: np.ndarray, token_row: int, stat: str) -> int:
    return int(tensor[token_row, FIELD_BOOST_START + STAT_NAMES.index(stat)])


def test_stat_boost_resolves_and_encodes():
    t = _seed()
    ev = parse_string("Gyarados's Attack and Speed rose!")
    assert t.apply_log_event(ev) is True

    mon = t.state.mons["p1a"]
    assert mon.species == "gyarados"
    assert mon.boosts == {"atk": 1, "spe": 1}

    snap = t.encode_snapshot()
    assert _boost_in_token(snap, TOKEN_OUR_ACTIVE, "atk") == boost_id(1)
    assert _boost_in_token(snap, TOKEN_OUR_ACTIVE, "spe") == boost_id(1)
    assert _boost_in_token(snap, TOKEN_OUR_ACTIVE, "def") == boost_id(0)


def test_stat_boost_accumulates_and_clamps():
    t = _seed()
    for _ in range(5):
        t.apply_log_event(parse_string("Gyarados's Attack rose drastically!"))  # +3 each
    assert t.state.mons["p1a"].boosts["atk"] == 6  # clamped


def test_opponent_stat_drop_targets_opponent_slot():
    t = _seed()
    ev = parse_string("The opposing Azumarill's Speed fell!")
    assert ev["is_opponent"] is True
    assert t.apply_log_event(ev) is True
    assert t.state.mons["p2a"].boosts == {"spe": -1}
    # Our gyarados untouched.
    assert t.state.mons["p1a"].boosts == {}

    snap = t.encode_snapshot()
    assert _boost_in_token(snap, TOKEN_OPP_ACTIVE, "spe") == boost_id(-1)


def test_faint_event_marks_fainted():
    t = _seed()
    assert t.apply_log_event(parse_string("The opposing Heracross fainted!")) is True
    mon = t.state.mons["p2b"]
    assert mon.fainted is True
    assert mon.hp == 0
    assert mon.active is False


def test_weather_event_sets_field():
    t = _seed()
    assert t.apply_log_event(parse_string("The sunlight turned harsh!")) is True
    assert t.state.field.weather == "sunnyday"


def test_move_event_reveals_move_for_opponent():
    t = _seed()
    ev = parse_string("The opposing Azumarill used Belly Drum!")
    assert t.apply_log_event(ev) is True
    assert "bellydrum" in t.state.mons["p2a"].moves


def test_unresolvable_subject_returns_false():
    t = _seed()
    # Species not on the field.
    assert t.apply_log_event(parse_string("Pikachu's Attack rose!")) is False


def test_get_model_inputs_shape_and_masks():
    t = _seed()
    obs, masks = t.get_model_inputs()
    assert obs.shape == (TRAJECTORY_DEPTH * N_TOKENS, N_FIELDS)
    assert masks is not None
    assert set(masks) == {"slot_a", "slot_b"}
