"""Validate battle-log text -> event-dict parsing (incl. OCR-noise leniency)."""

from __future__ import annotations

import pytest

from src.cv_bridge.battle_log_parser import parse_string


def test_single_stat_rose():
    ev = parse_string("Garchomp's Attack rose!")
    assert ev == {
        "type": "stat_boost",
        "target": "garchomp",
        "target_name": "Garchomp",
        "is_opponent": False,
        "stats": ["atk"],
        "amount": 1,
    }


def test_multi_stat_split_by_and():
    ev = parse_string("Gyarados's Attack and Speed rose!")
    assert ev["type"] == "stat_boost"
    assert ev["stats"] == ["atk", "spe"]
    assert ev["amount"] == 1
    assert ev["target"] == "gyarados"


def test_stat_fell():
    ev = parse_string("Garchomp's Attack fell!")
    assert ev["stats"] == ["atk"]
    assert ev["amount"] == -1


def test_rose_drastically_is_plus_three():
    ev = parse_string("Azumarill's Defense rose drastically!")
    assert ev["stats"] == ["def"]
    assert ev["amount"] == 3


def test_sharply_rose_is_plus_two():
    ev = parse_string("Pikachu's Speed sharply rose!")
    assert ev["stats"] == ["spe"]
    assert ev["amount"] == 2


def test_harshly_fell_is_minus_two():
    ev = parse_string("Heracross's Special Attack harshly fell!")
    assert ev["stats"] == ["spa"]
    assert ev["amount"] == -2


def test_opponent_prefix_resolution():
    ev = parse_string("The opposing Glimmora's Speed fell!")
    assert ev["is_opponent"] is True
    assert ev["target"] == "glimmora"
    assert ev["stats"] == ["spe"]


def test_faint_player():
    ev = parse_string("Kingambit fainted!")
    assert ev == {
        "type": "faint",
        "target": "kingambit",
        "target_name": "Kingambit",
        "is_opponent": False,
    }


def test_faint_opponent():
    ev = parse_string("The opposing Glimmora fainted!")
    assert ev["type"] == "faint"
    assert ev["is_opponent"] is True
    assert ev["target"] == "glimmora"


def test_move_usage():
    ev = parse_string("Whimsicott used Tailwind!")
    assert ev == {
        "type": "move",
        "user": "whimsicott",
        "user_name": "Whimsicott",
        "is_opponent": False,
        "move": "tailwind",
        "move_name": "Tailwind",
    }


def test_move_opponent_multiword():
    ev = parse_string("The opposing Gardevoir used Dazzling Gleam!")
    assert ev["type"] == "move"
    assert ev["is_opponent"] is True
    assert ev["user"] == "gardevoir"
    assert ev["move"] == "dazzlinggleam"


def test_weather_sun():
    assert parse_string("The sunlight turned harsh!") == {"type": "weather", "weather": "sunnyday"}


def test_weather_rain_and_sand_and_snow():
    assert parse_string("It started to rain!")["weather"] == "raindance"
    assert parse_string("A sandstorm kicked up!")["weather"] == "sandstorm"
    assert parse_string("It started to snow!")["weather"] == "snowscape"


def test_status_paralysis():
    ev = parse_string("The opposing Azumarill was paralyzed!")
    assert ev["type"] == "status"
    assert ev["status"] == "par"
    assert ev["is_opponent"] is True


def test_ocr_leniency_missing_punctuation_and_apostrophe():
    # No "!" and dropped apostrophe in possessive.
    ev = parse_string("Gyarados Attack and Speed rose")
    assert ev is not None
    assert ev["type"] == "stat_boost"
    assert ev["stats"] == ["atk", "spe"]
    assert ev["amount"] == 1


# --- Regression: exact strings captured from the live shadow-loop run where OCR
# misread the trailing "!" as "l" (faints + stat drops were silently dropped). ---


def test_ocr_faint_opponent_trailing_l_snorlax():
    ev = parse_string("The opposing Snorlax faintedl")
    assert ev["type"] == "faint"
    assert ev["target"] == "snorlax"
    assert ev["is_opponent"] is True


def test_ocr_faint_opponent_trailing_l_hydreigon():
    ev = parse_string("The opposing Hydreigon faintedl")
    assert ev["type"] == "faint"
    assert ev["target"] == "hydreigon"
    assert ev["is_opponent"] is True


def test_ocr_faint_opponent_trailing_l_hawlucha():
    ev = parse_string("The opposing Hawlucha faintedl")
    assert ev["type"] == "faint"
    assert ev["target"] == "hawlucha"
    assert ev["is_opponent"] is True


def test_ocr_move_trailing_l_reflect():
    ev = parse_string("Grimmsnarl used Reflectl")
    assert ev["type"] == "move"
    assert ev["user"] == "grimmsnarl"
    assert ev["move"] == "reflect"
    assert ev["move_name"] == "Reflect"
    assert ev["is_opponent"] is False


def test_ocr_move_opponent_trailing_l_screech():
    ev = parse_string("The opposing Snorlax used Screechl")
    assert ev["type"] == "move"
    assert ev["user"] == "snorlax"
    assert ev["move"] == "screech"
    assert ev["move_name"] == "Screech"
    assert ev["is_opponent"] is True


def test_ocr_stat_drop_double_l_verb_harshly():
    # "felll" must become "fell!" (not "fel!") and harshly => -2.
    ev = parse_string("Grimmsnarl's Defense harshly felll")
    assert ev["type"] == "stat_boost"
    assert ev["stats"] == ["def"]
    assert ev["amount"] == -2
    assert ev["is_opponent"] is False


def test_ocr_stat_drop_multi_with_sp_colon_typo():
    # "Sp: Atk" colon typo + "felll" trailing artifact, two stats at -1.
    ev = parse_string("The opposing Snorlax's Attack and Sp: Atk felll")
    assert ev["type"] == "stat_boost"
    assert ev["stats"] == ["atk", "spa"]
    assert ev["amount"] == -1
    assert ev["is_opponent"] is True
    assert ev["target"] == "snorlax"


def test_ocr_mega_evolution_trailing_l():
    ev = parse_string("Staraptor has Mega Evolved into Mega Staraptorl")
    assert ev["type"] == "mega_evolve"
    assert ev["target"] == "staraptor"
    assert ev["target_name"] == "Staraptor"
    assert ev["is_opponent"] is False


def test_whirlpool_move_name_not_corrupted():
    # Guard against naive rstrip('l') eating double-l move names.
    ev = parse_string("Gyarados used Whirlpool!")
    assert ev["type"] == "move"
    assert ev["move"] == "whirlpool"
    assert ev["move_name"] == "Whirlpool"


def test_noop_message_returns_none():
    assert parse_string("Gyarados's Attack won't go any higher!") is None


def test_unrelated_text_returns_none():
    assert parse_string("You defeated Kazumasa!") is None
    assert parse_string("") is None
    assert parse_string(None) is None
