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


def test_faint_excludes_mon_from_bench_and_switch_mask():
    # A mon that faints (via log) and is replaced must not reappear as an alive
    # bench token, or the switch mask would offer a dead mon as a target.
    from src.doubles.data.log_action_mask import log_turn_slot_mask

    t = LiveBattleTracker(battle_format="doubles", player_side="p1")
    t._known_team = {
        k: {"species": k, "ability": "", "item": "", "moves": ["protect"], "max_hp": 150, "mega": False}
        for k in ["sinistcha", "ceruledge", "ninetalesalola", "milotic"]
    }
    t.known_team_species = ["sinistcha", "ceruledge", "ninetalesalola", "milotic"]
    t._seed_known_team()
    t.record_brought_ally(["sinistcha", "ceruledge", "ninetalesalola", "milotic"])
    t.update_from_perception(
        {
            "state": "TURN_DECISION",
            "battle_format": "doubles",
            "ocr": {
                "player_slot_a": {"species_id": "ninetalesalola", "hp": 150, "max_hp": 150},
                "player_slot_b": {"species_id": "ceruledge", "hp": 181, "max_hp": 181},
            },
        }
    )
    assert t.apply_log_event(parse_string("Ninetales fainted!")) is True

    view = t._state_with_bench()
    # Exactly one ninetales token, and it's fainted (no stale alive duplicate).
    nines = [m for m in view.mons.values() if m.species == "ninetalesalola"]
    assert len(nines) == 1
    assert nines[0].fainted is True
    # The benched ninetales index is never a legal switch for the partner.
    mask_b = log_turn_slot_mask(view, "p1", "b")
    assert mask_b[3] == False  # roster index 3 == ninetalesalola


def test_switch_in_resets_volatiles_to_match_training():
    # The live tracker reuses one mon object per slot; a switch must wipe the
    # departed mon's boosts / Protect streak / last move / gimmick (training resets
    # these on switch-in), while keeping persistent status restored from memory.
    t = _seed()
    # Gyarados (p1a) racks up volatile state: a boost, a Protect commit (sets
    # last_move_id), and an *observed* Protect success (bumps protect_counter).
    t.apply_log_event(parse_string("Gyarados's Attack rose!"))
    t.record_committed_move("a", "Protect")
    t.apply_log_event(parse_string("Gyarados protected itself!"))
    p1a = t.state.mons["p1a"]
    assert p1a.boosts == {"atk": 1}
    assert p1a.protect_counter == 1
    assert p1a.last_move_id != 0

    # Garchomp switches into p1a (same slot, different species).
    t.update_from_perception(
        {
            "state": "TURN_DECISION",
            "battle_format": "doubles",
            "ocr": {"player_slot_a": {"species_id": "garchomp", "hp": 200, "max_hp": 200}},
        }
    )
    mon = t.state.mons["p1a"]
    assert mon.species == "garchomp"
    assert mon.boosts == {}
    assert mon.protect_counter == 0
    assert mon.last_move_id == 0
    assert mon.turns_active == 0
    assert "a" not in t._protect_streak


def test_protect_counter_only_counts_observed_successes():
    # Success-gated, matching BC training: committing Protect does NOT bump the
    # counter; only an observed "protected itself" line does. A failed Protect
    # (success line never seen) leaves the counter flat -- not reset to 0.
    t = _seed()
    p1a = t.state.mons["p1a"]

    # Turn 1: commit Protect, success observed -> counter 1.
    t.record_committed_move("a", "Protect")
    assert p1a.protect_counter == 0  # commit alone does not count
    t.apply_log_event(parse_string("Gyarados used Protect!"))  # move line, no bump
    assert p1a.protect_counter == 0
    t.apply_log_event(parse_string("Gyarados protected itself!"))
    assert p1a.protect_counter == 1

    # Turn 2: commit Protect again, but it FAILS (no success line) -> stays 1.
    t.record_committed_move("a", "Protect")
    t.apply_log_event(parse_string("Gyarados used Protect!"))
    assert p1a.protect_counter == 1  # flat on failure, not reset, not incremented

    # Turn 3: a non-Protect move resets the streak to 0.
    t.record_committed_move("a", "Waterfall")
    assert p1a.protect_counter == 0


def test_opponent_protect_success_increments_via_log():
    # Opponent protect is seen only via the log; success bumps its counter, and a
    # subsequent non-Protect move resets it (mirrors training's _record_move).
    t = _seed()
    p2a = t.state.mons["p2a"]
    t.apply_log_event(parse_string("The opposing Azumarill protected itself!"))
    assert p2a.protect_counter == 1
    t.apply_log_event(parse_string("The opposing Azumarill used Liquidation!"))
    assert p2a.protect_counter == 0


def test_mega_evolution_is_not_treated_as_switch():
    # A Mega form shares the base roster key, so volatiles must persist.
    t = _seed()
    t.apply_log_event(parse_string("Gyarados's Attack rose!"))
    t.update_from_perception(
        {
            "state": "TURN_DECISION",
            "battle_format": "doubles",
            "ocr": {"player_slot_a": {"species_id": "gyaradosmega", "hp": 172, "max_hp": 172}},
        }
    )
    # Boost retained (not a switch); species canonicalizes to the roster base.
    assert t.state.mons["p1a"].boosts == {"atk": 1}


def test_get_model_inputs_no_advance_reproduces_obs_without_growing_history():
    # A same-turn re-decision (recovery) must not push a duplicate trajectory frame,
    # and must reproduce the exact observation the model first saw.
    t = _seed()
    obs1, _ = t.get_model_inputs(advance=True)
    assert len(t._history) == 1
    obs2, _ = t.get_model_inputs(advance=False)
    assert len(t._history) == 1  # history not grown
    assert np.array_equal(obs1, obs2)  # identical input on re-decision
    # The next genuine turn advances history again.
    t.get_model_inputs(advance=True)
    assert len(t._history) == 2


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


def _legal_move_slots(mask) -> list[int]:
    from src.doubles.battle.move_order import decode_move_action_index

    slots = {decode_move_action_index(i)[0] for i in range(7, len(mask)) if mask[i]}
    return sorted(slots)


def _seed_choice_team() -> LiveBattleTracker:
    t = LiveBattleTracker(battle_format="doubles", player_side="p1")
    t.load_player_team_file("teams/champions_live_team.json")
    t.record_brought_ally(["ninetalesalola", "ceruledge", "milotic", "sinistcha"])
    t.update_from_perception(
        {
            "state": "TURN_DECISION",
            "battle_format": "doubles",
            "ocr": {
                "player_slot_a": {"species_id": "ninetalesalola", "hp": 150, "max_hp": 150},
                "player_slot_b": {"species_id": "ceruledge", "hp": 181, "max_hp": 181},
                "opp_slot_a": {"species_id": "incineroar", "hp_percent": 100.0},
                "opp_slot_b": {"species_id": "floetteeternal", "hp_percent": 100.0},
            },
        }
    )
    return t


def test_choice_item_locks_moves_after_commit():
    t = _seed_choice_team()
    # Free to pick any of its 4 moves before it has used one.
    _, masks = t.get_model_inputs()
    assert _legal_move_slots(masks["slot_a"]) == [1, 2, 3, 4]

    # Ninetales-Alola holds a Choice Scarf; after using Icy Wind it is locked.
    t.record_committed_move("a", "icywind")
    _, masks = t.get_model_inputs()
    # Icy Wind is canonical (alphabetical) slot 3 of [blizzard, freezedry, icywind, roar].
    assert _legal_move_slots(masks["slot_a"]) == [3]
    # Switching is still legal under a Choice lock.
    assert masks["slot_a"][1:7].any()


def test_choice_lock_clears_when_mon_switches_out():
    t = _seed_choice_team()
    t.record_committed_move("a", "icywind")
    assert t.choice_locked_move("a") == "icywind"

    # A different species occupies slot A -> the Choice lock no longer applies.
    t.update_from_perception(
        {
            "state": "TURN_DECISION",
            "battle_format": "doubles",
            "ocr": {
                "player_slot_a": {"species_id": "milotic", "hp": 202, "max_hp": 202},
                "player_slot_b": {"species_id": "ceruledge", "hp": 181, "max_hp": 181},
                "opp_slot_a": {"species_id": "incineroar", "hp_percent": 100.0},
                "opp_slot_b": {"species_id": "floetteeternal", "hp_percent": 100.0},
            },
        }
    )
    assert t.choice_locked_move("a") is None


def test_non_choice_item_is_not_locked():
    t = _seed_choice_team()
    # Ceruledge (slot B) holds a Colbur Berry, not a Choice item.
    t.record_committed_move("b", "bitterblade")
    assert t.choice_locked_move("b") is None
    _, masks = t.get_model_inputs()
    assert len(_legal_move_slots(masks["slot_b"])) > 1


def test_active_party_rows_excludes_on_field_partner_by_species():
    t = LiveBattleTracker(battle_format="doubles", player_side="p1")
    t.load_player_team_file("teams/champions_live_team.json")
    # Party-screen row order == brought (preview-selection) order.
    t.record_brought_ally(["milotic", "sinistcha", "ceruledge", "ninetalesalola"])
    t.update_from_perception(
        {
            "state": "TURN_DECISION",
            "battle_format": "doubles",
            "ocr": {
                # Tracker HP is intentionally stale vs. the party screen so the old
                # HP-matching heuristic would have failed to exclude the partner.
                "player_slot_a": {"species_id": "milotic", "hp": 202, "max_hp": 202},
                "player_slot_b": {"species_id": "sinistcha", "hp": 100, "max_hp": 178},
                "opp_slot_a": {"species_id": "incineroar", "hp_percent": 100.0},
                "opp_slot_b": {"species_id": "floetteeternal", "hp_percent": 100.0},
            },
        }
    )
    # Sinistcha (slot B / row 2) faints; Milotic (row 1) is still on the field.
    t.state.mons["p1b"].hp = 0
    t.state.mons["p1b"].fainted = True
    t.state.mons["p1b"].active = False
    assert t.active_party_rows() == {1}
