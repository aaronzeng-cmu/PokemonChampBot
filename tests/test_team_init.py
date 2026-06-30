"""Known-team initialization: seeding our moves/bench fixes the doubles mask.

Without our own team data the live CV path can't read our move list off the HUD,
so the doubles legal-action mask is empty and the policy is forced to ``pass``
(the ``(0, 0)`` bug). Loading the team profile must populate our actives' moves
and model the bench so real moves + switches become legal.
"""

from __future__ import annotations

from pathlib import Path

from src.cv_bridge.state_tracker import LiveBattleTracker

# Dedicated fixture so these assertions don't break when the live team profile
# (teams/champions_live_team.json) is re-parsed for a different lineup.
TEAM_FILE = Path(__file__).resolve().parent / "fixtures" / "champions_team.json"


def _doubles_tracker() -> LiveBattleTracker:
    tracker = LiveBattleTracker(battle_format="doubles", player_side="p1")
    tracker.load_player_team_file(TEAM_FILE)
    return tracker


def test_load_player_team_seeds_roster_and_memory():
    tracker = _doubles_tracker()
    assert tracker.known_team_species == [
        "raichu",
        "garchomp",
        "azumarill",
        "grimmsnarl",
        "dragonite",
        "staraptor",
    ]
    # Roster is populated and bench memory carries each mon's known moves.
    assert set(tracker.state.team_roster["p1"]) == set(tracker.known_team_species)
    raichu_mem = tracker._mon_memory["p1:raichu"]
    assert raichu_mem.moves == ["zapcannon", "fakeout", "focusblast", "protect"]
    assert raichu_mem.max_hp == 137
    assert raichu_mem.can_mega is True  # holds Raichunite Y


def test_reconcile_preview_fills_unknown_by_elimination():
    tracker = _doubles_tracker()
    parsed = ["raichu", "unknown", "azumarill", "grimmsnarl", "dragonite", "staraptor"]
    resolved = tracker.reconcile_preview_species(parsed)
    # The single garbled slot is filled with the one missing known species.
    assert resolved[1] == "garchomp"
    assert resolved == tracker.known_team_species


def test_active_gets_known_moves_so_move_mask_is_not_empty():
    tracker = _doubles_tracker()
    tracker.record_brought_ally(["raichu", "azumarill", "grimmsnarl", "staraptor"])
    tracker.update_from_perception(
        {
            "state": "TURN_DECISION",
            "battle_format": "doubles",
            "ocr": {
                "player_slot_a": {"species_id": "raichu", "hp": 137, "max_hp": 137},
                "player_slot_b": {"species_id": "azumarill", "hp": 207, "max_hp": 207},
                "opp_slot_a": {"species_id": "serperior", "hp": 100, "max_hp": 100},
                "opp_slot_b": {"species_id": "volcarona", "hp": 100, "max_hp": 100},
            },
        }
    )

    # Our active now carries its real move list (perception can't read this).
    assert tracker.state.mons["p1a"].moves[:2] == ["zapcannon", "fakeout"]

    masks = tracker._legal_action_masks(force_switch=False)
    assert masks is not None
    slot_a = masks["slot_a"]
    # Moves (index >= 7) are legal -> the policy is no longer forced to pass (0).
    assert slot_a[7:].any()
    # Switching to a brought bench mon (grimmsnarl/staraptor, indices 1-6) is legal.
    assert slot_a[1:7].any()


def test_no_team_loaded_yields_pass_only_active_mask():
    """Regression guard: this is the broken state team-init fixes."""
    tracker = LiveBattleTracker(battle_format="doubles", player_side="p1")
    tracker.update_from_perception(
        {
            "state": "TURN_DECISION",
            "battle_format": "doubles",
            "ocr": {
                "player_slot_a": {"species_id": "raichu", "hp": 137, "max_hp": 137},
                "player_slot_b": {"species_id": "azumarill", "hp": 207, "max_hp": 207},
                "opp_slot_a": {"species_id": "serperior", "hp": 100, "max_hp": 100},
                "opp_slot_b": {"species_id": "volcarona", "hp": 100, "max_hp": 100},
            },
        }
    )
    masks = tracker._legal_action_masks(force_switch=False)
    assert masks is not None
    # No known moves -> no move indices legal; only pass survives.
    assert not masks["slot_a"][7:].any()
