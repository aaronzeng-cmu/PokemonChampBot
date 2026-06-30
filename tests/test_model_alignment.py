"""CV-live vs training/eval alignment for the singles BC path.

Given the *same* battle context, the live CV path must encode the same per-turn
observation token, build the same legal-action mask, and decode the same action
as the training-data encoder (``encode_log_state``) and the eval mask
(``singles_turn_mask`` / ``singles_force_switch_mask``).
"""

from __future__ import annotations

import numpy as np
import torch

from src.core.data.log_tracker import BattleLogState
from src.core.data.perspective import MonPerspective
from src.core.data.state_tokenizer import encode_log_state
from src.cv_bridge.bc_policy import BCPolicy
from src.cv_bridge.state_tracker import LiveBattleTracker
from src.singles.bench_slots import log_our_bench_species
from src.singles.log_action_codec import MEGA_BASE, MOVE_BASE, SWITCH_BASE
from src.singles.log_action_mask import singles_force_switch_mask, singles_turn_mask


def _singles_state() -> BattleLogState:
    state = BattleLogState()
    state.team_roster["p1"] = ["gyarados", "raichu", "garchomp"]
    state.team_roster["p2"] = ["azumarill"]

    active = MonPerspective(slot="p1a", species="gyarados", hp=160, max_hp=160, active=True, seen=True)
    active.moves = ["waterfall", "icefang", "earthquake", "dragondance"]
    state.mons["p1a"] = active

    bench = MonPerspective(slot="p1b", species="raichu", hp=120, max_hp=120, active=False, seen=True)
    state.mons["p1b"] = bench

    opp = MonPerspective(slot="p2a", species="azumarill", hp=140, max_hp=140, active=True, seen=True)
    state.mons["p2a"] = opp
    return state


def _tracker_for(state: BattleLogState, ui_state: str) -> LiveBattleTracker:
    tracker = LiveBattleTracker(battle_format="singles", player_side="p1")
    tracker.state = state
    tracker.last_ui_state = ui_state
    return tracker


def test_turn_snapshot_matches_training_encoder():
    state = _singles_state()
    tracker = _tracker_for(state, "TURN_DECISION")

    live = tracker.encode_snapshot(force_switch=False)
    ref = encode_log_state(state, "p1", format="singles", force_switch=False)
    assert np.array_equal(live, ref)


def test_turn_mask_matches_eval_mask():
    state = _singles_state()
    tracker = _tracker_for(state, "TURN_DECISION")

    masks = tracker._legal_action_masks(force_switch=False)
    assert masks is not None
    ref = singles_turn_mask(state, "p1")
    assert np.array_equal(masks["slot_a"], ref)
    # Sanity: a real move is legal, all gimmick indices are masked off.
    assert masks["slot_a"][MOVE_BASE]
    assert not masks["slot_a"][MEGA_BASE:].any()


def test_force_switch_snapshot_and_mask_match_eval():
    state = _singles_state()
    tracker = _tracker_for(state, "FORCE_SWITCH")

    obs, masks = tracker.get_model_inputs()
    # Force-switch flag flips token-0 features vs a normal turn.
    turn_snap = encode_log_state(state, "p1", format="singles", force_switch=False)
    fs_snap = encode_log_state(state, "p1", format="singles", force_switch=True)
    assert not np.array_equal(turn_snap, fs_snap)
    assert np.array_equal(obs[-turn_snap.shape[0] :], fs_snap)

    assert masks is not None
    assert np.array_equal(masks["slot_a"], singles_force_switch_mask(state, "p1"))


def test_masked_decode_matches_eval_and_blocks_illegal_gimmick():
    state = _singles_state()
    mask = singles_turn_mask(state, "p1")

    # Logits peak on an illegal gimmick index (12), runner-up a legal move.
    logits = torch.full((18,), -5.0)
    logits[12] = 10.0
    logits[MOVE_BASE + 1] = 5.0

    unmasked = int(torch.argmax(logits).item())
    masked = BCPolicy._masked_argmax(logits, mask)

    eval_logits = logits.clone()
    eval_logits[~torch.as_tensor(mask, dtype=torch.bool)] = -float("inf")
    eval_pick = int(eval_logits.argmax().item())

    assert unmasked == 12  # old CV behavior would have chosen an illegal action
    assert masked == eval_pick == MOVE_BASE + 1  # now legal + matches eval decoder


def test_bench_modeled_from_brought_team_legalizes_switch():
    """Bring-3 bench (incl. not-yet-seen mons) is encoded + makes switches legal."""
    tracker = LiveBattleTracker(battle_format="singles", player_side="p1")
    tracker.record_team_preview(
        ["gyarados", "raichu", "garchomp", "azumarill", "dragonite", "snorlax"],
        ["volcarona", "kingambit", "greattusk", "ironvaliant", "gholdengo", "ogerpon"],
    )
    tracker.record_brought_ally(["gyarados", "raichu", "garchomp"])
    tracker.update_from_perception(
        {
            "state": "TURN_DECISION",
            "battle_format": "singles",
            "ocr": {
                "player_slot_a": {"species_id": "gyarados", "hp": 160, "max_hp": 160},
                "opp_slot_a": {"species_id": "volcarona", "hp": 100, "max_hp": 100},
            },
        }
    )

    # Bench tokens carry our other two brought mons in stable roster order.
    view = tracker._state_with_bench()
    assert log_our_bench_species(view, "p1")[:2] == ["raichu", "garchomp"]

    # Both bench mons are switch-legal even though only gyarados has been seen.
    _, masks = tracker.get_model_inputs()
    assert masks is not None
    assert masks["slot_a"][SWITCH_BASE]
    assert masks["slot_a"][SWITCH_BASE + 1]


def _doubles_known_team() -> dict:
    def mon(species: str, moves: list[str]) -> dict:
        return {"species": species, "ability": "", "item": "", "moves": moves, "max_hp": 0, "mega": False}

    return {
        "pokemon": [
            mon("raichu", ["thunderbolt", "voltswitch", "fakeout", "grassknot"]),
            mon("garchomp", ["earthquake", "rockslide", "dragonclaw", "protect"]),
            mon("azumarill", ["liquidation", "playrough", "aquajet", "protect"]),
            mon("grimmsnarl", ["spiritbreak", "thunderwave", "lightscreen", "reflect"]),
            mon("dragonite", ["dragonclaw", "extremespeed", "earthquake", "protect"]),
            mon("staraptor", ["bravebird", "closecombat", "doubleedge", "protect"]),
        ]
    }


def test_doubles_seeds_leads_and_excludes_onfield_from_switch():
    """Our chosen leads are active from turn 1, so we never offer to switch one
    of them in -- even if the live sprite read comes back ``unknown``."""
    tracker = LiveBattleTracker(battle_format="doubles", player_side="p1")
    tracker.load_player_team(_doubles_known_team())
    # Selection order: leads = azumarill + garchomp.
    tracker.record_brought_ally(["azumarill", "garchomp", "raichu", "grimmsnarl"])

    actives = {s: m.species for s, m in tracker.state.mons.items() if m.active}
    assert actives == {"p1a": "azumarill", "p1b": "garchomp"}

    # A failed sprite read must not clobber the seeded actives.
    tracker.update_from_perception(
        {
            "state": "TURN_DECISION",
            "battle_format": "doubles",
            "ocr": {
                "player_slot_a": {"species_id": "unknown", "hp": 207, "max_hp": 207},
                "player_slot_b": {"species_id": "unknown", "hp": 185, "max_hp": 185},
            },
        }
    )
    actives = {s: m.species for s, m in tracker.state.mons.items() if m.active}
    assert actives == {"p1a": "azumarill", "p1b": "garchomp"}

    _, masks = tracker.get_model_inputs()
    assert masks is not None
    roster = tracker.state.team_roster["p1"]
    gar_idx = roster.index("garchomp") + 1
    azu_idx = roster.index("azumarill") + 1
    # On-field mons are illegal switch targets on both slots.
    assert not masks["slot_a"][gar_idx] and not masks["slot_b"][gar_idx]
    assert not masks["slot_a"][azu_idx] and not masks["slot_b"][azu_idx]
    # ...and a real move is legal (we can fight, not just pass/switch).
    assert masks["slot_a"][7:].any()


def test_fainted_mon_excluded_from_switch_after_party_readout():
    """A fainted active that's been replaced must not linger as a switch target."""
    tracker = LiveBattleTracker(battle_format="doubles", player_side="p1")
    tracker.load_player_team(_doubles_known_team())
    # leads = staraptor + garchomp (max_hp 186 / garchomp uses default 100 here)
    tracker.record_brought_ally(["staraptor", "garchomp", "raichu", "azumarill"])

    # Give the team distinct max HPs so the party readout can identify species.
    for sp, mx in {"staraptor": 186, "garchomp": 185, "raichu": 137, "azumarill": 207}.items():
        tracker._known_team[sp]["max_hp"] = mx
    tracker._seed_known_team()
    tracker.record_brought_ally(["staraptor", "garchomp", "raichu", "azumarill"])

    # Force-switch party screen: garchomp fainted (0/185), raichu chosen.
    tracker.record_party_readout(
        [
            {"slot": 1, "hp": 186, "max_hp": 186, "alive": True},
            {"slot": 2, "hp": 0, "max_hp": 185, "alive": False},
            {"slot": 3, "hp": 137, "max_hp": 137, "alive": True},
            {"slot": 4, "hp": 207, "max_hp": 207, "alive": True},
        ]
    )
    # raichu has replaced garchomp on the field.
    tracker.update_from_perception(
        {
            "state": "TURN_DECISION",
            "battle_format": "doubles",
            "ocr": {
                "player_slot_a": {"species_id": "staraptor", "hp": 186, "max_hp": 186},
                "player_slot_b": {"species_id": "raichu", "hp": 137, "max_hp": 137},
            },
        }
    )

    roster = tracker.state.team_roster["p1"]
    gar_idx = roster.index("garchomp") + 1
    azu_idx = roster.index("azumarill") + 1
    _, masks = tracker.get_model_inputs()
    assert masks is not None
    # Fainted garchomp is illegal; bench azumarill is the only legal switch.
    assert not masks["slot_a"][gar_idx]
    assert masks["slot_a"][azu_idx]


def test_history_frame_drops_force_switch_flag():
    """Trajectory parity: stored history frame is force-switch-free (turn-start)."""
    state = _singles_state()
    tracker = _tracker_for(state, "FORCE_SWITCH")

    tracker.get_model_inputs()
    assert len(tracker._history) == 1
    stored = tracker._history[-1]

    view = tracker._state_with_bench()
    neutral = encode_log_state(view, "p1", format="singles", force_switch=False)
    flagged = encode_log_state(view, "p1", format="singles", force_switch=True)
    assert np.array_equal(stored, neutral)
    assert not np.array_equal(stored, flagged)
