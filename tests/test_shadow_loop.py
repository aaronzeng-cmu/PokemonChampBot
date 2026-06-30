"""Shadow loop body: log-event application + once-per-turn decision gating."""

from __future__ import annotations

import json

import numpy as np

from src.cv_bridge.shadow_loop import ShadowLoop
from src.cv_bridge.state_tracker import LiveBattleTracker


class _FakeBridge:
    def __init__(self):
        self.taps: list[tuple[int, int]] = []

    def get_screen(self):
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def tap(self, x, y):
        self.taps.append((x, y))


class _FakePerception:
    """Scriptable stand-in for PerceptionModule."""

    def __init__(self, script):
        self._script = list(script)
        self._i = -1
        self.party_slots: list[dict] = []
        self.party_species: dict[int, str] = {}
        self.popups: list[str] = []
        self.battle_data = {
            "player_slot_a": {"species_id": "gyarados", "hp": 172, "max_hp": 172},
            "player_slot_b": {"species_id": "raichu", "hp": 137, "max_hp": 137},
            "opp_slot_a": {"species_id": "azumarill", "hp_percent": 100.0},
            "opp_slot_b": {"species_id": "heracross", "hp_percent": 100.0},
        }

    def step(self):
        self._i += 1

    def get_current_state(self, frame):
        return self._script[self._i][0]

    def read_battle_log(self, frame):
        return self._script[self._i][1]

    def extract_battle_data(self, frame):
        return self.battle_data

    def read_party_slots(self, frame, *, max_slots=6):
        return self.party_slots

    def read_party_species(self, frame, *, max_slots=6):
        return self.party_species

    def read_ability_item_popups(self, frame):
        return self.popups


def _seed_tracker() -> LiveBattleTracker:
    t = LiveBattleTracker(battle_format="doubles", player_side="p1")
    t.update_from_perception(
        {
            "state": "TURN_DECISION",
            "battle_format": "doubles",
            "ocr": {
                "player_slot_a": {"species_id": "gyarados", "hp": 172, "max_hp": 172},
                "opp_slot_a": {"species_id": "azumarill", "hp_percent": 100.0},
            },
        }
    )
    return t


def _run(script, *, policy=None, track_log=False):
    perception = _FakePerception(script)
    tracker = _seed_tracker()
    loop = ShadowLoop(
        bridge=_FakeBridge(),
        perception=perception,
        tracker=tracker,
        policy=policy,
    )
    # Battle-log OCR is gated to the post-action window; open it for log tests that
    # feed bare ANIMATION frames (in live play a submitted move opens the window).
    loop._track_log = track_log
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    for _ in script:
        perception.step()
        loop.process_frame(frame)
    return loop, tracker


def test_decision_debounced_until_state_stable():
    calls = []

    def policy(obs, masks):
        calls.append(1)
        return None

    perception = _FakePerception([("TURN_DECISION", None)] * 3)
    loop = ShadowLoop(
        bridge=_FakeBridge(),
        perception=perception,
        tracker=_seed_tracker(),
        policy=policy,
        stability_frames=3,
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    obs = []
    for _ in range(3):
        perception.step()
        obs.append(loop.process_frame(frame))
    # Decision only fires once the state has held for 3 consecutive frames.
    assert [o["stable"] for o in obs] == [False, False, True]
    assert len(calls) == 1


def test_transient_decision_frame_is_skipped():
    calls = []

    def policy(obs, masks):
        calls.append(1)
        return None

    # A lone TURN_DECISION sandwiched by animation never reaches 2 in a row.
    script = [("ANIMATION", None), ("TURN_DECISION", None), ("ANIMATION", None)]
    perception = _FakePerception(script)
    loop = ShadowLoop(
        bridge=_FakeBridge(),
        perception=perception,
        tracker=_seed_tracker(),
        policy=policy,
        stability_frames=2,
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    for _ in script:
        perception.step()
        loop.process_frame(frame)
    assert calls == []


def test_tracker_prefers_sprite_icon_over_nameplate():
    # The icon/CNN match is authoritative (nickname-proof); the nameplate text is
    # only a fallback, so it must NOT override a good sprite read.
    t = LiveBattleTracker(battle_format="doubles", player_side="p1")
    t.update_from_perception(
        {
            "state": "TURN_DECISION",
            "battle_format": "doubles",
            "ocr": {
                "player_slot_a": {
                    "name": "sparky",          # nickname on the plate
                    "species_id": "garchomp",  # CNN icon match
                    "hp": 100,
                    "max_hp": 100,
                },
            },
        }
    )
    assert t.state.mons["p1a"].species == "garchomp"


def test_tracker_falls_back_to_nameplate_when_sprite_unknown():
    t = LiveBattleTracker(battle_format="doubles", player_side="p1")
    t.update_from_perception(
        {
            "state": "TURN_DECISION",
            "battle_format": "doubles",
            "ocr": {
                "player_slot_a": {
                    "name": "garchomp",       # nameplate happens to be the species
                    "species_id": "unknown",  # sprite unreadable this frame
                    "hp": 100,
                    "max_hp": 100,
                },
            },
        }
    )
    assert t.state.mons["p1a"].species == "garchomp"


def test_log_event_applied_once_and_deduped():
    script = [
        ("ANIMATION", "Gyarados's Attack rose!"),
        ("ANIMATION", "Gyarados's Attack rose!"),  # duplicate -> ignored
    ]
    _, tracker = _run(script, track_log=True)
    # Boost applied exactly once despite the repeated text.
    assert tracker.state.mons["p1a"].boosts["atk"] == 1


def test_decision_fires_once_per_turn():
    calls = []

    def policy(obs, masks):
        calls.append(obs.shape)
        return None

    script = [
        ("TURN_DECISION", None),
        ("TURN_DECISION", None),  # still same turn -> no second decision
        ("ANIMATION", None),       # leaves decision -> resets gate
        ("TURN_DECISION", None),  # new decision
    ]
    _run(script, policy=policy)
    assert len(calls) == 2


def test_new_log_after_first_is_applied():
    script = [
        ("ANIMATION", "Gyarados's Attack rose!"),
        ("ANIMATION", "The opposing Azumarill's Speed fell!"),
    ]
    _, tracker = _run(script, track_log=True)
    assert tracker.state.mons["p1a"].boosts["atk"] == 1
    assert tracker.state.mons["p2a"].boosts["spe"] == -1


def test_move_selection_recovery_retaps_move():
    from src.cv_bridge.action_executor import Tap

    perception = _FakePerception([("MOVE_SELECTION", None), ("MOVE_SELECTION", None)])
    bridge = _FakeBridge()
    loop = ShadowLoop(bridge=bridge, perception=perception, tracker=_seed_tracker(), execute_taps=True)
    loop.tap_delay = loop.submenu_settle = loop.post_action_delay = 0.0
    loop._pending_move_taps = [Tap(111, 222, "move.move_1")]

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    perception.step()
    loop.process_frame(frame)
    assert (111, 222) in bridge.taps  # re-tapped the move on the open list


def test_move_selection_recovery_respects_attempt_cap():
    from src.cv_bridge.action_executor import Tap

    perception = _FakePerception([("MOVE_SELECTION", None)] * 10)
    bridge = _FakeBridge()
    loop = ShadowLoop(bridge=bridge, perception=perception, tracker=_seed_tracker(), execute_taps=True)
    loop.tap_delay = loop.submenu_settle = loop.post_action_delay = 0.0
    loop._pending_move_taps = [Tap(111, 222, "move.move_1")]

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    for _ in range(10):
        perception.step()
        loop.process_frame(frame)
    assert bridge.taps.count((111, 222)) == loop._max_move_recover_attempts


def test_move_selection_no_recovery_in_dry_run():
    from src.cv_bridge.action_executor import Tap

    perception = _FakePerception([("MOVE_SELECTION", None)])
    bridge = _FakeBridge()
    loop = ShadowLoop(bridge=bridge, perception=perception, tracker=_seed_tracker(), execute_taps=False)
    loop._pending_move_taps = [Tap(111, 222, "move.move_1")]

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    perception.step()
    loop.process_frame(frame)
    assert bridge.taps == []


def test_force_switch_taps_first_alive_slot():
    perception = _FakePerception([("FORCE_SWITCH", None)])
    perception.party_slots = [
        {"slot": 1, "hp": 0, "max_hp": 137, "alive": False},
        {"slot": 2, "hp": 186, "max_hp": 186, "alive": True},
        {"slot": 3, "hp": 185, "max_hp": 185, "alive": True},
    ]
    bridge = _FakeBridge()
    loop = ShadowLoop(
        bridge=bridge,
        perception=perception,
        tracker=_seed_tracker(),
        battle_format="singles",
        execute_taps=True,
    )
    loop.tap_delay = loop.submenu_settle = loop.post_action_delay = 0.0

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    perception.step()
    obs = loop.process_frame(frame)

    # Slot 1 is fainted -> picks slot 2; two taps on that row (select +
    # open popup), then a third confirms via Switch in.
    assert obs["force_switch"]["slot"] == 2
    assert bridge.taps[0] == (258, 436)
    assert bridge.taps[1] == (258, 436)
    assert len(bridge.taps) == 3


def test_force_switch_skips_reordered_active_row_via_sprite():
    # Party list reorders: an on-field mon (milotic) floats to row 1, so the static
    # brought-order mapping would wrongly pick it. Sprite reading must exclude it.
    tracker = LiveBattleTracker(battle_format="doubles", player_side="p1")
    tracker.update_from_perception(
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
    perception = _FakePerception([("FORCE_SWITCH", None)])
    perception.party_slots = [
        {"slot": 1, "hp": 202, "max_hp": 202, "alive": True},   # milotic (ACTIVE)
        {"slot": 2, "hp": 0, "max_hp": 150, "alive": False},    # ninetales (fainted)
        {"slot": 3, "hp": 178, "max_hp": 178, "alive": True},   # sinistcha (bench)
    ]
    perception.party_species = {1: "milotic", 2: "ninetalesalola", 3: "sinistcha"}
    bridge = _FakeBridge()
    loop = ShadowLoop(
        bridge=bridge,
        perception=perception,
        tracker=tracker,
        battle_format="doubles",
        execute_taps=True,
    )
    loop.tap_delay = loop.submenu_settle = loop.post_action_delay = 0.0

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    perception.step()
    obs = loop.process_frame(frame)

    # Must NOT pick row 1 (active milotic); picks row 3 (benched sinistcha).
    assert obs["force_switch"]["slot"] == 3
    assert bridge.taps[0] == (258, 562)


def test_phased_switch_taps_live_row_for_target_species():
    # Voluntary switch: the model picks bench index 3 (sinistcha), but the live
    # party list has reordered so sinistcha is now on row 2. The phased executor
    # must open the party, read the sprites, and tap row 2 -- not the stale row 3.
    tracker = LiveBattleTracker(battle_format="doubles", player_side="p1")
    tracker.state.team_roster = {
        "p1": ["milotic", "ninetalesalola", "sinistcha", "ceruledge", "incineroar", "floetteeternal"]
    }
    perception = _FakePerception([("TURN_DECISION", None)])
    perception.party_species = {1: "milotic", 2: "sinistcha", 3: "ninetalesalola"}
    bridge = _FakeBridge()
    loop = ShadowLoop(
        bridge=bridge,
        perception=perception,
        tracker=tracker,
        battle_format="doubles",
        execute_taps=True,
    )
    loop.tap_delay = loop.submenu_settle = loop.post_action_delay = 0.0

    labels = loop._execute_switch_phase(bench_slot=3, slot=0)

    assert "switch.open" in labels[0]
    assert (258, 436) in bridge.taps  # force_switch.slot_2 (sinistcha's live row)
    assert (258, 562) not in bridge.taps  # stale brought-order row 3 must NOT be tapped


def test_same_turn_redecision_does_not_double_push_trajectory():
    # Recovery can clear the turn gate and re-decide the SAME turn without the state
    # ever leaving TURN_DECISION. That re-decision must not push a duplicate frame
    # into the rolling trajectory history (training appends once per game turn).
    calls = []

    def policy(obs, masks):
        calls.append(1)
        return (0, 0)  # pass / pass

    perception = _FakePerception([("TURN_DECISION", None), ("TURN_DECISION", None)])
    tracker = _seed_tracker()
    loop = ShadowLoop(
        bridge=_FakeBridge(),
        perception=perception,
        tracker=tracker,
        policy=policy,
        battle_format="doubles",
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    perception.step()
    loop.process_frame(frame)
    assert len(tracker._history) == 1

    # Simulate recovery re-opening the gate for the same turn (state unchanged).
    loop.turn_processed = False
    perception.step()
    loop.process_frame(frame)
    assert len(calls) == 2  # decided again
    assert len(tracker._history) == 1  # but no duplicate trajectory frame


def test_log_line_applied_once_per_window_despite_flicker():
    # At the higher frame rate a line is re-OCR'd many frames and can flicker out
    # and back; a stat boost must be applied exactly once per post-action window.
    script = [
        ("ANIMATION", "Gyarados's Attack rose!"),
        ("ANIMATION", None),
        ("ANIMATION", "Gyarados's Attack rose!"),
    ]
    loop, tracker = _run(script, track_log=True)
    assert tracker.state.mons["p1a"].boosts.get("atk") == 1


def test_force_switch_no_alive_does_not_tap():
    perception = _FakePerception([("FORCE_SWITCH", None)])
    perception.party_slots = [{"slot": 1, "hp": 0, "max_hp": 137, "alive": False}]
    bridge = _FakeBridge()
    loop = ShadowLoop(
        bridge=bridge,
        perception=perception,
        tracker=_seed_tracker(),
        battle_format="singles",
        execute_taps=True,
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    perception.step()
    obs = loop.process_frame(frame)
    assert obs["force_switch"]["status"] == "no_alive_slots"
    assert bridge.taps == []


def test_force_switch_respects_attempt_cap():
    perception = _FakePerception([("FORCE_SWITCH", None)] * 10)
    perception.party_slots = [
        {"slot": 1, "hp": 0, "max_hp": 137, "alive": False},
        {"slot": 2, "hp": 186, "max_hp": 186, "alive": True},
    ]
    bridge = _FakeBridge()
    loop = ShadowLoop(
        bridge=bridge,
        perception=perception,
        tracker=_seed_tracker(),
        battle_format="singles",
        execute_taps=True,
    )
    loop.tap_delay = loop.submenu_settle = loop.post_action_delay = 0.0
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    for _ in range(10):
        perception.step()
        loop.process_frame(frame)
    # Each attempt taps the row twice (select + open popup); capped at the limit.
    assert bridge.taps.count((258, 436)) == loop._max_force_switch_attempts * 2


def test_ability_popup_reveals_ability():
    perception = _FakePerception([("ANIMATION", None)])
    perception.popups = ["Gyarados's Intimidate"]
    tracker = _seed_tracker()
    loop = ShadowLoop(bridge=_FakeBridge(), perception=perception, tracker=tracker)
    loop._track_log = True  # post-action window (animations only follow a move)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    perception.step()
    obs = loop.process_frame(frame)
    assert obs["popups"][0]["subtype"] == "ability"
    assert tracker.state.mons["p1a"].ability == "intimidate"
    assert tracker.state.mons["p1a"].ability_revealed


def test_item_popup_reveals_opponent_item_by_species():
    perception = _FakePerception([("ANIMATION", None)])
    perception.popups = ["Azumarill's Sitrus Berry"]
    tracker = _seed_tracker()
    loop = ShadowLoop(bridge=_FakeBridge(), perception=perception, tracker=tracker)
    loop._track_log = True  # post-action window (animations only follow a move)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    perception.step()
    obs = loop.process_frame(frame)
    assert obs["popups"][0]["subtype"] == "item"
    # Azumarill is the active opponent (p2a) in the seeded tracker.
    assert tracker.state.mons["p2a"].item == "sitrusberry"
    assert tracker.state.mons["p2a"].item_revealed


def test_popup_processed_once_while_it_lingers():
    perception = _FakePerception([("ANIMATION", None), ("ANIMATION", None)])
    perception.popups = ["Gyarados's Intimidate"]
    loop = ShadowLoop(bridge=_FakeBridge(), perception=perception, tracker=_seed_tracker())
    loop._track_log = True  # post-action window (animations only follow a move)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    perception.step()
    first = loop.process_frame(frame)
    perception.step()
    second = loop.process_frame(frame)
    assert "popups" in first
    assert "popups" not in second  # same banner still on screen -> not re-fired


def test_log_skipped_outside_post_action_window():
    # The window is closed by default, so a stray ANIMATION frame's log box (which
    # could be showing the lingering timer) is not OCR'd / applied.
    perception = _FakePerception([("ANIMATION", "Azumarill fainted!")])
    tracker = _seed_tracker()
    loop = ShadowLoop(bridge=_FakeBridge(), perception=perception, tracker=tracker)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    perception.step()
    obs = loop.process_frame(frame)
    assert obs["track_log"] is False
    assert obs["log_text"] is None


def test_decision_point_closes_log_window():
    # A fresh decision point ends the previous turn's window (timer overlaps log box).
    perception = _FakePerception([("TURN_DECISION", "ignored 06:48")])
    loop = ShadowLoop(bridge=_FakeBridge(), perception=perception, tracker=_seed_tracker())
    loop._track_log = True
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    perception.step()
    obs = loop.process_frame(frame)
    assert obs["track_log"] is False
    assert obs["log_text"] is None


def test_screenshots_saved_at_interval_with_jsonl(tmp_path):
    script = [("ANIMATION", "Gyarados's Attack rose!")]
    perception = _FakePerception(script)
    loop = ShadowLoop(
        bridge=_FakeBridge(),
        perception=perception,
        tracker=_seed_tracker(),
        save_screenshots=True,
        screenshot_dir=tmp_path,
        screenshot_interval=0.0,  # save every iteration
    )
    loop._track_log = True  # post-action window (animations only follow a move)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    perception.step()
    obs = loop.process_frame(frame)
    loop._maybe_save_screenshot(frame, obs)

    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 1
    jsonl = tmp_path / "shadow_log.jsonl"
    assert jsonl.exists()
    line = json.loads(jsonl.read_text().strip())
    assert line["state"] == "ANIMATION"
    assert line["log_text"] == "Gyarados's Attack rose!"
    assert line["event"]["type"] == "stat_boost"


def test_screenshot_interval_throttles(tmp_path):
    loop = ShadowLoop(
        bridge=_FakeBridge(),
        perception=_FakePerception([]),
        tracker=_seed_tracker(),
        save_screenshots=True,
        screenshot_dir=tmp_path,
        screenshot_interval=999.0,  # only the first capture should pass
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    obs = {"state": "ANIMATION", "log_text": None, "event": None, "decided": False}
    loop._maybe_save_screenshot(frame, obs)
    loop._maybe_save_screenshot(frame, obs)
    assert len(list(tmp_path.glob("*.png"))) == 1


def test_screenshots_pruned_to_keep_limit(tmp_path):
    loop = ShadowLoop(
        bridge=_FakeBridge(),
        perception=_FakePerception([]),
        tracker=_seed_tracker(),
        save_screenshots=True,
        screenshot_dir=tmp_path,
        screenshot_interval=0.0,  # save every call
        screenshot_keep=3,
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    obs = {"state": "ANIMATION", "log_text": None, "event": None, "decided": False}
    for _ in range(10):
        loop._maybe_save_screenshot(frame, obs)
    assert len(list(tmp_path.glob("*.png"))) == 3


def test_screenshots_disabled(tmp_path):
    loop = ShadowLoop(
        bridge=_FakeBridge(),
        perception=_FakePerception([]),
        tracker=_seed_tracker(),
        save_screenshots=False,
        screenshot_dir=tmp_path,
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    obs = {"state": "ANIMATION", "log_text": None, "event": None, "decided": False}
    loop._maybe_save_screenshot(frame, obs)
    assert list(tmp_path.glob("*.png")) == []
