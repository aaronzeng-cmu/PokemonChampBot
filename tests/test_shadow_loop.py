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


def _run(script, *, policy=None):
    perception = _FakePerception(script)
    tracker = _seed_tracker()
    loop = ShadowLoop(
        bridge=_FakeBridge(),
        perception=perception,
        tracker=tracker,
        policy=policy,
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    for _ in script:
        perception.step()
        loop.process_frame(frame)
    return loop, tracker


def test_log_event_applied_once_and_deduped():
    script = [
        ("ANIMATION", "Gyarados's Attack rose!"),
        ("ANIMATION", "Gyarados's Attack rose!"),  # duplicate -> ignored
    ]
    _, tracker = _run(script)
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
    _, tracker = _run(script)
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
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    perception.step()
    first = loop.process_frame(frame)
    perception.step()
    second = loop.process_frame(frame)
    assert "popups" in first
    assert "popups" not in second  # same banner still on screen -> not re-fired


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
