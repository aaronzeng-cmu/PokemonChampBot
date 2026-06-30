"""Closed-set construction (base + Mega/Primal forms) and mega->base canon."""

from __future__ import annotations

import numpy as np

from src.cv_bridge.perception import PerceptionModule


class _FakeMatcher:
    """Stand-in recognizer exposing a fixed icon/class vocabulary."""

    def __init__(self, vocab: set[str]):
        self._vocab = vocab

    def known_species_ids(self) -> set[str]:
        return set(self._vocab)


def _module() -> PerceptionModule:
    # Bypass heavy __init__; wire just the recognizer vocabulary we need.
    p = PerceptionModule.__new__(PerceptionModule)
    p.sprite_matcher = _FakeMatcher(
        {"floettemega", "staraptormega", "milotic", "sinistcha", "garchompmega"}
    )
    return p


def test_set_own_team_widens_closed_set_with_megas():
    p = _module()
    p.set_own_team(["floetteeternal", "milotic", "sinistcha"])
    # Base members are present...
    assert {"floetteeternal", "milotic", "sinistcha"} <= p._own_team_keys
    # ...and the irregular Mega form of Floette-Eternal is added to the set.
    assert "floettemega" in p._own_team_keys
    assert p._own_mega_to_base == {"floettemega": "floetteeternal"}
    # Off-team megas (staraptor/garchomp) are not pulled in.
    assert "staraptormega" not in p._own_team_keys


def test_set_enemy_team_builds_independent_closed_set():
    p = _module()
    p.set_enemy_team(["staraptor", "garchomp"])
    assert {"staraptor", "garchomp"} <= p._enemy_team_keys
    assert p._enemy_mega_to_base == {
        "staraptormega": "staraptor",
        "garchompmega": "garchomp",
    }


def test_canon_species_maps_mega_back_to_roster_base():
    mega_map = {"floettemega": "floetteeternal"}
    assert PerceptionModule._canon_species("floettemega", mega_map) == "floetteeternal"
    # Robust to separators/casing via normalization.
    assert PerceptionModule._canon_species("Floette-Mega", mega_map) == "floetteeternal"
    # Non-mega ids pass through untouched.
    assert PerceptionModule._canon_species("milotic", mega_map) == "milotic"


def test_extract_slot_canonicalizes_megaevolved_active_to_base():
    p = _module()
    p.set_own_team(["floetteeternal", "milotic"])
    p.ocr_enabled = False
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    p._crop_region = lambda frame, key: np.zeros((4, 4, 3), dtype=np.uint8)
    p._read_ally_hp = lambda crop, mx: {
        "hp": None,
        "max_hp": None,
        "hp_percent": None,
        "hp_text": "",
    }
    p._read_nameplate_species = lambda frame, key: None
    # The recognizer sees the mega sprite and returns the mega id...
    p._identify_sprite_crop = (
        lambda crop, *, exclude_forms=False, allowed=None: "floettemega"
    )
    slot = p._extract_slot(frame, "hp", "sprite", name_key="n")
    # ...but the tracked identity is the roster base.
    assert slot["species_id"] == "floetteeternal"


def test_build_closed_set_survives_missing_recognizer():
    # __new__ instance has no sprite_matcher -> base-only closed set, no crash.
    p = PerceptionModule.__new__(PerceptionModule)
    p.set_own_team(["garchomp", "milotic"])
    assert p._own_team_keys == {"garchomp", "milotic"}
    assert p._own_mega_to_base == {}


def _anchor_module() -> PerceptionModule:
    p = PerceptionModule.__new__(PerceptionModule)
    # Only the anchor region matters for force-switch detection.
    p.regions = {"force_switch_anchor": [1542, 166, 324, 52]}
    p.ocr_enabled = False
    return p


def test_force_switch_detected_by_red_pill_colour():
    p = _anchor_module()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Paint the anchor region with the saturated red name-pill colour (BGR red).
    x, y, w, h = p.regions["force_switch_anchor"]
    frame[y : y + h, x : x + w] = (40, 40, 220)
    assert p._force_switch_anchor_red_fraction(frame) > 0.4
    assert p._detect_force_switch(frame) is True


def test_force_switch_not_detected_on_dark_menu():
    p = _anchor_module()
    # A near-black battle menu region has no red pill -> not the party screen.
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert p._force_switch_anchor_red_fraction(frame) < 0.05
    assert p._detect_force_switch(frame) is False
