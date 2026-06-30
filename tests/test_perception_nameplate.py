"""Nameplate OCR -> fuzzy match against our known closed set of 6."""

from __future__ import annotations

from src.cv_bridge.perception import PerceptionModule


def _module_with_team(species: list[str]) -> PerceptionModule:
    # Bypass heavy __init__ (templates / OCR / sprite index); we only exercise
    # the pure fuzzy-matching logic here.
    p = PerceptionModule.__new__(PerceptionModule)
    p.set_own_team(species)
    return p


def test_fuzzy_snaps_garbled_ocr_to_known_species():
    p = _module_with_team(
        ["garchomp", "staraptor", "tyranitar", "rotomwash", "azumarill", "ferrothorn"]
    )
    assert p._fuzzy_own_species("Garchompl") == "garchomp"
    assert p._fuzzy_own_species("STARAPTOR") == "staraptor"
    assert p._fuzzy_own_species("Tyranltar") == "tyranitar"


def test_fuzzy_handles_runon_two_nameplates():
    p = _module_with_team(["staraptor", "garchomp"])
    # Two nameplates merged into one OCR line still resolve via containment.
    assert p._fuzzy_own_species("Staraptor and Garchompl") == "staraptor"


def test_fuzzy_rejects_unrelated_text():
    p = _module_with_team(["garchomp", "staraptor"])
    assert p._fuzzy_own_species("Fight") is None
    assert p._fuzzy_own_species("") is None


def test_fuzzy_noop_without_team():
    p = _module_with_team([])
    assert p._fuzzy_own_species("Garchomp") is None
