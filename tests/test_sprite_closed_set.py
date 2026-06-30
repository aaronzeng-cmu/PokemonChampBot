"""Closed-set (allowed-species) sprite matching used for our-side actives."""

from __future__ import annotations

import types

import numpy as np

import src.cv_bridge.sprite_matcher as smod
from src.cv_bridge.sprite_matcher import SpriteMatcher, _norm_species, mega_forms_in_vocab


def test_norm_species_strips_separators():
    assert _norm_species("Ho-Oh") == "hooh"
    assert _norm_species("Mr. Mime") == "mrmime"
    assert _norm_species("Garchomp") == "garchomp"


def test_mega_forms_in_vocab_maps_regular_and_irregular_pairs():
    vocab = {
        "floettemega",
        "staraptormega",
        "raichumegax",
        "raichumegay",
        "charizardmegax",
        "milotic",  # not a form
        "meganium",  # ends in "ium", not a battle form despite containing "mega"
    }
    # Irregular pair: roster id 'floetteeternal' resolves to icon 'floettemega'.
    out = mega_forms_in_vocab(["floetteeternal", "staraptor", "raichu", "milotic"], vocab)
    assert out == {
        "floettemega": "floetteeternal",
        "staraptormega": "staraptor",
        "raichumegax": "raichu",
        "raichumegay": "raichu",
    }
    # Bases not on the roster (charizard) and non-forms (meganium) are ignored.
    assert "charizardmegax" not in out
    assert "meganium" not in out


def test_rank_allowed_restricts_to_closed_set(monkeypatch):
    sm = SpriteMatcher.__new__(SpriteMatcher)
    sm._index = [
        types.SimpleNamespace(species_id=s, phashes=s, histograms=None)
        for s in ("garchomp", "staraptor", "pikachu", "gardevoir")
    ]
    dists = {"garchomp": 0.30, "staraptor": 0.20, "pikachu": 0.05, "gardevoir": 0.10}
    monkeypatch.setattr(smod, "_choose_mask", lambda bgr: None)
    monkeypatch.setattr(smod, "_features_from_bgr", lambda bgr, mask: None)
    monkeypatch.setattr(smod, "_combined_distance", lambda q, feats: dists[feats[0]])

    img = np.zeros((8, 8, 3), dtype=np.uint8)
    ranked = sm._rank(img, allowed={"garchomp", "staraptor"})
    # Off-team species (pikachu/gardevoir) are excluded even though they're closer.
    assert {sid for _, sid in ranked} == {"garchomp", "staraptor"}
    assert ranked[0][1] == "staraptor"  # nearest within the closed set


def test_closed_set_match_returns_best_despite_threshold(monkeypatch):
    sm = SpriteMatcher.__new__(SpriteMatcher)
    sm.max_distance = 0.1  # strict gate that the open-set path would trip
    sm.min_margin = 0.5
    monkeypatch.setattr(
        sm, "_rank", lambda *a, **k: [(0.9, "garchomp"), (0.95, "staraptor")]
    )
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    # Open set: distance 0.9 > max -> unknown.
    assert sm.match_sprite(img).species_id == "unknown"
    # Closed set: it must be one of ours, so the nearest wins despite the strict
    # open-set gate (0.9 is within the lenient closed-set ceiling).
    assert sm.match_sprite(img, allowed={"garchomp", "staraptor"}).species_id == "garchomp"


def test_closed_set_rejects_crop_resembling_no_allowed_icon(monkeypatch):
    """An empty / mid-animation slot is far from every allowed icon -> unknown.

    Late-game guard: closed-set matching keeps a lenient absolute-distance gate so
    a junk crop doesn't latch the nearest allowed species (and clobber the tracker).
    """
    sm = SpriteMatcher.__new__(SpriteMatcher)
    sm.max_distance = 0.78
    sm.min_margin = 0.006
    sm.closed_max_distance = 0.92
    # Best candidate distance 0.97 > closed ceiling -> rejected even in closed set.
    monkeypatch.setattr(
        sm, "_rank", lambda *a, **k: [(0.97, "garchomp"), (0.99, "staraptor")]
    )
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    assert sm.match_sprite(img, allowed={"garchomp", "staraptor"}).species_id == "unknown"
