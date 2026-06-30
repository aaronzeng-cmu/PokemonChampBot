"""SpeciesRecognizer: CNN-first with pHash fallback + closed-set handling."""

from __future__ import annotations

import numpy as np

from src.cv_bridge.species_classifier.recognizer import SpeciesRecognizer


class _FakeMatcher:
    ready = True

    def __init__(self):
        self.calls = 0

    def build_index(self):
        pass

    def identify_sprite(self, crop, *, exclude_forms=False, allowed=None):
        self.calls += 1
        return "fallback_species"

    def rank_sprite(self, crop, *, top_n=10, exclude_forms=False, allowed=None):
        return {"ranked": [("fallback_species", 1.0)], "best_species_id": "fallback_species"}


class _FakeCNN:
    def __init__(self, species, prob):
        self.species, self.prob = species, prob

    def rank_sprite(self, crop, *, top_n=1, exclude_forms=False, allowed=None):
        return {
            "ranked": [(self.species, self.prob)],
            "best_species_id": self.species,
            "best_prob": self.prob,
            "margin": self.prob,
        }


def _rec(cnn, matcher):
    r = SpeciesRecognizer(matcher=matcher, conf_min=0.5)
    r._cnn = cnn  # inject, skip real load
    return r


CROP = np.zeros((8, 8, 3), dtype=np.uint8)


def test_high_confidence_cnn_wins():
    m = _FakeMatcher()
    r = _rec(_FakeCNN("garchomp", 0.95), m)
    assert r.identify_sprite(CROP) == "garchomp"
    assert m.calls == 0  # fallback untouched


def test_low_confidence_cnn_falls_back_to_phash():
    m = _FakeMatcher()
    r = _rec(_FakeCNN("garchomp", 0.20), m)
    assert r.identify_sprite(CROP) == "fallback_species"
    assert m.calls == 1


def test_closed_set_trusts_confident_cnn_pick():
    m = _FakeMatcher()
    r = _rec(_FakeCNN("garchomp", 0.80), m)
    assert r.identify_sprite(CROP, allowed={"garchomp", "staraptor"}) == "garchomp"
    assert m.calls == 0  # closed set: CNN authoritative, no pHash fallback


def test_closed_set_rejects_low_conf_pick_as_unknown():
    """Empty / mid-animation slot scores low even within the closed set.

    Returning "unknown" (instead of the nearest allowed species) lets the tracker
    keep the prior species rather than latching a confident-but-wrong late-game read.
    """
    m = _FakeMatcher()
    r = _rec(_FakeCNN("garchomp", 0.20), m)
    assert r.identify_sprite(CROP, allowed={"garchomp", "staraptor"}) == "unknown"
    assert m.calls == 0  # no pHash fallback in closed set; gate is authoritative


def test_falls_back_when_cnn_unavailable():
    m = _FakeMatcher()
    r = SpeciesRecognizer(matcher=m)
    r._cnn_failed = True  # simulate missing weights / ultralytics
    assert r.identify_sprite(CROP) == "fallback_species"
    assert m.calls == 1
