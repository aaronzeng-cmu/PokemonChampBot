"""Lightweight CNN species classifier (YOLOv8-cls) as an alternative to the
pHash + HSV-histogram sprite matcher.

The classifier is trained on synthetic composites (official menu sprites pasted
onto randomized backgrounds with scale/rotation/blur/lighting augmentation) and
exposes the same ``identify_sprite`` surface as ``SpriteMatcher`` so it can drop
into ``perception.py`` / ``team_init.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.cv_bridge.species_classifier.classifier import SpeciesClassifier
    from src.cv_bridge.species_classifier.recognizer import SpeciesRecognizer

__all__ = ["SpeciesClassifier", "SpeciesRecognizer"]


def __getattr__(name: str):  # lazy import so synth_data doesn't pull in ultralytics
    if name == "SpeciesClassifier":
        from src.cv_bridge.species_classifier.classifier import SpeciesClassifier

        return SpeciesClassifier
    if name == "SpeciesRecognizer":
        from src.cv_bridge.species_classifier.recognizer import SpeciesRecognizer

        return SpeciesRecognizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
