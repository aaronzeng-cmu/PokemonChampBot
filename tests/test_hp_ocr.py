"""HP OCR: prove preprocessing recovers exact fractions and percentages.

These run EasyOCR against saved HUD crops. They are skipped automatically if the
reference crops or the easyocr model are unavailable.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.cv_bridge.ocr_utils import (
    get_hp_percentage_from_bar,
    parse_hp_percent,
    parse_hp_text,
    preprocess_hp_crop,
)

_HP_DEBUG = Path("logs/cv_bridge/analysis/hp_debug")
_SCREENSHOT = Path("logs/cv_bridge/screenshots/20260624_232432.png")
_HP_REGIONS = {
    "player_active_hp_slot_a": (248, 1005, 156, 47),
    "player_active_hp_slot_b": (645, 1005, 155, 51),
    "opp_active_hp_slot_a": (1312, 111, 138, 61),
    "opp_active_hp_slot_b": (1720, 122, 124, 52),
}


@lru_cache(maxsize=1)
def _reader():
    try:
        import easyocr
    except ImportError:  # pragma: no cover
        pytest.skip("easyocr not installed")
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def _crop(key: str) -> np.ndarray:
    saved = _HP_DEBUG / f"{key}.png"
    if saved.exists():
        img = cv2.imread(str(saved))
        if img is not None:
            return img
    if not _SCREENSHOT.exists():
        pytest.skip(f"No crop or screenshot available for {key}")
    frame = cv2.imread(str(_SCREENSHOT))
    if frame is None:
        pytest.skip("Reference screenshot unreadable")
    x, y, w, h = _HP_REGIONS[key]
    return frame[y : y + h, x : x + w]


def test_preprocess_outputs_binary_image():
    proc = preprocess_hp_crop(_crop("player_active_hp_slot_a"))
    assert proc is not None
    assert proc.ndim == 2
    assert set(np.unique(proc)).issubset({0, 255})


def test_preprocess_handles_empty():
    assert preprocess_hp_crop(None) is None
    assert preprocess_hp_crop(np.zeros((0, 0, 3), dtype=np.uint8)) is None


def test_parse_hp_text_reads_137_fraction():
    assert parse_hp_text(_crop("player_active_hp_slot_a"), _reader()) == (137, 137)


def test_parse_hp_text_reads_172_fraction():
    assert parse_hp_text(_crop("player_active_hp_slot_b"), _reader()) == (172, 172)


def test_parse_enemy_hp_percent():
    assert parse_hp_percent(_crop("opp_active_hp_slot_a"), _reader()) == 100.0
    assert parse_hp_percent(_crop("opp_active_hp_slot_b"), _reader()) == 100.0


def test_known_max_recovers_current_when_slash_missing():
    # No reader call: empty crop path is separate; here we check the digit-join logic
    # by constructing a synthetic crop is overkill, so validate the public contract
    # via a tiny fake reader.
    class _FakeReader:
        def __init__(self, text):
            self._text = text

        def readtext(self, *_a, **_k):
            return [self._text]

    crop = np.full((20, 60, 3), 255, dtype=np.uint8)
    # Slash dropped, digits run together -> "120137"; known_max=137 -> current 120.
    assert parse_hp_text(crop, _FakeReader("120137"), known_max=137) == (120, 137)
    # Single number with known max is clamped.
    assert parse_hp_text(crop, _FakeReader("88"), known_max=137) == (88, 137)


def test_bar_percentage_full_and_empty():
    full = np.zeros((20, 100, 3), dtype=np.uint8)
    full[:, :, 1] = 255  # solid green
    assert get_hp_percentage_from_bar(full) > 0.95

    empty = np.zeros((20, 100, 3), dtype=np.uint8)
    assert get_hp_percentage_from_bar(empty) == 0.0

    half = np.zeros((20, 100, 3), dtype=np.uint8)
    half[:, :50, 1] = 255  # green left half
    assert 0.4 <= get_hp_percentage_from_bar(half) <= 0.6
