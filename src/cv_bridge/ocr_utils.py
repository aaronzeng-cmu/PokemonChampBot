"""Robust OCR helpers for HP fractions and health-bar fallback.

The Champions HUD draws HP as **white glyphs with a dark navy outline** on top of a
colored HP bar (green / yellow / red) that fades into a dark background. Plain
grayscale + Otsu thresholding bleeds the bright bar into the text and garbles the
``/`` separator (e.g. ``137/137`` -> ``1374137``).

The fix is to isolate "whiteness" via the per-pixel channel minimum: white text is
high in every channel, while green/yellow/red bar pixels and the dark background all
have a low minimum. Thresholding that map cleanly separates the digits regardless of
the current bar colour.
"""

from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np

_HP_FRACTION_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
_PERCENT_RE = re.compile(r"(\d+)\s*%")
_DIGITS_RE = re.compile(r"\d+")

_HP_ALLOWLIST = "0123456789/"
_PERCENT_ALLOWLIST = "0123456789%"

# A current HP value should never exceed max by more than this slack (OCR sanity).
_MAX_HP_SLACK = 1.2


def preprocess_hp_crop(crop: np.ndarray | None, *, scale: int = 3) -> np.ndarray | None:
    """Upscale + isolate white text as black-on-white for OCR.

    Returns a single-channel uint8 image (black digits on a white field, padded)
    or ``None`` if the crop is empty.
    """
    if crop is None or crop.size == 0:
        return None

    if crop.ndim == 3:
        # White text is high in R, G and B; coloured bar / dark bg are not.
        whiteness = crop.min(axis=2)
    else:
        whiteness = crop

    if scale and scale > 1:
        whiteness = cv2.resize(
            whiteness, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    # Otsu adapts to the crop; text becomes white (255) on black (0).
    _, binary = cv2.threshold(whiteness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # EasyOCR is most reliable on dark text over a light field.
    binary = cv2.bitwise_not(binary)
    return cv2.copyMakeBorder(binary, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255)


def _read_text(reader: Any, image: np.ndarray, allowlist: str, *, min_conf: float = 0.0) -> str:
    if min_conf <= 0.0:
        lines = reader.readtext(image, allowlist=allowlist, detail=0)
        return " ".join(str(line) for line in lines).strip()
    items = reader.readtext(image, allowlist=allowlist, detail=1)
    return " ".join(str(t) for (_b, t, c) in items if float(c) >= min_conf).strip()


def read_text_lines(
    cropped_image: np.ndarray | None,
    reader: Any,
    *,
    scale: int = 2,
    min_conf: float = 0.0,
) -> str:
    """OCR free-form light text (e.g. the battle log) via whiteness isolation.

    No allowlist: returns the joined recognized text, or ``""`` when nothing is read.
    ``min_conf`` drops tokens EasyOCR isn't confident about (junk on inactive regions
    typically scores < 0.25 while real text scores > 0.7).
    """
    processed = preprocess_hp_crop(cropped_image, scale=scale)
    if processed is None:
        return ""
    if min_conf <= 0.0:
        lines = reader.readtext(processed, detail=0)
        return " ".join(str(line) for line in lines).strip()
    items = reader.readtext(processed, detail=1)
    return " ".join(str(t) for (_b, t, c) in items if float(c) >= min_conf).strip()


def parse_hp_text(
    cropped_image: np.ndarray,
    reader: Any,
    *,
    known_max: int | None = None,
    scale: int = 3,
    min_conf: float = 0.0,
) -> tuple[int, int] | None:
    """Read a ``current/max`` HP fraction from a HUD crop.

    Returns ``(current_hp, max_hp)`` or ``None`` when nothing usable is read.
    ``known_max`` (e.g. from the teambuilder) lets us recover the current HP when
    the ``/`` separator is missed and the digits run together.
    """
    processed = preprocess_hp_crop(cropped_image, scale=scale)
    if processed is None:
        return None

    text = _read_text(reader, processed, _HP_ALLOWLIST, min_conf=min_conf)

    match = _HP_FRACTION_RE.search(text)
    if match:
        current, maximum = int(match.group(1)), int(match.group(2))
        if maximum > 0 and current <= maximum * _MAX_HP_SLACK:
            return current, maximum

    digits = _DIGITS_RE.findall(text)
    if not digits:
        return None

    # Fallback: slash dropped. Try to recover using a known max HP.
    if known_max and known_max > 0:
        joined = "".join(digits)
        max_str = str(known_max)
        if joined.endswith(max_str) and len(joined) > len(max_str):
            current = int(joined[: -len(max_str)])
            return min(current, known_max), known_max
        return min(int(digits[0]), known_max), known_max

    if len(digits) >= 2:
        current, maximum = int(digits[0]), int(digits[1])
        if maximum > 0 and current <= maximum * _MAX_HP_SLACK:
            return current, maximum

    # Single number with no reference: assume it is the current value at full HP.
    value = int(digits[0])
    return value, value


def parse_hp_percent(
    cropped_image: np.ndarray,
    reader: Any,
    *,
    scale: int = 3,
    min_conf: float = 0.0,
) -> float | None:
    """Read an enemy ``NN%`` HP readout. Returns percent in ``[0, 100]`` or ``None``."""
    processed = preprocess_hp_crop(cropped_image, scale=scale)
    if processed is None:
        return None

    text = _read_text(reader, processed, _PERCENT_ALLOWLIST, min_conf=min_conf)

    match = _PERCENT_RE.search(text)
    if match:
        return float(min(100, max(0, int(match.group(1)))))

    digits = _DIGITS_RE.findall(text)
    if digits:
        return float(min(100, max(0, int(digits[0]))))
    return None


def get_hp_percentage_from_bar(cropped_image: np.ndarray) -> float:
    """Estimate HP fill from a coloured health bar via column-wise colour masking.

    Masks green / yellow / red bar pixels and returns the fraction of bar columns
    that contain bar colour (fill is left-anchored), in ``[0.0, 1.0]``. Use this for
    enemies that show only a visual bar with no numeric readout.
    """
    if cropped_image is None or cropped_image.size == 0:
        return 0.0
    if cropped_image.ndim != 3:
        return 0.0

    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    # Saturated, bright pixels only (excludes dark background / white text outline).
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    hue = hsv[:, :, 0]

    bright = (sat > 80) & (val > 80)
    green = (hue >= 35) & (hue <= 90)
    yellow = (hue >= 18) & (hue < 35)
    red = (hue <= 12) | (hue >= 168)
    bar_mask = bright & (green | yellow | red)

    width = bar_mask.shape[1]
    if width == 0:
        return 0.0

    # A column counts as "filled" if a meaningful number of its pixels are bar colour.
    col_counts = bar_mask.sum(axis=0)
    height = bar_mask.shape[0]
    filled_cols = int(np.count_nonzero(col_counts > max(1, height * 0.15)))
    return float(filled_cols) / float(width)
