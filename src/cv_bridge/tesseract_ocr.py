"""Tesseract (pytesseract) OCR path for structured game text.

Implements the fast, traditional-OCR pipeline:

* **ROI** is supplied by the caller (we never OCR the whole frame).
* **Preprocess**: grayscale -> integer upscale with *nearest-neighbour* (keeps the
  pixel-font edges crisp instead of blurring them) -> Otsu binarization to pure
  black text on a pure white field.
* **Config**: ``--psm 7`` for single lines (HP / nameplate / timer) and ``--psm 6``
  for uniform text blocks (battle log / popups), with an optional char whitelist.

This mirrors the ``SpriteMatcher``/EasyOCR surface enough to be dropped into the
comparison harness (``scripts/compare_ocr_engines.py``).
"""

from __future__ import annotations

import re
import shutil
import time
from typing import Any

import cv2
import numpy as np

# Single line of text vs. a uniform block -- the two modes that fit HUD text.
PSM_SINGLE_LINE = 7
PSM_BLOCK = 6

_HP_FRACTION_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
_PERCENT_RE = re.compile(r"(\d+)\s*%")

_tesseract_ready: bool | None = None


def tesseract_available() -> bool:
    """True when the Tesseract binary + pytesseract wrapper are importable.

    Also pins ``pytesseract.tesseract_cmd`` to the resolved binary so the path
    works even when the conda env isn't on ``PATH`` for the calling process.
    """
    global _tesseract_ready
    if _tesseract_ready is not None:
        return _tesseract_ready
    try:
        import pytesseract

        binary = shutil.which("tesseract")
        if binary:
            pytesseract.pytesseract.tesseract_cmd = binary
        pytesseract.get_tesseract_version()
        _tesseract_ready = True
    except Exception:
        _tesseract_ready = False
    return _tesseract_ready


def preprocess_for_tesseract(
    crop: np.ndarray | None, *, scale: int = 3, invert: bool = True
) -> np.ndarray | None:
    """Grayscale -> nearest-neighbour upscale -> Otsu binarize (black-on-white).

    ``invert=True`` (the default) maps bright glyphs (the HUD's white text) to
    black on a white field, which is the polarity Tesseract reads best.
    """
    if crop is None or crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if scale and scale > 1:
        gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    thresh = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)[1]
    return cv2.copyMakeBorder(thresh, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)


def _config(psm: int, whitelist: str | None) -> str:
    cfg = f"--oem 1 --psm {psm}"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    return cfg


def ocr_tokens(
    processed: np.ndarray | None,
    *,
    psm: int = PSM_BLOCK,
    whitelist: str | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Run Tesseract on a *preprocessed* image; return (tokens, latency_ms).

    Tokens are ``{"text", "conf"}`` with conf in ``[0, 1]`` (Tesseract reports
    0-100; non-text boxes report -1 and are dropped).
    """
    if processed is None or processed.size == 0:
        return [], 0.0
    import pytesseract
    from pytesseract import Output

    cfg = _config(psm, whitelist)
    t0 = time.perf_counter()
    data = pytesseract.image_to_data(processed, config=cfg, output_type=Output.DICT)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    tokens: list[dict[str, Any]] = []
    for text, conf in zip(data.get("text", []), data.get("conf", []), strict=False):
        text = str(text).strip()
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            conf_f = -1.0
        if text and conf_f >= 0:
            tokens.append({"text": text, "conf": conf_f / 100.0})
    return tokens, latency_ms


def ocr_text(
    crop: np.ndarray | None,
    *,
    psm: int = PSM_BLOCK,
    whitelist: str | None = None,
    scale: int = 3,
    invert: bool = True,
    min_conf: float = 0.0,
) -> str:
    """End-to-end: preprocess ``crop`` then OCR it, joining confident tokens."""
    processed = preprocess_for_tesseract(crop, scale=scale, invert=invert)
    tokens, _ = ocr_tokens(processed, psm=psm, whitelist=whitelist)
    return " ".join(t["text"] for t in tokens if t["conf"] >= min_conf).strip()


def parse_hp_text(crop: np.ndarray | None, **kw: Any) -> tuple[int, int] | None:
    """Read a ``current/max`` HP fraction via Tesseract (digits + ``/`` only)."""
    text = ocr_text(crop, psm=PSM_SINGLE_LINE, whitelist="0123456789/", **kw)
    m = _HP_FRACTION_RE.search(text)
    if not m:
        return None
    cur, mx = int(m.group(1)), int(m.group(2))
    if mx <= 0 or cur > mx * 1.2:
        return None
    return cur, mx


def parse_hp_percent(crop: np.ndarray | None, **kw: Any) -> float | None:
    """Read an enemy ``NN%`` HP readout via Tesseract (digits + ``%`` only)."""
    text = ocr_text(crop, psm=PSM_SINGLE_LINE, whitelist="0123456789%", **kw)
    m = _PERCENT_RE.search(text)
    if not m:
        return None
    pct = float(m.group(1))
    return pct if 0 <= pct <= 100 else None
