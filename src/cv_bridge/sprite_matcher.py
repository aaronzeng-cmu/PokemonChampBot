"""Identify Pokémon menu sprites via color pHash + HSV histogram matching."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imagehash
import numpy as np
from PIL import Image

_DEFAULT_ICONS = Path(__file__).resolve().parent / "assets" / "pokemon_icons"
_DEFAULT_INDEX = Path(__file__).resolve().parent / "assets" / "icon_index.pkl"

_HASH_SIZE = 16
_SCALE_SIZES = (56, 64, 72, 80, 88)
_HIST_BINS = (16, 16, 16)
_INDEX_FEATURE_VERSION = 10
_PHASH_WEIGHT = 0.45
_HIST_WEIGHT = 0.55
_DEFAULT_MAX_DISTANCE = 0.78
_DEFAULT_MIN_MARGIN = 0.006


@dataclass(frozen=True)
class SpriteMatch:
    species_id: str
    distance: float
    margin: float


@dataclass
class _IndexedSprite:
    species_id: str
    phashes: tuple[imagehash.ImageHash, ...]
    histograms: tuple[np.ndarray, ...]


def _species_id_from_filename(path: Path) -> str:
    return path.stem.lower()


def _mask_enemy_background(bgr: np.ndarray) -> np.ndarray:
    """Mask magenta/red enemy-preview panels."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    bg = cv2.inRange(hsv, np.array([0, 55, 20], dtype=np.uint8), np.array([18, 255, 160], dtype=np.uint8))
    bg |= cv2.inRange(hsv, np.array([150, 55, 20], dtype=np.uint8), np.array([180, 255, 160], dtype=np.uint8))
    return cv2.bitwise_not(bg)


def _mask_ally_background(bgr: np.ndarray) -> np.ndarray:
    """Mask purple ally-preview row panels."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    bg = cv2.inRange(hsv, np.array([120, 40, 30], dtype=np.uint8), np.array([165, 255, 220], dtype=np.uint8))
    return cv2.bitwise_not(bg)


def _choose_mask(bgr: np.ndarray) -> np.ndarray:
    total = bgr.shape[0] * bgr.shape[1]
    full = np.full(bgr.shape[:2], 255, dtype=np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ui_bg = cv2.inRange(hsv, np.array([0, 55, 20], dtype=np.uint8), np.array([18, 255, 160], dtype=np.uint8))
    ui_bg |= cv2.inRange(hsv, np.array([150, 55, 20], dtype=np.uint8), np.array([180, 255, 160], dtype=np.uint8))
    ui_bg |= cv2.inRange(hsv, np.array([120, 40, 30], dtype=np.uint8), np.array([165, 255, 220], dtype=np.uint8))
    if float(ui_bg.mean()) < 20.0:
        return full

    best = full
    best_fg = total
    for mask in (_mask_enemy_background(bgr), _mask_ally_background(bgr), full):
        fg = int(mask.sum()) // 255
        if 0.18 * total <= fg <= 0.88 * total and fg < best_fg:
            best = mask
            best_fg = fg
    return best


def _foreground_bbox(mask: np.ndarray, *, pad: int = 2) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(mask.shape[1] - 1, int(xs.max()) + pad)
    y1 = min(mask.shape[0] - 1, int(ys.max()) + pad)
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _composite_on_white(bgr: np.ndarray, mask: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    alpha = np.clip(mask, 0, 255).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    pil = Image.fromarray(rgba, mode="RGBA")
    background = Image.new("RGBA", pil.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, pil).convert("RGB")


def _color_histogram(rgb: Image.Image) -> np.ndarray:
    arr = np.asarray(rgb)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    foreground = cv2.inRange(hsv, np.array([0, 20, 0], dtype=np.uint8), np.array([180, 255, 250], dtype=np.uint8))
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        foreground,
        list(_HIST_BINS),
        [0, 180, 0, 256, 0, 256],
    )
    return cv2.normalize(hist, hist).flatten()


def _multi_scale_features(rgb: Image.Image) -> tuple[tuple[imagehash.ImageHash, ...], tuple[np.ndarray, ...]]:
    phashes: list[imagehash.ImageHash] = []
    histograms: list[np.ndarray] = []
    for size in _SCALE_SIZES:
        scaled = rgb.resize((size, size), Image.Resampling.LANCZOS)
        phashes.append(imagehash.phash(scaled, hash_size=_HASH_SIZE))
        histograms.append(_color_histogram(scaled))
    return tuple(phashes), tuple(histograms)


def _features_from_bgr(bgr: np.ndarray, mask: np.ndarray | None = None) -> tuple[tuple[imagehash.ImageHash, ...], tuple[np.ndarray, ...]]:
    if mask is None:
        mask = np.full(bgr.shape[:2], 255, dtype=np.uint8)
    bbox = _foreground_bbox(mask)
    if bbox is not None:
        x, y, w, h = bbox
        bgr = bgr[y : y + h, x : x + w]
        mask = mask[y : y + h, x : x + w]
    rgb = _composite_on_white(bgr, mask)
    return _multi_scale_features(rgb)


def _features_from_icon_path(path: Path) -> tuple[tuple[imagehash.ImageHash, ...], tuple[np.ndarray, ...]] | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        return None

    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)
    elif image.shape[2] == 4:
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]
    else:
        bgr = image
        alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)

    opaque = (alpha > 16).astype(np.uint8) * 255
    if not np.any(opaque):
        return None
    return _features_from_bgr(bgr, opaque)


def _combined_distance(
    query: tuple[tuple[imagehash.ImageHash, ...], tuple[np.ndarray, ...]],
    reference: tuple[tuple[imagehash.ImageHash, ...], tuple[np.ndarray, ...]],
) -> float:
    q_phashes, q_hists = query
    r_phashes, r_hists = reference
    best = float("inf")
    for q_phash, q_hist in zip(q_phashes, q_hists, strict=True):
        for r_phash, r_hist in zip(r_phashes, r_hists, strict=True):
            ph_dist = (float(q_phash - r_phash)) / max(1.0, _HASH_SIZE * _HASH_SIZE)
            hist_dist = float(
                cv2.compareHist(
                    q_hist.astype(np.float32),
                    r_hist.astype(np.float32),
                    cv2.HISTCMP_BHATTACHARYYA,
                )
            )
            best = min(best, _PHASH_WEIGHT * ph_dist + _HIST_WEIGHT * hist_dist)
    return best


class SpriteMatcher:
    """Nearest-neighbour sprite identification against offline menu icons."""

    def __init__(
        self,
        *,
        icons_dir: Path | str | None = None,
        index_path: Path | str | None = None,
        max_distance: float = _DEFAULT_MAX_DISTANCE,
        min_margin: float = _DEFAULT_MIN_MARGIN,
    ) -> None:
        self.icons_dir = Path(icons_dir or _DEFAULT_ICONS)
        self.index_path = Path(index_path or _DEFAULT_INDEX)
        self.max_distance = max_distance
        self.min_margin = min_margin
        self._index: list[_IndexedSprite] = []

    @property
    def ready(self) -> bool:
        return bool(self._index)

    def build_index(self, *, force: bool = False) -> int:
        """Build (or load) the icon feature index. Returns entry count."""
        if not force and self.index_path.exists():
            self._index = self._load_index(self.index_path)
            if self._index:
                return len(self._index)

        if not self.icons_dir.exists():
            raise FileNotFoundError(
                f"Icon directory not found: {self.icons_dir}. "
                "Run: python -m src.cv_bridge.tools.download_icons"
            )

        entries: list[_IndexedSprite] = []
        for path in sorted(self.icons_dir.glob("*.png")):
            features = _features_from_icon_path(path)
            if features is None:
                continue
            phashes, histograms = features
            entries.append(
                _IndexedSprite(
                    species_id=_species_id_from_filename(path),
                    phashes=phashes,
                    histograms=histograms,
                )
            )

        if not entries:
            raise RuntimeError(f"No usable PNG icons found in {self.icons_dir}")

        self._index = entries
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_version": _INDEX_FEATURE_VERSION,
            "entries": [
                {
                    "species_id": entry.species_id,
                    "phashes": [str(ph) for ph in entry.phashes],
                    "histograms": [hist.astype(np.float32) for hist in entry.histograms],
                }
                for entry in entries
            ],
        }
        with self.index_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return len(entries)

    @staticmethod
    def _load_index(path: Path) -> list[_IndexedSprite]:
        with path.open("rb") as handle:
            raw = pickle.load(handle)
        if isinstance(raw, dict):
            if raw.get("feature_version") != _INDEX_FEATURE_VERSION:
                return []
            payload: list[dict[str, Any]] = raw.get("entries", [])
        else:
            payload = raw
        entries: list[_IndexedSprite] = []
        for item in payload:
            if "phashes" in item:
                phashes = tuple(imagehash.hex_to_hash(str(text)) for text in item["phashes"])
                histograms = tuple(np.asarray(hist, dtype=np.float32) for hist in item["histograms"])
            else:
                phashes = (imagehash.hex_to_hash(str(item["phash"])),)
                histograms = (np.asarray(item["histogram"], dtype=np.float32),)
            if histograms and histograms[0].size != int(np.prod(_HIST_BINS)):
                return []
            entries.append(
                _IndexedSprite(
                    species_id=str(item["species_id"]),
                    phashes=phashes,
                    histograms=histograms,
                )
            )
        return entries

    def _prepare_bgr(self, cropped_image: np.ndarray) -> np.ndarray:
        if cropped_image.ndim == 2:
            return cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
        if cropped_image.shape[2] == 4:
            return cropped_image[:, :, :3]
        return cropped_image

    def _rank(self, bgr: np.ndarray) -> list[tuple[float, str]]:
        if not self._index:
            self.build_index()
        bgr = self._prepare_bgr(bgr)
        mask = _choose_mask(bgr)
        query = _features_from_bgr(bgr, mask)
        ranked: list[tuple[float, str]] = []
        for entry in self._index:
            distance = _combined_distance(query, (entry.phashes, entry.histograms))
            ranked.append((distance, entry.species_id))
        ranked.sort(key=lambda item: item[0])
        return ranked

    def identify_sprite(self, cropped_image: np.ndarray) -> str:
        match = self.match_sprite(cropped_image)
        if match is None:
            return "unknown"
        return match.species_id

    def rank_sprite(self, cropped_image: np.ndarray, *, top_n: int = 10) -> dict[str, Any]:
        empty: dict[str, Any] = {
            "ranked": [],
            "decision": None,
            "best_distance": None,
            "best_species_id": None,
            "margin": None,
        }
        if cropped_image is None or cropped_image.size == 0:
            return empty
        ranked = self._rank(self._prepare_bgr(cropped_image))
        if not ranked:
            return empty
        best_distance, best_id = ranked[0]
        second_distance = ranked[1][0] if len(ranked) > 1 else float("inf")
        margin = second_distance - best_distance
        return {
            "ranked": [(species_id, float(dist)) for dist, species_id in ranked[:top_n]],
            "decision": self.match_sprite(cropped_image),
            "best_distance": float(best_distance),
            "best_species_id": best_id,
            "margin": float(margin),
        }

    def match_sprite(self, cropped_image: np.ndarray) -> SpriteMatch | None:
        if cropped_image is None or cropped_image.size == 0:
            return None
        ranked = self._rank(self._prepare_bgr(cropped_image))
        if not ranked:
            return None
        best_distance, best_id = ranked[0]
        second_distance = ranked[1][0] if len(ranked) > 1 else float("inf")
        margin = second_distance - best_distance
        if best_distance > self.max_distance or margin < self.min_margin:
            return SpriteMatch(species_id="unknown", distance=best_distance, margin=margin)
        return SpriteMatch(species_id=best_id, distance=best_distance, margin=margin)


def build_index(
    icons_dir: Path | str | None = None,
    index_path: Path | str | None = None,
    *,
    force: bool = False,
) -> int:
    matcher = SpriteMatcher(icons_dir=icons_dir, index_path=index_path)
    return matcher.build_index(force=force)


def identify_sprite(cropped_image: np.ndarray) -> str:
    return _default_matcher().identify_sprite(cropped_image)


_default: SpriteMatcher | None = None


def _default_matcher() -> SpriteMatcher:
    global _default
    if _default is None:
        _default = SpriteMatcher()
        if _default.index_path.exists() or _default.icons_dir.exists():
            try:
                _default.build_index()
            except (FileNotFoundError, RuntimeError):
                pass
    return _default
