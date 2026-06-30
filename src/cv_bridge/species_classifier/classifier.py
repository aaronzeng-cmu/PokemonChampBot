"""Inference wrapper around a trained YOLOv8-cls species model.

Mirrors the ``SpriteMatcher`` surface (``identify_sprite`` / ``rank_sprite``,
with ``exclude_forms`` and closed-set ``allowed``) so it can be swapped in
wherever the pHash matcher is used.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "assets" / "species_cls.pt"
_BATTLE_FORM_RE = re.compile(r"(mega[xy]?|primal|gmax|eternamax)$")


def _norm_species(species_id: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(species_id).lower())


def _is_battle_only_form(species_id: str) -> bool:
    return bool(_BATTLE_FORM_RE.search(species_id))


class SpeciesClassifier:
    """YOLOv8-cls species recognizer with a SpriteMatcher-compatible API."""

    def __init__(
        self,
        weights: str | Path = _DEFAULT_WEIGHTS,
        *,
        device: str | None = None,
        conf_min: float = 0.30,
    ):
        from ultralytics import YOLO

        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(
                f"species classifier weights not found: {self.weights}. "
                "Train with scripts/train_species_classifier.py first."
            )
        self.model = YOLO(str(self.weights))
        self.names: dict[int, str] = dict(self.model.names)
        self.device = device
        self.conf_min = conf_min

    @property
    def ready(self) -> bool:
        return True

    def build_index(self) -> None:  # API parity with SpriteMatcher; no-op
        return None

    def known_species_ids(self) -> set[str]:
        """Normalized ids of every class the model can output."""
        return {_norm_species(name) for name in self.names.values()}

    def _probs(self, crop: np.ndarray) -> np.ndarray:
        res = self.model.predict(crop, verbose=False, device=self.device)[0]
        return res.probs.data.detach().cpu().numpy().astype(np.float32)

    def rank_sprite(
        self,
        cropped_image: np.ndarray,
        *,
        top_n: int = 10,
        exclude_forms: bool = False,
        allowed: set[str] | None = None,
    ) -> dict[str, Any]:
        empty: dict[str, Any] = {
            "ranked": [],
            "best_species_id": None,
            "best_prob": None,
            "margin": None,
        }
        if cropped_image is None or cropped_image.size == 0:
            return empty
        probs = self._probs(cropped_image)

        def keep(idx: int) -> bool:
            name = self.names[idx]
            if exclude_forms and _is_battle_only_form(name):
                return False
            if allowed is not None and _norm_species(name) not in allowed:
                return False
            return True

        cand = [i for i in range(len(probs)) if keep(i)]
        cand.sort(key=lambda i: -probs[i])
        if not cand:
            return empty
        ranked = [(self.names[i], float(probs[i])) for i in cand[:top_n]]
        best_prob = ranked[0][1]
        margin = best_prob - ranked[1][1] if len(ranked) > 1 else best_prob
        return {
            "ranked": ranked,
            "best_species_id": ranked[0][0],
            "best_prob": best_prob,
            "margin": float(margin),
        }

    def identify_sprite(
        self,
        cropped_image: np.ndarray,
        *,
        exclude_forms: bool = False,
        allowed: set[str] | None = None,
        conf_min: float | None = None,
    ) -> str:
        result = self.rank_sprite(
            cropped_image, top_n=2, exclude_forms=exclude_forms, allowed=allowed
        )
        best = result["best_species_id"]
        prob = result["best_prob"]
        if best is None:
            return "unknown"
        # Closed-set: it must be one of the allowed species -> take the best.
        if allowed is not None:
            return best
        threshold = self.conf_min if conf_min is None else conf_min
        if prob is not None and prob < threshold:
            return "unknown"
        return best
