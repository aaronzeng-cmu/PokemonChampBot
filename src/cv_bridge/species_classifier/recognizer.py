"""Unified species recognizer: YOLOv8-cls CNN first, pHash matcher as fallback.

Drop-in for ``SpriteMatcher`` (same ``identify_sprite`` / ``rank_sprite`` /
``ready`` / ``build_index`` surface). The CNN handles the hard, backgrounded
crops; if its confidence is low (or the model / ultralytics isn't available),
we fall back to the classic pHash + histogram matcher so the system degrades
gracefully rather than breaking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.cv_bridge.sprite_matcher import SpriteMatcher

_DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "assets" / "species_cls.pt"


class SpeciesRecognizer:
    def __init__(
        self,
        *,
        weights: str | Path = _DEFAULT_WEIGHTS,
        matcher: SpriteMatcher | None = None,
        conf_min: float = 0.5,
        closed_conf_min: float = 0.40,
        device: str | None = None,
    ):
        self.matcher = matcher or SpriteMatcher()
        self.weights = Path(weights)
        self.conf_min = conf_min
        # Confidence floor for *closed-set* picks. Even when the crop must be one of
        # the allowed species, an empty / animating / fainting slot is out-of-set and
        # scores low; gating here returns "unknown" so the tracker keeps the prior
        # species instead of latching a confident-but-wrong read (the late-game
        # "opponent shows our species / species flips" failure mode).
        self.closed_conf_min = closed_conf_min
        self.device = device
        self._cnn: Any | None = None
        self._cnn_failed = False

    @property
    def ready(self) -> bool:
        return self.matcher.ready

    def build_index(self) -> None:
        self.matcher.build_index()

    def known_species_ids(self) -> set[str]:
        """Union of the pHash icon vocabulary and the CNN's class names.

        The pHash index carries Mega/Primal icons even when the CNN was trained
        without those classes, so the union is the widest set of forms we can
        reason about for closed-set expansion.
        """
        ids = self.matcher.known_species_ids()
        cnn = self._cnn_model()
        if cnn is not None:
            ids = ids | cnn.known_species_ids()
        return ids

    def _cnn_model(self) -> Any | None:
        if self._cnn is None and not self._cnn_failed:
            try:
                from src.cv_bridge.species_classifier.classifier import SpeciesClassifier

                self._cnn = SpeciesClassifier(
                    self.weights, device=self.device, conf_min=self.conf_min
                )
                print(f"[recognizer] species CNN loaded: {self.weights.name}")
            except Exception as exc:  # missing weights / ultralytics / load error
                self._cnn_failed = True
                print(f"[recognizer] CNN unavailable ({exc!r}); using pHash matcher only")
        return self._cnn

    def identify_sprite(
        self,
        cropped_image: np.ndarray,
        *,
        exclude_forms: bool = False,
        allowed: set[str] | None = None,
    ) -> str:
        cnn = self._cnn_model()
        if cnn is not None:
            res = cnn.rank_sprite(
                cropped_image, top_n=1, exclude_forms=exclude_forms, allowed=allowed
            )
            best = res["best_species_id"]
            prob = res["best_prob"] or 0.0
            if allowed is not None:
                # Closed set: the CNN is authoritative, but still require a minimum
                # confidence so an empty / mid-animation slot doesn't latch the
                # nearest allowed species. Below the floor -> "unknown" (keep prior).
                return best if (best is not None and prob >= self.closed_conf_min) else "unknown"
            # Open set: only trust a confident CNN call; else fall back to pHash.
            if best is not None and prob >= self.conf_min:
                return best
        return self.matcher.identify_sprite(
            cropped_image, exclude_forms=exclude_forms, allowed=allowed
        )

    def rank_sprite(
        self,
        cropped_image: np.ndarray,
        *,
        top_n: int = 10,
        exclude_forms: bool = False,
        allowed: set[str] | None = None,
    ) -> dict[str, Any]:
        cnn = self._cnn_model()
        if cnn is not None:
            return cnn.rank_sprite(
                cropped_image, top_n=top_n, exclude_forms=exclude_forms, allowed=allowed
            )
        return self.matcher.rank_sprite(
            cropped_image, top_n=top_n, exclude_forms=exclude_forms, allowed=allowed
        )
