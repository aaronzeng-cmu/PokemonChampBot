"""Trained-model policies for the live CV battle loop.

Two thin wrappers turn raw model outputs into the action shapes the
``ActionExecutor`` expects:

* :class:`BCPolicy` -- the per-turn BC transformer. Doubles uses sequential
  masked argmax (slot A, then slot B re-masked given A); singles bypasses slot B
  and returns a single canonical index.
* :class:`PreviewPolicy` -- the team-preview model. Returns the 1-based roster
  slots to bring (2 leads + 2 back for doubles, 1 lead + back for singles BSS).

Both are defensive: any inference failure or all-zero mask degrades to a safe
"pass / first legal" action rather than crashing the loop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from poke_env.data import to_id_str

from src.core.model.preview_model import (
    load_preview_model,
    predict_preview_slots,
    predict_singles_preview_slots,
)
from src.core.model.transformer_bot import load_model
from src.doubles.battle.move_order import apply_joint_slot1_mask_numpy
from src.doubles.data.log_action_mask import pick_masked_argmax

Action = int | tuple[int, int]
Masks = dict[str, np.ndarray] | None


class BCPolicy:
    """Per-turn policy backed by the trained BC transformer checkpoint."""

    def __init__(self, model_path: Path | str, *, device: str = "cpu"):
        self.device = device
        self.model = load_model(Path(model_path), device=device)
        self.model.eval()
        self.is_singles = self.model.action_space == "singles"

    def __call__(self, obs: np.ndarray, masks: Masks) -> Action:
        try:
            return self._select(obs, masks)
        except Exception as exc:  # desync / shape / all-zero mask -> stay safe.
            safe: Action = 0 if self.is_singles else (0, 0)
            print(f"[POLICY] inference failed ({exc!r}); falling back to {safe}")
            return safe

    def _select(self, obs: np.ndarray, masks: Masks) -> Action:
        x = torch.as_tensor(np.asarray(obs), dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.model(x)

        if self.is_singles:
            logits = (out[0] if isinstance(out, tuple) else out)[0]
            return self._masked_argmax(logits, masks.get("slot_a") if masks else None)

        logits0, logits1 = out
        mask_a = masks.get("slot_a") if masks else None
        mask_b = masks.get("slot_b") if masks else None

        ca0 = self._masked_argmax(logits0[0], mask_a)
        if mask_b is not None:
            mask_b = apply_joint_slot1_mask_numpy(mask_b, a0_canonical=ca0, force_switch=False)
        ca1 = self._masked_argmax(logits1[0], mask_b)
        return ca0, ca1

    @staticmethod
    def _masked_argmax(logits: torch.Tensor, mask: np.ndarray | None) -> int:
        if mask is not None:
            # pick_masked_argmax already returns 0 (pass) when the mask is empty.
            return pick_masked_argmax(logits, mask)
        return int(torch.argmax(logits).item())


class PreviewPolicy:
    """Team-preview policy backed by the trained TeamPreviewModel checkpoint."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        battle_format: str = "doubles",
        device: str = "cpu",
        pick_count: int | None = None,
    ):
        self.device = device
        self.battle_format = battle_format
        self.pick_count = pick_count
        self.model = load_preview_model(Path(model_path), device=device)

    def __call__(self, ally_team: list[str], enemy_team: list[str]) -> list[int]:
        # Keep the full 6-length lists (incl. "unknown") so returned slot indices
        # line up 1:1 with the physical roster_slot_N tap coordinates.
        ally = [to_id_str(s) for s in ally_team]
        enemy = [to_id_str(s) for s in enemy_team]
        if self.battle_format == "singles":
            return predict_singles_preview_slots(
                self.model, ally, enemy, device=self.device, n_brought=self.pick_count or 3
            )
        return predict_preview_slots(self.model, ally, enemy, device=self.device)
