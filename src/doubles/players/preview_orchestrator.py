"""Team preview (Turn 0) via trained TeamPreviewModel."""

from __future__ import annotations

import logging
from pathlib import Path

from poke_env.battle.double_battle import DoubleBattle
from poke_env.data import to_id_str

from src.core.model.preview_model import load_preview_model, predict_preview_slots
from src.doubles.teams.teampreview import random_teampreview_command

_logger = logging.getLogger(__name__)


class PreviewOrchestrator:
    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        device: str = "cpu",
    ):
        from config.settings import PREVIEW_MODEL_PATH

        path = Path(model_path) if model_path is not None else PREVIEW_MODEL_PATH
        self.device = device
        self._model = None
        self._model_path = path
        if path.is_file():
            self._model = load_preview_model(path, device=device)

    def _species_list(self, team_values) -> list[str]:
        return [to_id_str(p.base_species) for p in team_values]

    def teampreview(self, battle: DoubleBattle) -> str:
        if self._model is None:
            return random_teampreview_command(battle)

        # A crash during preview = an instant forfeit, so any failure in the
        # PyTorch model path falls back to a random 4-mon selection. The
        # exception is logged so the bad input/state can be debugged later.
        try:
            our = self._species_list(battle.team.values())
            opp = self._species_list(battle.opponent_team.values())
            slots = predict_preview_slots(
                self._model,
                our,
                opp,
                device=self.device,
            )

            team_list = list(battle.team.values())
            for idx in slots:
                if 1 <= idx <= len(team_list):
                    team_list[idx - 1]._selected_in_teampreview = True
            return "/team " + "".join(str(s) for s in slots)
        except Exception as exc:  # noqa: BLE001 - never forfeit on preview
            _logger.error(
                "Preview model failed for battle %s; falling back to random "
                "team selection. TODO debug: %s",
                getattr(battle, "battle_tag", "unknown"),
                exc,
                exc_info=True,
            )
            return random_teampreview_command(battle)
