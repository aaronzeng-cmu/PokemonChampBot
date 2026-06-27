"""Team preview (Turn 0) via trained TeamPreviewModel for BSS Bring-3."""

from __future__ import annotations

from pathlib import Path

from poke_env.battle.battle import Battle
from poke_env.data import to_id_str

from src.core.model.preview_model import load_preview_model, predict_singles_preview_slots
from src.singles.teampreview import random_teampreview_command


class SinglesPreviewOrchestrator:
    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        device: str = "cpu",
    ):
        from config.settings import SINGLES_PREVIEW_MODEL_PATH

        path = Path(model_path) if model_path is not None else SINGLES_PREVIEW_MODEL_PATH
        self.device = device
        self._model = None
        self._model_path = path
        if path.is_file():
            self._model = load_preview_model(path, device=device)

    def _species_list(self, team_values) -> list[str]:
        return [to_id_str(p.base_species) for p in team_values]

    def teampreview(self, battle: Battle) -> str:
        if self._model is None:
            return random_teampreview_command(battle)

        our = self._species_list(battle.team.values())
        opp = self._species_list(battle.opponent_team.values())
        slots = predict_singles_preview_slots(
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
