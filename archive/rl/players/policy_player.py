"""Player that acts using a trained MaskablePPO policy."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from poke_env.battle.double_battle import DoubleBattle
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder
from poke_env.player.player import Player

from config.settings import BATTLE_FORMAT, DEFAULT_TEAM_PREVIEW, LEARN_TEAM_PREVIEW, TEAM_PATH
from archive.rl.env.champions_vgc_env import ChampionsVGCRLEnv
from archive.rl.env.observation import embed_battle
from src.teams.teampreview import policy_teampreview_command, random_teampreview_command


class PolicyPlayer(Player):
    def __init__(
        self,
        model: Any = None,
        *,
        policy: Any = None,
        battle_format: str = BATTLE_FORMAT,
        team: Optional[str] = None,
        **kwargs: Any,
    ):
        if team is None:
            team = TEAM_PATH.read_text(encoding="utf-8")
        super().__init__(battle_format=battle_format, team=team, **kwargs)
        self.model = model
        self.policy = policy if policy is not None else (model.policy if model else None)

    def _predict_combo_index(self, battle: DoubleBattle) -> int:
        obs = embed_battle(battle)
        mask = np.array(ChampionsVGCRLEnv.get_action_mask(battle), dtype=bool)
        if self.model is not None:
            action, _ = self.model.predict(
                obs, action_masks=mask, deterministic=True
            )
            return int(action)
        assert self.policy is not None
        device = self.policy.device
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(mask, device=device).unsqueeze(0)
        dist = self.policy.get_distribution(obs_t, action_masks=mask_t)
        return int(dist.get_actions(deterministic=True).cpu().numpy()[0])

    def teampreview(self, battle: DoubleBattle) -> str:
        if not LEARN_TEAM_PREVIEW:
            return DEFAULT_TEAM_PREVIEW
        if self.model is None and self.policy is None:
            return random_teampreview_command(battle)
        return policy_teampreview_command(
            battle,
            predict_combo_index=self._predict_combo_index,
        )

    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        if battle.wait:
            return DefaultBattleOrder()
        combo_idx = self._predict_combo_index(battle)
        return ChampionsVGCRLEnv.action_to_order(combo_idx, battle, strict=False)
