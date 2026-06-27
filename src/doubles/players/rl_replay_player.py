"""Record Showdown HTML replays using a trained MaskablePPO policy."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from poke_env.battle.double_battle import DoubleBattle
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder
from sb3_contrib import MaskablePPO

from src.doubles.battle.canonical_inference import canonical_indices_to_battle_order
from src.doubles.evaluation.battle_inference_trace import (
    format_live_battle_brief,
    summarize_server_request,
)
from src.doubles.rl.rl_eval_player import RLEvalPlayer


class RLReplayPlayer(RLEvalPlayer):
    """
    TransformerPlayer observation/mask path with MaskablePPO inference.

    Used for HTML replay recording and live inference traces (no Gym queues).
    """

    def __init__(
        self,
        rl_model: MaskablePPO,
        *,
        deterministic: bool = True,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("log_illegal_top1", False)
        kwargs.setdefault("trace_inference", False)
        super().__init__(**kwargs)
        self._rl_model = rl_model
        self._deterministic = deterministic

    def _rl_slot_logits(self, obs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        policy = self._rl_model.policy
        obs_t = torch.as_tensor(
            obs, dtype=torch.float32, device=policy.device
        ).unsqueeze(0)
        with torch.no_grad():
            features = policy.extract_features(obs_t)
            latent_pi, _ = policy.mlp_extractor(features)
            cloner = policy.features_extractor.cloner
            return cloner.head1(latent_pi)[0], cloner.head2(latent_pi)[0]

    async def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        if battle.wait and not any(battle.force_switch):
            if self.trace_inference:
                tag = battle.battle_tag
                idx = self._decision_counters.get(tag, 0) + 1
                self._decision_counters[tag] = idx
                self._inference_traces.setdefault(tag, []).append(
                    {
                        "decision_index": idx,
                        "kind": "wait",
                        "battle_tag": tag,
                        "turn": int(battle.turn),
                        "state_text": format_live_battle_brief(battle),
                        "server_request": summarize_server_request(battle),
                        "battle_events_since_last": self._events_since_last_decision(
                            battle
                        ),
                    }
                )
            return DefaultBattleOrder()

        x, trajectory_frames, view, sample_kind, snapshot = self._stacked_input(battle)
        obs = x.squeeze(0).detach().cpu().numpy().astype(np.float32)

        mask0 = self._canonical_mask(battle, 0, view=view, sample_kind=sample_kind)
        mask1 = self._canonical_mask(battle, 1, view=view, sample_kind=sample_kind)
        mask = np.concatenate(
            [
                mask0.detach().cpu().numpy().astype(bool),
                mask1.detach().cpu().numpy().astype(bool),
            ]
        )
        if not mask.any():
            mask[0] = True

        logits0, logits1 = self._rl_slot_logits(obs)
        raw0 = int(logits0.argmax().item())
        raw1 = int(logits1.argmax().item())

        action, _ = self._rl_model.predict(
            obs,
            deterministic=self._deterministic,
            action_masks=mask,
        )
        pair = np.asarray(action, dtype=np.int64).reshape(2)
        ca0, ca1 = int(pair[0]), int(pair[1])

        if self.trace_inference:
            self._record_inference_trace(
                battle,
                logits0=logits0.detach().cpu(),
                logits1=logits1.detach().cpu(),
                mask0=mask0,
                mask1=mask1,
                ca0=ca0,
                ca1=ca1,
                raw0=raw0,
                raw1=raw1,
                trajectory_frames=trajectory_frames,
            )

        order = canonical_indices_to_battle_order(battle, ca0, ca1)
        if not isinstance(order, DefaultBattleOrder) and not any(battle.force_switch):
            self._commit_trajectory(battle, snapshot)
        return order
