"""Record Showdown battles and inference traces with a trained Singles MaskablePPO policy."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
from poke_env.battle.battle import Battle
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder
from poke_env.player.player import Player
from sb3_contrib import MaskablePPO

from src.core.data.state_tokenizer import trajectory_frame_fingerprints
from src.doubles.evaluation.battle_inference_trace import (
    format_battle_timeline,
    summarize_protocol_lines,
)
from src.singles.action_mask import pick_masked_argmax
from src.singles.battle.canonical_inference import (
    canonical_index_to_battle_order,
    submission_debug,
)
from src.singles.battle.live_protocol_support import SinglesLiveProtocolSupport
from src.singles.evaluation.inference_trace import (
    action_record,
    format_singles_live_battle_brief,
    summarize_server_request_singles,
    topk_singles_live_choices,
)
from src.singles.preview_orchestrator import SinglesPreviewOrchestrator


class SinglesRLReplayPlayer(SinglesLiveProtocolSupport, Player):
    """
    Live Showdown player driven by MaskablePPO (no Gym queues).

    Inference-trace format matches SinglesTransformerPlayer for alignment audits.
    """

    def __init__(
        self,
        rl_model: MaskablePPO,
        *,
        preview: SinglesPreviewOrchestrator | None = None,
        device: str = "cpu",
        deterministic: bool = True,
        trace_inference: bool = False,
        trace_top_k: int = 5,
        capture_battle_log: bool = False,
        log_illegal_top1: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._rl_model = rl_model
        self.device = device
        self._deterministic = deterministic
        self.preview = preview or SinglesPreviewOrchestrator(device=device)
        self._init_live_protocol_state()
        self.trace_inference = trace_inference
        self.trace_top_k = trace_top_k
        self.capture_battle_log = capture_battle_log or trace_inference
        self.log_illegal_top1 = log_illegal_top1
        self.illegal_top1_events: list[dict] = []
        self._inference_traces: dict[str, list[dict]] = {}
        self._protocol_cursor: dict[str, int] = {}
        self._decision_counters: dict[str, int] = {}

    def teampreview(self, battle: Battle) -> str:
        cmd = self.preview.teampreview(battle)
        self._record_teampreview_protocol(battle, cmd)
        return cmd

    async def _handle_battle_message(self, split_messages):
        if self.capture_battle_log:
            await self._capture_protocol_message(split_messages)
        await super()._handle_battle_message(split_messages)

    def drain_illegal_top1_events(self) -> list[dict]:
        out = list(self.illegal_top1_events)
        self.illegal_top1_events.clear()
        return out

    def drain_inference_trace(self, battle_tag: str) -> dict:
        log = self._protocol_logs.get(battle_tag, [])
        cursor = self._protocol_cursor.get(battle_tag, 0)
        if cursor < len(log):
            tail = summarize_protocol_lines(log[cursor:])
            if tail and self._inference_traces.get(battle_tag):
                self._inference_traces[battle_tag][-1]["battle_events_since_last"] = (
                    self._inference_traces[battle_tag][-1].get(
                        "battle_events_since_last", []
                    )
                    + tail
                )
            self._protocol_cursor[battle_tag] = len(log)

        self._flush_pending_turn_trajectories(battle_tag)
        all_events = summarize_protocol_lines(log)
        return {
            "battle_tag": battle_tag,
            "teampreview": self._teampreview_cmds.pop(battle_tag, None),
            "decisions": self._inference_traces.pop(battle_tag, []),
            "protocol_log": self._protocol_logs.pop(battle_tag, []),
            "battle_timeline": format_battle_timeline(all_events),
            "battle_events": all_events,
        }

    def _events_since_last_decision(self, battle: Battle) -> list[str]:
        tag = battle.battle_tag
        log = self._protocol_logs.get(tag, [])
        cursor = self._protocol_cursor.get(tag, 0)
        events = summarize_protocol_lines(log[cursor:])
        self._protocol_cursor[tag] = len(log)
        return events

    def _policy_logits(self, obs: np.ndarray) -> np.ndarray:
        policy = self._rl_model.policy
        obs_t = torch.as_tensor(
            obs, dtype=torch.float32, device=policy.device
        ).unsqueeze(0)
        with torch.no_grad():
            features = policy.extract_features(obs_t)
            latent_pi, _ = policy.mlp_extractor(features)
            cloner = policy.features_extractor.cloner
            logits = cloner.head_singles(latent_pi)[0]
        return logits.detach().cpu().numpy()

    def _log_illegal_top1(
        self,
        battle: Battle,
        *,
        raw_top1: int,
        picked: int,
        mask: np.ndarray,
    ) -> None:
        if not self.log_illegal_top1:
            return
        if 0 <= raw_top1 < mask.shape[0] and bool(mask[raw_top1]):
            return
        from src.singles.evaluation.inference_trace import format_singles_live_action

        self.illegal_top1_events.append(
            {
                "battle_tag": battle.battle_tag,
                "turn": int(battle.turn),
                "force_switch": bool(battle.force_switch),
                "raw_top1_index": raw_top1,
                "raw_top1_label": format_singles_live_action(battle, raw_top1),
                "picked_index": picked,
                "picked_label": format_singles_live_action(battle, picked),
                "fallback_used": raw_top1 != picked,
            }
        )

    def _record_inference_trace(
        self,
        battle: Battle,
        *,
        logits: np.ndarray,
        mask: np.ndarray,
        picked: int,
        raw: int,
        trajectory_frames: list[str],
        stacked_tokens: np.ndarray,
    ) -> None:
        tag = battle.battle_tag
        idx = self._decision_counters.get(tag, 0) + 1
        self._decision_counters[tag] = idx
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        logits_t = torch.as_tensor(logits, dtype=torch.float32, device=self.device)

        self._inference_traces.setdefault(tag, []).append(
            {
                "decision_index": idx,
                "kind": "inference",
                "battle_tag": tag,
                "turn": int(battle.turn),
                "wait": bool(battle.wait),
                "force_switch": bool(battle.force_switch),
                "trajectory_depth": len(self._history_for_tag(tag)),
                "trajectory_frames": trajectory_frames,
                "protocol_len": len(self._protocol_logs.get(tag, [])),
                "token_digest": hashlib.md5(
                    np.ascontiguousarray(stacked_tokens).tobytes()
                ).hexdigest()[:16],
                "stacked_tokens": stacked_tokens.astype(np.int64).tolist(),
                "state_text": format_singles_live_battle_brief(battle),
                "server_request": summarize_server_request_singles(battle),
                "battle_events_since_last": self._events_since_last_decision(battle),
                "raw_top1": action_record(
                    battle,
                    index=raw,
                    legal=0 <= raw < mask.shape[0] and bool(mask[raw]),
                ),
                "picked": action_record(battle, index=picked),
                "submission": submission_debug(battle, picked),
                "topk_legal": topk_singles_live_choices(
                    logits_t,
                    mask_t,
                    battle,
                    k=self.trace_top_k,
                    legal_only=True,
                ),
                "fallback": raw != picked,
            }
        )

    def choose_move(self, battle: Battle) -> BattleOrder:
        if battle.wait and not battle.force_switch:
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
                        "state_text": format_singles_live_battle_brief(battle),
                        "server_request": summarize_server_request_singles(battle),
                        "battle_events_since_last": self._events_since_last_decision(
                            battle
                        ),
                    }
                )
            return DefaultBattleOrder()

        stacked = self._stacked_obs(battle)
        obs = stacked.astype(np.float32)
        mask = self._live_action_mask(battle)
        trajectory_frames = trajectory_frame_fingerprints(stacked)

        logits = self._policy_logits(obs)
        raw = int(np.argmax(logits))
        action, _ = self._rl_model.predict(
            obs,
            deterministic=self._deterministic,
            action_masks=mask,
        )
        picked = int(action)
        if not mask[picked]:
            picked = pick_masked_argmax(logits, mask)

        self._log_illegal_top1(battle, raw_top1=raw, picked=picked, mask=mask)

        if self.trace_inference:
            self._record_inference_trace(
                battle,
                logits=logits,
                mask=mask,
                picked=picked,
                raw=raw,
                trajectory_frames=trajectory_frames,
                stacked_tokens=stacked,
            )

        order = canonical_index_to_battle_order(battle, picked)
        if not isinstance(order, DefaultBattleOrder):
            self._record_champchoice(battle, picked)
        return order
