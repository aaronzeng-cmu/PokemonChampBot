"""Singles RL eval player: queue bridge between Showdown and Gymnasium."""

from __future__ import annotations

import asyncio
import hashlib
import queue
import threading
from typing import Any

import numpy as np
from poke_env.battle.battle import Battle
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder
from poke_env.player.player import Player

from src.core.data.state_tokenizer import (
    N_FIELDS,
    STACKED_N_TOKENS,
    trajectory_frame_fingerprints,
)
from src.core.model.transformer_bot import SINGLES_ACTION_SIZE
from src.doubles.evaluation.battle_inference_trace import summarize_protocol_lines
from src.doubles.rl.rewards import BattleSnapshot, calc_step_reward
from src.singles.battle.canonical_inference import canonical_index_to_battle_order
from src.singles.battle.live_protocol_support import SinglesLiveProtocolSupport
from src.singles.evaluation.inference_trace import (
    action_record,
    format_singles_live_battle_brief,
)
from src.singles.preview_orchestrator import SinglesPreviewOrchestrator


class SinglesRLEvalPlayer(SinglesLiveProtocolSupport, Player):
    """One Showdown decision ↔ one Gym step via obs/action queues (BC-aligned tensors)."""

    def __init__(
        self,
        *,
        preview: SinglesPreviewOrchestrator | None = None,
        device: str = "cpu",
        trace_decisions: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.device = device
        self.preview = preview or SinglesPreviewOrchestrator(device=device)
        self.trace_decisions = trace_decisions
        self._init_live_protocol_state()
        self.obs_queue: queue.Queue[
            tuple[np.ndarray, np.ndarray, float, bool, dict[str, Any]]
        ] = queue.Queue()
        self.action_queue: queue.Queue[int] = queue.Queue()
        self._last_reward_snap: BattleSnapshot | None = None
        self._rl_traces: dict[str, list[dict]] = {}
        self._decision_counters: dict[str, int] = {}

    def reset_rl_state(self) -> None:
        self._last_reward_snap = None
        self._clear_live_protocol_state()
        self._rl_traces.clear()
        self._decision_counters.clear()

    def teampreview(self, battle: Battle) -> str:
        cmd = self.preview.teampreview(battle)
        self._record_teampreview_protocol(battle, cmd)
        return cmd

    async def _handle_battle_message(self, split_messages):
        await self._capture_protocol_message(split_messages)
        await super()._handle_battle_message(split_messages)

    def _record_rl_decision(
        self,
        battle: Battle,
        *,
        stacked: np.ndarray,
        mask: np.ndarray,
        action: int,
    ) -> None:
        if not self.trace_decisions:
            return
        tag = battle.battle_tag
        idx = self._decision_counters.get(tag, 0) + 1
        self._decision_counters[tag] = idx
        protocol = self._protocol_for_encoding(battle)
        self._rl_traces.setdefault(tag, []).append(
            {
                "decision_index": idx,
                "kind": "inference",
                "battle_tag": tag,
                "turn": int(battle.turn),
                "force_switch": bool(battle.force_switch),
                "trajectory_depth": len(self._history_for_tag(tag)),
                "trajectory_frames": trajectory_frame_fingerprints(stacked),
                "protocol_len": len(protocol),
                "token_digest": hashlib.md5(
                    np.ascontiguousarray(stacked).tobytes()
                ).hexdigest()[:16],
                "stacked_tokens": stacked.astype(np.int64).tolist(),
                "state_text": format_singles_live_battle_brief(battle),
                "picked": action_record(battle, index=action, legal=bool(mask[action])),
            }
        )

    def drain_rl_trace(self, battle_tag: str) -> dict:
        """Export captured protocol + RL decisions (inference-trace compatible JSON)."""
        self._flush_pending_turn_trajectories(battle_tag)
        log = self._protocol_logs.pop(battle_tag, [])
        return {
            "battle_tag": battle_tag,
            "teampreview": self._teampreview_cmds.pop(battle_tag, None),
            "decisions": self._rl_traces.pop(battle_tag, []),
            "protocol_log": log,
            "battle_events": summarize_protocol_lines(log),
        }

    def _build_rl_packet(
        self, battle: Battle
    ) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
        stacked = self._stacked_obs(battle)
        obs = stacked.astype(np.float32)
        mask = self._live_action_mask(battle)
        reward, self._last_reward_snap = calc_step_reward(self._last_reward_snap, battle)  # type: ignore[arg-type]
        info = {
            "battle_turn": int(battle.turn),
            "battle_won": False,
            "battle_lost": False,
            "force_switch": bool(battle.force_switch),
        }
        return obs, mask, reward, info

    async def choose_move(self, battle: Battle) -> BattleOrder:
        if battle.wait and not battle.force_switch:
            return DefaultBattleOrder()

        stacked = self._stacked_obs(battle)
        obs = stacked.astype(np.float32)
        mask = self._live_action_mask(battle)
        reward, self._last_reward_snap = calc_step_reward(self._last_reward_snap, battle)  # type: ignore[arg-type]
        info = {
            "battle_turn": int(battle.turn),
            "battle_won": False,
            "battle_lost": False,
            "force_switch": bool(battle.force_switch),
        }
        self.obs_queue.put((obs, mask, reward, False, info))

        loop = asyncio.get_running_loop()
        action = await loop.run_in_executor(None, self.action_queue.get)
        self._record_rl_decision(battle, stacked=stacked, mask=mask, action=int(action))
        order = canonical_index_to_battle_order(battle, int(action))
        if not isinstance(order, DefaultBattleOrder):
            self._record_champchoice(battle, int(action))
        return order

    def _battle_finished_callback(self, battle: Battle) -> None:
        super()._battle_finished_callback(battle)
        self._flush_pending_turn_trajectories(battle.battle_tag)
        reward, self._last_reward_snap = calc_step_reward(self._last_reward_snap, battle)  # type: ignore[arg-type]
        try:
            obs = self._stacked_obs(battle).astype(np.float32)
        except Exception:
            obs = np.zeros((STACKED_N_TOKENS, N_FIELDS), dtype=np.float32)
        mask = np.ones(SINGLES_ACTION_SIZE, dtype=bool)
        self.obs_queue.put(
            (
                obs,
                mask,
                reward,
                True,
                {
                    "battle_won": bool(battle.won),
                    "battle_lost": bool(battle.lost),
                    "battle_turn": int(battle.turn),
                },
            )
        )


def start_bc_action_feeder(
    agent: SinglesRLEvalPlayer,
    *,
    model_path,
    device: str = "cpu",
) -> threading.Thread:
    """Background thread: BC masked-argmax drives RL trace battles."""
    import torch
    from pathlib import Path

    from src.core.model.transformer_bot import load_model
    from src.singles.action_mask import pick_masked_argmax

    model = load_model(Path(model_path), device=device)
    model.eval()
    stop = threading.Event()

    def _loop() -> None:
        while not stop.is_set():
            try:
                obs, mask, _reward, done, _info = agent.obs_queue.get(timeout=300.0)
            except queue.Empty:
                continue
            if done:
                continue
            x = torch.as_tensor(obs, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = model(x).cpu().numpy().squeeze(0)
            action = pick_masked_argmax(logits, mask)
            agent.action_queue.put(int(action))

    thread = threading.Thread(target=_loop, name="RLTraceBCFeeder", daemon=True)
    thread.start()
    thread._stop_event = stop  # type: ignore[attr-defined]
    return thread
