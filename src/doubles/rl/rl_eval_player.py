"""TransformerPlayer bridge for RL eval: 1:1 Showdown decision ↔ Gym step via queues."""

from __future__ import annotations

import asyncio
import queue
from typing import Any

import numpy as np
from poke_env.battle.double_battle import DoubleBattle
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder

from src.doubles.battle.canonical_inference import canonical_indices_to_battle_order
from src.doubles.data.action_space_spec import ACTION_SIZE
from src.core.data.state_tokenizer import N_FIELDS, STACKED_N_TOKENS
from src.doubles.players.transformer_player import TransformerPlayer
from src.doubles.rl.rewards import BattleSnapshot, calc_step_reward


class RLEvalPlayer(TransformerPlayer):
    """
    Proven TransformerPlayer observation/mask path; RL policy supplies actions via queues.

    Each ``choose_move`` call enqueues (obs, mask, reward) and blocks until the Gym env
    supplies a canonical (slot0, slot1) action pair.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("log_illegal_top1", False)
        kwargs.setdefault("trace_inference", False)
        super().__init__(**kwargs)
        self.obs_queue: queue.Queue[
            tuple[np.ndarray, np.ndarray, float, bool, dict[str, Any]]
        ] = queue.Queue()
        self.action_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._last_reward_snap: BattleSnapshot | None = None

    def reset_rl_state(self) -> None:
        """Clear per-episode reward bookkeeping (call on env reset)."""
        self._last_reward_snap = None

    def _build_rl_packet(
        self, battle: DoubleBattle
    ) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any], np.ndarray]:
        """Mirror TransformerPlayer inputs without running BC inference."""
        x, _, view, sample_kind, snapshot = self._stacked_input(battle)
        obs = x.squeeze(0).detach().cpu().numpy().astype(np.float32)

        mask0 = self._canonical_mask(battle, 0, view=view, sample_kind=sample_kind)
        mask1 = self._canonical_mask(battle, 1, view=view, sample_kind=sample_kind)
        mask = np.concatenate(
            [mask0.detach().cpu().numpy().astype(bool), mask1.detach().cpu().numpy().astype(bool)]
        )
        if not mask.any():
            mask[0] = True

        reward, self._last_reward_snap = calc_step_reward(self._last_reward_snap, battle)
        info = {
            "battle_turn": int(battle.turn),
            "battle_won": False,
            "battle_lost": False,
            "force_switch": list(battle.force_switch),
        }
        return obs, mask, reward, info, snapshot

    async def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        if battle.wait and not any(battle.force_switch):
            return DefaultBattleOrder()

        obs, mask, reward, info, snapshot = self._build_rl_packet(battle)
        self.obs_queue.put((obs, mask, reward, False, info))

        loop = asyncio.get_running_loop()
        rl_action = await loop.run_in_executor(None, self.action_queue.get)
        pair = np.asarray(rl_action, dtype=np.int64).reshape(2)
        ca0, ca1 = int(pair[0]), int(pair[1])

        order = canonical_indices_to_battle_order(battle, ca0, ca1)
        if not isinstance(order, DefaultBattleOrder) and not any(battle.force_switch):
            self._commit_trajectory(battle, snapshot)
        return order

    def _battle_finished_callback(self, battle: DoubleBattle) -> None:
        super()._battle_finished_callback(battle)
        reward, self._last_reward_snap = calc_step_reward(self._last_reward_snap, battle)
        try:
            x, _, _, _, _ = self._stacked_input(battle)
            obs = x.squeeze(0).detach().cpu().numpy().astype(np.float32)
        except Exception:
            obs = np.zeros((STACKED_N_TOKENS, N_FIELDS), dtype=np.float32)
        mask = np.ones(ACTION_SIZE * 2, dtype=bool)
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
