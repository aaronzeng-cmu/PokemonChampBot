"""Gymnasium env backed by RLEvalPlayer — one step per Showdown decision."""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, MultiDiscrete
from poke_env.ps_client.account_configuration import AccountConfiguration

from config.settings import BATTLE_FORMAT, USE_OPPONENT_TEAM_POOL
from src.doubles.data.action_space_spec import ACTION_SIZE
from src.core.data.state_tokenizer import N_FIELDS, STACKED_N_TOKENS
from src.doubles.players.max_damage_player import MaxDamagePlayer
from src.doubles.rl.rl_eval_player import RLEvalPlayer
from src.doubles.teams.team_pool import load_agent_team, load_opponent_team_builder

QUEUE_TIMEOUT_S = 300.0


class CleanVGCRLEnv(gym.Env):
    """
    Pure gymnasium.Env: steps only when Showdown calls RLEvalPlayer.choose_move.

    No poke-env DoublesEnv / legacy Gym wrappers.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        battle_format: str = BATTLE_FORMAT,
        device: str = "cpu",
        use_opponent_pool: bool = USE_OPPONENT_TEAM_POOL,
    ) -> None:
        super().__init__()
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(STACKED_N_TOKENS, N_FIELDS),
            dtype=np.float32,
        )
        self.action_space = MultiDiscrete([ACTION_SIZE, ACTION_SIZE])

        agent_team = load_agent_team()
        opponent_team = load_opponent_team_builder(use_pool=use_opponent_pool)

        self.player = RLEvalPlayer(
            battle_format=battle_format,
            team=agent_team,
            device=device,
            max_concurrent_battles=1,
            start_listening=True,
            account_configuration=AccountConfiguration.generate("RLEvalAgent", rand=True),
        )
        self.opponent = MaxDamagePlayer(
            battle_format=battle_format,
            team=opponent_team,
            max_concurrent_battles=1,
            start_listening=True,
            account_configuration=AccountConfiguration.generate("RLEvalOpp", rand=True),
        )

        self._last_mask = np.ones(ACTION_SIZE * 2, dtype=bool)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._battle_future: Any = None
        self._start_async_loop()

    def _await_battle_done(self) -> None:
        if self._battle_future is None:
            return
        try:
            self._battle_future.result(timeout=QUEUE_TIMEOUT_S)
        except Exception:
            pass
        self._battle_future = None

    def _start_async_loop(self) -> None:
        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            loop.run_forever()

        self._thread = threading.Thread(target=_run, daemon=True, name="CleanVGCRLEnv")
        self._thread.start()
        while self._loop is None:
            time.sleep(0.01)

    def _drain_queues(self) -> None:
        for q in (self.player.obs_queue, self.player.action_queue):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def action_masks(self) -> np.ndarray:
        return self._last_mask

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del seed, options
        self._await_battle_done()
        self._drain_queues()
        self.player.reset_battles()
        self.opponent.reset_battles()
        self.player.reset_rl_state()

        assert self._loop is not None
        self._battle_future = asyncio.run_coroutine_threadsafe(
            self.player.battle_against(self.opponent, n_battles=1),
            self._loop,
        )

        obs, mask, _reward, done, info = self.player.obs_queue.get(timeout=QUEUE_TIMEOUT_S)
        if done:
            raise RuntimeError("Battle finished before the first RL decision")
        self._last_mask = mask
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.player.action_queue.put(np.asarray(action, dtype=np.int64))
        obs, mask, reward, done, info = self.player.obs_queue.get(timeout=QUEUE_TIMEOUT_S)
        self._last_mask = mask
        info = dict(info)
        info["action_masks"] = mask
        if done:
            info["battle_won"] = bool(info.get("battle_won"))
            info["battle_lost"] = bool(info.get("battle_lost"))
        return obs, float(reward), bool(done), False, info

    def close(self) -> None:
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
