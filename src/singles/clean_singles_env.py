"""Gymnasium env for Champions BSS singles — one step per Showdown decision."""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.teambuilder import Teambuilder

from config.settings import (
    SINGLES_BATTLE_FORMAT,
    SINGLES_PREVIEW_MODEL_PATH,
    USE_SINGLES_OPPONENT_TEAM_POOL,
)
from src.singles.preview_orchestrator import SinglesPreviewOrchestrator
from src.core.data.state_tokenizer import N_FIELDS, STACKED_N_TOKENS
from src.core.model.transformer_bot import SINGLES_ACTION_SIZE
from src.doubles.teams.team_pool import PoolTeambuilder, opponent_pool_description
from src.singles.max_damage_player import SinglesMaxDamagePlayer
from src.singles.teams.team_pool import (
    load_agent_team,
    load_meta_team_pool,
    load_opponent_team_builder,
)
from src.singles.rl_eval_player import SinglesRLEvalPlayer

QUEUE_TIMEOUT_S = 300.0


class CleanSinglesEnv(gym.Env):
    """Threaded Gym env backed by SinglesRLEvalPlayer (no poke-env SinglesEnv wrapper)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        battle_format: str = SINGLES_BATTLE_FORMAT,
        device: str = "cpu",
        use_meta_pool: bool = True,
        use_opponent_pool: bool = USE_SINGLES_OPPONENT_TEAM_POOL,
    ) -> None:
        super().__init__()
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(STACKED_N_TOKENS, N_FIELDS),
            dtype=np.float32,
        )
        self.action_space = Discrete(SINGLES_ACTION_SIZE)
        self.use_meta_pool = use_meta_pool

        if use_meta_pool:
            agent_team: str | Teambuilder = load_meta_team_pool(use_curriculum=False)
            opponent_team: str | Teambuilder = load_meta_team_pool(use_curriculum=False)
        else:
            agent_team = load_agent_team()
            opponent_team = load_opponent_team_builder(use_pool=use_opponent_pool)

        self._agent_pool_info = opponent_pool_description(agent_team)
        self._opponent_pool_info = opponent_pool_description(opponent_team)

        preview = SinglesPreviewOrchestrator(
            model_path=SINGLES_PREVIEW_MODEL_PATH,
            device=device,
        )
        self.player = SinglesRLEvalPlayer(
            battle_format=battle_format,
            team=agent_team,
            device=device,
            preview=preview,
            max_concurrent_battles=1,
            start_listening=True,
            account_configuration=AccountConfiguration.generate("SinglesRLAgent", rand=True),
        )
        self.opponent = SinglesMaxDamagePlayer(
            battle_format=battle_format,
            team=opponent_team,
            max_concurrent_battles=1,
            start_listening=True,
            account_configuration=AccountConfiguration.generate("SinglesRLOpp", rand=True),
        )

        self._last_mask = np.ones(SINGLES_ACTION_SIZE, dtype=bool)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._battle_future: Any = None
        self._start_async_loop()

    @property
    def team_pool_info(self) -> dict:
        return {
            "agent": self._agent_pool_info,
            "opponent": self._opponent_pool_info,
            "use_meta_pool": self.use_meta_pool,
        }

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

        self._thread = threading.Thread(target=_run, daemon=True, name="CleanSinglesEnv")
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
        self, action: int | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.player.action_queue.put(int(action))
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
