"""Gymnasium wrappers for MaskablePPO + VGCRLEnv."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from poke_env.player.player import Player

from src.doubles.rl.vgc_env import VGCRLEnv


def find_vgc_env(env: gym.Env) -> VGCRLEnv | None:
    cur: gym.Env | None = env
    while cur is not None:
        if isinstance(cur, VGCRLEnv):
            return cur
        cur = getattr(cur, "env", None)
    return None


class VGCSingleAgentWrapper(gym.Env):
    """Wrap VGCRLEnv for single-agent training against a scripted opponent."""

    def __init__(self, env: VGCRLEnv, opponent: Player):
        self.env = env
        self.opponent = opponent
        self.observation_space = env.observation_spaces[env.possible_agents[0]]
        self.action_space = env.action_spaces[env.possible_agents[0]]
        self.metadata = getattr(env, "metadata", {})
        self.second_teampreview_action: np.ndarray | None = None

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks(self.env.battle1)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        assert self.env.battle2 is not None
        from poke_env.player.battle_order import DefaultBattleOrder
        from typing import Awaitable

        if self.env.battle2.wait:
            opp_action = self.env.order_to_action(
                DefaultBattleOrder(),
                self.env.battle2,
                fake=self.env._fake,
                strict=self.env._strict,
            )
        elif not self.env.battle2.teampreview:
            opp_order = self.opponent.choose_move(self.env.battle2)
            assert not isinstance(opp_order, Awaitable)
            opp_action = self.env.order_to_action(
                opp_order,
                self.env.battle2,
                fake=self.env._fake,
                strict=self.env._strict,
            )
        elif (
            self.env.battle2.format is None
            or "vgc" not in self.env.battle2.format
        ):
            opp_action = self._random_preview_action(self.env.battle2)
        elif self.second_teampreview_action is None:
            tp_order = self.opponent.teampreview(self.env.battle2)
            assert not isinstance(tp_order, Awaitable)
            assert len(tp_order) == 10, f"{tp_order} must specify 4 slots in VGC!"
            teampreview_order_list = [int(i) for i in tp_order[-4:]]
            opp_action = np.array(teampreview_order_list[:2], dtype=np.int64)
            self.second_teampreview_action = np.array(
                teampreview_order_list[2:], dtype=np.int64
            )
            for i, pokemon in enumerate(self.env.battle2.team.values(), start=1):
                pokemon._selected_in_teampreview = i in teampreview_order_list[:2]
        else:
            opp_action = self.second_teampreview_action
            for i in opp_action:
                mon = list(self.env.battle2.team.values())[int(i) - 1]
                mon._selected_in_teampreview = True
            self.second_teampreview_action = None

        actions = {
            self.env.possible_agents[0]: np.asarray(action, dtype=np.int64),
            self.env.possible_agents[1]: opp_action,
        }
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        agent = self.env.possible_agents[0]
        if self.env.battle1 is not None:
            self.env.commit_trajectory(self.env.battle1)
        return (
            self._obs(obs[agent]),
            float(rewards[agent]),
            bool(terms[agent]),
            bool(truncs[agent]),
            infos[agent],
        )

    def _random_preview_action(self, battle) -> np.ndarray:
        legal = [
            i
            for i, p in enumerate(battle.team.values(), start=1)
            if not p.selected_in_teampreview
        ]
        if not legal:
            return np.array([0, 0], dtype=np.int64)
        pick = int(np.random.randint(0, len(legal)))
        return np.array([legal[pick], 0], dtype=np.int64)

    def reset(self, *, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.second_teampreview_action = None
        self.opponent.reset_battles()
        assert self.env.battle2 is not None
        self.opponent._battles[self.env.battle2.battle_tag] = self.env.battle2
        agent = self.env.possible_agents[0]
        return self._obs(obs[agent]), infos[agent]

    @staticmethod
    def _obs(raw: Any) -> np.ndarray:
        if isinstance(raw, dict):
            return np.asarray(raw["observation"], dtype=np.float32)
        return np.asarray(raw, dtype=np.float32)

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        self.env.close()


class VGCObsWrapper(gym.ObservationWrapper):
    """Ensure flat (39, 24) float observation for SB3."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        sample = env.observation_space
        if isinstance(sample, spaces.Dict):
            shape = sample["observation"].shape
        else:
            shape = sample.shape
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=shape,
            dtype=np.float32,
        )

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()

    def observation(self, observation):
        if isinstance(observation, dict):
            return np.asarray(observation["observation"], dtype=np.float32)
        return np.asarray(observation, dtype=np.float32)


def wrap_vgc_for_sb3(env: gym.Env) -> gym.Env:
    """Apply wrappers expected by sb3-contrib MaskablePPO."""
    from sb3_contrib.common.wrappers import ActionMasker

    env = VGCObsWrapper(env)
    env = ActionMasker(env, lambda e: e.action_masks())
    return env
