"""Vectorized environment factory for parallel Showdown battles."""

from __future__ import annotations

from typing import Callable, Type

from poke_env.player.player import Player
from src.players.vgc_random_player import VGCRandomPlayer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from config.settings import BATTLE_FORMAT, N_ENV, USE_OPPONENT_TEAM_POOL
from archive.rl.env.champions_vgc_env import ChampionsVGCRLEnv, load_team
from src.teams.team_pool import load_opponent_team_builder
from archive.rl.env.gym_wrappers import wrap_for_sb3
from src.players.max_damage_player import MaxDamagePlayer
from archive.rl.players.policy_player import PolicyPlayer


def _make_env(
    opponent_cls: Type[Player],
    *,
    log_level: int = 40,
    opponent_kwargs: dict | None = None,
) -> Callable[[], Monitor]:
    opponent_kwargs = opponent_kwargs or {}

    def _init():
        env = ChampionsVGCRLEnv(
            battle_format=BATTLE_FORMAT,
            team=load_team(),
            opponent_team=load_opponent_team_builder(use_pool=USE_OPPONENT_TEAM_POOL),
            log_level=log_level,
            open_timeout=None,
        )
        opponent = opponent_cls(
            battle_format=BATTLE_FORMAT,
            team=load_team(),
            start_listening=False,
            **opponent_kwargs,
        )
        from archive.rl.env.single_agent_wrapper import SingleAgentWrapper

        wrapped = wrap_for_sb3(SingleAgentWrapper(env, opponent))
        return Monitor(wrapped)

    return _init


def make_vec_env(
    opponent_cls: Type[Player] = VGCRandomPlayer,
    n_envs: int = N_ENV,
    *,
    use_subproc: bool = True,
    opponent_kwargs: dict | None = None,
) -> VecEnv:
    factories = [
        _make_env(opponent_cls, opponent_kwargs=opponent_kwargs) for _ in range(n_envs)
    ]
    if n_envs == 1 or not use_subproc:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories)


def make_policy_opponent_env(
    model_path: str,
    n_envs: int = N_ENV,
    *,
    use_subproc: bool = True,
    device: str = "cpu",
) -> VecEnv:
    """Create env with PolicyPlayer opponent loaded from checkpoint."""

    def _init():
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(model_path, device=device)
        env = ChampionsVGCRLEnv(
            battle_format=BATTLE_FORMAT,
            team=load_team(),
            opponent_team=load_opponent_team_builder(use_pool=USE_OPPONENT_TEAM_POOL),
            log_level=40,
            open_timeout=None,
        )
        opponent = PolicyPlayer(
            model=model,
            battle_format=BATTLE_FORMAT,
            team=load_team(),
            start_listening=False,
        )
        from archive.rl.env.single_agent_wrapper import SingleAgentWrapper

        wrapped = wrap_for_sb3(SingleAgentWrapper(env, opponent))
        return Monitor(wrapped)

    factories = [_init for _ in range(n_envs)]
    if n_envs == 1 or not use_subproc:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories)
