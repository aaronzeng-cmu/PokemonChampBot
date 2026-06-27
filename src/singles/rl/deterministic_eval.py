"""Deterministic MaskablePPO evaluation against SinglesMaxDamage (CleanSinglesEnv)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from src.singles.clean_singles_env import CleanSinglesEnv


def make_masked_env(*, device: str = "cpu", use_meta_pool: bool = True) -> gym.Env:
    env = CleanSinglesEnv(device=device, use_meta_pool=use_meta_pool)
    return ActionMasker(env, lambda e: e.action_masks())


def _action_masks(env: gym.Env) -> np.ndarray:
    cur: gym.Env | None = env
    while cur is not None:
        if hasattr(cur, "action_masks"):
            return np.asarray(cur.action_masks(), dtype=bool)
        cur = getattr(cur, "env", None)
    raise RuntimeError("No action_masks() on env wrapper chain")


def run_deterministic_eval(
    model: MaskablePPO,
    *,
    battles: int = 100,
    max_steps: int = 250,
) -> tuple[float, int, list[dict]]:
    """
    Run deterministic (argmax) eval; returns (win_rate, wins, battle_rows).
    Uses a fresh CleanSinglesEnv so training env state is untouched.
    """
    env = make_masked_env(device="cpu")
    rows: list[dict] = []
    wins = 0

    try:
        for bi in range(1, battles + 1):
            obs, _ = env.reset()
            done = False
            steps = 0
            info: dict = {}
            while not done and steps < max_steps:
                masks = _action_masks(env)
                action, _ = model.predict(
                    obs, deterministic=True, action_masks=masks
                )
                obs, _reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                steps += 1

            won = bool(info.get("battle_won"))
            wins += int(won)
            rows.append(
                {
                    "index": bi,
                    "won": won,
                    "steps": steps,
                    "turn": info.get("battle_turn"),
                }
            )
    finally:
        env.close()

    rate = wins / battles if battles else 0.0
    return rate, wins, rows
