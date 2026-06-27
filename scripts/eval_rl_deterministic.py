#!/usr/bin/env python3
"""Evaluate MaskablePPO with deterministic (argmax) actions on CleanVGCRLEnv.

Default: fresh policy with BC actor weights only (--bc-model).
With --checkpoint: load a trained RL .zip (same path as record_rl_replays.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, MultiDiscrete
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from config.settings import BC_MODEL_PATH, BATTLE_FORMAT, USE_OPPONENT_TEAM_POOL
from src.doubles.data.action_space_spec import ACTION_SIZE
from src.core.data.state_tokenizer import N_FIELDS, STACKED_N_TOKENS
from src.doubles.rl.checkpoints import load_rl_checkpoint
from src.doubles.rl.clean_vgc_env import CleanVGCRLEnv
from src.doubles.rl.custom_policy import VGCBehaviorMaskablePolicy, init_bc_actor_weights

RL_CHECKPOINTS_DIR = Path("models/rl_checkpoints")
DEFAULT_BEST = RL_CHECKPOINTS_DIR / "best_wr86_steps200000_20260615_112649.zip"


def _resolve_checkpoint(path: Path | None) -> Path:
    from src.doubles.rl.checkpoints import resolve_rl_checkpoint

    return resolve_rl_checkpoint(path)


class _SpaceOnlyEnv(gym.Env):
    """Stub env for MaskablePPO construction (avoids DummyVecEnv auto-reset battles)."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(STACKED_N_TOKENS, N_FIELDS),
            dtype=np.float32,
        )
        self.action_space = MultiDiscrete([ACTION_SIZE, ACTION_SIZE])

    def reset(self, *, seed=None, options=None):
        del seed, options
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action):
        del action
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, 0.0, True, False, {}

    def action_masks(self) -> np.ndarray:
        return np.ones(int(np.prod(self.action_space.nvec)), dtype=bool)


def make_eval_env(*, device: str = "cpu") -> gym.Env:
    env = CleanVGCRLEnv(device=device)
    return ActionMasker(env, lambda e: e.action_masks())


def _action_masks(env: gym.Env) -> np.ndarray:
    cur: gym.Env | None = env
    while cur is not None:
        if hasattr(cur, "action_masks"):
            return np.asarray(cur.action_masks(), dtype=bool)
        cur = getattr(cur, "env", None)
    raise RuntimeError("No action_masks() on env wrapper chain")


def load_checkpoint(*, checkpoint: Path | None, device: str) -> tuple[MaskablePPO, Path]:
    return load_rl_checkpoint(checkpoint, device=device)


def build_model(*, bc_model: Path, device: str) -> MaskablePPO:
    stub = _SpaceOnlyEnv()
    model = MaskablePPO(
        VGCBehaviorMaskablePolicy,
        stub,
        learning_rate=5e-6,
        n_steps=64,
        batch_size=64,
        device=device,
        policy_kwargs=dict(
            bc_model_path=str(bc_model),
            net_arch=dict(pi=[], vf=[64, 64]),
            activation_fn=torch.nn.Tanh,
            ortho_init=False,
        ),
        verbose=0,
    )
    init_bc_actor_weights(model.policy, bc_model)
    return model


def run_eval(
    model: MaskablePPO,
    *,
    battles: int,
    policy_source: str,
    checkpoint: Path | None = None,
    bc_model: Path | None = None,
) -> dict:
    env = make_eval_env()

    rows: list[dict] = []
    wins = 0

    for bi in range(1, battles + 1):
        obs, _ = env.reset()
        done = False
        steps = 0
        info: dict = {}
        max_steps = 250
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
        if bi % 10 == 0 or bi == battles:
            print(f"Progress: {bi}/{battles} wins={wins} ({100 * wins / bi:.1f}%)")

    env.close()
    return {
        "battles": battles,
        "wins": wins,
        "losses": battles - wins,
        "win_rate": wins / battles if battles else 0.0,
        "deterministic": True,
        "env": "CleanVGCRLEnv",
        "format": BATTLE_FORMAT,
        "opponent": "MaxDamagePlayer",
        "opponent_team_pool": USE_OPPONENT_TEAM_POOL,
        "policy_source": policy_source,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "bc_model": str(
            bc_model
            or getattr(model.policy, "_bc_model_path", None)
            or "unknown"
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "battles_detail": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battles", type=int, default=100)
    parser.add_argument("--bc-model", type=Path, default=BC_MODEL_PATH)
    parser.add_argument(
        "--checkpoint",
        nargs="?",
        const="auto",
        default=None,
        metavar="PATH",
        help="Evaluate a trained RL .zip (omit PATH for best_wr* in models/rl_checkpoints)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("logs/eval/rl_deterministic"),
    )
    args = parser.parse_args()

    checkpoint_path: Path | None = None
    if args.checkpoint is not None:
        ckpt_arg = None if args.checkpoint == "auto" else Path(args.checkpoint)
        model, checkpoint_path = load_checkpoint(
            checkpoint=ckpt_arg, device=args.device
        )
        print(f"Loaded RL checkpoint: {checkpoint_path} (device={args.device})")
        policy_source = "checkpoint"
    else:
        print(f"Building policy from BC weights: {args.bc_model} (device={args.device})")
        model = build_model(bc_model=args.bc_model, device=args.device)
        policy_source = "bc_init"

    print(f"Deterministic eval: {args.battles} battles vs MaxDamage (CleanVGCRLEnv)")
    report = run_eval(
        model,
        battles=args.battles,
        policy_source=policy_source,
        checkpoint=checkpoint_path,
        bc_model=args.bc_model if policy_source == "bc_init" else None,
    )

    print(
        f"\nWin rate: {report['win_rate']:.1%} "
        f"({report['wins']}/{report['battles']})"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = args.out_dir / f"rl_det_eval_{stamp}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report -> {out.resolve()}")


if __name__ == "__main__":
    main()
