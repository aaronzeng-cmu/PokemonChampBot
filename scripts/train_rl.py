#!/usr/bin/env python3
"""RL fine-tuning with MaskablePPO bootstrapped from BC Transformer weights."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from config.settings import (
    BC_MODEL_PATH,
    LOGS_DIR,
    MODELS_DIR,
    SINGLES_BC_MODEL_PATH,
)
from src.doubles.rl.clean_vgc_env import CleanVGCRLEnv
from src.doubles.rl.custom_policy import VGCBehaviorMaskablePolicy, init_bc_actor_weights
from src.doubles.rl.deterministic_eval import run_deterministic_eval as run_doubles_deterministic_eval
from src.singles.clean_singles_env import CleanSinglesEnv
from src.singles.rl.custom_policy import (
    SinglesBehaviorMaskablePolicy,
    init_bc_actor_weights as init_singles_bc_actor_weights,
)
from src.singles.rl.deterministic_eval import run_deterministic_eval as run_singles_deterministic_eval

RL_CHECKPOINTS_DIR_DOUBLES = MODELS_DIR / "rl_checkpoints"
RL_CHECKPOINTS_DIR_SINGLES = MODELS_DIR / "rl_checkpoints_singles"
_STEPS_RE = re.compile(r"steps(\d+)")


class WinRateCallback(BaseCallback):
    """Log rolling win rate and episode reward to TensorBoard."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._wins = 0
        self._episodes = 0
        self._ep_rewards: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones") or []
        for done, info in zip(dones, infos):
            if not isinstance(info, dict):
                continue
            if info.get("episode"):
                ep = info["episode"]
                self._ep_rewards.append(float(ep.get("r", 0.0)))
            if done:
                self._episodes += 1
                if info.get("battle_won"):
                    self._wins += 1
                self.logger.record(
                    "rollout/last_battle_won", float(bool(info.get("battle_won")))
                )
                self.logger.record(
                    "rollout/last_battle_lost", float(bool(info.get("battle_lost")))
                )
        if self._episodes > 0 and self.num_timesteps % 1000 == 0:
            win_rate = self._wins / max(1, self._episodes)
            mean_reward = (
                float(np.mean(self._ep_rewards[-50:])) if self._ep_rewards else 0.0
            )
            self.logger.record("rollout/win_rate", win_rate)
            self.logger.record("rollout/episode_count", self._episodes)
            self.logger.record("rollout/mean_ep_reward", mean_reward)
        return True


class DeterministicEvalCallback(BaseCallback):
    """Pause training periodically; run deterministic eval; save best."""

    def __init__(
        self,
        *,
        eval_fn,
        eval_freq: int = 20_000,
        n_battles: int = 100,
        checkpoint_dir: Path,
        initial_best_win_rate: float = -1.0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_fn = eval_fn
        self.eval_freq = eval_freq
        self.n_battles = n_battles
        self.checkpoint_dir = Path(checkpoint_dir)
        self.best_win_rate = initial_best_win_rate
        self._last_eval_at = -1

    def _on_step(self) -> bool:
        if self.num_timesteps <= 0 or self.num_timesteps % self.eval_freq != 0:
            return True
        if self.num_timesteps == self._last_eval_at:
            return True
        self._last_eval_at = self.num_timesteps

        if self.verbose:
            print(
                f"\n[Eval] Deterministic eval at {self.num_timesteps:,} steps "
                f"({self.n_battles} battles)...",
                flush=True,
            )

        win_rate, wins, rows = self.eval_fn(
            self.model,
            battles=self.n_battles,
        )
        self.logger.record("eval/win_rate", win_rate)
        self.logger.record("eval/wins", wins)
        self.logger.record("eval/battles", self.n_battles)

        if self.verbose:
            print(
                f"[Eval] Win rate: {win_rate:.1%} ({wins}/{self.n_battles})",
                flush=True,
            )

        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            wr_pct = int(round(win_rate * 100))
            path = self.checkpoint_dir / (
                f"best_wr{wr_pct}_steps{self.num_timesteps}_{stamp}"
            )
            self.model.save(str(path))
            meta = {
                "timesteps": self.num_timesteps,
                "win_rate": win_rate,
                "wins": wins,
                "battles": self.n_battles,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "battles_detail": rows,
            }
            path.with_suffix(".json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            best_model = self.checkpoint_dir / "best_model"
            shutil.copy2(_zip_path(path), _zip_path(best_model))
            best_model.with_suffix(".json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            if self.verbose:
                print(f"[Eval] New best — saved {path}.zip", flush=True)
                print(f"[Eval] Updated {best_model}.zip", flush=True)

        return True


class ResumeCheckpointCallback(BaseCallback):
    """Save resume_latest.zip so training can continue after an interrupted run."""

    def __init__(
        self,
        *,
        save_freq: int = 20_000,
        checkpoint_stem: Path,
        best_win_rate: float = -1.0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.checkpoint_stem = Path(checkpoint_stem)
        self.best_win_rate = best_win_rate
        self._last_save_at = -1

    def _on_step(self) -> bool:
        if self.num_timesteps <= 0 or self.num_timesteps % self.save_freq != 0:
            return True
        if self.num_timesteps == self._last_save_at:
            return True
        self._last_save_at = self.num_timesteps

        self.checkpoint_stem.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.checkpoint_stem))
        meta = {
            "timesteps": int(self.num_timesteps),
            "best_win_rate": self.best_win_rate,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.checkpoint_stem.with_suffix(".json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        if self.verbose:
            print(
                f"[Resume] Saved checkpoint at {self.num_timesteps:,} steps "
                f"-> {self.checkpoint_stem}.zip",
                flush=True,
            )
        return True

    def sync_best_win_rate(self, best: float) -> None:
        self.best_win_rate = best


def make_doubles_env(*, device: str = "cpu") -> Monitor:
    """CleanVGCRLEnv + ActionMasker + Monitor for MaskablePPO training."""
    env = CleanVGCRLEnv(device=device)
    env = ActionMasker(env, lambda e: e.action_masks())
    return Monitor(env)


def make_singles_env(*, device: str = "cpu", use_meta_pool: bool = True) -> Monitor:
    """CleanSinglesEnv + ActionMasker + Monitor for MaskablePPO training."""
    env = CleanSinglesEnv(device=device, use_meta_pool=use_meta_pool)
    env = ActionMasker(env, lambda e: e.action_masks())
    return Monitor(env)


def _singles_bc_preflight(bc_model: Path) -> None:
    """Verify singles BC checkpoint loads cleanly into the RL policy actor."""
    from src.core.model.transformer_bot import SINGLES_ACTION_SIZE, load_model

    print("[Preflight] Singles BC alignment check...", flush=True)
    if not bc_model.is_file():
        raise FileNotFoundError(f"Singles BC model not found: {bc_model}")

    bc = load_model(bc_model, device="cpu")
    if bc.config.action_space != "singles":
        raise RuntimeError(
            f"BC model action_space={bc.config.action_space!r}, expected 'singles'"
        )
    if bc.head_singles is None or bc.head_singles.out_features != SINGLES_ACTION_SIZE:
        raise RuntimeError(
            f"BC head_singles out_features != SINGLES_ACTION_SIZE ({SINGLES_ACTION_SIZE})"
        )

    from gymnasium.spaces import Box, Discrete
    from src.core.data.state_tokenizer import N_FIELDS, STACKED_N_TOKENS

    obs_space = Box(
        low=-np.inf,
        high=np.inf,
        shape=(STACKED_N_TOKENS, N_FIELDS),
        dtype=np.float32,
    )
    policy = SinglesBehaviorMaskablePolicy(
        obs_space,
        Discrete(SINGLES_ACTION_SIZE),
        lr_schedule=lambda _: 5e-6,
        bc_model_path=str(bc_model),
    )
    init_singles_bc_actor_weights(policy, bc_model)
    print(
        f"[Preflight] BC loaded into SinglesBehaviorMaskablePolicy "
        f"({SINGLES_ACTION_SIZE}-class head, strict=True)",
        flush=True,
    )

    try:
        from src.singles.evaluation.bc_examples import generate_bc_examples

        examples = generate_bc_examples(
            model_path=bc_model,
            n_examples=5,
            device="cpu",
        )
        hits = sum(1 for ex in examples if ex.correct)
        rate = hits / len(examples) if examples else 0.0
        print(
            f"[Preflight] BC examples: {hits}/{len(examples)} top-1 ({rate:.1%})",
            flush=True,
        )
    except Exception as exc:
        print(f"[Preflight] BC examples skipped: {exc}", flush=True)


def _zip_path(path: Path) -> Path:
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def _checkpoint_exists(path: Path) -> bool:
    return _zip_path(path).is_file()


def _steps_from_stem(stem: str) -> int:
    match = _STEPS_RE.search(stem)
    return int(match.group(1)) if match else -1


def _restore_best_win_rate(checkpoint_dir: Path) -> float:
    """Only restore from best_wr* eval checkpoints, not resume_latest metadata."""
    best = -1.0
    for meta_path in checkpoint_dir.glob("best_wr*.json"):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        wr = data.get("win_rate")
        if wr is not None:
            best = max(best, float(wr))
    return best


def _default_resume_path(checkpoint_dir: Path) -> Path | None:
    resume = checkpoint_dir / "resume_latest.zip"
    if resume.is_file():
        return resume

    candidates: list[tuple[int, Path]] = []
    for z in checkpoint_dir.glob("best_wr*.zip"):
        steps = _steps_from_stem(z.stem)
        if steps >= 0:
            candidates.append((steps, z))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def _learn_timesteps(*, resume: bool, current: int, target: int) -> int:
    if resume:
        return max(0, target - current)
    return target


def _build_model(
    env: Monitor,
    *,
    fmt: str,
    bc_model: Path,
    device: str,
    learning_rate: float,
    n_steps: int,
    batch_size: int,
    ent_coef: float,
    clip_range: float,
    tensorboard_log: Path,
    load_path: Path | None,
) -> MaskablePPO:
    if fmt == "singles":
        policy_cls = SinglesBehaviorMaskablePolicy
        init_fn = init_singles_bc_actor_weights
    else:
        policy_cls = VGCBehaviorMaskablePolicy
        init_fn = init_bc_actor_weights

    policy_kwargs = dict(
        bc_model_path=str(bc_model),
        net_arch=dict(pi=[], vf=[64, 64]),
        activation_fn=torch.nn.Tanh,
        ortho_init=False,
    )

    if load_path is not None and _checkpoint_exists(load_path):
        zip_file = _zip_path(load_path)
        print(f"Loading checkpoint: {zip_file}", flush=True)
        model = MaskablePPO.load(str(zip_file), env=env, device=device)
        print(f"  Resuming from {model.num_timesteps:,} timesteps", flush=True)
        return model

    model = MaskablePPO(
        policy_cls,
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        ent_coef=ent_coef,
        gamma=0.99,
        clip_range=clip_range,
        device=device,
        tensorboard_log=str(tensorboard_log),
        policy_kwargs=policy_kwargs,
        verbose=1,
    )
    init_fn(model.policy, bc_model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="RL fine-tune BC Transformer with MaskablePPO")
    parser.add_argument(
        "--format",
        choices=("doubles", "singles"),
        default="doubles",
        help="Battle format / env (default: doubles)",
    )
    parser.add_argument("--bc-model", type=Path, default=None)
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total target timesteps (when --resume, trains until this cumulative count)",
    )
    parser.add_argument("--sanity", action="store_true", help="Run 10k-step sanity check")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load a checkpoint and continue training (see --load)",
    )
    parser.add_argument(
        "--load",
        type=Path,
        default=None,
        help="Checkpoint .zip to load (default with --resume: resume_latest or highest-step best_wr*)",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--eval-battles", type=int, default=100)
    parser.add_argument(
        "--fixed-team",
        action="store_true",
        help="Singles only: use fixed agent team instead of meta pool randomization",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tensorboard-log", type=Path, default=None)
    parser.add_argument("--save-path", type=Path, default=None)
    args = parser.parse_args()

    fmt = args.format
    is_singles = fmt == "singles"
    if args.learning_rate is None:
        args.learning_rate = 1e-5 if is_singles else 5e-6
    if args.ent_coef is None:
        args.ent_coef = 0.01 if is_singles else 0.001
    if args.n_steps is None:
        args.n_steps = 4096 if is_singles else 2048
    if args.batch_size is None:
        args.batch_size = 512 if is_singles else 256
    if args.bc_model is None:
        args.bc_model = SINGLES_BC_MODEL_PATH if is_singles else BC_MODEL_PATH
    if args.tensorboard_log is None:
        args.tensorboard_log = LOGS_DIR / ("rl_ppo_singles" if is_singles else "rl_ppo")
    if args.save_path is None:
        args.save_path = MODELS_DIR / (
            "maskable_ppo_singles" if is_singles else "maskable_ppo_vgc"
        )

    checkpoint_dir = RL_CHECKPOINTS_DIR_SINGLES if is_singles else RL_CHECKPOINTS_DIR_DOUBLES
    resume_checkpoint_stem = checkpoint_dir / "resume_latest"
    eval_fn = run_singles_deterministic_eval if is_singles else run_doubles_deterministic_eval
    use_meta_pool = is_singles and not args.fixed_team

    def make_env_fn(*, device: str = "cpu") -> Monitor:
        if is_singles:
            return make_singles_env(device=device, use_meta_pool=use_meta_pool)
        return make_doubles_env(device=device)

    target_timesteps = 10_000 if args.sanity else args.timesteps
    if args.eval_freq is not None:
        eval_freq = args.eval_freq
    elif args.sanity:
        eval_freq = 5_000
    elif is_singles:
        eval_freq = 20_000
    else:
        eval_freq = 20_000
    eval_battles = 20 if args.sanity else args.eval_battles

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_log.mkdir(parents=True, exist_ok=True)

    if is_singles and not args.resume:
        _singles_bc_preflight(args.bc_model)

    load_path: Path | None = None
    if args.resume:
        if args.load is not None:
            load_path = args.load
            if not _checkpoint_exists(load_path):
                raise FileNotFoundError(f"Checkpoint not found: {_zip_path(load_path)}")
        else:
            load_path = _default_resume_path(checkpoint_dir)
            if load_path is None:
                raise FileNotFoundError(
                    "No checkpoint to resume from. Pass --load PATH or run training first."
                )
            print(f"Auto-selected resume checkpoint: {load_path}", flush=True)

    env = make_env_fn(device="cpu")
    if is_singles and use_meta_pool:
        inner = env
        while hasattr(inner, "env"):
            if isinstance(inner, CleanSinglesEnv):
                pool = inner.team_pool_info
                print(
                    f"Meta-pool training: agent pool={pool['agent']['pool_size']} teams, "
                    f"opponent pool={pool['opponent']['pool_size']} teams",
                    flush=True,
                )
                break
            inner = inner.env

    model = _build_model(
        env,
        fmt=fmt,
        bc_model=args.bc_model,
        device=args.device,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        tensorboard_log=args.tensorboard_log,
        load_path=load_path if args.resume else None,
    )

    initial_best_wr = _restore_best_win_rate(checkpoint_dir)
    learn_steps = _learn_timesteps(
        resume=args.resume,
        current=int(model.num_timesteps),
        target=target_timesteps,
    )

    if learn_steps <= 0:
        print(
            f"Already at or past target ({model.num_timesteps:,} >= {target_timesteps:,}); "
            "saving and exiting.",
            flush=True,
        )
        out = args.save_path.with_name(f"{args.save_path.name}_latest")
        model.save(str(out))
        env.close()
        return

    eval_cb = DeterministicEvalCallback(
        eval_fn=eval_fn,
        eval_freq=eval_freq,
        n_battles=eval_battles,
        checkpoint_dir=checkpoint_dir,
        initial_best_win_rate=initial_best_wr,
    )
    resume_cb = ResumeCheckpointCallback(
        save_freq=eval_freq,
        checkpoint_stem=resume_checkpoint_stem,
        best_win_rate=initial_best_wr,
    )

    class _SyncBestCallback(BaseCallback):
        def _on_step(self) -> bool:
            resume_cb.sync_best_win_rate(eval_cb.best_win_rate)
            return True

    if args.resume:
        print(
            f"Resuming MaskablePPO from {model.num_timesteps:,} steps\n"
            f"  Training {learn_steps:,} more steps (target {target_timesteps:,})\n"
            f"  restored best eval WR: {initial_best_wr:.1%}",
            flush=True,
        )
    else:
        env_name = "CleanSinglesEnv" if is_singles else "CleanVGCRLEnv"
        print(
            f"Starting MaskablePPO ({env_name}, {fmt}) for {learn_steps:,} steps\n"
            f"  lr={args.learning_rate} ent_coef={args.ent_coef} "
            f"clip_range={args.clip_range}\n"
            f"  n_steps={args.n_steps} batch_size={args.batch_size} "
            f"device={args.device}\n"
            f"  eval every {eval_freq:,} steps ({eval_battles} battles)\n"
            f"  bc_model={args.bc_model}\n"
            f"  tensorboard: {args.tensorboard_log}",
            flush=True,
        )

    callbacks = CallbackList(
        [
            WinRateCallback(),
            eval_cb,
            resume_cb,
            _SyncBestCallback(),
        ]
    )

    try:
        model.learn(
            total_timesteps=learn_steps,
            callback=callbacks,
            progress_bar=False,
            reset_num_timesteps=not args.resume,
        )
    except KeyboardInterrupt:
        print("\n[Interrupt] Saving resume checkpoint before exit...", flush=True)
        resume_checkpoint_stem.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(resume_checkpoint_stem))
        meta = {
            "timesteps": int(model.num_timesteps),
            "best_win_rate": eval_cb.best_win_rate,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "interrupted": True,
        }
        resume_checkpoint_stem.with_suffix(".json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(f"Saved {resume_checkpoint_stem}.zip", flush=True)
        env.close()
        raise

    win_cb = callbacks.callbacks[0]
    if isinstance(win_cb, WinRateCallback) and win_cb._episodes > 0:
        wr = win_cb._wins / win_cb._episodes
        print(
            f"Training summary: {win_cb._episodes} episodes, "
            f"win_rate={wr:.1%} ({win_cb._wins}/{win_cb._episodes})",
            flush=True,
        )

    suffix = "sanity" if args.sanity else "latest"
    out = args.save_path.with_name(f"{args.save_path.name}_{suffix}")
    model.save(str(out))
    resume_checkpoint_stem.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(resume_checkpoint_stem))
    print(f"Saved final model to {out}", flush=True)
    print(f"Saved resume checkpoint to {resume_checkpoint_stem}.zip", flush=True)
    env.close()


if __name__ == "__main__":
    main()
