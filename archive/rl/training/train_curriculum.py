"""Curriculum training: Random -> MaxDamage -> Self-play."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from config.settings import (
    CHECKPOINT_SAVE_FREQ,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    MODELS_DIR,
    N_ENV,
    OPPONENT_POOL_CURRICULUM,
    OPPONENT_POOL_DIR,
    OPPONENT_POOL_GROW_EVERY_STEPS,
    PPO_BATCH_SIZE,
    PPO_DEVICE,
    PPO_ENT_COEF,
    PPO_GAMMA,
    PPO_LEARNING_RATE,
    PPO_N_STEPS,
    STAGE1_TIMESTEPS,
    STAGE2_TIMESTEPS,
    STAGE3_TIMESTEPS,
    USE_OPPONENT_TEAM_POOL,
)
from src.players.max_damage_player import MaxDamagePlayer
from src.players.vgc_random_player import VGCRandomPlayer
from archive.rl.training.device import resolve_device
from archive.rl.training.pool_curriculum import (
    init_pool_curriculum,
    pool_limit_for_timesteps,
    write_pool_team_limit,
)
from archive.rl.training.vec_env import make_policy_opponent_env, make_vec_env

_STEPS_SUFFIX = re.compile(r"_(\d+)_steps\.zip$")


def _zip_path(path: Path) -> Path:
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def _checkpoint_exists(path: Path | None) -> bool:
    if path is None:
        return False
    return _zip_path(path).is_file()


def _timesteps_from_checkpoint(path: Path) -> int:
    match = _STEPS_SUFFIX.search(path.name)
    if match:
        return int(match.group(1))
    return 0


def _latest_stage_checkpoint(stage_name: str, final_save: Path) -> Path | None:
    """Newest periodic checkpoint for this stage, else the final stage zip."""
    prefix = f"maskable_ppo_{stage_name}_"
    candidates: list[Path] = []
    if CHECKPOINTS_DIR.is_dir():
        candidates.extend(CHECKPOINTS_DIR.glob(f"{prefix}*_steps.zip"))
    final_zip = _zip_path(final_save)
    if final_zip.is_file():
        candidates.append(final_zip)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda p: (_timesteps_from_checkpoint(p), p.stat().st_mtime),
    )


def _opponent_pool_size() -> int:
    if not (OPPONENT_POOL_CURRICULUM and USE_OPPONENT_TEAM_POOL and OPPONENT_POOL_DIR.is_dir()):
        return 0
    from src.teams.team_pool import PoolTeambuilder

    try:
        return PoolTeambuilder.from_directory(OPPONENT_POOL_DIR).team_count
    except FileNotFoundError:
        return 0


class PoolExpansionCallback(BaseCallback):
    """Expand opponent pool on a fixed step schedule (workers read a JSON file)."""

    def __init__(self, total_available: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_available = total_available
        self._last_limit = 0

    def _on_training_start(self) -> None:
        limit = pool_limit_for_timesteps(0, self.total_available)
        self._last_limit = limit
        write_pool_team_limit(limit, total_available=self.total_available)
        if self.verbose:
            print(f"Opponent pool curriculum: {limit}/{self.total_available} teams")

    def _on_step(self) -> bool:
        if self.num_timesteps <= 0:
            return True
        if self.num_timesteps % OPPONENT_POOL_GROW_EVERY_STEPS != 0:
            return True
        limit = pool_limit_for_timesteps(self.num_timesteps, self.total_available)
        if limit != self._last_limit:
            self._last_limit = limit
            write_pool_team_limit(limit, total_available=self.total_available)
            if self.verbose:
                print(
                    f"Opponent pool expanded to {limit}/{self.total_available} "
                    f"at {self.num_timesteps} steps"
                )
        return True


def _make_training_callbacks(
    stage_name: str,
    n_envs: int,
    checkpoint_freq: int,
    *,
    pool_size: int,
) -> CallbackList | None:
    callbacks = []
    checkpoint = _make_checkpoint_callback(stage_name, n_envs, checkpoint_freq)
    if checkpoint is not None:
        callbacks.append(checkpoint)
    if pool_size > 0 and OPPONENT_POOL_CURRICULUM:
        callbacks.append(PoolExpansionCallback(pool_size, verbose=1))
    if not callbacks:
        return None
    return CallbackList(callbacks)


def _make_checkpoint_callback(
    stage_name: str,
    n_envs: int,
    freq_timesteps: int,
) -> CheckpointCallback | None:
    if freq_timesteps <= 0:
        return None
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    save_freq = max(freq_timesteps // n_envs, 1)
    return CheckpointCallback(
        save_freq=save_freq,
        save_path=str(CHECKPOINTS_DIR),
        name_prefix=f"maskable_ppo_{stage_name}",
        verbose=1,
    )


def _learn_timesteps(
    *,
    resume: bool,
    current_timesteps: int,
    cli_timesteps: int | None,
    default_target: int,
) -> int:
    if resume:
        if cli_timesteps is not None:
            return cli_timesteps
        return max(0, default_target - current_timesteps)
    return cli_timesteps or default_target


def train_stage(
    name: str,
    opponent_cls,
    *,
    device: str,
    load_path: Path | None = None,
    resume: bool = False,
    cli_timesteps: int | None = None,
    target_timesteps: int,
    opponent_kwargs: dict | None = None,
    n_envs: int = N_ENV,
    use_subproc: bool = True,
    checkpoint_freq: int = CHECKPOINT_SAVE_FREQ,
) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / f"maskable_ppo_{name}"

    env = make_vec_env(
        opponent_cls,
        n_envs=n_envs,
        use_subproc=use_subproc,
        opponent_kwargs=opponent_kwargs,
    )
    if _checkpoint_exists(load_path):
        model = MaskablePPO.load(str(_zip_path(load_path)), env=env, device=device)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=PPO_LEARNING_RATE,
            n_steps=max(PPO_N_STEPS // n_envs, 64),
            batch_size=PPO_BATCH_SIZE,
            gamma=PPO_GAMMA,
            ent_coef=PPO_ENT_COEF,
            device=device,
            tensorboard_log=str(LOGS_DIR),
            verbose=1,
        )

    learn_steps = _learn_timesteps(
        resume=resume,
        current_timesteps=model.num_timesteps,
        cli_timesteps=cli_timesteps,
        default_target=target_timesteps,
    )
    if learn_steps <= 0:
        print(f"Already at or past target ({model.num_timesteps} >= {target_timesteps}); saving and exiting.")
        model.save(str(save_path))
        env.close()
        return save_path

    if resume:
        print(
            f"Resuming from {model.num_timesteps} steps; "
            f"training {learn_steps} more (target {target_timesteps})"
        )
    else:
        print(f"Training {learn_steps} steps (target {target_timesteps})")

    from archive.rl.env.observation import N_FEATURES

    print(f"Observation size: {N_FEATURES}-d")

    pool_size = _opponent_pool_size()
    if pool_size > 0 and OPPONENT_POOL_CURRICULUM and not resume:
        init_pool_curriculum(pool_size)
    callbacks = _make_training_callbacks(
        name, n_envs, checkpoint_freq, pool_size=pool_size
    )

    model.learn(
        total_timesteps=learn_steps,
        tb_log_name=name,
        reset_num_timesteps=not resume,
        callback=callbacks,
    )
    model.save(str(save_path))
    env.close()
    return save_path


def train_self_play(
    opponent_model_path: Path,
    *,
    device: str,
    name: str = "stage3_selfplay",
    resume: bool = False,
    cli_timesteps: int | None = None,
    target_timesteps: int,
    n_envs: int = N_ENV,
    use_subproc: bool = True,
    checkpoint_freq: int = CHECKPOINT_SAVE_FREQ,
) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / f"maskable_ppo_{name}"

    env = make_policy_opponent_env(
        str(opponent_model_path),
        n_envs=n_envs,
        use_subproc=use_subproc,
        device=device,
    )

    resume_path = _latest_stage_checkpoint(name, save_path) if resume else None
    if resume_path is not None:
        model = MaskablePPO.load(str(resume_path), env=env, device=device)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=PPO_LEARNING_RATE,
            n_steps=max(PPO_N_STEPS // n_envs, 64),
            batch_size=PPO_BATCH_SIZE,
            gamma=PPO_GAMMA,
            ent_coef=PPO_ENT_COEF,
            device=device,
            tensorboard_log=str(LOGS_DIR),
            verbose=1,
        )

    learn_steps = _learn_timesteps(
        resume=resume and resume_path is not None,
        current_timesteps=model.num_timesteps,
        cli_timesteps=cli_timesteps,
        default_target=target_timesteps,
    )
    if learn_steps <= 0:
        print(f"Already at or past target ({model.num_timesteps} >= {target_timesteps}); saving and exiting.")
        model.save(str(save_path))
        env.close()
        return save_path

    if resume and resume_path is not None:
        print(
            f"Resuming self-play from {model.num_timesteps} steps; "
            f"training {learn_steps} more (target {target_timesteps})"
        )
    else:
        print(f"Self-play training {learn_steps} steps (target {target_timesteps})")

    callback = _make_checkpoint_callback(name, n_envs, checkpoint_freq)
    callbacks = [callback] if callback is not None else None

    model.learn(
        total_timesteps=learn_steps,
        tb_log_name=name,
        reset_num_timesteps=not (resume and resume_path is not None),
        callback=callbacks,
    )
    model.save(str(save_path))
    env.close()
    return save_path


def _resolve_stage_load(
    stage_name: str,
    stage_save: Path,
    prior_stage: Path | None,
    *,
    resume: bool,
) -> tuple[Path | None, bool]:
    """Return (load_path, is_resume)."""
    if resume:
        latest = _latest_stage_checkpoint(stage_name, stage_save)
        if latest is not None:
            return latest, True
        print("Warning: --resume set but no stage checkpoint found; loading prior stage if available.")

    if _checkpoint_exists(prior_stage):
        return _zip_path(prior_stage), False
    return None, False


def main():
    parser = argparse.ArgumentParser(description="Champions VGC curriculum training")
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 0],
        default=0,
        help="Stage to run (0 = full curriculum)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Step budget: fresh run trains this many; --resume adds this many (or runs to stage default)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from latest periodic/final checkpoint for this stage",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=CHECKPOINT_SAVE_FREQ,
        help=f"Save checkpoint every N timesteps (0=disable, default {CHECKPOINT_SAVE_FREQ})",
    )
    parser.add_argument(
        "--n-env",
        type=int,
        default=N_ENV,
        help=f"Parallel Showdown battles (default: {N_ENV})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=PPO_DEVICE,
        choices=["auto", "cuda", "cpu"],
        help="PPO policy device (auto picks CUDA when available)",
    )
    parser.add_argument(
        "--no-subproc",
        action="store_true",
        help="Use DummyVecEnv instead of SubprocVecEnv (safer on Windows)",
    )
    args = parser.parse_args()
    device = resolve_device(args.device)
    use_subproc = not args.no_subproc

    print(f"Parallel envs: {args.n_env} ({'SubprocVecEnv' if use_subproc else 'DummyVecEnv'})")
    if args.checkpoint_freq > 0:
        print(f"Checkpoints every {args.checkpoint_freq} timesteps -> {CHECKPOINTS_DIR}")

    stage1_path = MODELS_DIR / "maskable_ppo_stage1_random"
    stage2_path = MODELS_DIR / "maskable_ppo_stage2_maxdamage"

    if args.stage in (0, 1):
        print(f"=== Stage 1: vs RandomPlayer (target {STAGE1_TIMESTEPS} steps) ===")
        load_from, resume = _resolve_stage_load(
            "stage1_random", stage1_path, None, resume=args.resume
        )
        if load_from:
            print(f"Loading checkpoint: {load_from}")
        stage1_path = train_stage(
            "stage1_random",
            VGCRandomPlayer,
            device=device,
            load_path=load_from,
            resume=resume,
            cli_timesteps=args.timesteps,
            target_timesteps=STAGE1_TIMESTEPS,
            n_envs=args.n_env,
            use_subproc=use_subproc,
            checkpoint_freq=args.checkpoint_freq,
        )

    if args.stage in (0, 2):
        print(f"=== Stage 2: vs MaxDamagePlayer (target {STAGE2_TIMESTEPS} steps) ===")
        load_from, resume = _resolve_stage_load(
            "stage2_maxdamage",
            stage2_path,
            stage1_path,
            resume=args.resume,
        )
        if load_from:
            print(f"Loading checkpoint: {load_from}")
        stage2_path = train_stage(
            "stage2_maxdamage",
            MaxDamagePlayer,
            device=device,
            load_path=load_from,
            resume=resume,
            cli_timesteps=args.timesteps,
            target_timesteps=STAGE2_TIMESTEPS,
            n_envs=args.n_env,
            use_subproc=use_subproc,
            checkpoint_freq=args.checkpoint_freq,
        )

    if args.stage in (0, 3):
        opp = (
            stage2_path
            if _checkpoint_exists(stage2_path)
            else stage1_path
        )
        print(f"=== Stage 3: self-play vs {opp} (target {STAGE3_TIMESTEPS} steps) ===")
        train_self_play(
            opp,
            device=device,
            resume=args.resume,
            cli_timesteps=args.timesteps,
            target_timesteps=STAGE3_TIMESTEPS,
            n_envs=args.n_env,
            use_subproc=use_subproc,
            checkpoint_freq=args.checkpoint_freq,
        )


if __name__ == "__main__":
    main()
