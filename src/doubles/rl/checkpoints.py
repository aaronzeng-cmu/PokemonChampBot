"""Resolve and load MaskablePPO RL checkpoints."""

from __future__ import annotations

from pathlib import Path

from sb3_contrib import MaskablePPO

RL_CHECKPOINTS_DIR = Path("models/rl_checkpoints")
DEFAULT_BEST = RL_CHECKPOINTS_DIR / "best_wr86_steps200000_20260615_112649.zip"


def resolve_rl_checkpoint(path: Path | None) -> Path:
    if path is not None:
        p = path if path.suffix == ".zip" else path.with_suffix(".zip")
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p
    if DEFAULT_BEST.is_file():
        return DEFAULT_BEST
    candidates = sorted(RL_CHECKPOINTS_DIR.glob("best_wr*.zip"))
    if not candidates:
        raise FileNotFoundError(f"No RL checkpoints in {RL_CHECKPOINTS_DIR}")
    return candidates[-1]


def load_rl_checkpoint(path: Path | None, *, device: str) -> tuple[MaskablePPO, Path]:
    ckpt = resolve_rl_checkpoint(path)
    model = MaskablePPO.load(str(ckpt), device=device)
    return model, ckpt
