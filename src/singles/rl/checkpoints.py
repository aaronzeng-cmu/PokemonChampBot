"""Resolve trained Singles MaskablePPO checkpoints."""

from __future__ import annotations

from pathlib import Path

RL_CHECKPOINTS_DIR_SINGLES = Path("models/rl_checkpoints_singles")
DEFAULT_BEST_MODEL = RL_CHECKPOINTS_DIR_SINGLES / "best_model.zip"


def resolve_rl_checkpoint(path: Path | None = None) -> Path:
    if path is not None:
        ckpt = path if path.suffix == ".zip" else path.with_suffix(".zip")
        if not ckpt.is_file():
            raise FileNotFoundError(f"RL checkpoint not found: {ckpt}")
        return ckpt
    if DEFAULT_BEST_MODEL.is_file():
        return DEFAULT_BEST_MODEL
    candidates = sorted(RL_CHECKPOINTS_DIR_SINGLES.glob("best_wr*.zip"))
    if not candidates:
        raise FileNotFoundError(
            f"No RL checkpoints in {RL_CHECKPOINTS_DIR_SINGLES} "
            "(expected best_model.zip or best_wr*.zip)"
        )
    return candidates[-1]
