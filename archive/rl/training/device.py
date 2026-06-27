"""Resolve PPO training device (CPU / CUDA)."""

from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> str:
    """Return a Stable-Baselines3 device string."""
    choice = requested.lower()
    if choice == "auto":
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"Using CUDA device: {name}")
            return "cuda"
        print("CUDA not available; using CPU.")
        return "cpu"
    if choice == "cuda":
        if not torch.cuda.is_available():
            print("Warning: --device cuda requested but CUDA is unavailable; using CPU.")
            return "cpu"
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return "cuda"
    if choice == "cpu":
        print("Using CPU device")
        return "cpu"
    raise ValueError(f"Unknown device {requested!r}; use auto, cuda, or cpu.")
