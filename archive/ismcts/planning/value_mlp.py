"""Small MLP value network for ISMCTS leaf evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class ValueMLPConfig:
    input_dim: int
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ValueMLPConfig:
        return cls(
            input_dim=int(data["input_dim"]),
            hidden_dims=tuple(int(x) for x in data.get("hidden_dims", (256, 128, 64))),
            dropout=float(data.get("dropout", 0.1)),
        )


class ValueMLP(nn.Module):
    """Maps belief-augmented state vector -> win probability in [-1, 1]."""

    def __init__(self, config: ValueMLPConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        prev = config.input_dim
        for width in config.hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, width),
                    nn.LayerNorm(width),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
            prev = width
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def save_value_mlp(
    model: ValueMLP,
    path: Path,
    *,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)
    meta_path = path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "config": model.config.to_dict(),
                "extra": extra or {},
                "weights": str(path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_value_mlp(path: Path, *, device: str = "cpu") -> ValueMLP:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = ValueMLPConfig.from_dict(payload["config"])
    model = ValueMLP(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model
