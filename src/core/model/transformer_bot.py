"""Transformer behavior cloning model (doubles + singles heads)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from src.doubles.data.action_space_spec import ACTION_SIZE
from src.core.data.state_tokenizer import N_FIELDS, N_TOKENS, STACKED_N_TOKENS

ActionSpaceKind = Literal["doubles", "singles"]
# 0-1 bench switch | 2-5 move | 6-9 mega | 10-13 z | 14-17 dynamax/tera
SINGLES_ACTION_SIZE = 18


@dataclass
class VGCBehaviorClonerConfig:
    n_tokens: int = STACKED_N_TOKENS
    n_fields: int = N_FIELDS
    vocab_size: int = 4096
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1
    action_space: ActionSpaceKind = "doubles"
    action_size: int = ACTION_SIZE


class VGCBehaviorCloner(nn.Module):
    def __init__(self, config: VGCBehaviorClonerConfig | None = None):
        super().__init__()
        self.config = config or VGCBehaviorClonerConfig()
        c = self.config
        field_dim = 32
        self.field_embeddings = nn.ModuleList(
            [nn.Embedding(c.vocab_size, field_dim) for _ in range(c.n_fields)]
        )
        self.token_proj = nn.Linear(field_dim * c.n_fields, c.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, c.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.nhead,
            dim_feedforward=c.dim_feedforward,
            dropout=c.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=c.num_layers)

        if c.action_space == "singles":
            self.head_singles = nn.Linear(c.d_model, SINGLES_ACTION_SIZE)
            self.head1 = None
            self.head2 = None
        else:
            self.head1 = nn.Linear(c.d_model, c.action_size)
            self.head2 = nn.Linear(c.d_model, c.action_size)
            self.head_singles = None

    @property
    def action_space(self) -> ActionSpaceKind:
        return self.config.action_space

    def _embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        parts = []
        for i in range(self.config.n_fields):
            parts.append(self.field_embeddings[i](token_ids[:, :, i]))
        return self.token_proj(torch.cat(parts, dim=-1))

    def _encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self._embed_tokens(token_ids)
        b = x.size(0)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        return x[:, 0]

    def forward(
        self, token_ids: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        pooled = self._encode(token_ids)
        if self.config.action_space == "singles":
            assert self.head_singles is not None
            return self.head_singles(pooled)
        assert self.head1 is not None and self.head2 is not None
        return self.head1(pooled), self.head2(pooled)


def save_model(
    model: VGCBehaviorCloner,
    path: Path,
    *,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_model(path: Path, device: str = "cpu") -> VGCBehaviorCloner:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = VGCBehaviorClonerConfig(**payload["config"])
    model = VGCBehaviorCloner(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def save_config_json(config: VGCBehaviorClonerConfig, path: Path) -> None:
    path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
