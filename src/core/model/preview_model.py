"""Feed-forward team preview model (leads + brought multi-label heads)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from src.core.data.perspective import hash_token


@dataclass
class TeamPreviewModelConfig:
    n_species: int = 12
    vocab_size: int = 4096
    embed_dim: int = 64
    hidden_dim: int = 256
    team_size: int = 6


class TeamPreviewModel(nn.Module):
    def __init__(self, config: TeamPreviewModelConfig | None = None):
        super().__init__()
        self.config = config or TeamPreviewModelConfig()
        c = self.config
        self.embed = nn.Embedding(c.vocab_size, c.embed_dim)
        self.fc1 = nn.Linear(c.n_species * c.embed_dim, c.hidden_dim)
        self.fc2 = nn.Linear(c.hidden_dim, c.hidden_dim)
        self.head_leads = nn.Linear(c.hidden_dim, c.team_size)
        self.head_brought = nn.Linear(c.hidden_dim, c.team_size)
        self.act = nn.ReLU()

    def forward(self, species_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embed(species_ids)
        flat = emb.reshape(species_ids.size(0), -1)
        h = self.act(self.fc1(flat))
        h = self.act(self.fc2(h))
        return self.head_leads(h), self.head_brought(h)


def species_ids_from_teams(our_species: list[str], opp_species: list[str]) -> torch.Tensor:
    ours = [hash_token(s) for s in our_species[:6]]
    opps = [hash_token(s) for s in opp_species[:6]]
    while len(ours) < 6:
        ours.append(0)
    while len(opps) < 6:
        opps.append(0)
    return torch.tensor([ours + opps], dtype=torch.long)


def predict_preview_slots(
    model: TeamPreviewModel,
    our_species: list[str],
    opp_species: list[str],
    *,
    device: str = "cpu",
) -> list[int]:
    """Return 1-based roster slots: 2 leads + 2 backline."""
    model.eval()
    x = species_ids_from_teams(our_species, opp_species).to(device)
    with torch.no_grad():
        lead_logits, brought_logits = model(x)
    lead_logits = lead_logits[0]
    brought_logits = brought_logits[0]

    lead_order = torch.argsort(lead_logits, descending=True).tolist()
    leads = lead_order[:2]

    remaining = [i for i in range(6) if i not in leads]
    back_order = sorted(remaining, key=lambda i: float(brought_logits[i]), reverse=True)
    backline = back_order[:2]
    return [i + 1 for i in leads + backline]


def predict_singles_preview_slots(
    model: TeamPreviewModel,
    our_species: list[str],
    opp_species: list[str],
    *,
    device: str = "cpu",
    n_brought: int = 3,
) -> list[int]:
    """Return 1-based roster slots: 1 lead + (n_brought - 1) backline for BSS."""
    model.eval()
    x = species_ids_from_teams(our_species, opp_species).to(device)
    with torch.no_grad():
        lead_logits, brought_logits = model(x)
    lead_logits = lead_logits[0]
    brought_logits = brought_logits[0]

    lead_idx = int(torch.argmax(lead_logits).item())
    remaining = [i for i in range(6) if i != lead_idx]
    back_order = sorted(remaining, key=lambda i: float(brought_logits[i]), reverse=True)
    slots = [lead_idx] + back_order[: max(0, n_brought - 1)]
    return [i + 1 for i in slots]


def save_preview_model(
    model: TeamPreviewModel,
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


def load_preview_model(path: Path, device: str = "cpu") -> TeamPreviewModel:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = TeamPreviewModelConfig(**payload["config"])
    model = TeamPreviewModel(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def save_preview_config_json(config: TeamPreviewModelConfig, path: Path) -> None:
    path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
