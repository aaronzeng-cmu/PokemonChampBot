#!/usr/bin/env python3
"""Train belief-augmented value MLP on collect_value_data output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config.settings import MODELS_DIR
from archive.ismcts.planning.value_mlp import ValueMLP, ValueMLPConfig, save_value_mlp


def _find_latest_npz(data_dir: Path) -> Path | None:
    files = sorted(data_dir.glob("value_data_*.npz"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def train(
    *,
    npz_path: Path,
    out_path: Path,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_frac: float = 0.1,
    hidden_dims: tuple[int, ...] = (256, 128, 64),
    device: str = "cpu",
) -> dict:
    data = np.load(npz_path)
    states = data["states"].astype(np.float32)
    outcomes = data["outcomes"].astype(np.float32)
    # Map {0,1} labels to {-1,1} for Tanh output
    targets = outcomes * 2.0 - 1.0

    n = states.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    split = int(n * (1.0 - val_frac))
    train_idx, val_idx = idx[:split], idx[split:]

    x_train = torch.as_tensor(states[train_idx], device=device)
    y_train = torch.as_tensor(targets[train_idx], device=device)
    x_val = torch.as_tensor(states[val_idx], device=device)
    y_val = torch.as_tensor(targets[val_idx], device=device)

    config = ValueMLPConfig(
        input_dim=states.shape[1],
        hidden_dims=hidden_dims,
    )
    model = ValueMLP(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    best_val = float("inf")
    best_state = None
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * len(xb)
        train_loss /= max(len(train_idx), 1)

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = float(loss_fn(val_pred, y_val).item())
            val_acc = float(
                ((val_pred >= 0) == (y_val >= 0)).float().mean().item()
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_sign_accuracy": val_acc,
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"epoch {epoch:3d} | train_mse={train_loss:.4f} "
                f"val_mse={val_loss:.4f} val_acc={val_acc:.1%}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    extra = {
        "npz_path": str(npz_path.resolve()),
        "samples": int(n),
        "best_val_mse": best_val,
        "history": history[-5:],
    }
    save_value_mlp(model, out_path, extra=extra)
    return extra


def main() -> None:
    parser = argparse.ArgumentParser(description="Train value MLP on gauntlet rollout data")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="NPZ from collect_value_data (default: latest in logs/value_data/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=MODELS_DIR / "value_mlp.pt",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parents[3] / "logs" / "value_data"
    npz_path = args.data or _find_latest_npz(data_dir)
    if npz_path is None or not npz_path.is_file():
        raise FileNotFoundError(
            f"No value data NPZ found. Run: python scripts/collect_value_data.py --games 100"
        )

    print(f"Training on: {npz_path}")
    meta = train(
        npz_path=npz_path,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    print(f"\nSaved: {args.out.resolve()}")
    print(f"Samples: {meta['samples']} | best val MSE: {meta['best_val_mse']:.4f}")
    summary_path = args.out.parent / "value_mlp_train_summary.json"
    summary_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
