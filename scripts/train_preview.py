#!/usr/bin/env python3
"""Train TeamPreviewModel on preview_dataset.pt."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from config.settings import (
    PREVIEW_DATASET_PATH,
    PREVIEW_MODEL_PATH,
    SINGLES_PREVIEW_DATASET_PATH,
    SINGLES_PREVIEW_MODEL_PATH,
)
from src.core.model.preview_model import TeamPreviewModel, save_preview_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train team preview model")
    parser.add_argument(
        "--format",
        choices=("doubles", "singles"),
        default="doubles",
        help="doubles=bring-4 preview; singles=BSS bring-3",
    )
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (val loss)")
    args = parser.parse_args()

    if args.format == "singles":
        if args.dataset is None:
            args.dataset = SINGLES_PREVIEW_DATASET_PATH
        if args.out is None:
            args.out = SINGLES_PREVIEW_MODEL_PATH
        log_subdir = "singles"
    else:
        if args.dataset is None:
            args.dataset = PREVIEW_DATASET_PATH
        if args.out is None:
            args.out = PREVIEW_MODEL_PATH
        log_subdir = "doubles"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    x = data["species_ids"]
    y_leads = data["leads"]
    y_brought = data["brought"]
    n = x.size(0)
    val_n = max(1, int(n * args.val_frac))
    train_n = n - val_n
    ds = TensorDataset(x, y_leads, y_brought)
    train_ds, val_ds = random_split(ds, [train_n, val_n], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = TeamPreviewModel().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    stale_epochs = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, lb, bb in train_loader:
            xb, lb, bb = xb.to(device), lb.to(device), bb.to(device)
            optim.zero_grad()
            pred_l, pred_b = model(xb)
            loss = loss_fn(pred_l, lb) + loss_fn(pred_b, bb)
            loss.backward()
            optim.step()
            train_loss += loss.item() * xb.size(0)

        model.eval()
        val_loss = 0.0
        lead_hits = 0.0
        brought_hits = 0.0
        joint_hits = 0.0
        total = 0
        with torch.no_grad():
            for xb, lb, bb in val_loader:
                xb, lb, bb = xb.to(device), lb.to(device), bb.to(device)
                pred_l, pred_b = model(xb)
                loss = loss_fn(pred_l, lb) + loss_fn(pred_b, bb)
                val_loss += loss.item() * xb.size(0)
                batch_n = xb.size(0)
                total += batch_n
                lead_pred = (pred_l.sigmoid() > 0.5).float()
                brought_pred = (pred_b.sigmoid() > 0.5).float()
                lead_hits += (lead_pred == lb).all(dim=1).float().sum().item()
                brought_hits += (brought_pred == bb).all(dim=1).float().sum().item()
                joint_hits += ((lead_pred == lb) & (brought_pred == bb)).all(dim=1).float().sum().item()

        train_loss /= train_n
        val_loss /= val_n
        metrics = {
            "epoch": epoch,
            "format": args.format,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_lead_exact": lead_hits / total,
            "val_brought_exact": brought_hits / total,
            "val_joint_exact": joint_hits / total,
        }
        history.append(metrics)
        print(json.dumps(metrics))
        if val_loss < best_val:
            best_val = val_loss
            stale_epochs = 0
            save_preview_model(
                model,
                args.out,
                extra={"best_epoch": epoch, "val_loss": val_loss, "format": args.format},
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping: val loss plateaued for {args.patience} epochs.")
                break

    log_dir = args.out.parent.parent / "logs" / "preview_training" / log_subdir
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary_path = log_dir / f"train_summary_{stamp}.json"
    summary_path.write_text(json.dumps({"history": history, "best_val": best_val}, indent=2), encoding="utf-8")
    print(f"Saved model -> {args.out}")


if __name__ == "__main__":
    main()
