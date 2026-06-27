#!/usr/bin/env python3
"""Supervised behavior cloning training for VGCBehaviorCloner (dual 107-class heads)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config.settings import (
    BC_DATASET_PATH,
    BC_MODEL_PATH,
    BC_TRAINING_LOG_DIR,
    CHECKPOINTS_DIR,
    SINGLES_BC_DATASET_PATH,
    SINGLES_BC_MODEL_PATH,
)
from src.doubles.data.action_space_spec import ACTION_UNKNOWN
from src.singles.log_action_codec import ACTION_UNKNOWN as SINGLES_ACTION_UNKNOWN
from src.singles.evaluation.eval_pipeline import EvalPipelineConfig, run_eval_pipeline
from src.core.model.transformer_bot import (
    SINGLES_ACTION_SIZE,
    VGCBehaviorCloner,
    VGCBehaviorClonerConfig,
    save_model,
)
from src.core.training.mmap_dataset import (
    build_chunked_datasets,
    has_mmap_dataset,
    load_doubles_sample_count,
    mmap_dataset_dir,
)

_MASK_NEG = -1e9


def _apply_action_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, _MASK_NEG)


def _run_epoch(
    model: VGCBehaviorCloner,
    loader: DataLoader,
    *,
    device: str,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, dict[str, float]]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total = 0
    slot_hits = 0.0
    joint_hits = 0.0

    for batch in loader:
        xb, y0, y1, m0, m1 = [t.to(device) for t in batch]
        if train_mode:
            optimizer.zero_grad()

        logits0, logits1 = model(xb)

        loss0 = loss_fn(logits0, y0)
        loss1 = loss_fn(logits1, y1)
        loss = loss0 + loss1

        if train_mode:
            loss.backward()
            optimizer.step()

        batch_n = xb.size(0)
        total_loss += loss.item() * batch_n
        total += batch_n

        valid0 = y0 != ACTION_UNKNOWN
        valid1 = y1 != ACTION_UNKNOWN
        with torch.no_grad():
            pred0 = logits0.argmax(1)
            pred1 = logits1.argmax(1)
            slot_hits += (pred0 == y0)[valid0].float().sum().item()
            slot_hits += (pred1 == y1)[valid1].float().sum().item()
            joint_mask = valid0 & valid1
            if joint_mask.any():
                joint_hits += (
                    (pred0 == y0) & (pred1 == y1)
                )[joint_mask].float().sum().item()

    denom = max(1, total)
    valid_slots = max(1, total * 2)
    return total_loss / denom, {
        "slot_top1": slot_hits / valid_slots,
        "joint_top1": joint_hits / denom,
    }


def _run_singles_epoch(
    model: VGCBehaviorCloner,
    loader: DataLoader,
    *,
    device: str,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    masked_loss: bool = False,
) -> tuple[float, dict[str, float]]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total = 0
    hits = 0.0
    valid = 0.0

    for batch in loader:
        xb, y, mask = [t.to(device) for t in batch]
        if train_mode:
            optimizer.zero_grad()

        logits = model(xb)
        loss_logits = _apply_action_mask(logits, mask) if masked_loss else logits
        loss = loss_fn(loss_logits, y)

        if train_mode:
            loss.backward()
            optimizer.step()

        batch_n = xb.size(0)
        total_loss += loss.item() * batch_n
        total += batch_n

        label_valid = y != SINGLES_ACTION_UNKNOWN
        with torch.no_grad():
            pred = logits.argmax(1)
            hits += (pred == y)[label_valid].float().sum().item()
            valid += label_valid.float().sum().item()

    denom = max(1, total)
    return total_loss / denom, {
        "top1": hits / max(1.0, valid),
    }


def _plot_curves(history: list[dict], out_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [row["val_slot_top1"] for row in history], label="val slot top-1")
    axes[1].plot(epochs, [row["val_joint_top1"] for row in history], label="val joint top-1")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_singles_curves(history: list[dict], out_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [row["val_top1"] for row in history], label="val top-1")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _train_singles(args: argparse.Namespace) -> None:
    dataset_path = args.dataset or SINGLES_BC_DATASET_PATH
    out_path = args.out or SINGLES_BC_MODEL_PATH
    log_dir = BC_TRAINING_LOG_DIR / "singles"

    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if "action_mask" not in data:
        raise SystemExit(
            f"Dataset {dataset_path} lacks action_mask. "
            "Re-run: python scripts/parse_replays.py --format singles"
        )

    x = torch.as_tensor(data["token_ids"], dtype=torch.long)
    y = torch.as_tensor(data["action"], dtype=torch.long)
    mask = torch.as_tensor(data["action_mask"], dtype=torch.bool)

    n = x.shape[0]
    if n == 0:
        raise SystemExit(f"No samples in {dataset_path}")

    idx = torch.randperm(n)
    split = int(n * (1.0 - args.val_frac))
    train_idx, val_idx = idx[:split], idx[split:]

    train_ds = TensorDataset(x[train_idx], y[train_idx], mask[train_idx])
    val_ds = TensorDataset(x[val_idx], y[val_idx], mask[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=True)

    model = VGCBehaviorCloner(
        VGCBehaviorClonerConfig(action_space="singles", action_size=SINGLES_ACTION_SIZE)
    ).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=SINGLES_ACTION_UNKNOWN)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    stale_epochs = 0
    history: list[dict] = []

    masked_loss = False
    print(
        f"[singles] Training on {len(train_ds)} samples, validating on {len(val_ds)} "
        f"(batch_size={args.batch_size}, device={args.device}, masked_loss={masked_loss})"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = _run_singles_epoch(
            model,
            train_loader,
            device=args.device,
            loss_fn=loss_fn,
            optimizer=opt,
            masked_loss=masked_loss,
        )
        val_loss, val_metrics = _run_singles_epoch(
            model,
            val_loader,
            device=args.device,
            loss_fn=loss_fn,
            optimizer=None,
            masked_loss=masked_loss,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_top1": train_metrics["top1"],
            "val_top1": val_metrics["top1"],
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_top1={val_metrics['top1']:.3f}"
        )

        epoch_ckpt = CHECKPOINTS_DIR / f"singles_bc_epoch_{epoch:02d}.pt"
        save_model(model, epoch_ckpt, extra={"epoch": epoch, **row})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale_epochs = 0
            save_model(
                model,
                out_path,
                extra={"best_val_loss": val_loss, "epoch": epoch, **row},
            )
        else:
            stale_epochs += 1

        _plot_singles_curves(history, log_dir / "latest.png")

        if stale_epochs >= args.patience:
            print(f"Early stopping: val loss plateaued for {args.patience} epochs.")
            break

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary = {
        "format": "singles",
        "dataset": str(dataset_path),
        "samples": n,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "batch_size": args.batch_size,
        "device": args.device,
        "masked_loss": masked_loss,
        "best_val_loss": best_val_loss,
        "epochs_run": len(history),
        "history": history,
        "model": str(out_path),
        "timestamp": stamp,
    }
    summary_path = log_dir / f"train_summary_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (log_dir / "train_summary_latest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Saved summary to {summary_path}")
    print(f"Best model: {out_path} (val_loss={best_val_loss:.4f})")

    if not args.skip_post_eval:
        print("\n--- Post-training eval (BC examples + inference trace) ---")
        run_eval_pipeline(
            EvalPipelineConfig(
                model_path=out_path,
                device=args.device,
                skip_replays=True,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BC Transformer")
    parser.add_argument(
        "--format",
        choices=("doubles", "singles"),
        default="doubles",
        help="doubles=VGC dual-head; singles=22-class head_singles",
    )
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=0, help="0 = auto (512 cuda, 256 cpu)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience (val loss)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--skip-post-eval",
        action="store_true",
        help="Skip post-training BC examples + inference trace",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        args.batch_size = 512 if args.device.startswith("cuda") else 256

    if args.format == "singles":
        if args.dataset is None:
            args.dataset = SINGLES_BC_DATASET_PATH
        if args.out is None:
            args.out = SINGLES_BC_MODEL_PATH
        _train_singles(args)
        return

    if args.dataset is None:
        args.dataset = BC_DATASET_PATH
    if args.out is None:
        args.out = BC_MODEL_PATH

    mmap_dir = mmap_dataset_dir(args.dataset)
    if not has_mmap_dataset(mmap_dir):
        raise SystemExit(
            f"No mmap dataset at {mmap_dir}. "
            "Run: python scripts/build_bc_mmap.py"
        )

    n = load_doubles_sample_count(mmap_dir)
    if n == 0:
        raise SystemExit(f"No samples in {mmap_dir}")

    idx = torch.randperm(n).numpy()
    split = int(n * (1.0 - args.val_frac))
    train_idx, val_idx = idx[:split], idx[split:]

    train_ds, val_ds = build_chunked_datasets(mmap_dir, train_idx, val_idx)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=True, num_workers=0)

    model = VGCBehaviorCloner().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=ACTION_UNKNOWN)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    BC_TRAINING_LOG_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    stale_epochs = 0
    history: list[dict] = []

    print(
        f"Training on {len(train_ds)} samples, validating on {len(val_ds)} "
        f"(batch_size={args.batch_size}, device={args.device}, masked_loss=False)"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = _run_epoch(
            model, train_loader, device=args.device, loss_fn=loss_fn, optimizer=opt
        )
        val_loss, val_metrics = _run_epoch(
            model, val_loader, device=args.device, loss_fn=loss_fn, optimizer=None
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_slot_top1": train_metrics["slot_top1"],
            "val_slot_top1": val_metrics["slot_top1"],
            "train_joint_top1": train_metrics["joint_top1"],
            "val_joint_top1": val_metrics["joint_top1"],
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_slot_top1={val_metrics['slot_top1']:.3f} "
            f"val_joint_top1={val_metrics['joint_top1']:.3f}"
        )

        epoch_ckpt = CHECKPOINTS_DIR / f"bc_epoch_{epoch:02d}.pt"
        save_model(model, epoch_ckpt, extra={"epoch": epoch, **row})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale_epochs = 0
            save_model(
                model,
                args.out,
                extra={"best_val_loss": val_loss, "epoch": epoch, **row},
            )
        else:
            stale_epochs += 1

        _plot_curves(history, BC_TRAINING_LOG_DIR / "latest.png")

        if stale_epochs >= args.patience:
            print(f"Early stopping: val loss plateaued for {args.patience} epochs.")
            break

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary = {
        "dataset": str(args.dataset),
        "samples": n,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "batch_size": args.batch_size,
        "device": args.device,
        "masked_loss": False,
        "best_val_loss": best_val_loss,
        "epochs_run": len(history),
        "history": history,
        "model": str(args.out),
        "timestamp": stamp,
    }
    summary_path = BC_TRAINING_LOG_DIR / f"train_summary_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (BC_TRAINING_LOG_DIR / "train_summary_latest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Saved summary to {summary_path}")
    print(f"Best model: {args.out} (val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    main()
