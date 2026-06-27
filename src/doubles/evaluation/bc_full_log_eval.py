"""Full validation-set BC eval with log-reconstructed legal masking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from config.settings import BC_DATASET_PATH, BC_MODEL_PATH, RAW_LOGS_DIR
from src.core.training.mmap_dataset import (
    has_mmap_dataset,
    mmap_dataset_dir,
    open_doubles_mmap_store,
    load_doubles_bc_data,
)
from src.doubles.data.action_space_spec import ACTION_UNKNOWN
from src.doubles.data.log_action_mask import pick_masked_log_actions
from src.doubles.data.replay_parser import find_sample_view_state
from src.core.model.transformer_bot import load_model


def _val_indices(n: int, val_frac: float, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    split = int(n * (1.0 - val_frac))
    return perm[split:]


@dataclass
class FullLogEvalMetrics:
    n_val: int
    n_with_log: int
    n_missing_log: int
    slot_top1: float
    joint_top1: float
    masking_overrides: int

    def to_dict(self) -> dict:
        return {
            "n_val": self.n_val,
            "n_with_log": self.n_with_log,
            "n_missing_log": self.n_missing_log,
            "slot_top1": self.slot_top1,
            "joint_top1": self.joint_top1,
            "masking_overrides": self.masking_overrides,
        }


def evaluate_bc_full_log(
    *,
    model_path: Path = BC_MODEL_PATH,
    dataset_path: Path = BC_DATASET_PATH,
    log_dir: Path = RAW_LOGS_DIR,
    val_frac: float = 0.1,
    seed: int = 42,
    device: str = "cpu",
    batch_size: int = 256,
) -> FullLogEvalMetrics:
    mmap_dir = mmap_dataset_dir(dataset_path)
    if has_mmap_dataset(mmap_dir):
        store = open_doubles_mmap_store(dataset_path)
        n = len(store)
        if store.has_meta():
            val_idx = _val_indices(n, val_frac, seed)
            val_tokens, val_y0, val_y1, meta = store.get_batch_tensors(val_idx)
        elif dataset_path.is_file():
            try:
                data = torch.load(
                    dataset_path, map_location="cpu", weights_only=False, mmap=True
                )
            except TypeError:
                data = torch.load(dataset_path, map_location="cpu", weights_only=False)
            tokens = torch.as_tensor(data["token_ids"], dtype=torch.long)
            y0 = torch.as_tensor(data["action_slot0"], dtype=torch.long)
            y1 = torch.as_tensor(data["action_slot1"], dtype=torch.long)
            meta = data["meta"]
            val_idx = _val_indices(len(meta), val_frac, seed)
            val_tokens = tokens[val_idx]
            val_y0 = y0[val_idx]
            val_y1 = y1[val_idx]
        else:
            raise RuntimeError("mmap meta missing; run scripts/backfill_mmap_meta.py")
    else:
        tokens, y0, y1, meta = load_doubles_bc_data(dataset_path)
        val_idx = _val_indices(len(meta), val_frac, seed)
        val_tokens = tokens[val_idx]
        val_y0 = y0[val_idx]
        val_y1 = y1[val_idx]

    model = load_model(model_path, device=device)

    loader = DataLoader(
        TensorDataset(val_tokens, val_y0, val_y1),
        batch_size=batch_size,
        shuffle=False,
    )

    slot_hits = 0.0
    joint_hits = 0.0
    masking_overrides = 0
    n_with_log = 0
    n_missing_log = 0
    offset = 0

    view_cache: dict[tuple[str, int, str, str], object | None] = {}

    with torch.no_grad():
        for batch in loader:
            xb, by0, by1 = [t.to(device) for t in batch]
            logits0, logits1 = model(xb)

            bsz = xb.size(0)
            for bi in range(bsz):
                ds_idx = val_idx[offset + bi]
                m = meta[ds_idx]
                sample_kind = str(m.get("sample_kind", "turn"))
                cache_key = (m["replay_id"], m["turn"], m["side"], sample_kind)
                if cache_key not in view_cache:
                    view_cache[cache_key] = find_sample_view_state(
                        log_dir,
                        replay_id=m["replay_id"],
                        turn=m["turn"],
                        side=m["side"],
                        sample_kind=sample_kind,
                    )
                view = view_cache[cache_key]
                true0 = int(by0[bi].item())
                true1 = int(by1[bi].item())

                if view is None:
                    n_missing_log += 1
                    pred0 = int(logits0[bi].argmax().item())
                    pred1 = int(logits1[bi].argmax().item())
                else:
                    n_with_log += 1
                    pred0, pred1 = pick_masked_log_actions(
                        logits0[bi],
                        logits1[bi],
                        view=view,
                        side=m["side"],
                        sample_kind=sample_kind,
                    )
                    raw0 = int(logits0[bi].argmax().item())
                    raw1 = int(logits1[bi].argmax().item())
                    if pred0 != raw0 or pred1 != raw1:
                        masking_overrides += 1

                s0_ok = true0 == ACTION_UNKNOWN or pred0 == true0
                s1_ok = true1 == ACTION_UNKNOWN or pred1 == true1
                if s0_ok:
                    slot_hits += 1
                if s1_ok:
                    slot_hits += 1
                if s0_ok and s1_ok:
                    joint_hits += 1

            offset += bsz

    n_val = len(val_idx)
    valid_slots = max(1, n_val * 2)
    return FullLogEvalMetrics(
        n_val=n_val,
        n_with_log=n_with_log,
        n_missing_log=n_missing_log,
        slot_top1=slot_hits / valid_slots,
        joint_top1=joint_hits / max(1, n_val),
        masking_overrides=masking_overrides,
    )


def write_full_log_eval_report(
    metrics: FullLogEvalMetrics,
    out_dir: Path,
    *,
    model_path: Path,
    dataset_path: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "dataset": str(dataset_path),
        **metrics.to_dict(),
    }
    out_path = out_dir / f"bc_full_log_eval_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest = out_dir / "bc_full_log_eval_latest.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
