#!/usr/bin/env python3
"""Rebuild singles BC dataset action masks with full log-legal actions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm import tqdm

from config.settings import SINGLES_BC_DATASET_PATH, SINGLES_RAW_LOGS_DIR
from src.singles.log_action_mask import training_singles_mask
from src.singles.replay_parser import parse_singles_log_file


def _build_view_lookup(log_dir: Path) -> dict[tuple[str, int, str, str], object]:
    lookup: dict[tuple[str, int, str, str], object] = {}
    paths = sorted(log_dir.glob("*.log"))
    for path in tqdm(paths, desc="Indexing replay views"):
        for sample in parse_singles_log_file(path, skip_rating=True, keep_view_state=True):
            if sample.view_state is None:
                continue
            key = (path.stem, sample.turn, sample.side, sample.sample_kind)
            lookup[key] = sample.view_state
    return lookup


def patch_singles_masks(
    *,
    dataset_path: Path,
    log_dir: Path,
) -> dict:
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    meta: list[dict] = data["meta"]
    actions = np.asarray(data["action"], dtype=np.int64)
    lookup = _build_view_lookup(log_dir)

    masks: list[np.ndarray] = []
    missing = 0
    legal_counts: list[int] = []
    for i, m in enumerate(tqdm(meta, desc="Rebuilding masks")):
        key = (
            m["replay_id"],
            int(m["turn"]),
            m["side"],
            str(m.get("sample_kind", "turn")),
        )
        view = lookup.get(key)
        gt = int(actions[i])
        if view is None:
            missing += 1
            from src.core.model.transformer_bot import SINGLES_ACTION_SIZE

            mask = np.zeros(SINGLES_ACTION_SIZE, dtype=bool)
            if 0 <= gt < SINGLES_ACTION_SIZE:
                mask[gt] = True
            else:
                mask[0] = True
        else:
            mask = training_singles_mask(
                view,
                m["side"],
                str(m.get("sample_kind", "turn")),
                ground_truth=gt,
            )
        masks.append(mask)
        legal_counts.append(int(mask.sum()))

    data["action_mask"] = np.stack(masks)
    torch.save(data, dataset_path)

    avg_legal = float(np.mean(legal_counts)) if legal_counts else 0.0
    return {
        "dataset": str(dataset_path),
        "samples": len(meta),
        "missing_views": missing,
        "avg_legal_actions": avg_legal,
        "min_legal": int(min(legal_counts)) if legal_counts else 0,
        "max_legal": int(max(legal_counts)) if legal_counts else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch singles BC dataset action masks")
    parser.add_argument("--dataset", type=Path, default=SINGLES_BC_DATASET_PATH)
    parser.add_argument("--log-dir", type=Path, default=SINGLES_RAW_LOGS_DIR)
    args = parser.parse_args()

    summary = patch_singles_masks(dataset_path=args.dataset, log_dir=args.log_dir)
    print(f"Patched {summary['samples']} samples in {summary['dataset']}")
    print(f"avg legal actions: {summary['avg_legal_actions']:.2f}")
    print(f"range: {summary['min_legal']}-{summary['max_legal']}")
    if summary["missing_views"]:
        print(f"warning: {summary['missing_views']} samples had no matching view")


if __name__ == "__main__":
    main()
