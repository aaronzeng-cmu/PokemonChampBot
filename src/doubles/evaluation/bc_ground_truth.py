"""Ground-truth-only BC dataset audit (parser sanity, no model)."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from config.settings import BC_DATASET_PATH, BC_EVAL_LOG_DIR, RAW_LOGS_DIR
from src.core.training.mmap_dataset import (
    has_mmap_dataset,
    mmap_dataset_dir,
    open_doubles_mmap_store,
    load_doubles_bc_data,
)
from src.doubles.data.action_codec import decode_log_slot_action, format_log_action_pair
from src.doubles.data.action_space_spec import ACTION_UNKNOWN
from src.doubles.data.replay_parser import find_sample_view_state
from src.core.data.state_tokenizer import human_readable_state
from src.doubles.evaluation.bc_examples import format_state_brief

_IMPOSSIBLE_PATTERNS = [
    (re.compile(r"protect\s*->\s*ally", re.I), "Protect must not target ally"),
    (re.compile(r"trickroom\s*->\s*ally", re.I), "Trick Room must not target ally"),
    (re.compile(r"tailwind\s*->\s*ally", re.I), "Tailwind must not target ally"),
    (re.compile(r"detect\s*->\s*ally", re.I), "Detect must not target ally"),
    (re.compile(r"swordsdance\s*->\s*ally", re.I), "Swords Dance must not target ally"),
    (re.compile(r"knockoff\s*->\s*ally", re.I), "Knock Off must not target ally"),
    (re.compile(r"knockoff\s*->\s*self", re.I), "Knock Off must not target self"),
]


@dataclass
class GroundTruthExample:
    index: int
    dataset_index: int
    replay_id: str
    turn: int
    side: str
    state_text: str
    ground_truth: str
    slot0_text: str
    slot1_text: str
    true_slot0: int
    true_slot1: int
    warnings: list[str]
    log_found: bool

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "dataset_index": self.dataset_index,
            "replay_id": self.replay_id,
            "turn": self.turn,
            "side": self.side,
            "state_text": self.state_text,
            "ground_truth": self.ground_truth,
            "slot0_text": self.slot0_text,
            "slot1_text": self.slot1_text,
            "true_slot0": self.true_slot0,
            "true_slot1": self.true_slot1,
            "warnings": self.warnings,
            "log_found": self.log_found,
        }

    def to_text_block(self) -> str:
        warn = f"\nWarnings: {', '.join(self.warnings)}" if self.warnings else ""
        return (
            f"--- Example {self.index} | {self.replay_id} turn {self.turn} ({self.side}) ---\n"
            f"{self.state_text}\n"
            f"Ground truth: {self.ground_truth}\n"
            f"  slot0 [{self.true_slot0}]: {self.slot0_text}\n"
            f"  slot1 [{self.true_slot1}]: {self.slot1_text}{warn}\n"
        )


def _ground_truth_warnings(text: str) -> list[str]:
    warnings: list[str] = []
    for pattern, message in _IMPOSSIBLE_PATTERNS:
        if pattern.search(text):
            warnings.append(message)
    return warnings


def _matches_filter(ground_truth: str, flt: str) -> bool:
    gt = ground_truth.lower()
    if flt == "random":
        return True
    if flt == "protect":
        return "protect" in gt
    if flt == "trickroom":
        return "trickroom" in gt or "trick room" in gt
    if flt == "knockoff":
        return "knockoff" in gt or "knock off" in gt
    if flt == "spread":
        return "-> default" in gt
    if flt == "offensive":
        return "opp slot" in gt
    if flt == "diverse":
        return any(
            k in gt
            for k in (
                "protect",
                "trickroom",
                "trick room",
                "knockoff",
                "knock off",
                "-> default",
                "opp slot",
                "switch ->",
            )
        )
    if flt == "unknown":
        return "unknown (erased selection)" in gt
    return True


def generate_ground_truth_examples(
    *,
    dataset_path: Path = BC_DATASET_PATH,
    log_dir: Path = RAW_LOGS_DIR,
    n_examples: int = 50,
    seed: int = 42,
    sample_filter: str = "diverse",
) -> list[GroundTruthExample]:
    """Sample parsed dataset rows and render human-readable ground truth only."""
    mmap_dir = mmap_dataset_dir(dataset_path)
    if has_mmap_dataset(mmap_dir):
        store = open_doubles_mmap_store(dataset_path)
        n = len(store)
    else:
        store = None
        _tokens, y0_t, y1_t, meta_all = load_doubles_bc_data(dataset_path)
        y0 = y0_t.numpy()
        y1 = y1_t.numpy()
        meta_all_list = meta_all
        n = len(meta_all_list)

    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    examples: list[GroundTruthExample] = []
    for ds_idx in indices:
        if len(examples) >= n_examples:
            break

        if store is not None:
            row = store.get(ds_idx)
            m = row["meta"]
            true0 = int(row["action_slot0"])
            true1 = int(row["action_slot1"])
        else:
            m = meta_all_list[ds_idx]
            true0 = int(y0[ds_idx])
            true1 = int(y1[ds_idx])
        view = find_sample_view_state(
            log_dir,
            replay_id=m["replay_id"],
            turn=m["turn"],
            side=m["side"],
        )

        if view is not None:
            state_text = format_state_brief(human_readable_state(view, m["side"]))
            gt = format_log_action_pair(view, m["side"], true0, true1)
            s0 = decode_log_slot_action(view, m["side"], "a", true0)
            s1 = decode_log_slot_action(view, m["side"], "b", true1)
            log_found = True
        else:
            state_text = (
                f"(log not found for {m['replay_id']})\n"
                f"Turn {m['turn']} | perspective {m['side']}"
            )
            gt = f"slot0={true0}, slot1={true1}"
            s0 = f"action {true0}"
            s1 = f"action {true1}"
            log_found = False

        if not _matches_filter(gt, sample_filter):
            continue

        warnings = _ground_truth_warnings(gt)
        examples.append(
            GroundTruthExample(
                index=len(examples) + 1,
                dataset_index=ds_idx,
                replay_id=m["replay_id"],
                turn=m["turn"],
                side=m["side"],
                state_text=state_text,
                ground_truth=gt,
                slot0_text=s0,
                slot1_text=s1,
                true_slot0=true0,
                true_slot1=true1,
                warnings=warnings,
                log_found=log_found,
            )
        )

    return examples


def write_ground_truth_report(
    examples: list[GroundTruthExample],
    out_dir: Path,
    *,
    dataset_path: Path,
    sample_filter: str,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    n = len(examples)
    warned = sum(1 for e in examples if e.warnings)
    missing_log = sum(1 for e in examples if not e.log_found)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "n_examples": n,
        "filter": sample_filter,
        "examples_with_warnings": warned,
        "examples_missing_log": missing_log,
        "examples": [e.to_dict() for e in examples],
    }

    json_path = out_dir / f"bc_ground_truth_{stamp}.json"
    txt_path = out_dir / f"bc_ground_truth_{stamp}.txt"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    txt_path.write_text(
        "\n".join(e.to_text_block() for e in examples),
        encoding="utf-8",
    )
    return txt_path, json_path


def format_ground_truth_text(examples: list[GroundTruthExample]) -> str:
    return "\n".join(e.to_text_block() for e in examples)
