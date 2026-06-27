"""Generate behavior-cloning prediction vs ground-truth examples."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from config.settings import BC_DATASET_PATH, BC_MODEL_PATH, RAW_LOGS_DIR
from src.core.training.mmap_dataset import load_doubles_bc_data, open_doubles_mmap_store, DoublesMmapStore
from src.doubles.data.action_codec import decode_log_slot_action, format_log_action_pair
from src.doubles.data.action_space_spec import ACTION_UNKNOWN
from src.doubles.data.log_action_mask import pick_masked_log_actions, slot_mask_for_eval
from src.core.data.log_tracker import BattleLogState
from src.doubles.data.replay_parser import find_sample_view_state
from src.core.data.state_tokenizer import STACKED_N_TOKENS, human_readable_state
from src.core.model.transformer_bot import load_model


@dataclass
class ActionChoice:
    index: int
    probability: float
    label: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "probability": self.probability,
            "label": self.label,
        }


def _decode_action_label(
    view: BattleLogState | None,
    side: str,
    slot_suffix: str,
    action_idx: int,
) -> str:
    if view is not None:
        return decode_log_slot_action(view, side, slot_suffix, action_idx)
    return f"action {action_idx}"


def _topk_choices(
    logits_row: torch.Tensor,
    *,
    view: BattleLogState | None,
    side: str,
    slot_suffix: str,
    k: int = 3,
    legal_mask: object | None = None,
) -> list[ActionChoice]:
    row = logits_row.clone()
    if legal_mask is not None:
        mask = torch.as_tensor(legal_mask, dtype=torch.bool, device=row.device)
        row[~mask] = -float("inf")
    probs = torch.softmax(row, dim=-1)
    k = min(k, probs.numel())
    values, indices = torch.topk(probs, k=k)
    return [
        ActionChoice(
            index=int(indices[i].item()),
            probability=float(values[i].item()),
            label=_decode_action_label(view, side, slot_suffix, int(indices[i].item())),
        )
        for i in range(k)
    ]


def _format_topk_block(title: str, choices: list[ActionChoice], *, true_idx: int) -> str:
    lines = [title]
    if true_idx == ACTION_UNKNOWN:
        lines.append("  (ground truth UNKNOWN — erased selection, not scored)")
        return "\n".join(lines)
    for rank, choice in enumerate(choices, start=1):
        mark = " *" if choice.index == true_idx else ""
        lines.append(
            f"  {rank}. {100 * choice.probability:5.1f}% | {choice.label}{mark}"
        )
    if not any(c.index == true_idx for c in choices):
        lines.append(f"  (ground truth action {true_idx} not in top-{len(choices)})")
    return "\n".join(lines)


def format_state_brief(state_dict: dict) -> str:
    """Compact multi-line battle snapshot for text reports."""
    lines = [
        f"Turn {state_dict['turn']} | perspective {state_dict['perspective']}",
    ]
    field = state_dict.get("field") or {}
    field_bits = [
        field.get("weather"),
        field.get("terrain"),
        "Trick Room" if field.get("trick_room") else None,
        "Tailwind" if field.get("tailwind") else None,
    ]
    field_bits = [b for b in field_bits if b]
    if field_bits:
        lines.append(f"Field: {', '.join(str(b) for b in field_bits)}")

    for label, key in (("Our active", "our_actives"), ("Opp active", "opp_actives")):
        parts: list[str] = []
        for mon in state_dict.get(key) or []:
            if not mon.get("present"):
                continue
            moves = ", ".join(mon.get("moves") or []) or "?"
            hp = mon.get("hp") or "?"
            parts.append(f"{mon.get('species', '?')} ({hp}) [{moves}]")
        if parts:
            lines.append(f"{label}: {' | '.join(parts)}")
    return "\n".join(lines)


@dataclass
class BCExample:
    index: int
    replay_id: str
    turn: int
    side: str
    sample_kind: str
    tensor_shape: tuple[int, int]
    state_text: str
    ground_truth: str
    prediction: str
    top3_slot0: list[ActionChoice]
    top3_slot1: list[ActionChoice]
    correct_slot0: bool
    correct_slot1: bool
    correct_joint: bool
    top3_slot0_hit: bool
    top3_slot1_hit: bool
    pred_slot0: int
    pred_slot1: int
    true_slot0: int
    true_slot1: int
    raw_slot0: int
    raw_slot1: int
    raw_slot0_legal: bool
    raw_slot1_legal: bool
    raw_slot0_label: str
    raw_slot1_label: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "replay_id": self.replay_id,
            "turn": self.turn,
            "side": self.side,
            "sample_kind": self.sample_kind,
            "tensor_shape": list(self.tensor_shape),
            "state_text": self.state_text,
            "ground_truth": self.ground_truth,
            "prediction": self.prediction,
            "top3_slot0": [c.to_dict() for c in self.top3_slot0],
            "top3_slot1": [c.to_dict() for c in self.top3_slot1],
            "correct_slot0": self.correct_slot0,
            "correct_slot1": self.correct_slot1,
            "correct_joint": self.correct_joint,
            "top3_slot0_hit": self.top3_slot0_hit,
            "top3_slot1_hit": self.top3_slot1_hit,
            "pred_slot0": self.pred_slot0,
            "pred_slot1": self.pred_slot1,
            "true_slot0": self.true_slot0,
            "true_slot1": self.true_slot1,
            "raw_slot0": self.raw_slot0,
            "raw_slot1": self.raw_slot1,
            "raw_slot0_legal": bool(self.raw_slot0_legal),
            "raw_slot1_legal": bool(self.raw_slot1_legal),
            "raw_slot0_label": self.raw_slot0_label,
            "raw_slot1_label": self.raw_slot1_label,
        }

    def to_text_block(self) -> str:
        mark0 = "OK" if self.correct_slot0 else "MISS"
        mark1 = "OK" if self.correct_slot1 else "MISS"
        t3_0 = "top3" if self.top3_slot0_hit else "miss"
        t3_1 = "top3" if self.top3_slot1_hit else "miss"
        joint = "JOINT OK" if self.correct_joint else "JOINT MISS"
        return (
            f"--- Example {self.index} | {self.replay_id} turn {self.turn} ({self.side}) "
            f"[{self.sample_kind}] shape={self.tensor_shape} ---\n"
            f"{self.state_text}\n"
            f"Ground truth: {self.ground_truth}\n"
            f"Raw top-1:    [{self.raw_slot0}] {self.raw_slot0_label} "
            f"(legal={self.raw_slot0_legal}) | "
            f"[{self.raw_slot1}] {self.raw_slot1_label} "
            f"(legal={self.raw_slot1_legal})\n"
            f"Prediction:   {self.prediction}  [{mark0}/{mark1}, {joint}]\n"
            f"{_format_topk_block('Top-3 slot0 (* = ground truth):', self.top3_slot0, true_idx=self.true_slot0)}\n"
            f"{_format_topk_block('Top-3 slot1 (* = ground truth):', self.top3_slot1, true_idx=self.true_slot1)}\n"
        )


def _val_indices(n: int, val_frac: float, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    split = int(n * (1.0 - val_frac))
    return perm[split:]


def generate_bc_examples(
    *,
    model_path: Path = BC_MODEL_PATH,
    dataset_path: Path = BC_DATASET_PATH,
    log_dir: Path = RAW_LOGS_DIR,
    n_examples: int = 50,
    val_frac: float = 0.1,
    seed: int = 42,
    device: str = "cpu",
    mix: str = "random",
    top_k: int = 3,
) -> list[BCExample]:
    """
    Sample validation decisions and compare model argmax vs logged human actions.

    mix: 'random' | 'correct' | 'incorrect' — filter examples after sampling.
    """
    mmap_dir = dataset_path.with_name(f"{dataset_path.stem}_mmap")
    store: DoublesMmapStore | None = None
    tokens = y0 = y1 = None
    meta_all: list[dict] | None = None

    if mmap_dir.joinpath("manifest.json").is_file():
        candidate = open_doubles_mmap_store(dataset_path)
        if candidate.has_meta():
            store = candidate
            n = len(store)
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
            meta_all = data["meta"]
            n = len(meta_all)
        else:
            store = candidate
            n = len(store)
    else:
        tokens, y0, y1, meta_all = load_doubles_bc_data(dataset_path)
        n = int(tokens.shape[0])

    val_idx = _val_indices(n, val_frac, seed)
    rng = random.Random(seed)
    rng.shuffle(val_idx)

    model = load_model(model_path, device=device)
    examples: list[BCExample] = []
    want = n_examples

    with torch.no_grad():
        for ds_idx in val_idx:
            if len(examples) >= want:
                break
            if store is not None:
                row = store.get(ds_idx)
                x = row["tokens"].unsqueeze(0).to(device)
                true0 = int(row["action_slot0"])
                true1 = int(row["action_slot1"])
                m = row["meta"]
                shape = tuple(int(x) for x in row["tokens"].shape)
                if not m.get("replay_id"):
                    continue
            else:
                x = tokens[ds_idx : ds_idx + 1].to(device)  # type: ignore[index]
                true0 = int(y0[ds_idx])  # type: ignore[index]
                true1 = int(y1[ds_idx])  # type: ignore[index]
                m = meta_all[ds_idx]  # type: ignore[index]
                shape = tuple(int(x) for x in tokens[ds_idx].shape)  # type: ignore[index]
            logits0, logits1 = model(x)
            row0 = logits0[0]
            row1 = logits1[0]

            if shape != (STACKED_N_TOKENS, x.shape[-1]):
                raise ValueError(f"expected stacked tokens {STACKED_N_TOKENS}, got {shape}")
            sample_kind = str(m.get("sample_kind", "turn"))
            view = find_sample_view_state(
                log_dir,
                replay_id=m["replay_id"],
                turn=m["turn"],
                side=m["side"],
                sample_kind=sample_kind,
            )
            raw0 = int(row0.argmax().item())
            raw1 = int(row1.argmax().item())
            mask0 = slot_mask_for_eval(
                view, side=m["side"], sample_kind=sample_kind, slot_suffix="a"
            )
            if view is not None:
                pred0, pred1 = pick_masked_log_actions(
                    row0, row1, view=view, side=m["side"], sample_kind=sample_kind
                )
            else:
                pred0 = raw0
                pred1 = raw1
            mask1 = slot_mask_for_eval(
                view,
                side=m["side"],
                sample_kind=sample_kind,
                slot_suffix="b",
                slot0_pred=pred0,
            )
            raw0_legal = mask0 is not None and mask0[raw0]
            raw1_legal = mask1 is not None and mask1[raw1]
            raw0_label = _decode_action_label(view, m["side"], "a", raw0)
            raw1_label = _decode_action_label(view, m["side"], "b", raw1)
            if view is not None:
                state_text = format_state_brief(human_readable_state(view, m["side"]))
                gt = format_log_action_pair(view, m["side"], true0, true1)
                pred = format_log_action_pair(view, m["side"], pred0, pred1)
            else:
                state_text = (
                    f"(log not found for {m['replay_id']}; indices only)\n"
                    f"Turn {m['turn']} | perspective {m['side']}"
                )
                gt = f"slot0={true0}, slot1={true1}"
                pred = f"slot0={pred0}, slot1={pred1}"

            slot0_ok = true0 == ACTION_UNKNOWN or pred0 == true0
            slot1_ok = true1 == ACTION_UNKNOWN or pred1 == true1
            joint_ok = slot0_ok and slot1_ok
            if mix == "correct" and not joint_ok:
                continue
            if mix == "incorrect" and joint_ok:
                continue

            top3_0 = _topk_choices(
                row0,
                view=view,
                side=m["side"],
                slot_suffix="a",
                k=top_k,
                legal_mask=slot_mask_for_eval(
                    view, side=m["side"], sample_kind=sample_kind, slot_suffix="a"
                ),
            )
            top3_1 = _topk_choices(
                row1,
                view=view,
                side=m["side"],
                slot_suffix="b",
                k=top_k,
                legal_mask=slot_mask_for_eval(
                    view,
                    side=m["side"],
                    sample_kind=sample_kind,
                    slot_suffix="b",
                    slot0_pred=pred0,
                ),
            )

            ex = BCExample(
                index=len(examples) + 1,
                replay_id=m["replay_id"],
                turn=m["turn"],
                side=m["side"],
                sample_kind=sample_kind,
                tensor_shape=shape,
                state_text=state_text,
                ground_truth=gt,
                prediction=pred,
                top3_slot0=top3_0,
                top3_slot1=top3_1,
                correct_slot0=pred0 == true0,
                correct_slot1=pred1 == true1,
                correct_joint=joint_ok,
                top3_slot0_hit=any(c.index == true0 for c in top3_0),
                top3_slot1_hit=any(c.index == true1 for c in top3_1),
                pred_slot0=pred0,
                pred_slot1=pred1,
                true_slot0=true0,
                true_slot1=true1,
                raw_slot0=raw0,
                raw_slot1=raw1,
                raw_slot0_legal=raw0_legal,
                raw_slot1_legal=raw1_legal,
                raw_slot0_label=raw0_label,
                raw_slot1_label=raw1_label,
            )
            examples.append(ex)

    return examples


def write_bc_examples_report(
    examples: list[BCExample],
    out_dir: Path,
    *,
    model_path: Path,
    dataset_path: Path,
    mix: str,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    n = len(examples)
    correct_joint = sum(1 for e in examples if e.correct_joint)
    correct_s0 = sum(1 for e in examples if e.correct_slot0)
    correct_s1 = sum(1 for e in examples if e.correct_slot1)
    top3_s0 = sum(1 for e in examples if e.top3_slot0_hit)
    top3_s1 = sum(1 for e in examples if e.top3_slot1_hit)
    raw_legal_slots = sum(
        int(e.raw_slot0_legal) + int(e.raw_slot1_legal) for e in examples
    )

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "dataset": str(dataset_path),
        "n_examples": n,
        "mix": mix,
        "joint_top1": correct_joint / n if n else 0.0,
        "slot0_top1": correct_s0 / n if n else 0.0,
        "slot1_top1": correct_s1 / n if n else 0.0,
        "slot0_top3": top3_s0 / n if n else 0.0,
        "slot1_top3": top3_s1 / n if n else 0.0,
        "top3_avg": (top3_s0 + top3_s1) / (2 * n) if n else 0.0,
        "raw_top1_legal_rate": raw_legal_slots / (2 * n) if n else 0.0,
        "examples": [e.to_dict() for e in examples],
    }

    json_path = out_dir / f"bc_examples_{stamp}.json"
    txt_path = out_dir / f"bc_examples_{stamp}.txt"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    txt_path.write_text(
        "\n".join(e.to_text_block() for e in examples),
        encoding="utf-8",
    )
    return txt_path, json_path


def format_bc_examples_text(examples: list[BCExample]) -> str:
    """Return all examples as one printable string."""
    return "\n".join(e.to_text_block() for e in examples)
