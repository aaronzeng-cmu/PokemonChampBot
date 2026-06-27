"""Offline dataset examples for a trained Singles MaskablePPO policy."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from config.settings import SINGLES_BC_DATASET_PATH, SINGLES_RAW_LOGS_DIR
from src.core.data.state_tokenizer import STACKED_N_TOKENS, human_readable_state
from src.doubles.evaluation.bc_examples import ActionChoice, _format_topk_block, format_state_brief
from src.singles.evaluation.bc_examples import _topk_choices, _val_indices
from src.singles.log_action_codec import ACTION_UNKNOWN, format_singles_log_action
from src.singles.log_action_mask import singles_mask_for_eval
from src.singles.replay_parser import find_sample_view_state


@dataclass
class RLPolicyExample:
    index: int
    replay_id: str
    turn: int
    side: str
    sample_kind: str
    tensor_shape: tuple[int, int]
    state_text: str
    ground_truth: str
    prediction: str
    top3: list[ActionChoice]
    correct: bool
    top3_hit: bool
    pred_action: int
    true_action: int

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
            "top3": [c.to_dict() for c in self.top3],
            "correct": self.correct,
            "top3_hit": self.top3_hit,
            "pred_action": self.pred_action,
            "true_action": self.true_action,
        }

    def to_text_block(self) -> str:
        mark = "OK" if self.correct else "MISS"
        t3 = "top3" if self.top3_hit else "miss"
        return (
            f"--- Example {self.index} | {self.replay_id} turn {self.turn} ({self.side}) "
            f"[{self.sample_kind}] shape={self.tensor_shape} ---\n"
            f"{self.state_text}\n"
            f"Ground truth: {self.ground_truth}\n"
            f"Prediction:   {self.prediction}  [{mark}, {t3}]\n"
            f"{_format_topk_block('Top-3 (* = ground truth):', self.top3, true_idx=self.true_action)}\n"
        )


def _policy_logits(model: MaskablePPO, obs: np.ndarray) -> np.ndarray:
    policy = model.policy
    obs_t = torch.as_tensor(
        obs, dtype=torch.float32, device=policy.device
    ).unsqueeze(0)
    with torch.no_grad():
        features = policy.extract_features(obs_t)
        latent_pi, _ = policy.mlp_extractor(features)
        logits = policy.features_extractor.cloner.head_singles(latent_pi)[0]
    return logits.detach().cpu()


def generate_rl_policy_examples(
    *,
    rl_checkpoint: Path,
    dataset_path: Path = SINGLES_BC_DATASET_PATH,
    log_dir: Path = SINGLES_RAW_LOGS_DIR,
    n_examples: int = 50,
    val_frac: float = 0.1,
    seed: int = 42,
    device: str = "cpu",
    mix: str = "random",
    top_k: int = 3,
    deterministic: bool = True,
) -> list[RLPolicyExample]:
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    tokens = torch.as_tensor(data["token_ids"], dtype=torch.long)
    y = torch.as_tensor(data["action"], dtype=torch.long)
    meta: list[dict] = data["meta"]

    val_idx = _val_indices(len(meta), val_frac, seed)
    rng = random.Random(seed)
    rng.shuffle(val_idx)

    model = MaskablePPO.load(str(rl_checkpoint), device=device)
    examples: list[RLPolicyExample] = []

    with torch.no_grad():
        for ds_idx in val_idx:
            if len(examples) >= n_examples:
                break
            true_action = int(y[ds_idx])
            obs = tokens[ds_idx].numpy().astype(np.float32)
            logits_row = _policy_logits(model, obs)

            m = meta[ds_idx]
            shape = tuple(int(x) for x in tokens[ds_idx].shape)
            if shape != (STACKED_N_TOKENS, tokens.shape[-1]):
                raise ValueError(f"expected stacked tokens {STACKED_N_TOKENS}, got {shape}")

            sample_kind = str(m.get("sample_kind", "turn"))
            view = find_sample_view_state(
                log_dir,
                replay_id=m["replay_id"],
                turn=m["turn"],
                side=m["side"],
                sample_kind=sample_kind,
            )

            legal_mask = singles_mask_for_eval(
                view,
                side=m["side"],
                sample_kind=sample_kind,
            )
            if legal_mask is not None and legal_mask.any():
                action, _ = model.predict(
                    obs,
                    deterministic=deterministic,
                    action_masks=legal_mask,
                )
                pred_action = int(action)
            else:
                pred_action = int(logits_row.argmax().item())

            if view is not None:
                state_text = format_state_brief(human_readable_state(view, m["side"]))
                gt = format_singles_log_action(view, m["side"], true_action)
                pred = format_singles_log_action(view, m["side"], pred_action)
            else:
                state_text = (
                    f"(log not found for {m['replay_id']}; indices only)\n"
                    f"Turn {m['turn']} | perspective {m['side']}"
                )
                gt = f"action={true_action}"
                pred = f"action={pred_action}"

            correct = true_action == ACTION_UNKNOWN or pred_action == true_action
            if mix == "correct" and not correct:
                continue
            if mix == "incorrect" and correct:
                continue

            top3 = _topk_choices(
                logits_row,
                view=view,
                side=m["side"],
                k=top_k,
                legal_mask=legal_mask,
            )

            examples.append(
                RLPolicyExample(
                    index=len(examples) + 1,
                    replay_id=m["replay_id"],
                    turn=m["turn"],
                    side=m["side"],
                    sample_kind=sample_kind,
                    tensor_shape=shape,
                    state_text=state_text,
                    ground_truth=gt,
                    prediction=pred,
                    top3=top3,
                    correct=pred_action == true_action,
                    top3_hit=any(c.index == true_action for c in top3),
                    pred_action=pred_action,
                    true_action=true_action,
                )
            )

    return examples


def write_rl_policy_examples_report(
    examples: list[RLPolicyExample],
    out_dir: Path,
    *,
    rl_checkpoint: Path,
    dataset_path: Path,
    mix: str,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    n = len(examples)
    correct = sum(1 for e in examples if e.correct)
    top3 = sum(1 for e in examples if e.top3_hit)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "format": "singles",
        "agent": "maskable_ppo",
        "rl_checkpoint": str(rl_checkpoint),
        "dataset": str(dataset_path),
        "n_examples": n,
        "mix": mix,
        "top1": correct / n if n else 0.0,
        "top3": top3 / n if n else 0.0,
        "examples": [e.to_dict() for e in examples],
    }

    json_path = out_dir / f"policy_examples_{stamp}.json"
    txt_path = out_dir / f"policy_examples_{stamp}.txt"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    txt_path.write_text("\n".join(e.to_text_block() for e in examples), encoding="utf-8")
    latest_json = out_dir / "policy_examples_latest.json"
    latest_txt = out_dir / "policy_examples_latest.txt"
    latest_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    latest_txt.write_text("\n".join(e.to_text_block() for e in examples), encoding="utf-8")
    return txt_path, json_path
