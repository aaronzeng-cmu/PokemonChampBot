#!/usr/bin/env python3
"""Verify BC checkpoint weights match VGCBehaviorMaskablePolicy cloner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from gymnasium.spaces import Box, MultiDiscrete
from stable_baselines3.common.utils import ConstantSchedule

from config.settings import BC_MODEL_PATH
from src.doubles.data.action_space_spec import ACTION_SIZE
from src.core.data.state_tokenizer import N_FIELDS, STACKED_N_TOKENS
from src.core.model.transformer_bot import load_model
from src.doubles.rl.custom_policy import VGCBehaviorMaskablePolicy, init_bc_actor_weights

# BC modules we expect to transplant 1:1 into policy.features_extractor.cloner
BC_KEYS = [
    "field_embeddings",
    "token_proj",
    "cls_token",
    "encoder",
    "head1",
    "head2",
]


def _policy_cloner_state(policy: VGCBehaviorMaskablePolicy) -> dict[str, torch.Tensor]:
    return policy.features_extractor.cloner.state_dict()


def compare_state_dicts(
    bc_sd: dict[str, torch.Tensor],
    policy_sd: dict[str, torch.Tensor],
) -> tuple[list[str], list[str]]:
    """Return (perfect_matches, mismatches) as human-readable lines."""
    perfect: list[str] = []
    mismatches: list[str] = []

    bc_keys = sorted(bc_sd.keys())
    pol_keys = sorted(policy_sd.keys())

    if bc_keys != pol_keys:
        only_bc = set(bc_keys) - set(pol_keys)
        only_pol = set(pol_keys) - set(bc_keys)
        if only_bc:
            mismatches.append(f"KEYS only in BC: {sorted(only_bc)[:10]}...")
        if only_pol:
            mismatches.append(f"KEYS only in policy cloner: {sorted(only_pol)[:10]}...")

    for key in bc_keys:
        if key not in policy_sd:
            mismatches.append(f"MISSING in policy: {key}")
            continue
        b = bc_sd[key]
        p = policy_sd[key]
        if b.shape != p.shape:
            mismatches.append(f"SHAPE {key}: bc={tuple(b.shape)} policy={tuple(p.shape)}")
            continue
        if torch.equal(b.cpu(), p.cpu()):
            perfect.append(key)
        else:
            max_diff = (b.float() - p.float()).abs().max().item()
            mismatches.append(f"VALUE {key}: max_abs_diff={max_diff:.6g}")

    return perfect, mismatches


def forward_logits_parity(
    bc_model,
    policy: VGCBehaviorMaskablePolicy,
    *,
    device: str,
    batch: int = 4,
) -> dict:
    """Compare BC forward vs policy actor logits on identical token input."""
    torch.manual_seed(0)
    token_ids = torch.randint(0, 512, (batch, STACKED_N_TOKENS, N_FIELDS), device=device)

    bc_model.eval()
    policy.eval()
    with torch.no_grad():
        bc_l0, bc_l1 = bc_model(token_ids)
        obs = token_ids.float()
        features = policy.extract_features(obs)
        latent_pi, _ = policy.mlp_extractor(features)
        dist = policy._get_action_dist_from_latent(latent_pi)
        assert dist._logits0 is not None and dist._logits1 is not None
        pol_l0, pol_l1 = dist._logits0, dist._logits1

    l0_eq = torch.allclose(bc_l0, pol_l0, atol=0, rtol=0)
    l1_eq = torch.allclose(bc_l1, pol_l1, atol=0, rtol=0)
    l0_max = (bc_l0 - pol_l0).abs().max().item() if not l0_eq else 0.0
    l1_max = (bc_l1 - pol_l1).abs().max().item() if not l1_eq else 0.0
    return {
        "head1_exact": l0_eq,
        "head2_exact": l1_eq,
        "head1_max_diff": l0_max,
        "head2_max_diff": l1_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-model", type=Path, default=BC_MODEL_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = args.device
    bc = load_model(args.bc_model, device=device)
    bc_sd = bc.state_dict()

    obs_space = Box(
        low=-np.inf, high=np.inf, shape=(STACKED_N_TOKENS, N_FIELDS), dtype=np.float32
    )
    act_space = MultiDiscrete([ACTION_SIZE, ACTION_SIZE])

    policy = VGCBehaviorMaskablePolicy(
        obs_space,
        act_space,
        ConstantSchedule(5e-6),
        net_arch=dict(pi=[], vf=[64, 64]),
        features_extractor_kwargs={"bc_model_path": str(args.bc_model)},
        ortho_init=False,
    ).to(device)

    # Fresh policy without extractor preload — test init_bc_actor_weights
    policy.features_extractor.cloner = type(bc)(bc.config).to(device)
    init_bc_actor_weights(policy, args.bc_model)

    pol_sd = _policy_cloner_state(policy)
    perfect, mismatches = compare_state_dicts(bc_sd, pol_sd)

    fwd = forward_logits_parity(bc, policy, device=device)

    lines = [
        "RL weight verification",
        f"BC model: {args.bc_model}",
        f"Device: {device}",
        "",
        f"State dict keys matched exactly: {len(perfect)}/{len(bc_sd)}",
        f"State dict mismatches: {len(mismatches)}",
    ]
    if mismatches:
        lines.append("")
        lines.append("--- MISMATCHES ---")
        lines.extend(mismatches[:30])
        if len(mismatches) > 30:
            lines.append(f"... and {len(mismatches) - 30} more")
    else:
        lines.append("All BC tensors == policy.features_extractor.cloner (torch.equal)")

    lines.extend(
        [
            "",
            "--- FORWARD LOGIT PARITY (same token_ids) ---",
            f"head1 exact: {fwd['head1_exact']} (max_diff={fwd['head1_max_diff']})",
            f"head2 exact: {fwd['head2_exact']} (max_diff={fwd['head2_max_diff']})",
        ]
    )

    all_ok = len(mismatches) == 0 and fwd["head1_exact"] and fwd["head2_exact"]
    lines.append("")
    lines.append(f"VERDICT: {'PASS' if all_ok else 'FAIL'}")

    report = "\n".join(lines)
    print(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"\nSaved -> {args.out.resolve()}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
