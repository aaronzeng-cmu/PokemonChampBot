"""MaskablePPO policy bootstrapped from VGCBehaviorCloner BC weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

from config.settings import BC_MODEL_PATH
from sb3_contrib.common.maskable.distributions import (
    MaskableCategorical,
    MaskableDistribution,
    MaskableMultiCategoricalDistribution,
)
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from src.doubles.battle.move_order import apply_joint_slot1_mask_torch
from src.doubles.data.action_space_spec import ACTION_SIZE
from src.core.data.state_tokenizer import N_FIELDS, STACKED_N_TOKENS
from src.core.model.transformer_bot import VGCBehaviorCloner, load_model

SLOT0_SIZE = ACTION_SIZE
SLOT1_SIZE = ACTION_SIZE


def _infer_force_switch_batch(mask0: th.Tensor, mask1_base: th.Tensor) -> th.Tensor:
    """True when at least one slot has no legal move actions (force-switch context)."""
    no_moves0 = ~mask0[:, 7:].any(dim=1)
    no_moves1 = ~mask1_base[:, 7:].any(dim=1)
    return no_moves0 | no_moves1


class VGCSequentialMaskableDistribution(MaskableDistribution):
    """
    Autoregressive MultiDiscrete(107, 107) with joint slot-1 masking.

    Mirrors TransformerPlayer: sample slot A, apply joint constraints, sample slot B.
    """

    def __init__(self) -> None:
        super().__init__()
        self.action_dims = [SLOT0_SIZE, SLOT1_SIZE]
        self._logits0: th.Tensor | None = None
        self._logits1: th.Tensor | None = None
        self._mask0: th.Tensor | None = None
        self._mask1_base: th.Tensor | None = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, sum(self.action_dims))

    def proba_distribution(
        self, action_logits: th.Tensor
    ) -> VGCSequentialMaskableDistribution:
        self._logits0 = action_logits[:, :SLOT0_SIZE]
        self._logits1 = action_logits[:, SLOT0_SIZE:]
        return self

    def apply_masking(self, masks: np.ndarray | th.Tensor | None) -> None:
        if masks is None or self._logits0 is None:
            self._mask0 = None
            self._mask1_base = None
            return
        device = self._logits0.device
        masks_t = th.as_tensor(masks, dtype=th.bool, device=device)
        if masks_t.dim() == 1:
            masks_t = masks_t.unsqueeze(0)
        masks_t = masks_t.view(-1, SLOT0_SIZE + SLOT1_SIZE)
        self._mask0 = masks_t[:, :SLOT0_SIZE]
        self._mask1_base = masks_t[:, SLOT0_SIZE:]

    def _dist0(self) -> MaskableCategorical:
        assert self._logits0 is not None
        dist = MaskableCategorical(logits=self._logits0)
        dist.apply_masking(self._mask0)
        return dist

    def _joint_mask1_row(self, row: int, a0: int) -> th.Tensor:
        assert self._mask1_base is not None and self._mask0 is not None
        force = bool(
            _infer_force_switch_batch(self._mask0, self._mask1_base)[row].item()
        )
        return apply_joint_slot1_mask_torch(
            self._mask1_base[row],
            a0_canonical=a0,
            force_switch=force,
        )

    def _dist1_row(self, row: int, a0: int) -> MaskableCategorical:
        assert self._logits1 is not None
        logits = self._logits1[row : row + 1]
        dist = MaskableCategorical(logits=logits)
        dist.apply_masking(self._joint_mask1_row(row, a0).unsqueeze(0))
        return dist

    def actions_from_params(
        self, action_logits: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self, action_logits: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        return actions, self.log_prob(actions)

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        return self.mode() if deterministic else self.sample()

    def sample(self) -> th.Tensor:
        assert self._logits0 is not None and self._logits1 is not None
        batch = self._logits0.shape[0]
        dist0 = self._dist0()
        a0 = dist0.sample()
        a1 = th.empty(batch, dtype=th.long, device=self._logits0.device)
        for i in range(batch):
            a1[i] = self._dist1_row(i, int(a0[i].item())).sample().squeeze(0)
        return th.stack([a0, a1], dim=1)

    def mode(self) -> th.Tensor:
        assert self._logits0 is not None and self._logits1 is not None
        batch = self._logits0.shape[0]
        dist0 = self._dist0()
        a0 = th.argmax(dist0.probs, dim=1)
        a1 = th.empty(batch, dtype=th.long, device=self._logits0.device)
        for i in range(batch):
            a1[i] = th.argmax(self._dist1_row(i, int(a0[i].item())).probs, dim=1).squeeze(0)
        return th.stack([a0, a1], dim=1)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        assert self._logits0 is not None and self._logits1 is not None
        actions = actions.view(-1, 2)
        a0 = actions[:, 0]
        a1 = actions[:, 1]
        dist0 = self._dist0()
        lp0 = dist0.log_prob(a0)
        batch = actions.shape[0]
        lp1 = th.empty(batch, device=self._logits0.device, dtype=lp0.dtype)
        for i in range(batch):
            lp1[i] = self._dist1_row(i, int(a0[i].item())).log_prob(a1[i : i + 1])
        return lp0 + lp1

    def entropy(self) -> th.Tensor:
        assert self._logits0 is not None and self._logits1 is not None
        dist0 = self._dist0()
        h0 = dist0.entropy()
        batch = self._logits0.shape[0]
        h1 = th.zeros_like(h0)
        for i in range(batch):
            probs0 = dist0.probs[i]
            legal = th.where(self._mask0[i])[0] if self._mask0 is not None else th.arange(SLOT0_SIZE, device=probs0.device)
            ent = probs0.new_zeros(())
            for a0_idx in legal:
                a0_int = int(a0_idx.item())
                p = probs0[a0_int]
                ent = ent + p * self._dist1_row(i, a0_int).entropy().squeeze()
            h1[i] = ent
        return h0 + h1


class VGCFeaturesExtractor(BaseFeaturesExtractor):
    """BC Transformer trunk (pooled CLS embedding)."""

    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 256,
        bc_model_path: str | Path | None = None,
    ):
        super().__init__(observation_space, features_dim)
        self.cloner = VGCBehaviorCloner()
        path = Path(bc_model_path or BC_MODEL_PATH)
        if path.is_file():
            bc = load_model(path, device="cpu")
            self.cloner.load_state_dict(bc.state_dict())
        self._obs_shape = (STACKED_N_TOKENS, N_FIELDS)

    def _obs_to_tokens(self, obs: th.Tensor) -> th.Tensor:
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        if obs.shape[-1] == STACKED_N_TOKENS * N_FIELDS:
            obs = obs.view(obs.shape[0], STACKED_N_TOKENS, N_FIELDS)
        return obs.round().long().clamp(min=0)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        token_ids = self._obs_to_tokens(observations)
        x = self.cloner._embed_tokens(token_ids)
        batch = x.size(0)
        cls = self.cloner.cls_token.expand(batch, -1, -1)
        x = th.cat([cls, x], dim=1)
        x = self.cloner.encoder(x)
        return x[:, 0]


class VGCBehaviorMaskablePolicy(MaskableActorCriticPolicy):
    """BC dual-head actor with sequential joint-legal action sampling."""

    def __init__(
        self,
        *args: Any,
        bc_model_path: str | Path | None = None,
        **kwargs: Any,
    ):
        self._bc_model_path = bc_model_path or BC_MODEL_PATH
        if "features_extractor_kwargs" not in kwargs:
            kwargs["features_extractor_kwargs"] = {}
        kwargs["features_extractor_kwargs"].setdefault(
            "bc_model_path", str(self._bc_model_path)
        )
        if "net_arch" not in kwargs:
            kwargs["net_arch"] = dict(pi=[], vf=[64, 64])
        if "features_extractor_class" not in kwargs:
            kwargs["features_extractor_class"] = VGCFeaturesExtractor
        super().__init__(*args, **kwargs)
        self.action_dist = VGCSequentialMaskableDistribution()

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> VGCSequentialMaskableDistribution:
        cloner = self.features_extractor.cloner
        logits0 = cloner.head1(latent_pi)
        logits1 = cloner.head2(latent_pi)
        action_logits = th.cat([logits0, logits1], dim=1)
        return self.action_dist.proba_distribution(action_logits=action_logits)


def init_bc_actor_weights(
    policy: VGCBehaviorMaskablePolicy,
    model_path: Path | str = BC_MODEL_PATH,
) -> None:
    """Load BC checkpoint into the policy feature extractor / actor heads."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"BC model not found: {path}")
    bc = load_model(path, device="cpu")
    policy.features_extractor.cloner.load_state_dict(bc.state_dict())


def is_joint_legal(
    mask0: np.ndarray,
    mask1_base: np.ndarray,
    a0: int,
    a1: int,
    *,
    force_switch: bool = False,
) -> bool:
    """Check whether (a0, a1) is legal under sequential joint masking."""
    from src.doubles.battle.move_order import apply_joint_slot1_mask_numpy

    if not (0 <= a0 < ACTION_SIZE and 0 <= a1 < ACTION_SIZE):
        return False
    if not mask0[a0]:
        return False
    mask1 = apply_joint_slot1_mask_numpy(
        mask1_base, a0_canonical=a0, force_switch=force_switch
    )
    return bool(mask1[a1])
