"""MaskablePPO policy bootstrapped from Singles BC Transformer weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from config.settings import SINGLES_BC_MODEL_PATH
from src.core.data.state_tokenizer import N_FIELDS, STACKED_N_TOKENS
from src.core.model.transformer_bot import (
    SINGLES_ACTION_SIZE,
    VGCBehaviorCloner,
    VGCBehaviorClonerConfig,
    load_model,
)


class SinglesFeaturesExtractor(BaseFeaturesExtractor):
    """BC Transformer trunk (pooled CLS embedding) for singles."""

    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 256,
        bc_model_path: str | Path | None = None,
    ):
        super().__init__(observation_space, features_dim)
        path = Path(bc_model_path or SINGLES_BC_MODEL_PATH)
        if path.is_file():
            self.cloner = load_model(path, device="cpu")
            if self.cloner.config.action_space != "singles":
                raise ValueError(f"Expected singles BC model at {path}")
        else:
            self.cloner = VGCBehaviorCloner(
                VGCBehaviorClonerConfig(action_space="singles")
            )

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


class SinglesBehaviorMaskablePolicy(MaskableActorCriticPolicy):
    """BC singles head with standard MaskableCategorical (one decision slot)."""

    def __init__(
        self,
        *args: Any,
        bc_model_path: str | Path | None = None,
        **kwargs: Any,
    ):
        self._bc_model_path = bc_model_path or SINGLES_BC_MODEL_PATH
        if "features_extractor_kwargs" not in kwargs:
            kwargs["features_extractor_kwargs"] = {}
        kwargs["features_extractor_kwargs"].setdefault(
            "bc_model_path", str(self._bc_model_path)
        )
        if "net_arch" not in kwargs:
            kwargs["net_arch"] = dict(pi=[], vf=[64, 64])
        if "features_extractor_class" not in kwargs:
            kwargs["features_extractor_class"] = SinglesFeaturesExtractor
        super().__init__(*args, **kwargs)
        self.action_dist = MaskableCategoricalDistribution(SINGLES_ACTION_SIZE)

    def _get_action_dist_from_latent(
        self, latent_pi: th.Tensor
    ) -> MaskableCategoricalDistribution:
        cloner = self.features_extractor.cloner
        assert cloner.head_singles is not None
        logits = cloner.head_singles(latent_pi)
        return self.action_dist.proba_distribution(action_logits=logits)


def init_bc_actor_weights(
    policy: SinglesBehaviorMaskablePolicy,
    model_path: Path | str = SINGLES_BC_MODEL_PATH,
) -> None:
    """Load singles BC checkpoint into the policy feature extractor / actor head."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Singles BC model not found: {path}")
    bc = load_model(path, device="cpu")
    if bc.config.action_space != "singles":
        raise ValueError(f"Expected singles BC model at {path}, got {bc.config.action_space}")
    missing, unexpected = policy.features_extractor.cloner.load_state_dict(
        bc.state_dict(), strict=True
    )
    if missing or unexpected:
        raise RuntimeError(
            f"BC load_state_dict mismatch: missing={missing} unexpected={unexpected}"
        )
