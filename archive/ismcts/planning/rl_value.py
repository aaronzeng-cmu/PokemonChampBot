"""Value network evaluators for ISMCTS (MLP + optional PPO fallback)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

from config.settings import ISMCTS_RL_VALUE_MODEL, ISMCTS_VALUE_MLP_PATH
from archive.rl.env.observation import adapt_observation_for_model, embed_battle
from archive.ismcts.planning.value_mlp import load_value_mlp
from archive.ismcts.planning.value_state import embed_value_state

if TYPE_CHECKING:
    from poke_env.battle.double_battle import DoubleBattle

    from src.planning.belief_state import BeliefState

logger = logging.getLogger(__name__)


class ValueMLPEvaluator:
    """Belief-augmented MLP trained on asymmetrical gauntlet rollouts."""

    def __init__(self, model_path: str | Path | None) -> None:
        self._model = None
        self._device = "cpu"
        self._input_dim: int | None = None
        if model_path:
            path = Path(model_path)
            if path.is_file():
                self._load(path)
            else:
                logger.warning("ISMCTS value MLP not found: %s", path)

    def _load(self, path: Path) -> None:
        try:
            self._model = load_value_mlp(path, device="cpu")
            self._input_dim = self._model.config.input_dim
            logger.info("ISMCTS using value MLP: %s (dim=%s)", path, self._input_dim)
        except Exception as exc:
            logger.warning("Failed to load value MLP (%s)", exc)
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def evaluate(
        self,
        battle: DoubleBattle,
        belief: BeliefState | None = None,
    ) -> float:
        if self._model is None:
            return 0.0
        try:
            state = embed_value_state(battle, belief).astype(np.float32)
            if self._input_dim and state.shape[0] != self._input_dim:
                if state.shape[0] > self._input_dim:
                    state = state[: self._input_dim]
                else:
                    padded = np.zeros(self._input_dim, dtype=np.float32)
                    padded[: state.shape[0]] = state
                    state = padded
            x = torch.as_tensor(state, device=self._device).unsqueeze(0)
            with torch.no_grad():
                value = self._model(x)
            return float(value.cpu().numpy().reshape(-1)[0])
        except Exception as exc:
            logger.debug("Value MLP eval failed: %s", exc)
            return 0.0


class PPOValueEvaluator:
    """Legacy PPO checkpoint value head (battle obs only, no belief)."""

    def __init__(self, model_path: str | Path | None) -> None:
        self._model = None
        self._device = "cpu"
        if model_path:
            path = Path(model_path)
            if path.is_file():
                self._load(path)
            else:
                logger.warning("ISMCTS PPO value model not found: %s", path)

    def _load(self, path: Path) -> None:
        try:
            from sb3_contrib import MaskablePPO

            self._model = MaskablePPO.load(str(path), device="cpu")
            self._device = self._model.policy.device
            logger.info("ISMCTS using PPO value model: %s", path)
        except Exception as exc:
            logger.warning("Failed to load PPO value model (%s)", exc)
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def evaluate(self, battle: DoubleBattle, belief: BeliefState | None = None) -> float:
        if self._model is None:
            return 0.0
        try:
            obs = embed_battle(battle).astype(np.float32)
            target = int(self._model.observation_space.shape[0])
            if obs.shape[0] != target:
                obs = adapt_observation_for_model(obs, target)
            obs_t = torch.as_tensor(obs, device=self._device).unsqueeze(0)
            with torch.no_grad():
                value = self._model.policy.predict_values(obs_t)
            return float(value.cpu().numpy().reshape(-1)[0])
        except Exception as exc:
            logger.debug("PPO value eval failed: %s", exc)
            return 0.0


class CombinedValueEvaluator:
    """Prefer trained MLP; fall back to PPO value head."""

    def __init__(
        self,
        *,
        mlp_path: str | Path | None = None,
        ppo_path: str | Path | None = None,
    ) -> None:
        self.mlp = ValueMLPEvaluator(mlp_path)
        self.ppo = PPOValueEvaluator(ppo_path)

    @property
    def available(self) -> bool:
        return self.mlp.available or self.ppo.available

    def evaluate(
        self,
        battle: DoubleBattle,
        belief: BeliefState | None = None,
    ) -> float:
        if self.mlp.available:
            return self.mlp.evaluate(battle, belief)
        return self.ppo.evaluate(battle, belief)


def build_value_evaluator(
    *,
    mlp_path: str | Path | None = None,
    ppo_path: str | Path | None = None,
) -> CombinedValueEvaluator:
    mlp = mlp_path if mlp_path is not None else ISMCTS_VALUE_MLP_PATH
    ppo = ppo_path if ppo_path is not None else ISMCTS_RL_VALUE_MODEL
    return CombinedValueEvaluator(
        mlp_path=mlp or None,
        ppo_path=ppo or None,
    )


def build_value_fn_for_hybrid(
    evaluator: CombinedValueEvaluator,
    belief_lookup: Callable[[DoubleBattle], BeliefState | None],
) -> Callable[[DoubleBattle], float] | None:
    if not evaluator.available:
        return None

    def _fn(battle: DoubleBattle) -> float:
        return evaluator.evaluate(battle, belief_lookup(battle))

    return _fn


# Backward compatibility
RLValueEvaluator = PPOValueEvaluator


def build_value_fn(model_path: str | Path | None) -> Callable[[DoubleBattle], float] | None:
    evaluator = build_value_evaluator(ppo_path=model_path, mlp_path=ISMCTS_VALUE_MLP_PATH)
    if not evaluator.available:
        return None
    return lambda battle: evaluator.evaluate(battle, None)
