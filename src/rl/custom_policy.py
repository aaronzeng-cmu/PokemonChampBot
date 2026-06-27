"""Re-export policy classes for pre-monorepo RL checkpoint unpickling."""

from src.doubles.rl.custom_policy import (  # noqa: F401
    VGCBehaviorMaskablePolicy,
    init_bc_actor_weights as init_doubles_bc_actor_weights,
)
from src.singles.rl.custom_policy import (  # noqa: F401
    SinglesBehaviorMaskablePolicy,
    init_bc_actor_weights as init_singles_bc_actor_weights,
)

init_bc_actor_weights = init_doubles_bc_actor_weights

__all__ = [
    "VGCBehaviorMaskablePolicy",
    "SinglesBehaviorMaskablePolicy",
    "init_bc_actor_weights",
    "init_doubles_bc_actor_weights",
    "init_singles_bc_actor_weights",
]
