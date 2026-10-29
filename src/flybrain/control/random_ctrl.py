"""Random baseline controller."""

import numpy as np


class RandomController:
    """Independent Bernoulli distribution per action."""

    def __init__(self, p: float = 0.1) -> None:
        self.p = p
        self._rng = np.random.default_rng()

    def reset(self, seed: int, reset_weights: bool = False) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, obs: dict) -> np.ndarray:
        # MultiBinary(3): [left, right, fire]
        action = (self._rng.random(3) < self.p).astype(np.int8)
        return action
