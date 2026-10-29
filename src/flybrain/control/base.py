"""Base controller interface and protocol."""

from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class Controller(Protocol):
    """Protocol for all game controllers."""

    def reset(self, seed: int, reset_weights: bool = False) -> None:
        """Reset internal controller state with seed."""
        ...

    def act(self, obs: dict) -> np.ndarray:
        """Produce an action MultiBinary(3) given current observation."""
        ...
