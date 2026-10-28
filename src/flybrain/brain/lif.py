"""Leaky Integrate-and-Fire (LIF) neural interface and data structures."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import numpy as np


@dataclass(frozen=True)
class LIFParams:
    """Biophysical and simulation parameters for the LIF model.
    Defaults directly match Shiu et al. (2024) / Kakaria & de Bivort (2017).
    """
    v_rest_mv: float = -52.0
    v_reset_mv: float = -52.0
    v_thresh_mv: float = -45.0
    tau_mem_ms: float = 20.0
    tau_syn_ms: float = 5.0
    refractory_ms: float = 2.2
    delay_ms: float = 1.8
    w_syn_mv: float = 0.275
    poisson_weight_scale: float = 250.0  # f_poi in Shiu repo (250 * w_syn)


@dataclass
class SpikeBatch:
    """Batch of spikes recorded across all neurons over a duration."""
    t_ms: np.ndarray        # float32 [S], relative to batch start (0 to duration_ms)
    neuron_idx: np.ndarray  # int32 [S], neuron index (0 to N-1)
    duration_ms: float      # duration of the step/batch in ms


@runtime_checkable
class SpikingBrain(Protocol):
    """Abstract protocol for backend-agnostic whole-brain spiking simulations."""
    n_neurons: int

    def reset(self, seed: int) -> None:
        """Reset internal potentials and state variables to resting state with seed."""
        ...

    def set_poisson_input(self, neuron_idx: np.ndarray, rate_hz: np.ndarray) -> None:
        """Set continuous Poisson firing rates on target neurons (persists across steps)."""
        ...

    def step(self, duration_ms: float) -> SpikeBatch:
        """Advance simulation by duration_ms and return emitted spikes."""
        ...
