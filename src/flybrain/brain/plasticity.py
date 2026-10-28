"""Reward-modulated Spike-Timing-Dependent Plasticity (R-STDP) and Dopaminergic Learning.

Implements three-factor synaptic plasticity where pre- and post-synaptic spike correlations
generate synaptic eligibility traces, and a global reward/dopamine signal R(t) gates
synaptic weight modification:
    e_ij(t) = gamma * e_ij(t-1) + S_pre * S_post
    Delta W_ij = eta * R(t) * e_ij(t)
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import numpy as np

from flybrain.brain.backend_brian2 import Brian2Brain
from flybrain.brain.lif import SpikeBatch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlasticityConfig:
    lr: float = 0.005              # Learning rate eta
    decay: float = 0.85            # Eligibility trace decay gamma (tau_e ~ 300 ms)
    w_max_factor: float = 3.0      # Maximum allowed weight relative to initial
    w_min_factor: float = 0.1      # Minimum allowed weight relative to initial
    max_delta_per_step: float = 0.05  # Gradient clipping per step


class RSTDPPlasticity:
    """Manages dopamine-modulated synaptic plasticity on Brian2Brain synapses."""

    def __init__(self, brain: Brian2Brain, cfg: PlasticityConfig | None = None) -> None:
        self.brain = brain
        self.cfg = cfg or PlasticityConfig()

        # Synapse topology
        self.n_syn = len(brain.syn)
        self.pre_idx = np.array(brain.syn.i[:], dtype=np.int32)
        self.post_idx = np.array(brain.syn.j[:], dtype=np.int32)

        # Baseline weights (mV units)
        from brian2 import mV, volt
        self.w_init = np.array(brain.syn.w[:] / volt, dtype=np.float32)
        self.signs = np.sign(self.w_init).astype(np.int8)

        # Bounds enforcing Dale's principle
        abs_init = np.abs(self.w_init)
        self.w_min = abs_init * self.cfg.w_min_factor
        self.w_max = abs_init * self.cfg.w_max_factor

        # Eligibility traces for each synapse
        self.eligibility = np.zeros(self.n_syn, dtype=np.float32)

        # Cumulative learning statistics
        self.cumulative_delta = np.zeros(self.n_syn, dtype=np.float32)
        self.updates_count = 0

    def reset_traces(self) -> None:
        """Reset eligibility traces between episodes."""
        self.eligibility.fill(0.0)

    def step(self, batch: SpikeBatch, reward: float) -> dict[str, float]:
        """Update eligibility traces from spikes, apply dopamine reward modulation,
        and update synaptic weights in the brain.
        """
        from brian2 import volt

        # 1. Compute spike count vector across neurons for this step
        spike_counts = np.zeros(self.brain.n_neurons, dtype=np.float32)
        if len(batch.neuron_idx) > 0:
            np.add.at(spike_counts, batch.neuron_idx, 1.0)

        # 2. Update eligibility traces: e = gamma * e + s_pre * s_post
        s_pre = spike_counts[self.pre_idx]
        s_post = spike_counts[self.post_idx]
        coincidence = s_pre * s_post

        self.eligibility = self.cfg.decay * self.eligibility + coincidence

        # 3. Apply dopamine reward modulation: Delta W = eta * R * e
        delta_w = self.cfg.lr * reward * self.eligibility

        # Gradient clipping
        delta_w = np.clip(delta_w, -self.cfg.max_delta_per_step, self.cfg.max_delta_per_step)

        # 4. Update current synaptic weights
        current_w = np.array(self.brain.syn.w[:] / volt, dtype=np.float32)
        new_w = current_w + delta_w

        # 5. Enforce Dale's principle and weight boundaries
        # Excitatory (sign > 0): clamped to [w_min, w_max]
        # Inhibitory (sign < 0): clamped to [-w_max, -w_min]
        pos_mask = self.signs > 0
        neg_mask = self.signs < 0

        new_w[pos_mask] = np.clip(new_w[pos_mask], self.w_min[pos_mask], self.w_max[pos_mask])
        new_w[neg_mask] = np.clip(new_w[neg_mask], -self.w_max[neg_mask], -self.w_min[neg_mask])

        # Write back to Brian2 Synapses
        self.brain.syn.w = new_w * volt

        self.cumulative_delta += (new_w - current_w)
        self.updates_count += 1

        stats = {
            "mean_eligibility": float(np.mean(self.eligibility)),
            "max_eligibility": float(np.max(self.eligibility)),
            "mean_delta_w": float(np.mean(np.abs(delta_w))),
            "total_weight_delta_l1": float(np.sum(np.abs(self.cumulative_delta))),
        }
        return stats

    def save_weights(self, path: Path | str) -> None:
        """Save trained synaptic weights and signs to an NPZ file."""
        from brian2 import volt
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        current_w = np.array(self.brain.syn.w[:] / volt, dtype=np.float32)
        np.savez_compressed(
            path,
            w=current_w,
            w_init=self.w_init,
            signs=self.signs,
            cumulative_delta=self.cumulative_delta,
            pre_idx=self.pre_idx,
            post_idx=self.post_idx,
        )
        logger.info(f"Saved trained weights ({len(current_w)} synapses) to {path}")

    def load_weights(self, path: Path | str) -> None:
        """Load trained synaptic weights from NPZ file and apply to Brian2 network."""
        from brian2 import volt
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        data = np.load(path)
        trained_w = data["w"]
        if len(trained_w) != self.n_syn:
            raise ValueError(f"Checkpoint synapse count {len(trained_w)} does not match network {self.n_syn}")

        self.brain.syn.w = trained_w * volt
        logger.info(f"Loaded {len(trained_w)} trained synaptic weights from {path}")
