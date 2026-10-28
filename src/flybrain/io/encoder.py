"""Sensory rate encoder translating game observations to Poisson input rates."""

from dataclasses import dataclass
import numpy as np

from flybrain.io.select import IOSpec


@dataclass(frozen=True)
class EncoderConfig:
    max_rate_hz: float = 100.0
    base_rate_hz: float = 0.0
    dx_range: float = 1.0
    align_sigma: float = 0.15


class Encoder:
    """Translates observation dict to (neuron_idx, rate_hz) for set_poisson_input."""

    def __init__(self, io_spec: IOSpec, cfg: EncoderConfig | None = None) -> None:
        if cfg is None:
            cfg = EncoderConfig()
        self.cfg = cfg
        self.io = io_spec

        self.idx_left = io_spec.input_groups["enemy_left"]
        self.idx_right = io_spec.input_groups["enemy_right"]
        self.idx_aligned = io_spec.input_groups["enemy_aligned"]

    def __call__(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        target_dx = float(obs["target_dx"][0])

        # Rates:
        # enemy_left  = max * clip(-target_dx / dx_range, 0, 1)
        # enemy_right = max * clip( target_dx / dx_range, 0, 1)
        # enemy_aligned = max * exp(-(target_dx / align_sigma)^2)
        rate_left = float(self.cfg.max_rate_hz * np.clip(-target_dx / self.cfg.dx_range, 0.0, 1.0))
        rate_right = float(self.cfg.max_rate_hz * np.clip(target_dx / self.cfg.dx_range, 0.0, 1.0))
        rate_aligned = float(self.cfg.max_rate_hz * np.exp(-((target_dx / self.cfg.align_sigma) ** 2)))

        indices = np.concatenate([self.idx_left, self.idx_right, self.idx_aligned])
        rates = np.concatenate([
            np.full(len(self.idx_left), rate_left, dtype=np.float32),
            np.full(len(self.idx_right), rate_right, dtype=np.float32),
            np.full(len(self.idx_aligned), rate_aligned, dtype=np.float32),
        ])

        return indices, rates
