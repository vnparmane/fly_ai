"""Motor spike decoder translating spike batches into game actions."""

from dataclasses import dataclass, field
import numpy as np

from flybrain.brain.lif import SpikeBatch
from flybrain.io.select import IOSpec


@dataclass(frozen=True)
class DecoderConfig:
    window_ms: float = 150.0
    margin_hz: float = 2.0
    thresh_hz: dict[str, float] = field(
        default_factory=lambda: {
            "move_left": 5.0,
            "move_right": 5.0,
            "fire": 5.0,
        }
    )


class Decoder:
    """Rolling-window spike rate decoder.
    Translates SpikeBatch into MultiBinary(3) actions: [left, right, fire].
    """

    def __init__(self, io_spec: IOSpec, cfg: DecoderConfig | None = None) -> None:
        if cfg is None:
            cfg = DecoderConfig()
        self.cfg = cfg
        self.io = io_spec

        self.idx_left = set(int(i) for i in io_spec.output_groups["move_left"])
        self.idx_right = set(int(i) for i in io_spec.output_groups["move_right"])
        self.idx_fire = set(int(i) for i in io_spec.output_groups["fire"])

        self.size_left = max(1, len(self.idx_left))
        self.size_right = max(1, len(self.idx_right))
        self.size_fire = max(1, len(self.idx_fire))

        # Rolling history of spike timestamps and indices: list of (t_abs, neuron_idx)
        self._history: list[tuple[float, int]] = []
        self._current_time_ms = 0.0

    def reset(self) -> None:
        """Clear rolling spike window."""
        self._history = []
        self._current_time_ms = 0.0
        self.last_rates = {"move_left": 0.0, "move_right": 0.0, "fire": 0.0}

    def __call__(self, batch: SpikeBatch) -> np.ndarray:
        # Append incoming spikes with absolute timestamps
        t_base = self._current_time_ms
        for t_rel, nid in zip(batch.t_ms, batch.neuron_idx):
            self._history.append((float(t_base + t_rel), int(nid)))

        self._current_time_ms += batch.duration_ms
        cutoff_time = self._current_time_ms - self.cfg.window_ms

        # Prune old spikes outside rolling window
        self._history = [item for item in self._history if item[0] >= cutoff_time]

        # Count spikes per group in window
        count_left = 0
        count_right = 0
        count_fire = 0

        for _, nid in self._history:
            if nid in self.idx_left:
                count_left += 1
            elif nid in self.idx_right:
                count_right += 1
            elif nid in self.idx_fire:
                count_fire += 1

        window_sec = self.cfg.window_ms / 1000.0
        rate_left = count_left / (self.size_left * window_sec)
        rate_right = count_right / (self.size_right * window_sec)
        rate_fire = count_fire / (self.size_fire * window_sec)

        self.last_rates = {
            "move_left": float(rate_left),
            "move_right": float(rate_right),
            "fire": float(rate_fire),
        }

        thresh = self.cfg.thresh_hz
        margin = self.cfg.margin_hz

        action = np.zeros(3, dtype=np.int8)

        # Left active: rate_left - rate_right > margin AND rate_left > thresh_left
        if (rate_left - rate_right > margin) and (rate_left > thresh.get("move_left", 5.0)):
            action[0] = 1

        # Right active: rate_right - rate_left > margin AND rate_right > thresh_right
        if (rate_right - rate_left > margin) and (rate_right > thresh.get("move_right", 5.0)):
            action[1] = 1

        # Fire active: rate_fire > thresh_fire
        if rate_fire > thresh.get("fire", 5.0):
            action[2] = 1

        return action
