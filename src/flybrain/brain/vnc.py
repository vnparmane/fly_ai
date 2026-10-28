"""FAFB VNC Descending Motor Circuits & Premotor Integration.

Maps anatomically identified Descending Neurons (DNs) from the Full Adult Female Brain
(FAFB / FlyWire release 783) to Ventral Nerve Cord (VNC) motor pools driving locomotion
steering and trigger behaviors.
"""

from dataclasses import dataclass, field
import logging
import numpy as np

from flybrain.brain.lif import SpikeBatch, SpikingBrain
from flybrain.data.connectome import Connectome, neurons_where
from flybrain.io.encoder import Encoder
from flybrain.io.select import IOSpec, candidate_inputs

logger = logging.getLogger(__name__)

# Verified FAFB Descending Neuron types documented in Drosophila motor neuroscience
# Steering DNs (Rayshubskiy et al. 2020, Namiki et al. 2018):
FAFB_STEERING_TYPES = ["DNa02", "DNp04", "DNa03", "DNa15", "DNa16"]
# Fast bilateral / trigger DNs (Bidaye et al. 2014, Ache et al. 2019):
FAFB_FIRE_TYPES = ["DNg108", "DNp31", "DNg74_a", "DNb01"]


@dataclass(frozen=True)
class VNCConfig:
    tau_vnc_ms: float = 40.0
    steering_margin_hz: float = 1.0
    steering_thresh_hz: float = 2.0
    fire_thresh_hz: float = 2.0


def extract_fafb_vnc_spec(conn: Connectome) -> IOSpec:
    """Extract an IOSpec using anatomically verified FAFB descending neurons for VNC motor control.

    - Sensory inputs: Visual projection neurons (VPNs, looming / motion responsive types).
    - Steering outputs: Symmetrically paired lateral descending neurons (DNa02, DNp04, DNa03, DNa15, DNa16).
    - Trigger outputs: Bilateral descending neurons with symmetrical visual drive (DNg108, DNp31, DNg74_a, DNb01).
    """
    vpn_l = candidate_inputs(conn, "left")
    vpn_r = candidate_inputs(conn, "right")

    # Sensory input groups
    input_groups = {
        "enemy_left": vpn_l.astype(np.int32),
        "enemy_right": vpn_r.astype(np.int32),
        "enemy_aligned": np.union1d(vpn_l[:20], vpn_r[:20]).astype(np.int32),
    }

    # Left steering DNs
    left_dns = neurons_where(conn, super_class="descending", side="left", cell_type=FAFB_STEERING_TYPES)
    # Right steering DNs
    right_dns = neurons_where(conn, super_class="descending", side="right", cell_type=FAFB_STEERING_TYPES)
    # Bilateral trigger DNs
    fire_dns = neurons_where(conn, super_class="descending", cell_type=FAFB_FIRE_TYPES)

    output_groups = {
        "move_left": left_dns.astype(np.int32),
        "move_right": right_dns.astype(np.int32),
        "fire": fire_dns.astype(np.int32),
    }

    justification = {
        "move_left": {
            "source": "FAFB FlyWire v783 Descending Neurons",
            "types": FAFB_STEERING_TYPES,
            "side": "left",
            "count": len(left_dns),
            "mechanism": "Ipsilateral VNC steering motor command",
        },
        "move_right": {
            "source": "FAFB FlyWire v783 Descending Neurons",
            "types": FAFB_STEERING_TYPES,
            "side": "right",
            "count": len(right_dns),
            "mechanism": "Ipsilateral VNC steering motor command",
        },
        "fire": {
            "source": "FAFB FlyWire v783 Descending Neurons",
            "types": FAFB_FIRE_TYPES,
            "side": "bilateral",
            "count": len(fire_dns),
            "mechanism": "Bilateral descending escape / attack trigger",
        },
    }

    logger.info(
        f"Extracted FAFB VNC Spec: Left DNs={len(left_dns)}, Right DNs={len(right_dns)}, Fire DNs={len(fire_dns)}"
    )

    return IOSpec(
        input_groups=input_groups,
        output_groups=output_groups,
        justification=justification,
        config_hash="fafb_vnc_spec_v1",
    )


class VNCMotorPool:
    """Simulates Ventral Nerve Cord (VNC) motor pool integration:
    Accumulates descending drive through a leaky integration filter and outputs discrete actions.
    """

    def __init__(self, io_spec: IOSpec, cfg: VNCConfig | None = None) -> None:
        self.cfg = cfg or VNCConfig()
        self.io = io_spec

        self.idx_left = set(int(i) for i in io_spec.output_groups["move_left"])
        self.idx_right = set(int(i) for i in io_spec.output_groups["move_right"])
        self.idx_fire = set(int(i) for i in io_spec.output_groups["fire"])

        self.size_left = max(1, len(self.idx_left))
        self.size_right = max(1, len(self.idx_right))
        self.size_fire = max(1, len(self.idx_fire))

        self.steer_state = 0.0  # Positive -> left steering, Negative -> right steering
        self.fire_state = 0.0
        self.last_rates = {"move_left": 0.0, "move_right": 0.0, "fire": 0.0}

    def reset(self) -> None:
        self.steer_state = 0.0
        self.fire_state = 0.0
        self.last_rates = {"move_left": 0.0, "move_right": 0.0, "fire": 0.0}

    def decode(self, batch: SpikeBatch) -> np.ndarray:
        """Integrate spike batch and return action MultiBinary(3)."""
        dt_sec = max(0.001, batch.duration_ms / 1000.0)

        # Count spikes in this step batch
        count_left = 0
        count_right = 0
        count_fire = 0

        for nid in batch.neuron_idx:
            nid = int(nid)
            if nid in self.idx_left:
                count_left += 1
            elif nid in self.idx_right:
                count_right += 1
            elif nid in self.idx_fire:
                count_fire += 1

        rate_left = count_left / (self.size_left * dt_sec)
        rate_right = count_right / (self.size_right * dt_sec)
        rate_fire = count_fire / (self.size_fire * dt_sec)

        self.last_rates = {
            "move_left": float(rate_left),
            "move_right": float(rate_right),
            "fire": float(rate_fire),
        }

        # Leaky motor integration: state = state * decay + drive
        decay = float(np.exp(-batch.duration_ms / self.cfg.tau_vnc_ms))
        steer_drive = rate_left - rate_right
        self.steer_state = self.steer_state * decay + steer_drive

        fire_drive = rate_fire
        self.fire_state = self.fire_state * decay + fire_drive

        action = np.zeros(3, dtype=np.int8)

        # Steering decision
        if self.steer_state > self.cfg.steering_margin_hz and rate_left > self.cfg.steering_thresh_hz:
            action[0] = 1  # Move Left
        elif self.steer_state < -self.cfg.steering_margin_hz and rate_right > self.cfg.steering_thresh_hz:
            action[1] = 1  # Move Right

        # Fire decision
        if self.fire_state > self.cfg.fire_thresh_hz:
            action[2] = 1  # Fire

        return action


class VNCBrainController:
    """Closed-loop controller driving Space Invaders via FAFB Descending -> VNC Motor Pool."""

    def __init__(
        self,
        brain: SpikingBrain,
        encoder: Encoder,
        vnc_motor_pool: VNCMotorPool,
        step_ms: float = 50.0,
    ) -> None:
        self.brain = brain
        self.encoder = encoder
        self.vnc = vnc_motor_pool
        self.step_ms = step_ms
        self.last_batch = None
        self.last_rates = {"move_left": 0.0, "move_right": 0.0, "fire": 0.0}
        self.last_action = np.zeros(3, dtype=np.int8)

    def reset(self, seed: int) -> None:
        self.brain.reset(seed)
        self.vnc.reset()
        self.last_batch = None
        self.last_rates = {"move_left": 0.0, "move_right": 0.0, "fire": 0.0}
        self.last_action = np.zeros(3, dtype=np.int8)

    def act(self, obs: dict) -> np.ndarray:
        indices, rates = self.encoder(obs)
        self.brain.set_poisson_input(indices, rates)
        batch = self.brain.step(self.step_ms)
        self.last_batch = batch

        action = self.vnc.decode(batch)
        self.last_rates = self.vnc.last_rates
        self.last_action = action
        return action
