"""Unit tests for FAFB Descending Neurons to VNC Motor Circuit."""

import numpy as np
import pytest

from flybrain.brain.lif import LIFParams, SpikeBatch
from flybrain.brain.vnc import (
    FAFB_FIRE_TYPES,
    FAFB_STEERING_TYPES,
    VNCConfig,
    VNCMotorPool,
    extract_fafb_vnc_spec,
)
from flybrain.data.connectome import load_connectome
from flybrain.io.select import IOSpec


@pytest.fixture(scope="module")
def conn():
    return load_connectome(release="783", min_syn=5)


def test_extract_fafb_vnc_spec(conn):
    """Verify that extract_fafb_vnc_spec correctly identifies FAFB DN types."""
    spec = extract_fafb_vnc_spec(conn)
    assert isinstance(spec, IOSpec)

    # Check input groups
    assert "enemy_left" in spec.input_groups
    assert "enemy_right" in spec.input_groups
    assert "enemy_aligned" in spec.input_groups
    assert len(spec.input_groups["enemy_left"]) > 0
    assert len(spec.input_groups["enemy_right"]) > 0
    assert len(spec.input_groups["enemy_aligned"]) > 0

    # Ensure sensory input groups are pairwise disjoint
    in_l = set(spec.input_groups["enemy_left"])
    in_r = set(spec.input_groups["enemy_right"])
    in_a = set(spec.input_groups["enemy_aligned"])
    assert len(in_l & in_r) == 0, "FAFB VNC left and right inputs overlap"
    assert len(in_l & in_a) == 0, "FAFB VNC left and aligned inputs overlap"
    assert len(in_r & in_a) == 0, "FAFB VNC right and aligned inputs overlap"

    # Check output groups
    assert "move_left" in spec.output_groups
    assert "move_right" in spec.output_groups
    assert "fire" in spec.output_groups

    left_dns = spec.output_groups["move_left"]
    right_dns = spec.output_groups["move_right"]
    fire_dns = spec.output_groups["fire"]

    assert len(left_dns) > 0, "Should find left steering DNs in FAFB"
    assert len(right_dns) > 0, "Should find right steering DNs in FAFB"
    assert len(fire_dns) > 0, "Should find fire DNs in FAFB"

    # Verify that left DNs are indeed on the left side and right on right
    left_sides = conn.meta.iloc[left_dns]["side"].values
    right_sides = conn.meta.iloc[right_dns]["side"].values
    assert all(s == "left" for s in left_sides)
    assert all(s == "right" for s in right_sides)


def test_vnc_motor_pool_steering():
    """Verify leaky motor integration and action generation for steering."""
    dummy_io = IOSpec(
        input_groups={"enemy_left": np.array([1]), "enemy_right": np.array([2]), "enemy_aligned": np.array([1, 2])},
        output_groups={"move_left": np.array([10, 11]), "move_right": np.array([20, 21]), "fire": np.array([30])},
        justification={},
        config_hash="test",
    )
    vnc = VNCMotorPool(dummy_io, VNCConfig(tau_vnc_ms=50.0, steering_margin_hz=2.0, steering_thresh_hz=3.0))

    # 1. Left stimulation: high spikes on left DNs [10, 11]
    spikes_left = SpikeBatch(
        t_ms=np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        neuron_idx=np.array([10, 11, 10, 11], dtype=np.int32),
        duration_ms=50.0,
    )
    act_left = vnc.decode(spikes_left)
    assert act_left[0] == 1, "Should turn Left on left DN activation"
    assert act_left[1] == 0

    # 2. Reset and right stimulation: high spikes on right DNs [20, 21]
    vnc.reset()
    spikes_right = SpikeBatch(
        t_ms=np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        neuron_idx=np.array([20, 21, 20, 21], dtype=np.int32),
        duration_ms=50.0,
    )
    act_right = vnc.decode(spikes_right)
    assert act_right[0] == 0
    assert act_right[1] == 1, "Should turn Right on right DN activation"

    # 3. Fire stimulation on neuron [30]
    vnc.reset()
    spikes_fire = SpikeBatch(
        t_ms=np.array([5.0, 15.0, 25.0, 35.0], dtype=np.float32),
        neuron_idx=np.array([30, 30, 30, 30], dtype=np.int32),
        duration_ms=50.0,
    )
    act_fire = vnc.decode(spikes_fire)
    assert act_fire[2] == 1, "Should trigger Fire on fire DN activation"
