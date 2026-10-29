"""Tests for LIF brain protocol and Brian2 backend implementation."""

import numpy as np
import pytest

from flybrain.brain.backend_brian2 import Brian2Brain
from flybrain.brain.lif import LIFParams, SpikeBatch, SpikingBrain
from flybrain.brain.subcircuit import extract_subcircuit
from flybrain.data.connectome import load_connectome


@pytest.fixture(scope="module")
def small_brain():
    conn = load_connectome(release="783", min_syn=5)
    # Extract subcircuit with actual connected path
    seeds_in = np.arange(20, dtype=np.int32)
    seeds_out = np.arange(100, 120, dtype=np.int32)
    sub_conn, _ = extract_subcircuit(conn, seeds_in, seeds_out, max_hops=3, max_neurons=200)
    params = LIFParams()
    return Brian2Brain(sub_conn, params)


def test_protocol_conformance(small_brain):
    # Verify runtime checkable protocol
    assert isinstance(small_brain, SpikingBrain)
    assert hasattr(small_brain, "n_neurons")
    assert hasattr(small_brain, "reset")
    assert hasattr(small_brain, "set_poisson_input")
    assert hasattr(small_brain, "step")


def test_step_returns_spikebatch(small_brain):
    small_brain.reset(seed=123)
    batch = small_brain.step(50.0)
    assert isinstance(batch, SpikeBatch)
    assert batch.duration_ms == 50.0
    assert isinstance(batch.t_ms, np.ndarray)
    assert isinstance(batch.neuron_idx, np.ndarray)


def test_poisson_stimulation_increases_firing(small_brain):
    # Zero input step
    small_brain.reset(seed=42)
    small_brain.set_poisson_input(np.array([0], dtype=np.int32), np.array([0.0], dtype=np.float32))
    b_zero = small_brain.step(100.0)
    zero_spikes = len(b_zero.neuron_idx)

    # 150 Hz input step
    small_brain.reset(seed=42)
    small_brain.set_poisson_input(np.array([0], dtype=np.int32), np.array([150.0], dtype=np.float32))
    b_stim = small_brain.step(100.0)
    stim_spikes = len(b_stim.neuron_idx)

    assert stim_spikes >= zero_spikes


def test_reset_is_deterministic(small_brain):
    small_brain.reset(seed=999)
    small_brain.set_poisson_input(np.array([0], dtype=np.int32), np.array([100.0], dtype=np.float32))
    b1 = small_brain.step(100.0)

    small_brain.reset(seed=999)
    small_brain.set_poisson_input(np.array([0], dtype=np.int32), np.array([100.0], dtype=np.float32))
    b2 = small_brain.step(100.0)

    assert np.array_equal(b1.t_ms, b2.t_ms)
    assert np.array_equal(b1.neuron_idx, b2.neuron_idx)


def test_reset_preserves_learned_weights(small_brain):
    """Finding 1 verification: Dynamic state reset preserves learned/modified weights."""
    import brian2 as b2
    from brian2 import volt, mV

    init_w = np.array(small_brain.syn.w[:] / volt, dtype=np.float32)
    modified_w = init_w + 0.05  # Alter weights by 50 uV

    # Set modified weights
    small_brain.syn.w = modified_w * volt

    # Step to depolarize neurons away from rest
    small_brain.set_poisson_input(np.array([0], dtype=np.int32), np.array([100.0], dtype=np.float32))
    small_brain.step(50.0)

    # Standard reset (as called between episodes)
    small_brain.reset(seed=123)

    # 1. Weights must NOT be wiped out
    current_w = np.array(small_brain.syn.w[:] / volt, dtype=np.float32)
    assert np.allclose(current_w, modified_w), "Brain reset incorrectly wiped learned synaptic weights!"

    # 2. Membrane potentials must be restored to rest
    assert np.allclose(np.array(small_brain.neu.v[:] / volt), float(small_brain.ns["v_0"] / volt))

    # 3. Explicit weight reset restores baseline
    small_brain.reset_weights_to_initial()
    restored_w = np.array(small_brain.syn.w[:] / volt, dtype=np.float32)
    assert np.allclose(restored_w, init_w), "reset_weights_to_initial failed to restore baseline weights"


def test_set_poisson_input_order_independence(small_brain):
    """Finding 2 verification: Backend Poisson input rates are order-independent."""
    from brian2 import Hz

    # Channel A then Channel B
    indices_1 = np.array([0, 1, 0], dtype=np.int32)
    rates_1 = np.array([20.0, 50.0, 80.0], dtype=np.float32)
    small_brain.set_poisson_input(indices_1, rates_1)
    rate_0_order1 = float(small_brain.poi_group.rates[0] / Hz)

    # Channel B then Channel A (reversed order)
    indices_2 = np.array([0, 1, 0], dtype=np.int32)
    rates_2 = np.array([80.0, 50.0, 20.0], dtype=np.float32)
    small_brain.set_poisson_input(indices_2, rates_2)
    rate_0_order2 = float(small_brain.poi_group.rates[0] / Hz)

    assert rate_0_order1 == 80.0, f"Expected 80.0 Hz, got {rate_0_order1}"
    assert rate_0_order2 == 80.0, f"Expected 80.0 Hz, got {rate_0_order2}"
    assert rate_0_order1 == rate_0_order2, "set_poisson_input is sensitive to input group ordering!"


def test_lif_params_from_yaml_matches_default():
    """Finding 3 verification: Configured Poisson input scale matches biophysical default."""
    params_yaml = LIFParams.from_yaml("configs/default.yaml")
    params_code = LIFParams()

    assert params_yaml.poisson_weight_scale == 250.0
    assert params_yaml.poisson_weight_scale == params_code.poisson_weight_scale
    assert params_yaml.w_syn_mv == params_code.w_syn_mv

    # Single Poisson event depolarization should cross AP threshold (delta_V = 68.75 mV > 7.0 mV)
    w_poi_mv = params_yaml.w_syn_mv * params_yaml.poisson_weight_scale
    thresh_distance_mv = params_yaml.v_thresh_mv - params_yaml.v_rest_mv
    assert w_poi_mv == 68.75
    assert w_poi_mv > thresh_distance_mv, "Single Poisson event fails to evoke action potential"
