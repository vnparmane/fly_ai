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
