"""Unit tests for Reward-modulated STDP (R-STDP) and Dopaminergic Plasticity."""

from pathlib import Path
import numpy as np
import pytest

from flybrain.brain.backend_brian2 import Brian2Brain
from flybrain.brain.lif import LIFParams, SpikeBatch
from flybrain.brain.plasticity import PlasticityConfig, RSTDPPlasticity
from flybrain.brain.subcircuit import extract_subcircuit
from flybrain.data.connectome import load_connectome


@pytest.fixture(scope="module")
def small_brain():
    conn = load_connectome(release="783", min_syn=5)
    # Extract small 40-neuron subcircuit for fast testing
    sub, _ = extract_subcircuit(conn, np.array([0, 1]), np.array([2, 3]), max_neurons=40)
    return Brian2Brain(sub, LIFParams())


def test_rstdp_trace_and_weight_update(small_brain, tmp_path):
    """Verify eligibility trace accumulation, reward-modulated updates, and sign preservation."""
    plasticity = RSTDPPlasticity(small_brain, PlasticityConfig(lr=0.01, decay=0.8))

    pre_neuron = int(plasticity.pre_idx[0])
    post_neuron = int(plasticity.post_idx[0])
    initial_w0 = float(plasticity.w_init[0])

    # 1. Step with coincidence (both pre and post fire) and positive reward (+1.0)
    batch = SpikeBatch(
        t_ms=np.array([10.0, 20.0], dtype=np.float32),
        neuron_idx=np.array([pre_neuron, post_neuron], dtype=np.int32),
        duration_ms=50.0,
    )

    stats = plasticity.step(batch, reward=1.0)
    assert plasticity.eligibility[0] > 0.0, "Eligibility trace should increase on coincidence"

    from brian2 import volt
    w_after = float(small_brain.syn.w[0] / volt)

    if initial_w0 > 0:
        assert w_after >= initial_w0, "Excitatory weight should increase with positive reward"
    else:
        assert w_after <= initial_w0, "Inhibitory weight magnitude should increase (more negative)"

    # 2. Check Dale's principle preservation across all synapses
    all_w = np.array(small_brain.syn.w[:] / volt)
    all_signs = np.sign(all_w).astype(np.int8)
    assert np.all(all_signs == plasticity.signs), "Dale's principle: signs must never flip"

    # 3. Check save and load round-trip
    chk_path = tmp_path / "test_weights.npz"
    plasticity.save_weights(chk_path)
    assert chk_path.exists()

    # Create fresh plasticity object and reload
    plasticity2 = RSTDPPlasticity(small_brain, PlasticityConfig())
    plasticity2.load_weights(chk_path)
    loaded_w = np.array(small_brain.syn.w[:] / volt)
    assert np.allclose(loaded_w, all_w, atol=1e-5)


def test_learning_persists_across_episode_resets(small_brain):
    """Finding 1 verification: Weights persist across episode resets enabling cumulative multi-episode learning."""
    from brian2 import volt
    small_brain.reset_weights_to_initial()
    plasticity = RSTDPPlasticity(small_brain, PlasticityConfig(lr=0.0002, decay=0.8))

    pre_neuron = int(plasticity.pre_idx[0])
    post_neuron = int(plasticity.post_idx[0])

    batch = SpikeBatch(
        t_ms=np.array([10.0, 20.0], dtype=np.float32),
        neuron_idx=np.array([pre_neuron, post_neuron], dtype=np.int32),
        duration_ms=50.0,
    )

    # Episode 1: train
    plasticity.step(batch, reward=1.0)
    w_ep1 = np.array(small_brain.syn.w[:] / volt, dtype=np.float32)

    # End of Episode 1 / Start of Episode 2: reset
    small_brain.reset(seed=200)
    plasticity.reset_traces()

    # Verify weights were preserved across reset
    w_after_reset = np.array(small_brain.syn.w[:] / volt, dtype=np.float32)
    assert np.allclose(w_after_reset, w_ep1), "Synaptic weights were wiped during episode reset!"

    # Episode 2: continue training (cumulative learning)
    plasticity.step(batch, reward=1.0)
    w_ep2 = np.array(small_brain.syn.w[:] / volt, dtype=np.float32)

    # Ep 2 weights must have advanced further from Ep 1
    assert not np.allclose(w_ep2, w_ep1), "Cumulative weight update did not advance in Episode 2"
    if plasticity.w_init[0] > 0:
        assert w_ep2[0] > w_ep1[0], "Excitatory weight failed to cumulatively increase across episodes"

    # Cleanup
    small_brain.reset_weights_to_initial()
