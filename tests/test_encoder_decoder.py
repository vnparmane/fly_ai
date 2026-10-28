"""Tests for Encoder, Decoder, and BrainController (M4 Gate)."""

import numpy as np
import pytest
import yaml

from flybrain.brain.backend_brian2 import Brian2Brain
from flybrain.brain.lif import LIFParams, SpikeBatch
from flybrain.brain.subcircuit import extract_subcircuit
from flybrain.control.brain_ctrl import BrainController
from flybrain.data.connectome import load_connectome
from flybrain.env.invaders import InvadersConfig, SpaceInvadersEnv
from flybrain.io.decoder import Decoder, DecoderConfig
from flybrain.io.encoder import Encoder, EncoderConfig
from flybrain.io.select import load_io_spec


@pytest.fixture(scope="module")
def io_spec():
    return load_io_spec("data/io_spec.json")


def test_encoder_rates(io_spec):
    enc = Encoder(io_spec, EncoderConfig(max_rate_hz=100.0, align_sigma=0.15))

    # Center target: target_dx = 0
    obs_center = {"target_dx": np.array([0.0], dtype=np.float32)}
    idx, rates = enc(obs_center)
    # Aligned rate should be at max, left/right should be 0
    n_left = len(io_spec.input_groups["enemy_left"])
    n_right = len(io_spec.input_groups["enemy_right"])
    assert np.allclose(rates[:n_left], 0.0)
    assert np.allclose(rates[n_left : n_left + n_right], 0.0)
    assert np.allclose(rates[n_left + n_right :], 100.0)

    # Left target: target_dx = -0.5
    obs_left = {"target_dx": np.array([-0.5], dtype=np.float32)}
    _, rates_l = enc(obs_left)
    assert rates_l[0] == 50.0  # left rate > 0
    assert rates_l[n_left] == 0.0  # right rate == 0


def test_decoder_synthetic_spikes(io_spec):
    cfg = DecoderConfig(
        window_ms=100.0,
        margin_hz=2.0,
        thresh_hz={"move_left": 5.0, "move_right": 5.0, "fire": 5.0},
    )
    dec = Decoder(io_spec, cfg)

    # Synthetic left spikes across all left neurons to exceed 5 Hz
    left_nids = np.array(list(io_spec.output_groups["move_left"]), dtype=np.int32)
    spk_batch_left = SpikeBatch(
        t_ms=np.linspace(10, 90, len(left_nids) * 2, dtype=np.float32),
        neuron_idx=np.repeat(left_nids, 2),
        duration_ms=100.0,
    )
    act_left = dec(spk_batch_left)
    assert act_left[0] == 1  # left active
    assert act_left[1] == 0  # right inactive

    # Reset clears history
    dec.reset()
    empty_batch = SpikeBatch(t_ms=np.array([]), neuron_idx=np.array([]), duration_ms=100.0)
    act_empty = dec(empty_batch)
    assert np.all(act_empty == 0)


def test_m4_gate_deterministic_episode(io_spec):
    """M4 Gate: One full episode runs end to end deterministically with identical seed."""
    conn = load_connectome(release="783", min_syn=5)
    all_in = np.concatenate(list(io_spec.input_groups.values()))
    all_out = np.concatenate(list(io_spec.output_groups.values()))

    # Build subcircuit for game play
    sub_conn, sub_to_full = extract_subcircuit(
        conn,
        seeds_in=all_in,
        seeds_out=all_out,
        max_hops=3,
        max_neurons=1500,
    )
    full_to_sub = {int(full): sub for sub, full in enumerate(sub_to_full)}

    # Map IOSpec groups to subcircuit indices
    io_sub = load_io_spec("data/io_spec.json")
    for k in io_sub.input_groups:
        io_sub.input_groups[k] = np.array([full_to_sub[i] for i in io_sub.input_groups[k] if i in full_to_sub], dtype=np.int32)
    for k in io_sub.output_groups:
        io_sub.output_groups[k] = np.array([full_to_sub[i] for i in io_sub.output_groups[k] if i in full_to_sub], dtype=np.int32)

    params = LIFParams()
    brain = Brian2Brain(sub_conn, params)
    enc = Encoder(io_sub)
    dec = Decoder(io_sub)
    ctrl = BrainController(brain, enc, dec, step_ms=50.0)

    env = SpaceInvadersEnv(InvadersConfig(max_steps=20))

    # Run episode 1 with seed 42
    obs, _ = env.reset(seed=42)
    ctrl.reset(seed=42)
    actions_run1 = []
    for _ in range(20):
        act = ctrl.act(obs)
        actions_run1.append(act.copy())
        obs, _, term, trunc, _ = env.step(act)
        if term or trunc:
            break

    # Run episode 2 with identical seed 42
    obs, _ = env.reset(seed=42)
    ctrl.reset(seed=42)
    actions_run2 = []
    for _ in range(20):
        act = ctrl.act(obs)
        actions_run2.append(act.copy())
        obs, _, term, trunc, _ = env.step(act)
        if term or trunc:
            break

    assert len(actions_run1) == len(actions_run2)
    for a1, a2 in zip(actions_run1, actions_run2):
        assert np.array_equal(a1, a2)

    print(f"\nM4 Gate Passed: Completed deterministic {len(actions_run1)}-step episode!")
