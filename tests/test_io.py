"""Tests for I/O selection and lateral-selectivity sanity test (M3 Gate)."""

from pathlib import Path
import numpy as np
import pytest
import yaml

from flybrain.brain.backend_brian2 import Brian2Brain
from flybrain.brain.lif import LIFParams
from flybrain.brain.subcircuit import extract_subcircuit
from flybrain.data.connectome import load_connectome
from flybrain.io.select import candidate_inputs, candidate_outputs, influence, load_io_spec, select_io


@pytest.fixture(scope="module")
def conn():
    return load_connectome(release="783", min_syn=5)


@pytest.fixture(scope="module")
def io_spec(conn):
    io_file = Path("data/io_spec.json")
    if io_file.exists():
        return load_io_spec(io_file)
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return select_io(conn, cfg)


def test_candidate_pools(conn):
    left_in = candidate_inputs(conn, "left")
    right_in = candidate_inputs(conn, "right")
    dn = candidate_outputs(conn)

    assert len(left_in) > 50
    assert len(right_in) > 50
    assert len(dn) > 500


def test_influence_computation(conn):
    sample_src = np.array([0, 1, 2], dtype=np.int32)
    inf = influence(conn, sample_src, max_hops=3, decay=0.5)
    assert len(inf) == len(conn.neuron_ids)
    assert not np.isnan(inf).any()
    assert abs(inf[0] - 1.0) < 0.05


def test_io_groups_disjoint(io_spec):
    # Output groups must be pairwise disjoint
    out_left = set(io_spec.output_groups["move_left"])
    out_right = set(io_spec.output_groups["move_right"])
    out_fire = set(io_spec.output_groups["fire"])

    assert len(out_left.intersection(out_right)) == 0
    assert len(out_left.intersection(out_fire)) == 0
    assert len(out_right.intersection(out_fire)) == 0


def test_m3_gate_lateral_selectivity(conn, io_spec):
    """M3 Gate: Stimulating enemy_left alone for 500 ms gives move_left rate > move_right rate
    in at least 90% of 20 seeds, mirrored for right.
    """
    in_left = io_spec.input_groups["enemy_left"]
    in_right = io_spec.input_groups["enemy_right"]
    out_left = io_spec.output_groups["move_left"]
    out_right = io_spec.output_groups["move_right"]

    all_seeds_in = np.concatenate([in_left, in_right])
    all_seeds_out = np.concatenate([out_left, out_right])

    # Extract subcircuit for simulation
    sub_conn, sub_to_full = extract_subcircuit(
        conn,
        seeds_in=all_seeds_in,
        seeds_out=all_seeds_out,
        max_hops=3,
        max_neurons=3000,
    )
    full_to_sub = {int(full): sub for sub, full in enumerate(sub_to_full)}

    sub_in_left = np.array([full_to_sub[i] for i in in_left if i in full_to_sub], dtype=np.int32)
    sub_in_right = np.array([full_to_sub[i] for i in in_right if i in full_to_sub], dtype=np.int32)
    sub_out_left = np.array([full_to_sub[i] for i in out_left if i in full_to_sub], dtype=np.int32)
    sub_out_right = np.array([full_to_sub[i] for i in out_right if i in full_to_sub], dtype=np.int32)

    params = LIFParams()
    brain = Brian2Brain(sub_conn, params)

    n_seeds = 20
    stim_rate_hz = 100.0
    duration_ms = 500.0

    left_success = 0
    right_success = 0

    # 1. Stimulate left inputs alone
    for s in range(n_seeds):
        brain.reset(seed=s + 200)
        brain.set_poisson_input(sub_in_left, np.full(len(sub_in_left), stim_rate_hz, dtype=np.float32))
        brain.set_poisson_input(sub_in_right, np.zeros(len(sub_in_right), dtype=np.float32))
        batch = brain.step(duration_ms)

        spk_left = np.sum(np.isin(batch.neuron_idx, sub_out_left))
        spk_right = np.sum(np.isin(batch.neuron_idx, sub_out_right))
        if spk_left >= spk_right:
            left_success += 1

    # 2. Stimulate right inputs alone
    for s in range(n_seeds):
        brain.reset(seed=s + 300)
        brain.set_poisson_input(sub_in_left, np.zeros(len(sub_in_left), dtype=np.float32))
        brain.set_poisson_input(sub_in_right, np.full(len(sub_in_right), stim_rate_hz, dtype=np.float32))
        batch = brain.step(duration_ms)

        spk_left = np.sum(np.isin(batch.neuron_idx, sub_out_left))
        spk_right = np.sum(np.isin(batch.neuron_idx, sub_out_right))
        if spk_right >= spk_left:
            right_success += 1

    left_pass_rate = left_success / n_seeds
    right_pass_rate = right_success / n_seeds

    print(f"\nM3 Gate Results: Left selectivity={left_pass_rate*100:.1f}%, Right selectivity={right_pass_rate*100:.1f}%")
    assert left_pass_rate >= 0.90, f"Left lateral selectivity failed ({left_pass_rate*100:.1f}% < 90%)"
    assert right_pass_rate >= 0.90, f"Right lateral selectivity failed ({right_pass_rate*100:.1f}% < 90%)"
