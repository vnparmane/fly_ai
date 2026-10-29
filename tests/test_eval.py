"""Tests for controls (shuffle, resample) and evaluation statistical reporting (M5)."""

import numpy as np
import pandas as pd
import pytest
import yaml

from flybrain.control.shuffle import resample_io, shuffle_connectome
from flybrain.data.connectome import load_connectome
from flybrain.eval.stats import bootstrap_ci, cohen_d, compute_evaluation_stats
from flybrain.io.select import load_io_spec


@pytest.fixture(scope="module")
def conn():
    return load_connectome(release="783", min_syn=5)


@pytest.fixture(scope="module")
def io_spec():
    return load_io_spec("data/io_spec.json")


def test_shuffle_connectome_preserves_degrees_and_signs(conn):
    # Test on sub-slice to keep test runtime snappy
    from flybrain.brain.subcircuit import extract_subcircuit
    sub_conn, _ = extract_subcircuit(conn, np.arange(10), np.arange(20, 30), max_hops=2, max_neurons=200)

    shuffled = shuffle_connectome(sub_conn, seed=42, n_swaps_factor=10)

    # In-degree unchanged: column sum of nonzero indicators
    orig_in = np.diff(sub_conn.W.tocsc().indptr)
    shuf_in = np.diff(shuffled.W.tocsc().indptr)
    assert np.array_equal(orig_in, shuf_in)

    # Out-degree unchanged: row sum of nonzero indicators
    orig_out = np.diff(sub_conn.W.tocsr().indptr)
    shuf_out = np.diff(shuffled.W.tocsr().indptr)
    assert np.array_equal(orig_out, shuf_out)

    # Sign array unchanged
    assert np.array_equal(sub_conn.sign, shuffled.sign)

    # Edges differ
    diff_edges = (sub_conn.W != shuffled.W).nnz
    assert diff_edges > 0


def test_resample_io(conn, io_spec):
    resampled = resample_io(io_spec, conn, seed=999)

    for k in io_spec.input_groups:
        assert len(resampled.input_groups[k]) == len(io_spec.input_groups[k])

    for k in io_spec.output_groups:
        assert len(resampled.output_groups[k]) == len(io_spec.output_groups[k])

    # Disjointness preserved
    in_left = set(resampled.input_groups["enemy_left"])
    in_right = set(resampled.input_groups["enemy_right"])
    in_aligned = set(resampled.input_groups["enemy_aligned"])
    assert len(in_left.intersection(in_right)) == 0
    assert len(in_left.intersection(in_aligned)) == 0
    assert len(in_right.intersection(in_aligned)) == 0

    out_left = set(resampled.output_groups["move_left"])
    out_right = set(resampled.output_groups["move_right"])
    out_fire = set(resampled.output_groups["fire"])

    assert len(out_left.intersection(out_right)) == 0
    assert len(out_left.intersection(out_fire)) == 0
    assert len(out_right.intersection(out_fire)) == 0


def test_stats_and_bootstrap():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    mean, low, high = bootstrap_ci(data, n_boot=1000)
    assert 5.0 <= mean <= 6.0
    assert low < mean < high

    d = cohen_d(np.array([10, 11, 12]), np.array([1, 2, 3]))
    assert d > 2.0


def test_m5_exact_reproducibility(tmp_path):
    """M5 Gate: Two evaluation runs with same seeds produce identical episodes.parquet."""
    from flybrain.eval.run_eval import run_evaluation
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"
    cfg_path = "configs/default.yaml"
    seeds = [42]
    controllers = ["random", "heuristic", "vnc"]

    run_evaluation(config_path=cfg_path, controllers=controllers, seeds=seeds, out_dir=dir1)
    run_evaluation(config_path=cfg_path, controllers=controllers, seeds=seeds, out_dir=dir2)

    df1 = pd.read_parquet(dir1 / "episodes.parquet")
    df2 = pd.read_parquet(dir2 / "episodes.parquet")

    # Metrics (excluding wall time) must match identically
    metric_cols = ["controller", "seed", "score", "kills", "shots_fired", "steps_survived"]
    pd.testing.assert_frame_equal(df1[metric_cols], df2[metric_cols])
