"""Tests for data layer (M1a): Connectome loading and verification."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest

from flybrain.data.connectome import load_connectome, neurons_where, Connectome


@pytest.fixture(scope="module")
def conn():
    # Load with min_syn=5 and cache=True
    return load_connectome(release="783", min_syn=5, cache=True)


def test_neuron_count_and_shape(conn):
    # N is ~139k (138,639)
    assert 138000 <= len(conn.neuron_ids) <= 140000
    N = len(conn.neuron_ids)
    assert conn.W.shape == (N, N)
    assert isinstance(conn.W, sp.csr_matrix)


def test_no_nan_and_sparse_weights(conn):
    assert not np.isnan(conn.W.data).any()
    assert np.all(np.isfinite(conn.W.data))
    assert conn.W.nnz > 1000000


def test_presynaptic_signs(conn):
    assert len(conn.sign) == len(conn.neuron_ids)
    # Every presynaptic neuron has exactly one sign (+1 or -1)
    assert set(np.unique(conn.sign)).issubset({-1, 1})


def test_metadata_columns(conn):
    expected_cols = ["root_id", "super_class", "cell_class", "cell_type", "side", "nt_type", "flow"]
    for col in expected_cols:
        assert col in conn.meta.columns
    assert len(conn.meta) == len(conn.neuron_ids)


def test_spot_check_random_edges(conn):
    # Verify 20 random edges match the raw parquet values
    df_con = pd.read_parquet("data/raw/Connectivity_783.parquet")
    df_filtered = df_con[df_con["Connectivity"] >= 5]
    sample = df_filtered.sample(n=20, random_state=42)

    for _, row in sample.iterrows():
        pre_idx = int(row["Presynaptic_Index"])
        post_idx = int(row["Postsynaptic_Index"])
        expected_weight = float(row["Excitatory x Connectivity"])
        actual_weight = float(conn.W[post_idx, pre_idx])
        assert actual_weight == expected_weight


def test_cache_roundtrip(conn):
    # Re-loading with cache=True should return identical matrix and arrays
    cached_conn = load_connectome(release="783", min_syn=5, cache=True)
    assert np.array_equal(conn.neuron_ids, cached_conn.neuron_ids)
    assert np.array_equal(conn.sign, cached_conn.sign)
    assert (conn.W != cached_conn.W).nnz == 0


def test_neurons_where(conn):
    descending_idx = neurons_where(conn, super_class="descending")
    assert len(descending_idx) > 100
    for idx in descending_idx[:10]:
        assert conn.meta.iloc[idx]["super_class"] == "descending"
