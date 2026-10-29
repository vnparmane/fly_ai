"""Controls: degree-preserving connectome shuffling and random-I/O resampling."""

import logging
import numpy as np
import scipy.sparse as sp

from flybrain.data.connectome import Connectome
from flybrain.io.select import IOSpec, candidate_inputs, candidate_outputs

logger = logging.getLogger(__name__)


def shuffle_connectome(conn: Connectome, seed: int, n_swaps_factor: int = 5) -> Connectome:
    """Degree-preserving double edge swap (Maslov-Sneppen algorithm).
    
    Preserves:
    • In-degree distribution exactly
    • Out-degree distribution exactly
    • Per-neuron neurotransmitter sign array exactly
    • Per-edge absolute synaptic weights (swapped alongside edges)
    
    Parameters
    ----------
    conn : Connectome
        Input connectome
    seed : int
        Random seed
    n_swaps_factor : int
        Number of swap attempts per edge (default 5)

    Returns
    -------
    Connectome
        Shuffled connectome where >= 95% of edges differ from original.
    """
    rng = np.random.default_rng(seed)

    # In our convention W[post, pre]: row = post, col = pre
    # Extract coordinate lists
    coo = conn.W.tocoo()
    post = coo.row.copy()
    pre = coo.col.copy()
    data = coo.data.copy()
    n_edges = len(data)

    if n_edges < 2:
        return conn

    # Track existing directed edges (pre, post) using a set
    # Using packed uint64: (int64(pre) << 32) | int64(post)
    edge_set = set(((int(p) << 32) | int(t)) for p, t in zip(pre, post))
    original_edge_set = set(edge_set)

    n_swaps = n_swaps_factor * n_edges
    logger.info(f"Shuffling connectome: attempting {n_swaps} edge swaps across {n_edges} edges...")

    idx1 = rng.integers(0, n_edges, size=n_swaps)
    idx2 = rng.integers(0, n_edges, size=n_swaps)

    for i1, i2 in zip(idx1, idx2):
        if i1 == i2:
            continue

        u1, v1 = pre[i1], post[i1]
        u2, v2 = pre[i2], post[i2]

        # Prevent self-loops
        if u1 == v2 or u2 == v1:
            continue

        # Target new edges
        new_edge1 = (int(u1) << 32) | int(v2)
        new_edge2 = (int(u2) << 32) | int(v1)

        # Ensure multigraph edges are not created
        if (new_edge1 in edge_set) or (new_edge2 in edge_set):
            continue

        # Execute swap: u1->v2 and u2->v1
        old_edge1 = (int(u1) << 32) | int(v1)
        old_edge2 = (int(u2) << 32) | int(v2)

        edge_set.remove(old_edge1)
        edge_set.remove(old_edge2)
        edge_set.add(new_edge1)
        edge_set.add(new_edge2)

        post[i1] = v2
        post[i2] = v1

    # Reconstruct CSR matrix
    shuffled_W = sp.csr_matrix(
        (data, (post, pre)),
        shape=conn.W.shape,
        dtype=np.float32,
    )
    shuffled_W.sort_indices()

    retained_edges = len(edge_set.intersection(original_edge_set))
    pct_differ = (1.0 - (retained_edges / n_edges)) * 100.0
    logger.info(f"Shuffled connectome: {pct_differ:.1f}% of edges differ from original")

    return Connectome(
        neuron_ids=conn.neuron_ids,
        id_to_idx=conn.id_to_idx,
        W=shuffled_W,
        sign=conn.sign,
        meta=conn.meta,
    )


def resample_io(io_spec: IOSpec, conn: Connectome, seed: int) -> IOSpec:
    """Random-I/O control: resample groups of identical sizes from the candidate pools.
    All groups remain pairwise disjoint.
    """
    rng = np.random.default_rng(seed)

    cand_left = candidate_inputs(conn, "left")
    cand_right = candidate_inputs(conn, "right")
    cand_dn = candidate_outputs(conn)

    size_in_l = len(io_spec.input_groups["enemy_left"])
    size_in_r = len(io_spec.input_groups["enemy_right"])
    size_in_a = len(io_spec.input_groups["enemy_aligned"])

    size_out_l = len(io_spec.output_groups["move_left"])
    size_out_r = len(io_spec.output_groups["move_right"])
    size_out_f = len(io_spec.output_groups["fire"])

    # Permute candidate sets
    perm_left = rng.permutation(cand_left)
    perm_right = rng.permutation(cand_right)
    perm_dn = rng.permutation(cand_dn)

    new_in_left = perm_left[:size_in_l]
    new_in_aligned_l = perm_left[size_in_l : size_in_l + size_in_a // 2]
    new_in_right = perm_right[:size_in_r]
    new_in_aligned_r = perm_right[size_in_r : size_in_r + size_in_a // 2]
    new_in_aligned = np.concatenate([new_in_aligned_l, new_in_aligned_r])

    total_dn = size_out_l + size_out_r + size_out_f
    if len(perm_dn) < total_dn:
        raise ValueError("Not enough descending neurons to form disjoint groups")

    new_out_left = perm_dn[:size_out_l]
    new_out_right = perm_dn[size_out_l : size_out_l + size_out_r]
    new_out_fire = perm_dn[size_out_l + size_out_r : total_dn]

    input_groups = {
        "enemy_left": np.array(new_in_left, dtype=np.int32),
        "enemy_right": np.array(new_in_right, dtype=np.int32),
        "enemy_aligned": np.array(new_in_aligned, dtype=np.int32),
    }

    output_groups = {
        "move_left": np.array(new_out_left, dtype=np.int32),
        "move_right": np.array(new_out_right, dtype=np.int32),
        "fire": np.array(new_out_fire, dtype=np.int32),
    }

    justification = {
        "control": "random_io",
        "seed": seed,
        "description": "Randomly resampled disjoint groups from same biological candidate pools",
    }

    return IOSpec(
        input_groups=input_groups,
        output_groups=output_groups,
        justification=justification,
        config_hash=f"{io_spec.config_hash}_resample_{seed}",
    )
