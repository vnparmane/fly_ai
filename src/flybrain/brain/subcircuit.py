"""Subcircuit extraction for performance scaling (PRD M2b contingency)."""

import logging
from collections import deque
import numpy as np
import scipy.sparse as sp
import pandas as pd

from flybrain.data.connectome import Connectome

logger = logging.getLogger(__name__)


def extract_subcircuit(
    conn: Connectome,
    seeds_in: np.ndarray,
    seeds_out: np.ndarray,
    max_hops: int = 3,
    max_neurons: int = 20000,
) -> tuple[Connectome, np.ndarray]:
    """Extract subcircuit retaining directed paths between input and output seeds.

    Parameters
    ----------
    conn : Connectome
        Full connectome
    seeds_in : np.ndarray
        Array of input neuron indices
    seeds_out : np.ndarray
        Array of output neuron indices
    max_hops : int
        Maximum path length from seeds_in to seeds_out
    max_neurons : int
        Maximum size threshold for the extracted subcircuit

    Returns
    -------
    tuple[Connectome, np.ndarray]
        (sub_connectome, sub_idx_to_full_idx)
    """
    seeds_in = np.asarray(seeds_in, dtype=np.int32)
    seeds_out = np.asarray(seeds_out, dtype=np.int32)

    # Graph adjacency for path search
    # conn.W is W[post, pre]: nonzeros in column 'pre' are postsynaptic targets
    # For forward propagation from inputs, we use W as adjacency (pre -> post)
    # Convert to CSC for fast column slicing (out-edges from pre)
    W_csc = conn.W.tocsc()
    # W_csr has rows as post: fast row slicing gives in-edges (pre -> post)
    W_csr = conn.W.tocsr()

    # Forward BFS from seeds_in
    forward_visited: dict[int, int] = {int(s): 0 for s in seeds_in}
    q_fwd = deque([(int(s), 0) for s in seeds_in])

    while q_fwd:
        u, d = q_fwd.popleft()
        if d < max_hops:
            # Targets of u: nonzero row indices in column u of W
            targets = W_csc.getcol(u).indices
            for v in targets:
                v = int(v)
                if v not in forward_visited or forward_visited[v] > d + 1:
                    forward_visited[v] = d + 1
                    q_fwd.append((v, d + 1))

    # Backward BFS from seeds_out
    backward_visited: dict[int, int] = {int(s): 0 for s in seeds_out}
    q_bwd = deque([(int(s), 0) for s in seeds_out])

    while q_bwd:
        v, d = q_bwd.popleft()
        if d < max_hops:
            # Predecessors of v: nonzero column indices in row v of W
            predecessors = W_csr.getrow(v).indices
            for u in predecessors:
                u = int(u)
                if u not in backward_visited or backward_visited[u] > d + 1:
                    backward_visited[u] = d + 1
                    q_bwd.append((u, d + 1))

    # Path neurons: in both forward and backward reach, with total distance <= max_hops
    sub_neurons_set = set(seeds_in).union(set(seeds_out))
    for node, d_fwd in forward_visited.items():
        if node in backward_visited:
            d_bwd = backward_visited[node]
            if d_fwd + d_bwd <= max_hops:
                sub_neurons_set.add(node)

    # Limit to max_neurons if needed
    if len(sub_neurons_set) > max_neurons:
        # Prioritize seed neurons, then shortest total distance
        sorted_nodes = sorted(
            sub_neurons_set,
            key=lambda x: (
                0 if x in seeds_in or x in seeds_out else 1,
                forward_visited.get(x, 999) + backward_visited.get(x, 999),
            ),
        )
        sub_neurons_set = set(sorted_nodes[:max_neurons])

    sub_idx_to_full_idx = np.array(sorted(sub_neurons_set), dtype=np.int32)
    n_sub = len(sub_idx_to_full_idx)
    logger.info(f"Extracted subcircuit: {n_sub} neurons (from {len(conn.neuron_ids)})")

    # Slice W
    sub_W = conn.W[sub_idx_to_full_idx, :][:, sub_idx_to_full_idx].tocsr()
    sub_neuron_ids = conn.neuron_ids[sub_idx_to_full_idx]
    sub_sign = conn.sign[sub_idx_to_full_idx]
    sub_meta = conn.meta.iloc[sub_idx_to_full_idx].reset_index(drop=True)
    sub_id_to_idx = {int(nid): i for i, nid in enumerate(sub_neuron_ids)}

    sub_conn = Connectome(
        neuron_ids=sub_neuron_ids,
        id_to_idx=sub_id_to_idx,
        W=sub_W,
        sign=sub_sign,
        meta=sub_meta,
    )

    return sub_conn, sub_idx_to_full_idx
