"""Connectome graph representation and loader."""

from dataclasses import dataclass
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

from flybrain.data.fetch import ensure_flywire

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Connectome:
    neuron_ids: np.ndarray       # int64 [N], sorted root ids
    id_to_idx: dict[int, int]    # root_id -> position in neuron_ids
    W: sp.csr_matrix             # [N, N] float32, W[post, pre] = sign * syn_count
    sign: np.ndarray             # int8 [N], per presynaptic neuron, +1 / -1
    meta: pd.DataFrame           # columns: root_id, super_class, cell_class, cell_type, side, nt_type, flow


def load_connectome(
    release: str = "783",
    min_syn: int = 5,
    cache: bool = True,
    raw_dir: Path = Path("data/raw"),
    cache_dir: Path = Path("data/processed"),
) -> Connectome:
    """Load FlyWire connectome as a signed, CSR synaptic matrix.

    Conventions:
    • W[post, pre] (CSR, float32) holds sign * syn_count, unscaled.
      Input current is W @ spikes.
    • Neuron index i is position in Connectome.neuron_ids (sorted root ids).
    • Sign rule: matches Shiu et al. (ACh=+1, GABA=-1, Glu=-1).
    • Connections with syn_count < min_syn are dropped.

    Parameters
    ----------
    release : str
        FlyWire release identifier (default: "783")
    min_syn : int
        Minimum synapse count threshold (default: 5)
    cache : bool
        Whether to cache the processed Connectome to disk
    raw_dir : Path
        Directory for raw FlyWire files
    cache_dir : Path
        Directory for processed cache files

    Returns
    -------
    Connectome
    """
    raw_dir = Path(raw_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / f"connectome_{release}_minsyn{min_syn}.npz"
    meta_cache_file = cache_dir / f"meta_{release}.parquet"

    if cache and cache_file.exists() and meta_cache_file.exists():
        logger.info(f"Loading cached connectome from {cache_file}")
        npz = np.load(cache_file)
        neuron_ids = npz["neuron_ids"]
        sign = npz["sign"]
        W = sp.csr_matrix(
            (npz["data"], npz["indices"], npz["indptr"]),
            shape=tuple(npz["shape"]),
            dtype=np.float32,
        )
        meta = pd.read_parquet(meta_cache_file)
        id_to_idx = {int(nid): i for i, nid in enumerate(neuron_ids)}
        N = len(neuron_ids)
        logger.info(f"Loaded connectome from cache: N={N}, nnz={W.nnz}")
        return Connectome(
            neuron_ids=neuron_ids,
            id_to_idx=id_to_idx,
            W=W,
            sign=sign,
            meta=meta,
        )

    # Fetch raw data files
    files = ensure_flywire(release=release, dest=raw_dir)

    # 1. Load neuron IDs from Completeness
    logger.info("Loading Completeness table...")
    df_comp = pd.read_csv(files[f"Completeness_{release}.csv"], index_col=0)
    neuron_ids = np.array(df_comp.index.values, dtype=np.int64)
    N = len(neuron_ids)
    id_to_idx = {int(nid): i for i, nid in enumerate(neuron_ids)}
    logger.info(f"Total neurons N={N}")

    # 2. Load and filter connectivity edges
    logger.info("Loading Connectivity parquet...")
    df_con = pd.read_parquet(files[f"Connectivity_{release}.parquet"])
    if min_syn > 0:
        df_con = df_con[df_con["Connectivity"] >= min_syn]

    pre_idx = df_con["Presynaptic_Index"].values
    post_idx = df_con["Postsynaptic_Index"].values
    # Excitatory x Connectivity gives sign * syn_count
    # Preserves: W[post, pre] = sign * syn_count
    weights = df_con["Excitatory x Connectivity"].values.astype(np.float32)

    logger.info("Constructing sparse CSR matrix W[post, pre]...")
    W = sp.csr_matrix(
        (weights, (post_idx, pre_idx)),
        shape=(N, N),
        dtype=np.float32,
    )
    # Ensure canonical sorted format
    W.sort_indices()

    # 3. Determine sign per presynaptic neuron (+1 / -1)
    sign = np.zeros(N, dtype=np.int8)
    # Extract known signs directly from Excitatory column
    pre_unique_df = df_con.drop_duplicates(subset=["Presynaptic_Index"])
    sign[pre_unique_df["Presynaptic_Index"].values] = pre_unique_df["Excitatory"].values.astype(np.int8)

    # For zero-outdegree neurons or neurons not in filtered edges, fill from annotations table
    anno_file = files.get("Supplemental_file1_neuron_annotations.tsv")
    df_anno = pd.read_csv(anno_file, sep="\t", low_memory=False)
    nt_map = {
        "acetylcholine": 1,
        "dopamine": 1,
        "octopamine": 1,
        "gaba": -1,
        "glutamate": -1,
        "serotonin": -1,
        "histamine": -1,
    }
    anno_dict = df_anno.set_index("root_id")["top_nt"].to_dict()

    unassigned_mask = (sign == 0)
    for idx in np.where(unassigned_mask)[0]:
        nid = neuron_ids[idx]
        nt = anno_dict.get(nid)
        if isinstance(nt, str):
            sign[idx] = nt_map.get(nt.lower(), 1)
        else:
            sign[idx] = 1  # default to excitatory if completely unknown

    # 4. Construct standardized meta DataFrame
    anno_sub = df_anno.set_index("root_id")
    anno_reindexed = anno_sub.reindex(neuron_ids)
    meta = pd.DataFrame(
        {
            "root_id": neuron_ids,
            "super_class": anno_reindexed["super_class"].fillna("unknown").values,
            "cell_class": anno_reindexed["cell_class"].fillna("unknown").values,
            "cell_type": anno_reindexed["cell_type"].fillna("unknown").values,
            "side": anno_reindexed["side"].fillna("unknown").values,
            "nt_type": anno_reindexed["top_nt"].fillna("unknown").values,
            "flow": anno_reindexed["flow"].fillna("unknown").values,
        }
    )

    if cache:
        logger.info(f"Saving connectome cache to {cache_file}...")
        np.savez_compressed(
            cache_file,
            neuron_ids=neuron_ids,
            sign=sign,
            data=W.data,
            indices=W.indices,
            indptr=W.indptr,
            shape=W.shape,
        )
        meta.to_parquet(meta_cache_file)

    logger.info(f"Connectome loaded successfully: N={N}, nnz={W.nnz}")
    return Connectome(
        neuron_ids=neuron_ids,
        id_to_idx=id_to_idx,
        W=W,
        sign=sign,
        meta=meta,
    )


def neurons_where(conn: Connectome, **filters) -> np.ndarray:
    """Filter neuron indices by metadata column values.

    Example:
        neurons_where(conn, super_class='descending', side='right')

    Returns
    -------
    np.ndarray (int32)
        Indices of matching neurons in Connectome
    """
    mask = pd.Series(True, index=conn.meta.index)
    for col, val in filters.items():
        if col not in conn.meta.columns:
            raise KeyError(f"Column '{col}' not found in connectome metadata")
        if isinstance(val, (list, tuple, set, np.ndarray)):
            mask &= conn.meta[col].isin(val)
        else:
            mask &= (conn.meta[col] == val)
    return np.where(mask.values)[0].astype(np.int32)
