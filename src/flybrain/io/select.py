"""I/O Selection using signed-influence connectivity analysis."""

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
import numpy as np

from flybrain.data.connectome import Connectome, neurons_where

logger = logging.getLogger(__name__)


@dataclass
class IOSpec:
    input_groups: dict[str, np.ndarray]    # keys: enemy_left, enemy_right, enemy_aligned
    output_groups: dict[str, np.ndarray]   # keys: move_left, move_right, fire
    justification: dict[str, dict]         # per group: criteria, scores, neuron types, counts
    config_hash: str


def candidate_inputs(conn: Connectome, side: str) -> np.ndarray:
    """Select candidate visual projection neurons (looming / motion projection) for a side.
    LC4, LPLC2, LC10, LC16, LC17, LC18 are known Drosophila looming and visual projection types.
    """
    looming_types = ["LC4", "LPLC2", "LC10a", "LC10b", "LC10c", "LC10d", "LC16", "LC17", "LC18", "LC12"]
    indices = neurons_where(conn, super_class="visual_projection", side=side, cell_type=looming_types)
    if len(indices) < 20:
        # Fallback to all visual projection neurons on that side
        indices = neurons_where(conn, super_class="visual_projection", side=side)
    return indices


def candidate_outputs(conn: Connectome) -> np.ndarray:
    """Select descending neurons (super_class == 'descending')."""
    return neurons_where(conn, super_class="descending")


def influence(
    conn: Connectome,
    source_idx: np.ndarray,
    max_hops: int = 4,
    decay: float = 0.5,
) -> np.ndarray:
    """Compute signed forward influence from source neurons to all neurons.

    Propagation follows pre -> post synaptic directions:
    In our convention W[post, pre], the forward step from pre to post is:
        x_post = W @ x_pre

    Parameters
    ----------
    conn : Connectome
        Connectome with W[post, pre]
    source_idx : np.ndarray
        Indices of stimulating source neurons
    max_hops : int
        Number of propagation hops
    decay : float
        Decay factor per hop

    Returns
    -------
    np.ndarray (float32 [N])
        Signed influence vector
    """
    N = len(conn.neuron_ids)
    score = np.zeros(N, dtype=np.float32)
    score[source_idx] = 1.0

    current = np.zeros(N, dtype=np.float32)
    current[source_idx] = 1.0

    for hop in range(max_hops):
        # Forward step: post receives current from pre
        # Note: W is W[post, pre]
        step_signal = conn.W @ current
        # Normalize step_signal by max absolute value to keep numerical stability
        max_val = np.max(np.abs(step_signal))
        if max_val > 0:
            step_signal = step_signal / max_val
        current = step_signal
        score += (decay ** (hop + 1)) * current

    return score


def select_io(conn: Connectome, cfg: dict) -> IOSpec:
    """Select enemy sensory input groups and descending motor output groups."""
    io_cfg = cfg.get("io", {})
    top_k = int(io_cfg.get("top_k", 50))
    min_group = int(io_cfg.get("min_group", 10))
    max_group = int(io_cfg.get("max_group", 200))

    cfg_str = json.dumps(cfg, sort_keys=True)
    config_hash = hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:16]

    # 1. Candidate Inputs
    # Left visual field inputs (threat on left) -> left visual projection neurons
    # Right visual field inputs (threat on right) -> right visual projection neurons
    cand_in_left = candidate_inputs(conn, "left")
    cand_in_right = candidate_inputs(conn, "right")

    # Select top_k inputs for left and right
    input_left = cand_in_left[: min(top_k, len(cand_in_left))]
    input_right = cand_in_right[: min(top_k, len(cand_in_right))]
    # enemy_aligned: bilateral sample from center/both eyes
    input_aligned = np.concatenate([input_left[: top_k // 2], input_right[: top_k // 2]])

    # 2. Forward influence of sensory inputs
    inf_left = influence(conn, input_left, max_hops=4, decay=0.5)
    inf_right = influence(conn, input_right, max_hops=4, decay=0.5)
    inf_aligned = influence(conn, input_aligned, max_hops=4, decay=0.5)

    # 3. Candidate Outputs (descending neurons)
    dn_candidates = candidate_outputs(conn)

    # Lateral differential score for steering
    diff_left = inf_left[dn_candidates] - inf_right[dn_candidates]
    diff_right = inf_right[dn_candidates] - inf_left[dn_candidates]
    score_fire = inf_aligned[dn_candidates] - np.abs(diff_left)

    # Sort candidates
    sort_left = dn_candidates[np.argsort(-diff_left)]
    sort_right = dn_candidates[np.argsort(-diff_right)]
    sort_fire = dn_candidates[np.argsort(-score_fire)]

    # Pick disjoint groups
    group_size = int(np.clip(top_k, min_group, max_group))
    selected = set()

    move_left_group = []
    for idx in sort_left:
        if idx not in selected and len(move_left_group) < group_size:
            move_left_group.append(int(idx))
            selected.add(int(idx))

    move_right_group = []
    for idx in sort_right:
        if idx not in selected and len(move_right_group) < group_size:
            move_right_group.append(int(idx))
            selected.add(int(idx))

    fire_group = []
    for idx in sort_fire:
        if idx not in selected and len(fire_group) < group_size:
            fire_group.append(int(idx))
            selected.add(int(idx))

    input_groups = {
        "enemy_left": np.array(input_left, dtype=np.int32),
        "enemy_right": np.array(input_right, dtype=np.int32),
        "enemy_aligned": np.array(input_aligned, dtype=np.int32),
    }

    output_groups = {
        "move_left": np.array(move_left_group, dtype=np.int32),
        "move_right": np.array(move_right_group, dtype=np.int32),
        "fire": np.array(fire_group, dtype=np.int32),
    }

    justification = {
        "enemy_left": {
            "criteria": "Left visual projection neurons (looming / motion types)",
            "count": len(input_left),
            "neuron_types": list(conn.meta.iloc[input_left]["cell_type"].unique()[:5]),
        },
        "enemy_right": {
            "criteria": "Right visual projection neurons (looming / motion types)",
            "count": len(input_right),
            "neuron_types": list(conn.meta.iloc[input_right]["cell_type"].unique()[:5]),
        },
        "enemy_aligned": {
            "criteria": "Bilateral combination of left and right visual projection neurons",
            "count": len(input_aligned),
            "neuron_types": list(conn.meta.iloc[input_aligned]["cell_type"].unique()[:5]),
        },
        "move_left": {
            "criteria": "Descending neurons with maximum differential influence: inf(left) - inf(right)",
            "count": len(move_left_group),
            "top_margin": float(np.max(diff_left)),
        },
        "move_right": {
            "criteria": "Descending neurons with maximum differential influence: inf(right) - inf(left)",
            "count": len(move_right_group),
            "top_margin": float(np.max(diff_right)),
        },
        "fire": {
            "criteria": "Descending neurons activated by aligned input with minimal lateral asymmetry",
            "count": len(fire_group),
            "top_margin": float(np.max(score_fire)),
        },
    }

    return IOSpec(
        input_groups=input_groups,
        output_groups=output_groups,
        justification=justification,
        config_hash=config_hash,
    )


def save_io_spec(spec: IOSpec, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "config_hash": spec.config_hash,
        "input_groups": {k: v.tolist() for k, v in spec.input_groups.items()},
        "output_groups": {k: v.tolist() for k, v in spec.output_groups.items()},
        "justification": spec.justification,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_io_spec(path: Path) -> IOSpec:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return IOSpec(
        input_groups={k: np.array(v, dtype=np.int32) for k, v in data["input_groups"].items()},
        output_groups={k: np.array(v, dtype=np.int32) for k, v in data["output_groups"].items()},
        justification=data["justification"],
        config_hash=data["config_hash"],
    )
