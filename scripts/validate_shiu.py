"""Validation script to reproduce the Shiu et al. (2024) sugar-sensing -> MN9 result (M1 Gate)."""

import json
import logging
from pathlib import Path
import numpy as np
import scipy.stats as stats

from flybrain.brain.backend_brian2 import Brian2Brain
from flybrain.brain.lif import LIFParams
from flybrain.brain.subcircuit import extract_subcircuit
from flybrain.data.connectome import load_connectome

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sugar-sensing gustatory receptor neurons from Shiu et al. (example.ipynb)
SUGAR_GRN_ROOT_IDS = [
    720575940624963786,
    720575940630233916,
    720575940637568838,
    720575940638202345,
    720575940617000768,
    720575940630797113,
    720575940632889389,
    720575940621754367,
    720575940621502051,
    720575940640649691,
    720575940639332736,
    720575940616885538,
    720575940639198653,
    720575940620900446,
    720575940617937543,
    720575940632425919,
    720575940633143833,
    720575940612670570,
    720575940628853239,
    720575940629176663,
    720575940611875570,
]

# Feeding motor neuron MN9 (proboscis extension)
MN9_ROOT_ID = 720575940660219265


def validate_shiu():
    logger.info("Loading connectome...")
    conn = load_connectome(release="783", min_syn=5)

    grn_indices = np.array([conn.id_to_idx[rid] for rid in SUGAR_GRN_ROOT_IDS if rid in conn.id_to_idx], dtype=np.int32)
    mn9_idx = conn.id_to_idx[MN9_ROOT_ID]

    logger.info(f"Sugar GRNs found: {len(grn_indices)}/{len(SUGAR_GRN_ROOT_IDS)}, MN9 idx: {mn9_idx}")

    # Extract subcircuit connecting sugar GRNs to MN9
    sub_conn, sub_to_full = extract_subcircuit(
        conn,
        seeds_in=grn_indices,
        seeds_out=np.array([mn9_idx], dtype=np.int32),
        max_hops=3,
        max_neurons=4000,
    )
    full_to_sub = {int(full): sub for sub, full in enumerate(sub_to_full)}
    sub_grns = np.array([full_to_sub[i] for i in grn_indices if i in full_to_sub], dtype=np.int32)
    sub_mn9 = full_to_sub[mn9_idx]

    params = LIFParams()
    brain = Brian2Brain(sub_conn, params)

    n_seeds = 30
    duration_ms = 500.0  # 500 ms trial
    stim_rate_hz = 100.0  # 100 Hz sugar stimulation

    baseline_mn9_spikes = []
    stimulated_mn9_spikes = []
    stimulated_rates_all = []

    logger.info(f"Running baseline trials ({n_seeds} seeds)...")
    for s in range(n_seeds):
        brain.reset(seed=s)
        brain.set_poisson_input(sub_grns, np.zeros(len(sub_grns), dtype=np.float32))
        batch = brain.step(duration_ms)
        mn9_spikes = np.sum(batch.neuron_idx == sub_mn9)
        baseline_mn9_spikes.append(mn9_spikes)

    logger.info(f"Running stimulated trials at {stim_rate_hz} Hz ({n_seeds} seeds)...")
    for s in range(n_seeds):
        brain.reset(seed=s + 100)
        brain.set_poisson_input(sub_grns, np.full(len(sub_grns), stim_rate_hz, dtype=np.float32))
        batch = brain.step(duration_ms)
        mn9_spikes = np.sum(batch.neuron_idx == sub_mn9)
        stimulated_mn9_spikes.append(mn9_spikes)

        # Record per-neuron spike counts for correlation
        counts = np.bincount(batch.neuron_idx, minlength=sub_conn.W.shape[0])
        stimulated_rates_all.append(counts / (duration_ms / 1000.0))

    mean_base = float(np.mean(baseline_mn9_spikes))
    mean_stim = float(np.mean(stimulated_mn9_spikes))
    logger.info(f"MN9 spikes in 500ms: baseline={mean_base:.2f}, stimulated={mean_stim:.2f}")

    # Mann-Whitney U test
    u_stat, p_val = stats.mannwhitneyu(stimulated_mn9_spikes, baseline_mn9_spikes, alternative="greater")
    logger.info(f"Mann-Whitney U: U={u_stat}, p-value={p_val:.4e}")

    # Mean firing rates across seeds
    mean_rates = np.mean(stimulated_rates_all, axis=0)
    # Check self-consistency across odd and even seeds (Spearman rho)
    rates_odd = np.mean(stimulated_rates_all[0::2], axis=0)
    rates_even = np.mean(stimulated_rates_all[1::2], axis=0)
    spearman_rho, _ = stats.spearmanr(rates_odd, rates_even)
    logger.info(f"Spearman rho across trials: {spearman_rho:.4f}")

    gate_passed = (p_val < 0.05) and (spearman_rho >= 0.90)

    report = {
        "n_seeds": n_seeds,
        "duration_ms": duration_ms,
        "stim_rate_hz": stim_rate_hz,
        "mn9_baseline_mean_spikes": mean_base,
        "mn9_stimulated_mean_spikes": mean_stim,
        "mann_whitney_p": float(p_val),
        "spearman_rho": float(spearman_rho),
        "passed": bool(gate_passed),
    }

    out_path = Path("reports/validate_shiu.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Validation result: {report}")
    assert gate_passed, f"Shiu validation gate failed: p={p_val}, rho={spearman_rho}"
    print("Shiu validation PASSED!")


if __name__ == "__main__":
    validate_shiu()
