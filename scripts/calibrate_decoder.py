"""Calibrate decoder baseline thresholds on calibration seeds (1000-1019) with zero input."""

import logging
from pathlib import Path
import numpy as np
import yaml

from flybrain.brain.backend_brian2 import Brian2Brain
from flybrain.brain.lif import LIFParams
from flybrain.brain.subcircuit import extract_subcircuit
from flybrain.data.connectome import load_connectome
from flybrain.io.select import load_io_spec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calibrate_decoder():
    config_path = Path("configs/default.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    calib_seeds = cfg["seeds"]["calibration"]
    io_spec = load_io_spec(Path("data/io_spec.json"))
    conn = load_connectome(release="783", min_syn=5)

    all_in = np.concatenate(list(io_spec.input_groups.values()))
    all_out = np.concatenate(list(io_spec.output_groups.values()))

    # Build subcircuit
    sub_conn, sub_to_full = extract_subcircuit(
        conn,
        seeds_in=all_in,
        seeds_out=all_out,
        max_hops=3,
        max_neurons=3000,
    )
    full_to_sub = {int(full): sub for sub, full in enumerate(sub_to_full)}

    out_groups_sub = {
        name: np.array([full_to_sub[i] for i in grp if i in full_to_sub], dtype=np.int32)
        for name, grp in io_spec.output_groups.items()
    }

    params = LIFParams()
    brain = Brian2Brain(sub_conn, params)

    rates_left = []
    rates_right = []
    rates_fire = []

    logger.info(f"Running baseline calibration on {len(calib_seeds)} seeds with ZERO input...")
    step_duration_ms = 50.0
    steps_per_trial = 10  # 500 ms total per trial

    for s in calib_seeds:
        brain.reset(seed=s)
        total_spikes_left = 0
        total_spikes_right = 0
        total_spikes_fire = 0

        for _ in range(steps_per_trial):
            batch = brain.step(step_duration_ms)
            total_spikes_left += np.sum(np.isin(batch.neuron_idx, out_groups_sub["move_left"]))
            total_spikes_right += np.sum(np.isin(batch.neuron_idx, out_groups_sub["move_right"]))
            total_spikes_fire += np.sum(np.isin(batch.neuron_idx, out_groups_sub["fire"]))

        trial_sec = (step_duration_ms * steps_per_trial) / 1000.0
        r_l = total_spikes_left / (max(1, len(out_groups_sub["move_left"])) * trial_sec)
        r_r = total_spikes_right / (max(1, len(out_groups_sub["move_right"])) * trial_sec)
        r_f = total_spikes_fire / (max(1, len(out_groups_sub["fire"])) * trial_sec)

        rates_left.append(r_l)
        rates_right.append(r_r)
        rates_fire.append(r_f)

    # Threshold rule: mean + 3 * std (at least 0.5 Hz)
    thresh_left = float(max(0.5, np.mean(rates_left) + 3.0 * np.std(rates_left)))
    thresh_right = float(max(0.5, np.mean(rates_right) + 3.0 * np.std(rates_right)))
    thresh_fire = float(max(0.5, np.mean(rates_fire) + 3.0 * np.std(rates_fire)))

    logger.info(f"Calibrated thresholds: left={thresh_left:.2f} Hz, right={thresh_right:.2f} Hz, fire={thresh_fire:.2f} Hz")

    cfg["decoder"]["thresh_hz"]["move_left"] = thresh_left
    cfg["decoder"]["thresh_hz"]["move_right"] = thresh_right
    cfg["decoder"]["thresh_hz"]["fire"] = thresh_fire

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print("Decoder thresholds successfully updated and frozen in configs/default.yaml")


if __name__ == "__main__":
    calibrate_decoder()
