"""Benchmark script to measure simulation speed per 50 ms step (M1 Gate)."""

import json
import logging
import time
from pathlib import Path
import numpy as np

from flybrain.brain.backend_brian2 import Brian2Brain
from flybrain.brain.lif import LIFParams
from flybrain.brain.subcircuit import extract_subcircuit
from flybrain.data.connectome import load_connectome

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_benchmark():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    benchmark_file = reports_dir / "benchmark.json"

    logger.info("Loading connectome for benchmark...")
    conn = load_connectome(release="783", min_syn=5)

    # Pick representative visual inputs and descending outputs for test
    seeds_in = np.arange(50, dtype=np.int32)
    seeds_out = np.arange(100, 150, dtype=np.int32)

    # Check subcircuit extraction
    sub_conn, _ = extract_subcircuit(conn, seeds_in, seeds_out, max_hops=3, max_neurons=5000)

    params = LIFParams()
    logger.info("Instantiating Brian2Brain on subcircuit...")
    brain = Brian2Brain(sub_conn, params)
    brain.reset(seed=42)

    # Drive input neurons at 50 Hz
    brain.set_poisson_input(np.arange(len(seeds_in)), np.full(len(seeds_in), 50.0, dtype=np.float32))

    # Warm-up step
    logger.info("Running warm-up step...")
    brain.step(50.0)

    # Measure 20 steps
    n_steps = 20
    times = []
    logger.info(f"Benchmarking {n_steps} steps of 50 ms brain time...")
    for _ in range(n_steps):
        t0 = time.perf_counter()
        brain.step(50.0)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    min_ms = float(np.min(times))
    max_ms = float(np.max(times))

    gate_passed = mean_ms <= 250.0

    res = {
        "n_neurons_subcircuit": len(sub_conn.neuron_ids),
        "n_synapses_subcircuit": int(sub_conn.W.nnz),
        "step_duration_ms": 50.0,
        "n_steps": n_steps,
        "wall_time_mean_ms": mean_ms,
        "wall_time_std_ms": std_ms,
        "wall_time_min_ms": min_ms,
        "wall_time_max_ms": max_ms,
        "gate_threshold_ms": 250.0,
        "passed": bool(gate_passed),
    }

    with open(benchmark_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    logger.info(f"Benchmark results: Mean {mean_ms:.2f} ms/step (Gate: <= 250 ms, Passed: {gate_passed})")
    print(f"REPORT: {json.dumps(res, indent=2)}")
    return res


if __name__ == "__main__":
    run_benchmark()
