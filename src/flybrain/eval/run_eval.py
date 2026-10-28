"""Evaluation harness to run standardized controller benchmark experiments."""

import argparse
import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from flybrain.brain.backend_brian2 import Brian2Brain
from flybrain.brain.lif import LIFParams
from flybrain.brain.subcircuit import extract_subcircuit
from flybrain.control.brain_ctrl import BrainController
from flybrain.control.heuristic import HeuristicController
from flybrain.control.random_ctrl import RandomController
from flybrain.control.shuffle import resample_io, shuffle_connectome
from flybrain.data.connectome import load_connectome
from flybrain.env.invaders import InvadersConfig, SpaceInvadersEnv
from flybrain.eval.stats import compute_evaluation_stats
from flybrain.io.decoder import Decoder, DecoderConfig
from flybrain.io.encoder import Encoder, EncoderConfig
from flybrain.io.select import load_io_spec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "unknown"


def build_brain_controller(conn, io_spec, cfg, max_sub_neurons=1500):
    all_in = np.concatenate(list(io_spec.input_groups.values()))
    all_out = np.concatenate(list(io_spec.output_groups.values()))

    sub_conn, sub_to_full = extract_subcircuit(
        conn,
        seeds_in=all_in,
        seeds_out=all_out,
        max_hops=3,
        max_neurons=max_sub_neurons,
    )
    full_to_sub = {int(full): sub for sub, full in enumerate(sub_to_full)}

    io_sub = load_io_spec("data/io_spec.json")
    for k in io_sub.input_groups:
        io_sub.input_groups[k] = np.array([full_to_sub[i] for i in io_spec.input_groups[k] if i in full_to_sub], dtype=np.int32)
    for k in io_sub.output_groups:
        io_sub.output_groups[k] = np.array([full_to_sub[i] for i in io_spec.output_groups[k] if i in full_to_sub], dtype=np.int32)

    params = LIFParams()
    brain = Brian2Brain(sub_conn, params)
    enc_cfg = EncoderConfig(
        max_rate_hz=float(cfg["encoder"]["max_rate_hz"]),
        dx_range=float(cfg["encoder"]["dx_range"]),
        align_sigma=float(cfg["encoder"]["align_sigma"]),
    )
    dec_cfg = DecoderConfig(
        window_ms=float(cfg["decoder"]["window_ms"]),
        margin_hz=float(cfg["decoder"]["margin_hz"]),
        thresh_hz={k: float(v) for k, v in cfg["decoder"]["thresh_hz"].items()},
    )
    enc = Encoder(io_sub, enc_cfg)
    dec = Decoder(io_sub, dec_cfg)
    return BrainController(brain, enc, dec, step_ms=float(cfg["eval"]["step_ms"]))


def run_evaluation(config_path: Path, controllers: list[str], seeds: list[int], out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg_str = json.dumps(cfg, sort_keys=True)
    config_hash = hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:16]

    io_spec = load_io_spec(Path("data/io_spec.json"))
    manifest_path = Path("data/raw/MANIFEST.json")
    manifest = json.load(open(manifest_path, "r", encoding="utf-8")) if manifest_path.exists() else {}

    env_cfg = InvadersConfig(
        width=int(cfg["env"]["width"]),
        height=int(cfg["env"]["height"]),
        player_speed=float(cfg["env"]["player_speed"]),
        bullet_speed=float(cfg["env"]["bullet_speed"]),
        fire_cooldown_steps=int(cfg["env"]["fire_cooldown_steps"]),
        enemy_rows=int(cfg["env"]["enemy_rows"]),
        enemy_cols=int(cfg["env"]["enemy_cols"]),
        enemy_speed=float(cfg["env"]["enemy_speed"]),
        enemy_drop=float(cfg["env"]["enemy_drop"]),
        enemy_fire_prob=float(cfg["env"]["enemy_fire_prob"]),
        max_steps=int(cfg["env"]["max_steps"]),
    )

    conn = load_connectome(release="783", min_syn=int(cfg["data"]["min_syn"]))

    episode_rows = []

    for ctrl_name in controllers:
        logger.info(f"Evaluating controller '{ctrl_name}' on {len(seeds)} seeds...")

        # Initialize controller instance
        if ctrl_name == "random":
            ctrl = RandomController(p=float(cfg["eval"]["random_p"]))
        elif ctrl_name == "heuristic":
            ctrl = HeuristicController(fire_thresh=float(cfg["eval"]["heuristic_fire_thresh"]))
        elif ctrl_name == "brain":
            ctrl = build_brain_controller(conn, io_spec, cfg)
        elif ctrl_name == "shuffled":
            # Shuffled connectome: extract subcircuit first, then shuffle subcircuit!
            logger.info("Building shuffled-wiring brain controller (shuffling extracted subcircuit)...")
            all_in = np.concatenate(list(io_spec.input_groups.values()))
            all_out = np.concatenate(list(io_spec.output_groups.values()))
            sub_conn, sub_to_full = extract_subcircuit(conn, all_in, all_out, max_hops=3, max_neurons=1500)
            shuffled_sub = shuffle_connectome(sub_conn, seed=42, n_swaps_factor=5)

            full_to_sub = {int(full): sub for sub, full in enumerate(sub_to_full)}
            io_sub = load_io_spec("data/io_spec.json")
            for k in io_sub.input_groups:
                io_sub.input_groups[k] = np.array([full_to_sub[i] for i in io_spec.input_groups[k] if i in full_to_sub], dtype=np.int32)
            for k in io_sub.output_groups:
                io_sub.output_groups[k] = np.array([full_to_sub[i] for i in io_spec.output_groups[k] if i in full_to_sub], dtype=np.int32)

            brain = Brian2Brain(shuffled_sub, LIFParams())
            enc = Encoder(io_sub, EncoderConfig(max_rate_hz=float(cfg["encoder"]["max_rate_hz"])))
            dec = Decoder(io_sub, DecoderConfig(margin_hz=float(cfg["decoder"]["margin_hz"])))
            ctrl = BrainController(brain, enc, dec, step_ms=float(cfg["eval"]["step_ms"]))
        elif ctrl_name == "random_io":
            # Random-I/O control
            logger.info("Building random-I/O brain controller...")
            resampled_io = resample_io(io_spec, conn, seed=42)
            ctrl = build_brain_controller(conn, resampled_io, cfg)
        else:
            raise ValueError(f"Unknown controller: {ctrl_name}")

        for seed in seeds:
            t0 = time.perf_counter()
            env = SpaceInvadersEnv(env_cfg)
            obs, _ = env.reset(seed=seed)
            ctrl.reset(seed=seed)

            done = False
            while not done:
                action = ctrl.act(obs)
                obs, reward, term, trunc, info = env.step(action)
                done = term or trunc

            wall_s = time.perf_counter() - t0

            episode_rows.append({
                "controller": ctrl_name,
                "seed": seed,
                "score": info["score"],
                "kills": info["kills"],
                "shots_fired": info["shots_fired"],
                "steps_survived": info["steps"],
                "wall_s": wall_s,
            })

    df_episodes = pd.DataFrame(episode_rows)
    df_episodes.to_parquet(out_dir / "episodes.parquet", index=False)

    summary_df, stats_md = compute_evaluation_stats(df_episodes)
    with open(out_dir / "stats.md", "w", encoding="utf-8") as f:
        f.write("# Fly Brain Arcade — Evaluation Statistics\n\n")
        f.write("## Overall Performance Summary\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")
        f.write(stats_md)

    summary = {
        "git_commit": get_git_commit(),
        "config_hash": config_hash,
        "io_spec_hash": io_spec.config_hash,
        "data_manifest": manifest,
        "controllers": controllers,
        "seeds": seeds,
        "n_episodes": len(df_episodes),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Evaluation completed! Results saved to {out_dir}")
    print("\nSummary Results:")
    print(summary_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Fly Brain Arcade Evaluation Runner")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--controllers", nargs="+", default=["random", "heuristic", "brain", "shuffled", "random_io"])
    parser.add_argument("--seeds", default="0-29", help="Seed range or comma-separated list (e.g. 0-29)")
    parser.add_argument("--out", default=None, help="Output directory")
    args = parser.parse_args()

    # Parse seeds
    if "-" in args.seeds:
        start, end = map(int, args.seeds.split("-"))
        seeds = list(range(start, end + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    out_dir = Path(args.out) if args.out else Path(f"reports/run_{int(time.time())}")
    run_evaluation(Path(args.config), args.controllers, seeds, out_dir)


if __name__ == "__main__":
    main()
