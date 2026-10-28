"""Interactive live player & neural monitor for Space Invaders."""

import argparse
import logging
from pathlib import Path
import sys
import time
import numpy as np
import yaml

from flybrain.brain.backend_brian2 import Brian2Brain
from flybrain.brain.lif import LIFParams
from flybrain.brain.subcircuit import extract_subcircuit
from flybrain.control.base import Controller
from flybrain.control.brain_ctrl import BrainController
from flybrain.control.heuristic import HeuristicController
from flybrain.control.random_ctrl import RandomController
from flybrain.control.shuffle import resample_io, shuffle_connectome
from flybrain.data.connectome import load_connectome
from flybrain.env.invaders import InvadersConfig, SpaceInvadersEnv
from flybrain.env.render import ArcadeViewer
from flybrain.io.decoder import Decoder, DecoderConfig
from flybrain.io.encoder import Encoder, EncoderConfig
from flybrain.io.select import load_io_spec

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("flybrain.play")


def load_controllers_dict(conn, io_spec, cfg) -> dict[str, Controller]:
    """Pre-build controllers dictionary for fast live-switching."""
    controllers: dict[str, Controller] = {
        "random": RandomController(p=float(cfg["eval"]["random_p"])),
        "heuristic": HeuristicController(fire_thresh=float(cfg["eval"]["heuristic_fire_thresh"])),
    }

    # Extract subcircuit for brain-based controllers (1500 neurons)
    logger.info("Extracting subcircuit for live brain controllers...")
    all_in = np.concatenate(list(io_spec.input_groups.values()))
    all_out = np.concatenate(list(io_spec.output_groups.values()))
    sub_conn, sub_to_full = extract_subcircuit(conn, all_in, all_out, max_hops=3, max_neurons=1500)
    full_to_sub = {int(full): sub for sub, full in enumerate(sub_to_full)}

    io_sub = load_io_spec("data/io_spec.json")
    for k in io_sub.input_groups:
        io_sub.input_groups[k] = np.array([full_to_sub[i] for i in io_spec.input_groups[k] if i in full_to_sub], dtype=np.int32)
    for k in io_sub.output_groups:
        io_sub.output_groups[k] = np.array([full_to_sub[i] for i in io_spec.output_groups[k] if i in full_to_sub], dtype=np.int32)

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

    # 1. Real Brain Controller
    brain_core = Brian2Brain(sub_conn, LIFParams())
    enc = Encoder(io_sub, enc_cfg)
    dec = Decoder(io_sub, dec_cfg)
    controllers["brain"] = BrainController(brain_core, enc, dec, step_ms=float(cfg["eval"]["step_ms"]))

    # 2. Shuffled Brain Controller
    shuffled_sub = shuffle_connectome(sub_conn, seed=42, n_swaps_factor=5)
    shuffled_brain = Brian2Brain(shuffled_sub, LIFParams())
    controllers["shuffled"] = BrainController(
        shuffled_brain,
        Encoder(io_sub, enc_cfg),
        Decoder(io_sub, dec_cfg),
        step_ms=float(cfg["eval"]["step_ms"]),
    )

    return controllers, io_sub


def run_interactive(
    controller_name: str = "brain",
    seed: int = 42,
    config_path: str = "configs/default.yaml",
    headless_check: bool = False,
    max_check_steps: int = 50,
) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

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
    env = SpaceInvadersEnv(env_cfg)

    viewer = ArcadeViewer(headless=headless_check, fps=20)

    # Load connectome and controllers
    logger.info("Loading connectome...")
    conn = load_connectome(release="783", min_syn=int(cfg["data"]["min_syn"]))
    io_spec = load_io_spec("data/io_spec.json")

    controllers, io_sub = load_controllers_dict(conn, io_spec, cfg)

    current_ctrl_name = controller_name if controller_name in controllers else "brain"
    current_ctrl = controllers[current_ctrl_name]

    obs, _ = env.reset(seed=seed)
    current_ctrl.reset(seed=seed)

    logger.info(f"Starting interactive session with controller: '{current_ctrl_name}' (Seed: {seed})")
    is_paused = False
    step_num = 0
    t_last = time.perf_counter()
    fps_estimate = 20.0

    try:
        while True:
            # 1. Event Handling
            events = viewer.process_events()
            if events["quit"]:
                break

            if events["pause_toggle"]:
                is_paused = not is_paused

            if events["switch_ctrl"] and events["switch_ctrl"] in controllers:
                current_ctrl_name = events["switch_ctrl"]
                current_ctrl = controllers[current_ctrl_name]
                current_ctrl.reset(seed=seed + step_num)
                logger.info(f"Switched controller to: '{current_ctrl_name}'")

            if events["reset"]:
                seed += 1
                obs, _ = env.reset(seed=seed)
                current_ctrl.reset(seed=seed)
                step_num = 0
                logger.info(f"Episode reset to seed {seed}")

            # 2. Simulation Step
            if not is_paused:
                action = current_ctrl.act(obs)
                obs, reward, term, trunc, info = env.step(action)
                step_num += 1

                # Record spikes into viewer
                last_batch = getattr(current_ctrl, "last_batch", None)
                viewer.record_spikes(last_batch, io_sub)

                # Reset upon episode termination in continuous mode
                if term or trunc:
                    seed += 1
                    obs, _ = env.reset(seed=seed)
                    current_ctrl.reset(seed=seed)

            # 3. Telemetry & Rendering
            t_now = time.perf_counter()
            dt = t_now - t_last
            t_last = t_now
            if dt > 0:
                fps_estimate = 0.9 * fps_estimate + 0.1 * (1.0 / dt)

            rates = getattr(current_ctrl, "last_rates", None)
            last_act = getattr(current_ctrl, "last_action", np.zeros(3, dtype=np.int8))

            viewer.render_frame(
                env=env,
                action=last_act,
                controller_name=current_ctrl_name,
                rates=rates,
                fps_actual=fps_estimate,
                is_paused=is_paused,
            )

            if headless_check and step_num >= max_check_steps:
                logger.info(f"Headless check completed successfully after {step_num} steps.")
                break

    finally:
        viewer.close()


def main():
    parser = argparse.ArgumentParser(description="Fly Brain Arcade Live Player & Neural Monitor")
    parser.add_argument("--controller", default="brain", choices=["brain", "heuristic", "random", "shuffled"])
    parser.add_argument("--seed", type=int, default=42, help="Initial game seed")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--headless-check", action="store_true", help="Run 50 steps headlessly for verification")
    args = parser.parse_args()

    run_interactive(
        controller_name=args.controller,
        seed=args.seed,
        config_path=args.config,
        headless_check=args.headless_check,
    )


if __name__ == "__main__":
    main()
