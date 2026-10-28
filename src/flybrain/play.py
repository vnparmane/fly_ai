"""Interactive live player & neural monitor for Space Invaders."""

import argparse
import logging
import os
from pathlib import Path
import sys
import time
import numpy as np
import yaml

# If running interactive play, remove any inherited SDL_VIDEODRIVER=dummy from session
if "--headless-check" not in sys.argv:
    os.environ.pop("SDL_VIDEODRIVER", None)

from flybrain.brain.backend_brian2 import Brian2Brain
from flybrain.brain.lif import LIFParams
from flybrain.brain.plasticity import RSTDPPlasticity
from flybrain.brain.subcircuit import extract_subcircuit
from flybrain.brain.vnc import VNCBrainController, VNCConfig, VNCMotorPool, extract_fafb_vnc_spec
from flybrain.control.base import Controller
from flybrain.control.brain_ctrl import BrainController
from flybrain.control.heuristic import HeuristicController
from flybrain.control.random_ctrl import RandomController
from flybrain.control.shuffle import shuffle_connectome
from flybrain.data.connectome import load_connectome
from flybrain.env.invaders import InvadersConfig, SpaceInvadersEnv
from flybrain.env.render import ArcadeViewer
from flybrain.io.decoder import Decoder, DecoderConfig
from flybrain.io.encoder import Encoder, EncoderConfig
from flybrain.io.select import load_io_spec

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("flybrain.play")


class ControllerManager:
    """Manages on-demand lazy initialization of controllers."""

    def __init__(self, conn, cfg, viewer: ArcadeViewer | None = None) -> None:
        self.conn = conn
        self.cfg = cfg
        self.viewer = viewer
        self.cache: dict[str, Controller] = {}
        self.io_cache: dict[str, object] = {}

    def get(self, name: str) -> tuple[Controller, object]:
        if name in self.cache:
            return self.cache[name], self.io_cache[name]

        if self.viewer:
            self.viewer.show_loading(f"Loading '{name.upper()}' Controller...", "Initializing Neural Subcircuit...")

        enc_cfg = EncoderConfig(
            max_rate_hz=float(self.cfg["encoder"]["max_rate_hz"]),
            dx_range=float(self.cfg["encoder"]["dx_range"]),
            align_sigma=float(self.cfg["encoder"]["align_sigma"]),
        )
        dec_cfg = DecoderConfig(
            window_ms=float(self.cfg["decoder"]["window_ms"]),
            margin_hz=float(self.cfg["decoder"]["margin_hz"]),
            thresh_hz={k: float(v) for k, v in self.cfg["decoder"]["thresh_hz"].items()},
        )
        step_ms = float(self.cfg["eval"]["step_ms"])

        if name == "random":
            ctrl = RandomController(p=float(self.cfg["eval"]["random_p"]))
            io_spec = load_io_spec("data/io_spec.json")

        elif name == "heuristic":
            ctrl = HeuristicController(fire_thresh=float(self.cfg["eval"]["heuristic_fire_thresh"]))
            io_spec = load_io_spec("data/io_spec.json")

        elif name == "brain":
            io_spec = load_io_spec("data/io_spec.json")
            all_in = np.concatenate(list(io_spec.input_groups.values()))
            all_out = np.concatenate(list(io_spec.output_groups.values()))
            sub_conn, sub_to_full = extract_subcircuit(self.conn, all_in, all_out, max_hops=3, max_neurons=1500)
            full_to_sub = {int(full): sub for sub, full in enumerate(sub_to_full)}

            io_sub = load_io_spec("data/io_spec.json")
            for k in io_sub.input_groups:
                io_sub.input_groups[k] = np.array([full_to_sub[i] for i in io_spec.input_groups[k] if i in full_to_sub], dtype=np.int32)
            for k in io_sub.output_groups:
                io_sub.output_groups[k] = np.array([full_to_sub[i] for i in io_spec.output_groups[k] if i in full_to_sub], dtype=np.int32)

            brain_core = Brian2Brain(sub_conn, LIFParams())
            ctrl = BrainController(brain_core, Encoder(io_sub, enc_cfg), Decoder(io_sub, dec_cfg), step_ms=step_ms)
            io_spec = io_sub

        elif name == "shuffled":
            io_spec = load_io_spec("data/io_spec.json")
            all_in = np.concatenate(list(io_spec.input_groups.values()))
            all_out = np.concatenate(list(io_spec.output_groups.values()))
            sub_conn, sub_to_full = extract_subcircuit(self.conn, all_in, all_out, max_hops=3, max_neurons=1500)
            full_to_sub = {int(full): sub for sub, full in enumerate(sub_to_full)}

            io_sub = load_io_spec("data/io_spec.json")
            for k in io_sub.input_groups:
                io_sub.input_groups[k] = np.array([full_to_sub[i] for i in io_spec.input_groups[k] if i in full_to_sub], dtype=np.int32)
            for k in io_sub.output_groups:
                io_sub.output_groups[k] = np.array([full_to_sub[i] for i in io_spec.output_groups[k] if i in full_to_sub], dtype=np.int32)

            shuffled_sub = shuffle_connectome(sub_conn, seed=42, n_swaps_factor=5)
            shuffled_brain = Brian2Brain(shuffled_sub, LIFParams())
            ctrl = BrainController(shuffled_brain, Encoder(io_sub, enc_cfg), Decoder(io_sub, dec_cfg), step_ms=step_ms)
            io_spec = io_sub

        elif name == "vnc":
            vnc_spec = extract_fafb_vnc_spec(self.conn)
            all_in_vnc = np.concatenate(list(vnc_spec.input_groups.values()))
            all_out_vnc = np.concatenate(list(vnc_spec.output_groups.values()))
            sub_vnc_conn, sub_to_full_vnc = extract_subcircuit(self.conn, all_in_vnc, all_out_vnc, max_hops=3, max_neurons=1500)
            full_to_sub_vnc = {int(full): sub for sub, full in enumerate(sub_to_full_vnc)}

            vnc_io_sub = extract_fafb_vnc_spec(self.conn)
            for k in vnc_io_sub.input_groups:
                vnc_io_sub.input_groups[k] = np.array([full_to_sub_vnc[i] for i in vnc_spec.input_groups[k] if i in full_to_sub_vnc], dtype=np.int32)
            for k in vnc_io_sub.output_groups:
                vnc_io_sub.output_groups[k] = np.array([full_to_sub_vnc[i] for i in vnc_spec.output_groups[k] if i in full_to_sub_vnc], dtype=np.int32)

            vnc_brain = Brian2Brain(sub_vnc_conn, LIFParams())
            pool = VNCMotorPool(vnc_io_sub, VNCConfig())
            ctrl = VNCBrainController(vnc_brain, Encoder(vnc_io_sub, enc_cfg), pool, step_ms=step_ms)
            io_spec = vnc_io_sub

        elif name == "trained_brain":
            vnc_spec = extract_fafb_vnc_spec(self.conn)
            all_in_vnc = np.concatenate(list(vnc_spec.input_groups.values()))
            all_out_vnc = np.concatenate(list(vnc_spec.output_groups.values()))
            sub_vnc_conn, sub_to_full_vnc = extract_subcircuit(self.conn, all_in_vnc, all_out_vnc, max_hops=3, max_neurons=1500)
            full_to_sub_vnc = {int(full): sub for sub, full in enumerate(sub_to_full_vnc)}

            vnc_io_trained = extract_fafb_vnc_spec(self.conn)
            for k in vnc_io_trained.input_groups:
                vnc_io_trained.input_groups[k] = np.array([full_to_sub_vnc[i] for i in vnc_spec.input_groups[k] if i in full_to_sub_vnc], dtype=np.int32)
            for k in vnc_io_trained.output_groups:
                vnc_io_trained.output_groups[k] = np.array([full_to_sub_vnc[i] for i in vnc_spec.output_groups[k] if i in full_to_sub_vnc], dtype=np.int32)

            trained_brain_core = Brian2Brain(sub_vnc_conn, LIFParams())
            chk_path = Path("data/checkpoints/trained_brain.npz")
            if chk_path.exists():
                plast = RSTDPPlasticity(trained_brain_core)
                plast.load_weights(chk_path)
            else:
                logger.warning("Checkpoint trained_brain.npz not found; running with baseline weights.")

            pool = VNCMotorPool(vnc_io_trained, VNCConfig())
            ctrl = VNCBrainController(trained_brain_core, Encoder(vnc_io_trained, enc_cfg), pool, step_ms=step_ms)
            io_spec = vnc_io_trained

        else:
            raise ValueError(f"Unknown controller: {name}")

        self.cache[name] = ctrl
        self.io_cache[name] = io_spec
        return ctrl, io_spec


def run_interactive(
    controller_name: str = "trained_brain",
    seed: int = 42,
    config_path: str = "configs/default.yaml",
    headless_check: bool = False,
    max_check_steps: int = 50,
) -> None:
    # 1. Initialize viewer first so desktop window appears immediately
    viewer = ArcadeViewer(headless=headless_check, fps=20)
    viewer.show_loading("Starting Fly Brain Arcade...", "Loading Connectome & Configurations...")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_cfg = InvadersConfig(
        width=int(cfg["env"]["width"]),
        height=int(cfg["env"]["height"]),
        player_speed=float(cfg["env"]["player_speed"]),
        bullet_speed=float(cfg["env"]["bullet_speed"]),
        fire_cooldown_steps=4,
        enemy_rows=int(cfg["env"]["enemy_rows"]),
        enemy_cols=int(cfg["env"]["enemy_cols"]),
        enemy_speed=0.4,
        enemy_drop=float(cfg["env"]["enemy_drop"]),
        enemy_fire_prob=0.008,
        max_steps=int(cfg["env"]["max_steps"]),
        lives=3,
    )
    env = SpaceInvadersEnv(env_cfg)

    # 2. Load Connectome
    logger.info("Loading connectome...")
    conn = load_connectome(release="783", min_syn=int(cfg["data"]["min_syn"]))

    # 3. Initialize Controller Manager (on-demand loading)
    manager = ControllerManager(conn, cfg, viewer=viewer)

    current_ctrl_name = controller_name
    current_ctrl, io_sub = manager.get(current_ctrl_name)

    obs, _ = env.reset(seed=seed)
    current_ctrl.reset(seed=seed)

    logger.info(f"Starting interactive session with controller: '{current_ctrl_name}' (Seed: {seed})")
    is_paused = False
    step_num = 0
    prev_kills = 0
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

            if events["switch_ctrl"]:
                next_ctrl_name = events["switch_ctrl"]
                if next_ctrl_name != current_ctrl_name:
                    current_ctrl_name = next_ctrl_name
                    current_ctrl, io_sub = manager.get(current_ctrl_name)
                    current_ctrl.reset(seed=seed + step_num)
                    logger.info(f"Switched controller to: '{current_ctrl_name}'")

            if events["reset"]:
                seed += 1
                obs, _ = env.reset(seed=seed)
                current_ctrl.reset(seed=seed)
                step_num = 0
                prev_kills = 0
                logger.info(f"Episode reset to seed {seed}")

            # 2. Simulation Step
            if not is_paused:
                action = current_ctrl.act(obs)
                obs, reward, term, trunc, info = env.step(action)
                step_num += 1

                # Record spikes and trigger explosion on kill
                last_batch = getattr(current_ctrl, "last_batch", None)
                viewer.record_spikes(last_batch, io_sub)
                if info["kills"] > prev_kills:
                    viewer.trigger_explosion(env.player_x, 40.0)
                    prev_kills = info["kills"]

                # Game Over / Episode termination handling
                if term or trunc:
                    viewer.render_frame(
                        env=env,
                        action=action,
                        controller_name=current_ctrl_name,
                        rates=getattr(current_ctrl, "last_rates", None),
                        fps_actual=fps_estimate,
                        is_paused=False,
                    )
                    if not headless_check:
                        t_pause_start = time.perf_counter()
                        while time.perf_counter() - t_pause_start < 1.2:
                            ev = viewer.process_events()
                            if ev["quit"]:
                                return
                            time.sleep(0.05)

                    seed += 1
                    obs, _ = env.reset(seed=seed)
                    current_ctrl.reset(seed=seed)
                    prev_kills = 0

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
    parser.add_argument(
        "--controller",
        default="trained_brain",
        choices=["trained_brain", "brain", "vnc", "heuristic", "random", "shuffled"],
    )
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
