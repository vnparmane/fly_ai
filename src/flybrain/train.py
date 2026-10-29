"""Training harness for Dopamine-Modulated Synaptic Plasticity (R-STDP) in Space Invaders."""

import argparse
import json
import logging
from pathlib import Path
import time
import numpy as np
import yaml

from flybrain.brain.backend_brian2 import Brian2Brain
from flybrain.brain.lif import LIFParams
from flybrain.brain.plasticity import PlasticityConfig, RSTDPPlasticity
from flybrain.brain.subcircuit import extract_subcircuit
from flybrain.brain.vnc import VNCBrainController, VNCConfig, VNCMotorPool, extract_fafb_vnc_spec
from flybrain.control.brain_ctrl import BrainController
from flybrain.data.connectome import load_connectome
from flybrain.env.invaders import InvadersConfig, SpaceInvadersEnv
from flybrain.io.decoder import Decoder, DecoderConfig
from flybrain.io.encoder import Encoder, EncoderConfig
from flybrain.io.select import load_io_spec

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("flybrain.train")


def compute_dopamine_reward(
    env_reward: float,
    obs: dict,
    action: np.ndarray,
    term: bool,
    prev_kills: int,
    current_kills: int,
) -> float:
    """Compute biologically grounded dopamine reward signal R(t) for Drosophila:
    - Primary reward: alien elimination (+1.0)
    - Sensorimotor orientation: centering alien beneath cannon (+0.05 * (1 - |target_dx|))
    - Negative punishment: ship destroyed / collision (-0.5)
    - Metabolic cost / wasted fire: firing off-target (-0.01)
    """
    r = 0.0

    # 1. Kill reward
    kills_delta = current_kills - prev_kills
    if kills_delta > 0:
        r += 1.0 * kills_delta

    # 2. Target centering bonus
    target_dx = float(obs["target_dx"][0])
    r += 0.05 * (1.0 - abs(target_dx))

    # 3. Ship destruction penalty
    if term:
        r -= 0.5

    # 4. Off-target firing penalty
    if action[2] == 1 and abs(target_dx) > 0.2:
        r -= 0.01

    return r


def train_episodes(
    n_episodes: int = 20,
    seeds_start: int = 1000,
    config_path: str = "configs/default.yaml",
    checkpoint_out: str = "data/checkpoints/trained_brain.npz",
    use_vnc: bool = True,
) -> Path:
    out_path = Path(checkpoint_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
        lives=int(cfg["env"].get("lives", 3)),
    )

    logger.info("Loading connectome and preparing subcircuit...")
    conn = load_connectome(release="783", min_syn=int(cfg["data"]["min_syn"]))

    if use_vnc:
        io_spec = extract_fafb_vnc_spec(conn)
    else:
        io_spec = load_io_spec("data/io_spec.json")

    all_in = np.concatenate(list(io_spec.input_groups.values()))
    all_out = np.concatenate(list(io_spec.output_groups.values()))
    sub_conn, sub_to_full = extract_subcircuit(conn, all_in, all_out, max_hops=3, max_neurons=1500)
    full_to_sub = {int(full): sub for sub, full in enumerate(sub_to_full)}

    io_sub = load_io_spec("data/io_spec.json") if not use_vnc else extract_fafb_vnc_spec(conn)
    for k in io_sub.input_groups:
        io_sub.input_groups[k] = np.array([full_to_sub[i] for i in io_spec.input_groups[k] if i in full_to_sub], dtype=np.int32)
    for k in io_sub.output_groups:
        io_sub.output_groups[k] = np.array([full_to_sub[i] for i in io_spec.output_groups[k] if i in full_to_sub], dtype=np.int32)

    enc_cfg = EncoderConfig(
        max_rate_hz=float(cfg["encoder"]["max_rate_hz"]),
        dx_range=float(cfg["encoder"]["dx_range"]),
        align_sigma=float(cfg["encoder"]["align_sigma"]),
    )
    enc = Encoder(io_sub, enc_cfg)

    # Initialize Brain & Plasticity
    lif_params = LIFParams.from_dict(cfg.get("lif", {}))
    brain = Brian2Brain(sub_conn, lif_params)
    plasticity = RSTDPPlasticity(brain, PlasticityConfig(lr=0.005, decay=0.85))

    if use_vnc:
        vnc_pool = VNCMotorPool(io_sub, VNCConfig())
        ctrl = VNCBrainController(brain, enc, vnc_pool, step_ms=float(cfg["eval"]["step_ms"]))
    else:
        dec_cfg = DecoderConfig(
            window_ms=float(cfg["decoder"]["window_ms"]),
            margin_hz=float(cfg["decoder"]["margin_hz"]),
            thresh_hz={k: float(v) for k, v in cfg["decoder"]["thresh_hz"].items()},
        )
        dec = Decoder(io_sub, dec_cfg)
        ctrl = BrainController(brain, enc, dec, step_ms=float(cfg["eval"]["step_ms"]))

    history = []
    logger.info(f"Beginning R-STDP Dopaminergic training over {n_episodes} episodes...")

    for ep in range(n_episodes):
        ep_seed = seeds_start + ep
        env = SpaceInvadersEnv(env_cfg)
        obs, _ = env.reset(seed=ep_seed)
        ctrl.reset(seed=ep_seed)
        plasticity.reset_traces()

        ep_reward = 0.0
        prev_kills = 0
        step_count = 0
        t0 = time.perf_counter()

        while True:
            action = ctrl.act(obs)
            next_obs, env_reward, term, trunc, info = env.step(action)
            step_count += 1

            # Compute shaped dopamine reward
            r_dopamine = compute_dopamine_reward(
                env_reward=env_reward,
                obs=obs,
                action=action,
                term=term,
                prev_kills=prev_kills,
                current_kills=info["kills"],
            )
            ep_reward += r_dopamine
            prev_kills = info["kills"]

            # Update synaptic plasticity
            last_batch = getattr(ctrl, "last_batch", None)
            if last_batch is not None:
                plasticity.step(last_batch, reward=r_dopamine)

            obs = next_obs
            if term or trunc:
                break

        wall_s = time.perf_counter() - t0
        stats = {
            "episode": ep + 1,
            "seed": ep_seed,
            "kills": info["kills"],
            "score": info["score"],
            "steps": info["steps"],
            "cumulative_reward": round(ep_reward, 2),
            "total_weight_delta_l1": round(float(np.sum(np.abs(plasticity.cumulative_delta))), 4),
            "wall_s": round(wall_s, 2),
        }
        history.append(stats)
        logger.info(
            f"Ep {ep+1:2d}/{n_episodes} | Kills: {info['kills']} | Steps: {info['steps']:3d} | "
            f"Reward: {ep_reward:6.2f} | ΔW_L1: {stats['total_weight_delta_l1']} | Time: {wall_s:4.1f}s"
        )
        # Save checkpoint after each episode
        plasticity.save_weights(out_path)

    # Save trained checkpoint
    plasticity.save_weights(out_path)

    # Save training log JSON
    log_path = out_path.with_suffix(".json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training completed successfully! Saved to {out_path} and {log_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Train Fly Brain Connectome with R-STDP")
    parser.add_argument("--episodes", type=int, default=10, help="Number of training episodes")
    parser.add_argument("--seeds-start", type=int, default=1000, help="Starting seed for training")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--out", default="data/checkpoints/trained_brain.npz", help="Checkpoint output path")
    parser.add_argument("--use-vnc", action="store_true", default=True, help="Use FAFB VNC motor circuits")
    args = parser.parse_args()

    train_episodes(
        n_episodes=args.episodes,
        seeds_start=args.seeds_start,
        config_path=args.config,
        checkpoint_out=args.out,
        use_vnc=args.use_vnc,
    )


if __name__ == "__main__":
    main()
