"""Visualisation module: renders side-by-side video and GIF of gameplay and neural spikes."""

import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from flybrain.brain.backend_brian2 import Brian2Brain
from flybrain.brain.lif import LIFParams
from flybrain.brain.subcircuit import extract_subcircuit
from flybrain.control.brain_ctrl import BrainController
from flybrain.data.connectome import load_connectome
from flybrain.env.invaders import InvadersConfig, SpaceInvadersEnv
from flybrain.io.decoder import Decoder, DecoderConfig
from flybrain.io.encoder import Encoder, EncoderConfig
from flybrain.io.select import load_io_spec

logger = logging.getLogger(__name__)


def render_episode(controller, env, seed: int, out: Path, max_steps: int = 150) -> None:
    """Record an episode and produce side-by-side visualisation:
    Left: Space Invaders gameplay frame
    Right top: Spike raster per neuron group
    Right bottom: Group firing rate traces over time
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset(seed=seed)
    controller.reset(seed=seed)

    frames = []
    spike_batches = []
    actions = []
    rates_history = {"move_left": [], "move_right": [], "fire": []}

    # Intercept controller's brain step to collect spikes
    original_step = controller.brain.step

    def tracked_step(duration_ms):
        batch = original_step(duration_ms)
        spike_batches.append(batch)
        return batch

    controller.brain.step = tracked_step

    done = False
    step = 0
    while not done and step < max_steps:
        frame = env.render()
        frames.append(frame)

        action = controller.act(obs)
        actions.append(action)

        # Record group rates
        if spike_batches:
            curr_batch = spike_batches[-1]
            for grp_name, grp_nids in controller.decoder.io.output_groups.items():
                spks = np.sum(np.isin(curr_batch.neuron_idx, grp_nids))
                r = spks / (max(1, len(grp_nids)) * (curr_batch.duration_ms / 1000.0))
                rates_history[grp_name].append(r)
        else:
            for grp_name in rates_history:
                rates_history[grp_name].append(0.0)

        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
        step += 1

    # Render animation with matplotlib
    fig, (ax_game, ax_raster, ax_rate) = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#111111")

    ax_game.axis("off")
    im_game = ax_game.imshow(frames[0])
    ax_game.set_title("Space Invaders (Game Obs)", color="white", fontsize=12)

    ax_raster.set_facecolor("#1a1a1a")
    ax_raster.set_title("Neural Spike Raster", color="white", fontsize=12)
    ax_raster.set_xlabel("Time (steps)", color="white")
    ax_raster.set_ylabel("Group Index", color="white")
    ax_raster.tick_params(colors="white")

    ax_rate.set_facecolor("#1a1a1a")
    ax_rate.set_title("Output Group Firing Rates (Hz)", color="white", fontsize=12)
    ax_rate.set_xlabel("Time (steps)", color="white")
    ax_rate.set_ylabel("Rate (Hz)", color="white")
    ax_rate.tick_params(colors="white")

    # Animation update function
    def update(frame_idx):
        im_game.set_data(frames[frame_idx])

        # Raster update
        ax_raster.cla()
        ax_raster.set_facecolor("#1a1a1a")
        ax_raster.set_title("Neural Spike Raster (Recent)", color="white", fontsize=12)
        ax_raster.set_xlim(max(0, frame_idx - 30), max(30, frame_idx + 1))
        ax_raster.set_ylim(-0.5, 2.5)
        ax_raster.set_yticks([0, 1, 2])
        ax_raster.set_yticklabels(["Left", "Right", "Fire"], color="white")
        ax_raster.tick_params(colors="white")

        start_f = max(0, frame_idx - 30)
        for f in range(start_f, frame_idx + 1):
            if f < len(spike_batches):
                b = spike_batches[f]
                # Plot dots for output groups
                for y_idx, grp_name in enumerate(["move_left", "move_right", "fire"]):
                    grp_nids = controller.decoder.io.output_groups[grp_name]
                    n_spk = np.sum(np.isin(b.neuron_idx, grp_nids))
                    if n_spk > 0:
                        ax_raster.scatter([f] * n_spk, [y_idx] * n_spk, color="#00ffcc", s=15, alpha=0.7)

        # Rate plot update
        ax_rate.cla()
        ax_rate.set_facecolor("#1a1a1a")
        ax_rate.set_title("Output Group Firing Rates (Hz)", color="white", fontsize=12)
        ax_rate.tick_params(colors="white")
        t_axis = np.arange(frame_idx + 1)
        ax_rate.plot(t_axis, rates_history["move_left"][: frame_idx + 1], label="Left", color="#ff0055", lw=1.5)
        ax_rate.plot(t_axis, rates_history["move_right"][: frame_idx + 1], label="Right", color="#00aaff", lw=1.5)
        ax_rate.plot(t_axis, rates_history["fire"][: frame_idx + 1], label="Fire", color="#ffcc00", lw=1.5)
        ax_rate.set_xlim(0, max_steps)
        ax_rate.legend(loc="upper right", facecolor="#222222", labelcolor="white")

        return [im_game]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)

    # Save animated GIF via Pillow
    gif_path = out.with_suffix(".gif")
    ani.save(gif_path, writer="pillow", fps=15)
    logger.info(f"Saved demo animation to {gif_path}")
    plt.close(fig)


def main():
    logger.info("Setting up BrainController for demo visualization...")
    conn = load_connectome(release="783", min_syn=5)
    io_spec = load_io_spec("data/io_spec.json")

    all_in = np.concatenate(list(io_spec.input_groups.values()))
    all_out = np.concatenate(list(io_spec.output_groups.values()))

    sub_conn, sub_to_full = extract_subcircuit(conn, all_in, all_out, max_hops=3, max_neurons=1500)
    full_to_sub = {int(full): sub for sub, full in enumerate(sub_to_full)}

    io_sub = load_io_spec("data/io_spec.json")
    for k in io_sub.input_groups:
        io_sub.input_groups[k] = np.array([full_to_sub[i] for i in io_spec.input_groups[k] if i in full_to_sub], dtype=np.int32)
    for k in io_sub.output_groups:
        io_sub.output_groups[k] = np.array([full_to_sub[i] for i in io_spec.output_groups[k] if i in full_to_sub], dtype=np.int32)

    brain = Brian2Brain(sub_conn, LIFParams())
    enc = Encoder(io_sub, EncoderConfig(max_rate_hz=100.0))
    dec = Decoder(io_sub, DecoderConfig())
    ctrl = BrainController(brain, enc, dec, step_ms=50.0)

    env = SpaceInvadersEnv(InvadersConfig(max_steps=100))
    render_episode(ctrl, env, seed=42, out=Path("reports/demo.gif"), max_steps=100)
    print("Demo video/gif generated at reports/demo.gif")


if __name__ == "__main__":
    main()
