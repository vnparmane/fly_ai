"""Train full control matrix for Drosophila Connectome vs Ablations."""

import logging
from pathlib import Path
from flybrain.train import train_episodes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("flybrain.train_controls")


def main():
    n_episodes = 10
    seeds_start = 2000
    config_path = "configs/default.yaml"

    models = [
        ("trained_brain", "none", "data/checkpoints/trained_brain.npz"),
        ("trained_shuffled", "shuffled", "data/checkpoints/trained_shuffled.npz"),
        ("trained_random_io", "random_io", "data/checkpoints/trained_random_io.npz"),
    ]

    for name, control_mode, out_path in models:
        logger.info(f"=== Starting training for {name} (control={control_mode}) ===")
        train_episodes(
            n_episodes=n_episodes,
            seeds_start=seeds_start,
            config_path=config_path,
            checkpoint_out=out_path,
            use_vnc=True,
            control=control_mode,
        )
        logger.info(f"=== Completed training for {name} -> {out_path} ===")


if __name__ == "__main__":
    main()
