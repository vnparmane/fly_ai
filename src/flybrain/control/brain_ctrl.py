"""Closed-loop brain controller integrating SpikingBrain, Encoder, and Decoder."""

import numpy as np

from flybrain.brain.lif import SpikingBrain
from flybrain.io.decoder import Decoder
from flybrain.io.encoder import Encoder


class BrainController:
    """Controls Space Invaders using real connectome spiking dynamics."""

    def __init__(
        self,
        brain: SpikingBrain,
        encoder: Encoder,
        decoder: Decoder,
        step_ms: float = 50.0,
    ) -> None:
        self.brain = brain
        self.encoder = encoder
        self.decoder = decoder
        self.step_ms = step_ms

    def reset(self, seed: int) -> None:
        self.brain.reset(seed)
        self.decoder.reset()

    def act(self, obs: dict) -> np.ndarray:
        # 1. Encode game observation into Poisson firing rates
        indices, rates = self.encoder(obs)

        # 2. Drive input neurons in brain
        self.brain.set_poisson_input(indices, rates)

        # 3. Step spiking brain simulation for step_ms (50 ms)
        batch = self.brain.step(self.step_ms)

        # 4. Decode spike batch into game action MultiBinary(3)
        action = self.decoder(batch)
        return action
