"""Heuristic baseline controller."""

import numpy as np


class HeuristicController:
    """Move towards nearest target enemy; dodge bullets; fire when target is aligned."""

    def __init__(self, fire_thresh: float = 0.20) -> None:
        self.fire_thresh = fire_thresh

    def reset(self, seed: int, reset_weights: bool = False) -> None:
        pass

    def act(self, obs: dict) -> np.ndarray:
        target_dx = float(obs["target_dx"][0])
        threat_dx = float(obs["threat_dx"][0])
        threat_dy = float(obs["threat_dy"][0])
        action = np.zeros(3, dtype=np.int8)

        # 1. Threat avoidance: if bullet is directly approaching (within 0.1 dx and close dy)
        if abs(threat_dx) < 0.12 and abs(threat_dy) < 0.35:
            # Dodge away from bullet
            if threat_dx >= 0:
                action[0] = 1  # move left
            else:
                action[1] = 1  # move right
        else:
            # 2. Track target
            if target_dx < -0.02:
                action[0] = 1
            elif target_dx > 0.02:
                action[1] = 1

        # 3. Fire when aligned
        if abs(target_dx) <= self.fire_thresh:
            action[2] = 1

        return action
