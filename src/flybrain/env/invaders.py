"""Space Invaders environment conforming to gymnasium specifications."""

from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import numpy as np


@dataclass(frozen=True)
class InvadersConfig:
    width: int = 160
    height: int = 210
    player_speed: float = 2.0
    bullet_speed: float = 4.0
    fire_cooldown_steps: int = 12
    enemy_rows: int = 3
    enemy_cols: int = 6
    enemy_speed: float = 0.5
    enemy_drop: float = 6.0
    enemy_fire_prob: float = 0.01
    max_steps: int = 600
    waves: int = 1


class SpaceInvadersEnv(gym.Env):
    """Custom Space Invaders Gym environment.
    
    Action space: MultiBinary(3) -> [left, right, fire]
    Observation space: Dict of normalized features in [-1, 1] (or [0, 1] for counts/can_fire)
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: InvadersConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = InvadersConfig()
        self.cfg = config

        # Action: MultiBinary(3) -> [left, right, fire]
        self.action_space = spaces.MultiBinary(3)

        # Observation dict
        self.observation_space = spaces.Dict(
            {
                "player_x": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                "target_dx": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                "target_dy": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                "enemy_count": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "can_fire": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "threat_dx": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                "threat_dy": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

        self._rng = np.random.default_rng()
        self.player_x = 0.0
        self.player_y = 0.0
        self.player_bullets: list[list[float]] = []
        self.enemies: list[list[float]] = []  # [x, y, alive_bool]
        self.enemy_bullets: list[list[float]] = []
        self.enemy_dir = 1.0  # +1 right, -1 left
        self.cooldown = 0
        self.step_count = 0
        self.score = 0
        self.kills = 0
        self.shots_fired = 0

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.step_count = 0
        self.score = 0
        self.kills = 0
        self.shots_fired = 0
        self.cooldown = 0

        # Player at bottom center
        self.player_x = self.cfg.width / 2.0
        self.player_y = self.cfg.height - 20.0
        self.player_bullets = []
        self.enemy_bullets = []
        self.enemy_dir = 1.0

        # Initialize enemies grid
        self.enemies = []
        start_x = 20.0
        start_y = 30.0
        x_spacing = (self.cfg.width - 40.0) / max(1, self.cfg.enemy_cols - 1)
        y_spacing = 15.0

        for r in range(self.cfg.enemy_rows):
            for c in range(self.cfg.enemy_cols):
                ex = start_x + c * x_spacing
                ey = start_y + r * y_spacing
                self.enemies.append([ex, ey, 1.0])

        obs = self._get_obs()
        info = {"score": 0, "kills": 0, "shots_fired": 0, "steps": 0}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False

        action = np.asarray(action, dtype=np.int32)
        move_left = bool(action[0])
        move_right = bool(action[1])
        fire = bool(action[2])

        # Player horizontal movement (left and right cancel)
        dx = 0.0
        if move_left and not move_right:
            dx = -self.cfg.player_speed
        elif move_right and not move_left:
            dx = self.cfg.player_speed

        self.player_x = float(np.clip(self.player_x + dx, 10.0, self.cfg.width - 10.0))

        # Cooldown update
        if self.cooldown > 0:
            self.cooldown -= 1

        # Player fire
        if fire and self.cooldown == 0:
            self.player_bullets.append([self.player_x, self.player_y - 4.0])
            self.cooldown = self.cfg.fire_cooldown_steps
            self.shots_fired += 1

        # Advance player bullets
        new_bullets = []
        for bx, by in self.player_bullets:
            by_new = by - self.cfg.bullet_speed
            if by_new > 0:
                new_bullets.append([bx, by_new])
        self.player_bullets = new_bullets

        # Advance enemy fleet
        living_enemies = [e for e in self.enemies if e[2] > 0.5]
        hit_edge = False
        for e in living_enemies:
            e[0] += self.enemy_dir * self.cfg.enemy_speed
            if e[0] <= 10.0 or e[0] >= self.cfg.width - 10.0:
                hit_edge = True

        if hit_edge:
            self.enemy_dir *= -1.0
            for e in living_enemies:
                e[1] += self.cfg.enemy_drop

        # Enemy firing
        for e in living_enemies:
            if self._rng.random() < self.cfg.enemy_fire_prob:
                self.enemy_bullets.append([e[0], e[1] + 4.0])

        # Advance enemy bullets
        new_ebullets = []
        for ebx, eby in self.enemy_bullets:
            eby_new = eby + self.cfg.bullet_speed * 0.75
            if eby_new < self.cfg.height:
                new_ebullets.append([ebx, eby_new])
        self.enemy_bullets = new_ebullets

        # Check player bullets hitting enemies
        for b in list(self.player_bullets):
            bx, by = b
            for e in living_enemies:
                if e[2] > 0.5:
                    ex, ey = e[0], e[1]
                    if abs(bx - ex) <= 6.0 and abs(by - ey) <= 6.0:
                        e[2] = 0.0  # kill enemy
                        if b in self.player_bullets:
                            self.player_bullets.remove(b)
                        reward += 1.0
                        self.kills += 1
                        self.score += 100
                        break

        # Check termination: player hit by enemy bullet
        for ebx, eby in self.enemy_bullets:
            if abs(ebx - self.player_x) <= 6.0 and abs(eby - self.player_y) <= 6.0:
                terminated = True
                break

        # Check termination: enemies reach player row
        for e in self.enemies:
            if e[2] > 0.5 and e[1] >= self.player_y - 6.0:
                terminated = True
                break

        # Check termination: all enemies eliminated
        living_count = sum(1 for e in self.enemies if e[2] > 0.5)
        if living_count == 0:
            terminated = True

        # Check truncation: max steps reached
        if self.step_count >= self.cfg.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {
            "score": self.score,
            "kills": self.kills,
            "shots_fired": self.shots_fired,
            "steps": self.step_count,
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> dict:
        total_enemies = self.cfg.enemy_rows * self.cfg.enemy_cols
        living = [e for e in self.enemies if e[2] > 0.5]
        enemy_fraction = float(len(living) / total_enemies) if total_enemies > 0 else 0.0

        # Target rule: enemy minimising |dx|, tie broken by largest y (closest to player)
        target_dx = 0.0
        target_dy = 0.0
        if living:
            living_sorted = sorted(living, key=lambda e: (abs(e[0] - self.player_x), -e[1]))
            best_target = living_sorted[0]
            # Normalise to [-1, 1] relative to screen dimensions
            target_dx = float((best_target[0] - self.player_x) / (self.cfg.width / 2.0))
            target_dy = float((best_target[1] - self.player_y) / self.cfg.height)

        # Threat rule: nearest enemy bullet
        threat_dx = 0.0
        threat_dy = 0.0
        if self.enemy_bullets:
            bullets_sorted = sorted(
                self.enemy_bullets,
                key=lambda b: ((b[0] - self.player_x) ** 2 + (b[1] - self.player_y) ** 2),
            )
            nearest_b = bullets_sorted[0]
            threat_dx = float((nearest_b[0] - self.player_x) / (self.cfg.width / 2.0))
            threat_dy = float((nearest_b[1] - self.player_y) / self.cfg.height)

        norm_player_x = float((self.player_x / (self.cfg.width / 2.0)) - 1.0)
        can_fire_val = 1.0 if self.cooldown == 0 else 0.0

        return {
            "player_x": np.array([np.clip(norm_player_x, -1.0, 1.0)], dtype=np.float32),
            "target_dx": np.array([np.clip(target_dx, -1.0, 1.0)], dtype=np.float32),
            "target_dy": np.array([np.clip(target_dy, -1.0, 1.0)], dtype=np.float32),
            "enemy_count": np.array([np.clip(enemy_fraction, 0.0, 1.0)], dtype=np.float32),
            "can_fire": np.array([can_fire_val], dtype=np.float32),
            "threat_dx": np.array([np.clip(threat_dx, -1.0, 1.0)], dtype=np.float32),
            "threat_dy": np.array([np.clip(threat_dy, -1.0, 1.0)], dtype=np.float32),
        }

    def render(self) -> np.ndarray:
        """Render frame to RGB array [H, W, 3] (headless safe)."""
        w, h = self.cfg.width, self.cfg.height
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw player (green)
        px, py = int(self.player_x), int(self.player_y)
        frame[max(0, py - 3):min(h, py + 4), max(0, px - 5):min(w, px + 6)] = [0, 255, 0]

        # Draw living enemies (red/yellow)
        for ex, ey, alive in self.enemies:
            if alive > 0.5:
                ix, iy = int(ex), int(ey)
                frame[max(0, iy - 3):min(h, iy + 4), max(0, ix - 4):min(w, ix + 5)] = [255, 200, 0]

        # Draw player bullets (cyan)
        for bx, by in self.player_bullets:
            ix, iy = int(bx), int(by)
            frame[max(0, iy - 2):min(h, iy + 3), max(0, ix - 1):min(w, ix + 2)] = [0, 255, 255]

        # Draw enemy bullets (white)
        for ebx, eby in self.enemy_bullets:
            ix, iy = int(ebx), int(eby)
            frame[max(0, iy - 2):min(h, iy + 3), max(0, ix - 1):min(w, ix + 2)] = [255, 255, 255]

        return frame
