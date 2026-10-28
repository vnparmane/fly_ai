"""ArcadeViewer: Real-time dual-pane Pygame viewer for Space Invaders and Spiking Brain telemetry."""

import os
from pathlib import Path
import sys
import numpy as np
import pygame


class ArcadeViewer:
    """Renders a dual-pane window: Space Invaders gameplay on the left,
    neural spike raster and group rate gauges on the right.
    """

    def __init__(
        self,
        width: int = 980,
        height: int = 640,
        headless: bool = False,
        fps: int = 30,
    ) -> None:
        self.width = width
        self.height = height
        self.headless = headless
        self.fps = fps
        self.game_w = 480
        self.game_h = 630

        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        else:
            if "SDL_VIDEODRIVER" in os.environ:
                del os.environ["SDL_VIDEODRIVER"]

        pygame.init()
        pygame.font.init()

        if self.headless:
            self.screen = pygame.Surface((self.width, self.height))
        else:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Fly Brain Arcade 🪰🕹️ — Drosophila Connectome v783")
            pygame.event.pump()

        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("consolas", 13)
        self.font_med = pygame.font.SysFont("consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("consolas", 22, bold=True)

        # Starfield stars for retro arcade background
        np.random.seed(42)
        self.stars = [
            (
                int(np.random.randint(5, self.game_w - 5)),
                int(np.random.randint(5, self.game_h - 5)),
                int(np.random.randint(1, 3)),
            )
            for _ in range(40)
        ]

        # Rolling spike buffer for raster plot: list of (t_step, neuron_idx, group_type)
        self.spike_history: list[tuple[int, int, str]] = []
        self.max_history_steps = 30
        self.step_idx = 0

    def show_loading(self, text: str = "Loading Fly Brain Connectome...", subtext: str = "Initializing Spiking Network...") -> None:
        """Display an immediate responsive loading splash screen."""
        if self.headless:
            return
        self.screen.fill((13, 17, 23))
        title = self.font_large.render(text, True, (0, 255, 136))
        trect = title.get_rect(center=(self.width // 2, self.height // 2 - 20))
        self.screen.blit(title, trect)

        sub = self.font_med.render(subtext, True, (139, 148, 158))
        srect = sub.get_rect(center=(self.width // 2, self.height // 2 + 20))
        self.screen.blit(sub, srect)

        pygame.display.flip()
        pygame.event.pump()

    def process_events(self) -> dict:
        """Process keyboard / window events.
        Returns dict with control flags: {'quit': bool, 'pause_toggle': bool, 'reset': bool, 'switch_ctrl': int | None}
        """
        flags = {
            "quit": False,
            "pause_toggle": False,
            "reset": False,
            "switch_ctrl": None,
        }
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                flags["quit"] = True
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    flags["quit"] = True
                elif event.key == pygame.K_SPACE:
                    flags["pause_toggle"] = True
                elif event.key == pygame.K_r:
                    flags["reset"] = True
                elif event.key == pygame.K_1:
                    flags["switch_ctrl"] = "random"
                elif event.key == pygame.K_2:
                    flags["switch_ctrl"] = "heuristic"
                elif event.key == pygame.K_3:
                    flags["switch_ctrl"] = "brain"
                elif event.key == pygame.K_4:
                    flags["switch_ctrl"] = "shuffled"
                elif event.key == pygame.K_5:
                    flags["switch_ctrl"] = "vnc"
                elif event.key == pygame.K_6:
                    flags["switch_ctrl"] = "trained_brain"

        return flags

    def record_spikes(self, spike_batch, io_spec=None) -> None:
        """Record spike batch into rolling history."""
        self.step_idx += 1
        if spike_batch is None:
            return

        in_neurons = set()
        out_neurons = set()
        if io_spec is not None:
            for g in io_spec.input_groups.values():
                in_neurons.update(int(i) for i in g)
            for g in io_spec.output_groups.values():
                out_neurons.update(int(i) for i in g)

        for nid in spike_batch.neuron_idx:
            nid = int(nid)
            if nid in in_neurons:
                gtype = "input"
            elif nid in out_neurons:
                gtype = "output"
            else:
                gtype = "inter"
            self.spike_history.append((self.step_idx, nid, gtype))

        # Prune old spikes
        min_step = self.step_idx - self.max_history_steps
        self.spike_history = [s for s in self.spike_history if s[0] >= min_step]

    def render_frame(
        self,
        env,
        action: np.ndarray,
        controller_name: str,
        rates: dict[str, float] | None = None,
        fps_actual: float = 0.0,
        is_paused: bool = False,
    ) -> np.ndarray:
        """Render the complete dual-pane view."""
        self.screen.fill((13, 17, 23))  # Dark background

        # 1. Render Left Pane (Space Invaders Game)
        self._render_game_pane(env, is_paused)

        # 2. Render Divider Line
        pygame.draw.line(self.screen, (48, 54, 61), (self.game_w + 10, 5), (self.game_w + 10, self.height - 5), 2)

        # 3. Render Right Pane (Neural Dashboard)
        self._render_neural_pane(env, action, controller_name, rates, fps_actual)

        if not self.headless:
            pygame.display.flip()
            self.clock.tick(self.fps)

        # Return RGB array for recording / verification
        return pygame.surfarray.array3d(self.screen).transpose((1, 0, 2))

    def _render_game_pane(self, env, is_paused: bool) -> None:
        """Render arcade game viewport."""
        game_rect = pygame.Rect(5, 5, self.game_w, self.game_h)
        pygame.draw.rect(self.screen, (5, 8, 15), game_rect)
        pygame.draw.rect(self.screen, (56, 139, 253), game_rect, 2)

        # Draw stars
        for sx, sy, srad in self.stars:
            pygame.draw.circle(self.screen, (100, 110, 130), (5 + sx, 5 + sy), srad)

        scale_x = self.game_w / env.cfg.width
        scale_y = self.game_h / env.cfg.height

        # Header info inside game pane
        score_text = self.font_med.render(f"SCORE: {env.score:04d}", True, (255, 255, 255))
        kills_text = self.font_med.render(f"KILLS: {env.kills:02d}", True, (255, 200, 0))
        steps_text = self.font_small.render(f"STEP: {env.step_count}/{env.cfg.max_steps}", True, (150, 160, 175))
        self.screen.blit(score_text, (20, 15))
        self.screen.blit(kills_text, (200, 15))
        self.screen.blit(steps_text, (340, 17))

        pygame.draw.line(self.screen, (30, 40, 60), (10, 40), (self.game_w, 40), 1)

        # Draw living enemies
        for ex, ey, alive in env.enemies:
            if alive > 0.5:
                cx = int(5 + ex * scale_x)
                cy = int(5 + ey * scale_y)
                ew = int(7 * scale_x)
                eh = int(5 * scale_y)
                # Outer enemy body
                pygame.draw.rect(self.screen, (255, 80, 80), (cx - ew // 2, cy - eh // 2, ew, eh))
                # Inner eye highlight
                pygame.draw.rect(self.screen, (255, 220, 0), (cx - ew // 4, cy - eh // 4, ew // 2, eh // 2))

        # Draw player bullets (cyan laser)
        for bx, by in env.player_bullets:
            cx = int(5 + bx * scale_x)
            cy = int(5 + by * scale_y)
            pygame.draw.line(self.screen, (0, 255, 255), (cx, cy - 6), (cx, cy + 6), 3)

        # Draw enemy bullets (white/yellow bombs)
        for ebx, eby in env.enemy_bullets:
            cx = int(5 + ebx * scale_x)
            cy = int(5 + eby * scale_y)
            pygame.draw.circle(self.screen, (255, 255, 100), (cx, cy), 3)

        # Draw player cannon (green)
        px = int(5 + env.player_x * scale_x)
        py = int(5 + env.player_y * scale_y)
        pw = int(10 * scale_x)
        ph = int(6 * scale_y)
        pygame.draw.rect(self.screen, (0, 255, 128), (px - pw // 2, py - ph // 2, pw, ph))
        # Turret
        pygame.draw.rect(self.screen, (0, 255, 200), (px - 2, py - ph // 2 - 5, 4, 6))

        # Bottom baseline
        pygame.draw.line(self.screen, (0, 180, 80), (10, self.game_h - 10), (self.game_w, self.game_h - 10), 1)

        if is_paused:
            pause_surf = self.font_large.render("PAUSED (SPACE TO RESUME)", True, (255, 255, 0))
            rect = pause_surf.get_rect(center=(self.game_w // 2 + 5, self.game_h // 2))
            pygame.draw.rect(self.screen, (0, 0, 0), rect.inflate(20, 10))
            pygame.draw.rect(self.screen, (255, 255, 0), rect.inflate(20, 10), 2)
            self.screen.blit(pause_surf, rect)

    def _render_neural_pane(
        self,
        env,
        action: np.ndarray,
        controller_name: str,
        rates: dict[str, float] | None,
        fps_actual: float,
    ) -> None:
        """Render neural telemetry dashboard on the right pane."""
        rx = self.game_w + 20
        ry = 15
        rw = self.width - rx - 15

        # 1. Header: Controller Info & Status
        title_surf = self.font_large.render("NEURAL TELEMETRY", True, (88, 166, 255))
        self.screen.blit(title_surf, (rx, ry))
        ry += 30

        ctrl_badge = f"CONTROLLER: {controller_name.upper()}"
        ctrl_color = (0, 255, 136) if "brain" in controller_name else (255, 200, 50)
        self.screen.blit(self.font_med.render(ctrl_badge, True, ctrl_color), (rx, ry))

        fps_text = f"STEP: 50 ms brain | {fps_actual:4.1f} FPS"
        self.screen.blit(self.font_small.render(fps_text, True, (139, 148, 158)), (rx + rw - 220, ry + 2))
        ry += 30

        pygame.draw.line(self.screen, (48, 54, 61), (rx, ry), (rx + rw, ry), 1)
        ry += 15

        # 2. Action Indicators
        self.screen.blit(self.font_med.render("MOTOR OUTPUT ACTIONS", True, (201, 209, 217)), (rx, ry))
        ry += 24

        actions_info = [
            ("◄ LEFT", action[0] == 1, (0, 255, 136)),
            ("RIGHT ►", action[1] == 1, (0, 255, 136)),
            ("▲ FIRE", action[2] == 1, (255, 68, 85)),
        ]
        btn_w = (rw - 20) // 3
        for i, (label, active, active_col) in enumerate(actions_info):
            bx = rx + i * (btn_w + 10)
            brect = pygame.Rect(bx, ry, btn_w, 36)
            bg_col = active_col if active else (22, 27, 34)
            text_col = (0, 0, 0) if active else (100, 110, 125)
            border_col = active_col if active else (48, 54, 61)

            pygame.draw.rect(self.screen, bg_col, brect, border_radius=4)
            pygame.draw.rect(self.screen, border_col, brect, 2, border_radius=4)

            text_surf = self.font_med.render(label, True, text_col)
            trect = text_surf.get_rect(center=brect.center)
            self.screen.blit(text_surf, trect)

        ry += 50
        pygame.draw.line(self.screen, (48, 54, 61), (rx, ry), (rx + rw, ry), 1)
        ry += 15

        # 3. Group Firing Rates (Hz)
        self.screen.blit(self.font_med.render("GROUP FIRING RATES (Hz)", True, (201, 209, 217)), (rx, ry))
        ry += 25

        rates = rates or {"move_left": 0.0, "move_right": 0.0, "fire": 0.0}
        max_bar_rate = 30.0  # reference scaling

        for group, label, color in [
            ("move_left", "Left Motor (DN)", (0, 255, 136)),
            ("move_right", "Right Motor (DN)", (0, 200, 255)),
            ("fire", "Fire Trigger (GF/DN)", (255, 68, 85)),
        ]:
            rate_val = rates.get(group, 0.0)
            self.screen.blit(self.font_small.render(f"{label:18s} {rate_val:5.1f} Hz", True, (170, 180, 195)), (rx, ry))

            # Draw bar gauge
            bar_x = rx + 220
            bar_w = rw - 230
            bar_h = 14
            fill_w = int(np.clip(rate_val / max_bar_rate, 0.0, 1.0) * bar_w)

            pygame.draw.rect(self.screen, (22, 27, 34), (bar_x, ry, bar_w, bar_h))
            if fill_w > 0:
                pygame.draw.rect(self.screen, color, (bar_x, ry, fill_w, bar_h))
            pygame.draw.rect(self.screen, (48, 54, 61), (bar_x, ry, bar_w, bar_h), 1)
            ry += 24

        ry += 10
        pygame.draw.line(self.screen, (48, 54, 61), (rx, ry), (rx + rw, ry), 1)
        ry += 15

        # 4. Rolling Spike Raster Plot
        self.screen.blit(self.font_med.render("SUBCIRCUIT SPIKE RASTER", True, (201, 209, 217)), (rx, ry))
        ry += 24

        raster_rect = pygame.Rect(rx, ry, rw, 150)
        pygame.draw.rect(self.screen, (10, 14, 20), raster_rect)
        pygame.draw.rect(self.screen, (48, 54, 61), raster_rect, 1)

        # Plot spikes inside raster_rect
        if self.spike_history:
            min_s = self.step_idx - self.max_history_steps
            max_s = self.step_idx
            span_s = max(1, max_s - min_s)

            for step_num, nid, gtype in self.spike_history:
                norm_x = (step_num - min_s) / span_s
                sx = int(raster_rect.x + norm_x * (raster_rect.w - 6) + 3)
                norm_y = (nid % 1500) / 1500.0
                sy = int(raster_rect.y + norm_y * (raster_rect.h - 6) + 3)

                if gtype == "input":
                    spk_col = (255, 220, 0)
                elif gtype == "output":
                    spk_col = (255, 68, 85)
                else:
                    spk_col = (88, 166, 255)

                pygame.draw.circle(self.screen, spk_col, (sx, sy), 1)

        ry += 165

        # Legend
        self.screen.blit(self.font_small.render("● Input (VPN)", True, (255, 220, 0)), (rx, ry))
        self.screen.blit(self.font_small.render("● Interneurons", True, (88, 166, 255)), (rx + 140, ry))
        self.screen.blit(self.font_small.render("● Output (DN)", True, (255, 68, 85)), (rx + 280, ry))
        ry += 25

        # 5. Keybindings Footer
        pygame.draw.line(self.screen, (48, 54, 61), (rx, ry), (rx + rw, ry), 1)
        ry += 10
        keys_help = [
            "[SPACE] Pause/Play  |  [R] Reset Seed  |  [ESC] Quit",
            "[1] Random [2] Heuristic [3] Brain [4] Shuffled [5] VNC [6] Trained",
        ]
        for line in keys_help:
            self.screen.blit(self.font_small.render(line, True, (139, 148, 158)), (rx, ry))
            ry += 16

    def close(self) -> None:
        """Close pygame window."""
        pygame.quit()
