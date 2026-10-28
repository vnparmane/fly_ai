"""ArcadeViewer: Enhanced dual-pane Pygame arcade renderer with retro sprites and neural telemetry."""

import os
from pathlib import Path
import sys
import numpy as np
import pygame


class ArcadeViewer:
    """Renders an arcade experience: Space Invaders gameplay on the left,
    neural spike raster, group rates, and action explanations on the right.
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
        self.font_title = pygame.font.SysFont("consolas", 28, bold=True)

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

        # Muzzle flash and hit particle effects
        self.muzzle_flash_timer = 0
        self.explosions: list[list[float]] = []  # [x, y, radius, timer]

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
        """Process keyboard / window events."""
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
            for g in getattr(io_spec, "input_groups", {}).values():
                in_neurons.update(int(i) for i in g)
            for g in getattr(io_spec, "output_groups", {}).values():
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

        min_step = self.step_idx - self.max_history_steps
        self.spike_history = [s for s in self.spike_history if s[0] >= min_step]

    def trigger_explosion(self, x: float, y: float) -> None:
        scale_x = self.game_w / 160.0
        scale_y = self.game_h / 210.0
        cx = int(5 + x * scale_x)
        cy = int(5 + y * scale_y)
        self.explosions.append([cx, cy, 3.0, 8.0])

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
        self.screen.fill((10, 13, 18))

        if action[2] == 1:
            self.muzzle_flash_timer = 3
        elif self.muzzle_flash_timer > 0:
            self.muzzle_flash_timer -= 1

        # 1. Render Left Pane (Space Invaders Game)
        self._render_game_pane(env, is_paused)

        # 2. Render Divider Line
        pygame.draw.line(self.screen, (48, 54, 61), (self.game_w + 10, 5), (self.game_w + 10, self.height - 5), 2)

        # 3. Render Right Pane (Neural Dashboard)
        self._render_neural_pane(env, action, controller_name, rates, fps_actual)

        if not self.headless:
            pygame.display.flip()
            self.clock.tick(self.fps)

        return pygame.surfarray.array3d(self.screen).transpose((1, 0, 2))

    def _draw_invader(self, cx: int, cy: int, row: int, frame_idx: int) -> None:
        """Draw authentic retro arcade alien sprites."""
        # Row 0: Squid (magenta), Row 1: Crab (cyan), Row 2: Octopus (green)
        colors = [(255, 68, 170), (0, 255, 255), (57, 255, 20)]
        color = colors[row % 3]

        if row == 0:
            # Squid Alien (8x8 pixel blocks)
            # Head
            pygame.draw.rect(self.screen, color, (cx - 8, cy - 8, 16, 10))
            # Eyes
            pygame.draw.rect(self.screen, (10, 13, 18), (cx - 5, cy - 5, 3, 3))
            pygame.draw.rect(self.screen, (10, 13, 18), (cx + 2, cy - 5, 3, 3))
            # Tentacles (alternate frames)
            if frame_idx == 0:
                pygame.draw.line(self.screen, color, (cx - 7, cy + 2), (cx - 10, cy + 8), 2)
                pygame.draw.line(self.screen, color, (cx - 2, cy + 2), (cx - 2, cy + 8), 2)
                pygame.draw.line(self.screen, color, (cx + 2, cy + 2), (cx + 2, cy + 8), 2)
                pygame.draw.line(self.screen, color, (cx + 7, cy + 2), (cx + 10, cy + 8), 2)
            else:
                pygame.draw.line(self.screen, color, (cx - 7, cy + 2), (cx - 4, cy + 8), 2)
                pygame.draw.line(self.screen, color, (cx - 2, cy + 2), (cx - 5, cy + 8), 2)
                pygame.draw.line(self.screen, color, (cx + 2, cy + 2), (cx + 5, cy + 8), 2)
                pygame.draw.line(self.screen, color, (cx + 7, cy + 2), (cx + 4, cy + 8), 2)

        elif row == 1:
            # Crab Alien
            pygame.draw.rect(self.screen, color, (cx - 10, cy - 6, 20, 10), border_radius=2)
            # Eyes
            pygame.draw.rect(self.screen, (10, 13, 18), (cx - 6, cy - 4, 3, 3))
            pygame.draw.rect(self.screen, (10, 13, 18), (cx + 3, cy - 4, 3, 3))
            # Claws
            if frame_idx == 0:
                pygame.draw.rect(self.screen, color, (cx - 13, cy - 9, 4, 6))
                pygame.draw.rect(self.screen, color, (cx + 9, cy - 9, 4, 6))
                pygame.draw.line(self.screen, color, (cx - 6, cy + 4), (cx - 10, cy + 8), 2)
                pygame.draw.line(self.screen, color, (cx + 6, cy + 4), (cx + 10, cy + 8), 2)
            else:
                pygame.draw.rect(self.screen, color, (cx - 13, cy - 3, 4, 6))
                pygame.draw.rect(self.screen, color, (cx + 9, cy - 3, 4, 6))
                pygame.draw.line(self.screen, color, (cx - 6, cy + 4), (cx - 6, cy + 8), 2)
                pygame.draw.line(self.screen, color, (cx + 6, cy + 4), (cx + 6, cy + 8), 2)

        else:
            # Octopus Alien
            pygame.draw.circle(self.screen, color, (cx, cy - 2), 8)
            # Eyes
            pygame.draw.rect(self.screen, (10, 13, 18), (cx - 5, cy - 4, 3, 3))
            pygame.draw.rect(self.screen, (10, 13, 18), (cx + 2, cy - 4, 3, 3))
            # Tentacles
            t_offset = 6 if frame_idx == 0 else 3
            pygame.draw.line(self.screen, color, (cx - 6, cy + 4), (cx - t_offset, cy + 9), 2)
            pygame.draw.line(self.screen, color, (cx, cy + 4), (cx, cy + 9), 2)
            pygame.draw.line(self.screen, color, (cx + 6, cy + 4), (cx + t_offset, cy + 9), 2)

    def _render_game_pane(self, env, is_paused: bool) -> None:
        """Render arcade game viewport."""
        game_rect = pygame.Rect(5, 5, self.game_w, self.game_h)
        pygame.draw.rect(self.screen, (5, 8, 15), game_rect)
        pygame.draw.rect(self.screen, (56, 139, 253), game_rect, 2)

        # Starfield
        for sx, sy, srad in self.stars:
            pygame.draw.circle(self.screen, (90, 100, 120), (5 + sx, 5 + sy), srad)

        scale_x = self.game_w / env.cfg.width
        scale_y = self.game_h / env.cfg.height

        # --- Top HUD Header ---
        score_val = getattr(env, "score", 0)
        kills_val = getattr(env, "kills", 0)
        lives_val = getattr(env, "lives", 3)

        score_text = self.font_med.render(f"SCORE: {score_val:04d}", True, (255, 230, 0))
        kills_text = self.font_med.render(f"KILLS: {kills_val:02d}/18", True, (0, 255, 255))
        self.screen.blit(score_text, (20, 14))
        self.screen.blit(kills_text, (180, 14))

        # Hearts for Lives
        heart_symbols = "♥ " * lives_val
        lives_surf = self.font_med.render(f"LIVES: {heart_symbols}", True, (255, 50, 70))
        self.screen.blit(lives_surf, (340, 14))

        pygame.draw.line(self.screen, (30, 45, 65), (10, 38), (self.game_w, 38), 1)

        # --- Draw Defensive Bunkers ---
        for blk in getattr(env, "bunkers", []):
            bx, by, hp = blk
            if hp > 0:
                cx = int(5 + bx * scale_x)
                cy = int(5 + by * scale_y)
                bw = int(12 * scale_x)
                bh = int(8 * scale_y)
                # Color fades as bunker takes damage: 3 -> bright green, 2 -> yellow, 1 -> orange/red
                b_colors = {3: (46, 204, 113), 2: (241, 196, 15), 1: (231, 76, 60)}
                col = b_colors.get(hp, (46, 204, 113))
                pygame.draw.rect(self.screen, col, (cx - bw // 2, cy - bh // 2, bw, bh), border_radius=2)
                pygame.draw.rect(self.screen, (20, 30, 40), (cx - bw // 2, cy - bh // 2, bw, bh), 1)

        # --- Draw Animated Invaders ---
        anim_frame = (env.step_count // 8) % 2
        for idx, (ex, ey, alive) in enumerate(env.enemies):
            if alive > 0.5:
                cx = int(5 + ex * scale_x)
                cy = int(5 + ey * scale_y)
                row = idx // env.cfg.enemy_cols
                self._draw_invader(cx, cy, row, anim_frame)

        # --- Draw Explosions ---
        new_explosions = []
        for exp in self.explosions:
            cx, cy, rad, timer = exp
            pygame.draw.circle(self.screen, (255, 200, 50), (int(cx), int(cy)), int(rad), 2)
            pygame.draw.circle(self.screen, (255, 80, 0), (int(cx), int(cy)), max(1, int(rad * 0.6)))
            if timer > 1:
                new_explosions.append([cx, cy, rad + 1.5, timer - 1])
        self.explosions = new_explosions

        # --- Draw Player Bullets (Glowing Cyan Lasers) ---
        for bx, by in env.player_bullets:
            cx = int(5 + bx * scale_x)
            cy = int(5 + by * scale_y)
            # Outer cyan glow
            pygame.draw.line(self.screen, (0, 180, 255), (cx, cy - 8), (cx, cy + 8), 5)
            # Inner bright white laser core
            pygame.draw.line(self.screen, (255, 255, 255), (cx, cy - 6), (cx, cy + 6), 2)

        # --- Draw Enemy Bullets (Pulsing Red/Orange Bombs) ---
        for ebx, eby in env.enemy_bullets:
            cx = int(5 + ebx * scale_x)
            cy = int(5 + eby * scale_y)
            # Jagged bomb
            pygame.draw.circle(self.screen, (255, 60, 60), (cx, cy), 4)
            pygame.draw.circle(self.screen, (255, 220, 0), (cx, cy), 2)
            pygame.draw.line(self.screen, (255, 100, 0), (cx, cy - 6), (cx, cy + 6), 2)

        # --- Draw Player Cannon ---
        is_invulnerable = getattr(env, "invulnerable_steps", 0) > 0
        blink = (env.step_count // 3) % 2 == 0

        if not is_invulnerable or blink:
            px = int(5 + env.player_x * scale_x)
            py = int(5 + env.player_y * scale_y)
            pw = int(14 * scale_x)
            ph = int(7 * scale_y)

            # Tank base
            pygame.draw.rect(self.screen, (0, 200, 100), (px - pw // 2, py - ph // 2, pw, ph), border_radius=3)
            # Cannon turret
            pygame.draw.rect(self.screen, (0, 255, 150), (px - 2, py - ph // 2 - 6, 5, 8), border_radius=1)
            # Muzzle flash effect when firing
            if self.muzzle_flash_timer > 0:
                pygame.draw.circle(self.screen, (255, 255, 200), (px, py - ph // 2 - 8), 5)
                pygame.draw.line(self.screen, (0, 255, 255), (px, py - ph // 2 - 12), (px, py - ph // 2 - 4), 2)

        # Ground baseline
        pygame.draw.line(self.screen, (0, 255, 128), (10, self.game_h - 10), (self.game_w, self.game_h - 10), 2)

        # --- Game Over / Wave Clear / Pause Overlays ---
        if getattr(env, "lives", 1) <= 0:
            gov_surf = self.font_title.render("GAME OVER", True, (255, 50, 50))
            score_ov = self.font_med.render(f"FINAL SCORE: {env.score}   KILLS: {env.kills}", True, (255, 255, 255))
            rect = gov_surf.get_rect(center=(self.game_w // 2 + 5, self.game_h // 2 - 15))
            srect = score_ov.get_rect(center=(self.game_w // 2 + 5, self.game_h // 2 + 20))
            box = rect.union(srect).inflate(40, 30)
            pygame.draw.rect(self.screen, (10, 10, 20), box)
            pygame.draw.rect(self.screen, (255, 50, 50), box, 2)
            self.screen.blit(gov_surf, rect)
            self.screen.blit(score_ov, srect)

        elif is_paused:
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
        ry = 14
        rw = self.width - rx - 15

        # 1. Header: Controller Info & Status
        title_surf = self.font_title.render("NEURAL DASHBOARD", True, (88, 166, 255))
        self.screen.blit(title_surf, (rx, ry))
        ry += 32

        ctrl_names_map = {
            "trained_brain": "Trained Connectome (R-STDP)",
            "vnc": "FAFB Connectome + VNC Motor Circuit",
            "brain": "Full Connectome LIF (Zero-Plasticity)",
            "heuristic": "Engineered Heuristic Baseline",
            "random": "Random Bernoulli Baseline",
            "shuffled": "Shuffled Connectome Control",
        }
        badge_name = ctrl_names_map.get(controller_name, controller_name.upper())
        ctrl_color = (0, 255, 136) if "brain" in controller_name or "vnc" in controller_name else (255, 200, 50)
        self.screen.blit(self.font_med.render(f"ACTIVE: {badge_name}", True, ctrl_color), (rx, ry))

        fps_text = f"STEP: 50ms | {fps_actual:4.1f} FPS"
        self.screen.blit(self.font_small.render(fps_text, True, (139, 148, 158)), (rx + rw - 180, ry + 2))
        ry += 28

        pygame.draw.line(self.screen, (48, 54, 61), (rx, ry), (rx + rw, ry), 1)
        ry += 12

        # 2. Big Real-Time Action Status Box
        self.screen.blit(self.font_med.render("REAL-TIME MOTOR OUTPUT STATUS", True, (201, 209, 217)), (rx, ry))
        ry += 24

        # Action status banner
        move_left = action[0] == 1
        move_right = action[1] == 1
        fire_active = action[2] == 1

        if fire_active and (move_left or move_right):
            steer_txt = "◄ STEERING LEFT" if move_left else "STEERING RIGHT ►"
            status_text = f"🔥🔥 {steer_txt} + FIRING! 🔥🔥"
            box_bg = (120, 25, 30)
            box_border = (255, 68, 85)
            txt_col = (255, 255, 255)
        elif fire_active:
            status_text = "🔥🔥 FIRING LASER CANNON! 🔥🔥"
            box_bg = (120, 30, 20)
            box_border = (255, 80, 0)
            txt_col = (255, 255, 200)
        elif move_left:
            status_text = "◄◄◄ STEERING LEFT ◄◄◄"
            box_bg = (10, 50, 40)
            box_border = (0, 255, 136)
            txt_col = (0, 255, 136)
        elif move_right:
            status_text = "►►► STEERING RIGHT ►►►"
            box_bg = (10, 45, 60)
            box_border = (0, 200, 255)
            txt_col = (0, 200, 255)
        else:
            status_text = "● HOLDING POSITION ●"
            box_bg = (22, 27, 34)
            box_border = (48, 54, 61)
            txt_col = (139, 148, 158)

        status_rect = pygame.Rect(rx, ry, rw, 40)
        pygame.draw.rect(self.screen, box_bg, status_rect, border_radius=4)
        pygame.draw.rect(self.screen, box_border, status_rect, 2, border_radius=4)
        s_surf = self.font_med.render(status_text, True, txt_col)
        self.screen.blit(s_surf, s_surf.get_rect(center=status_rect.center))
        ry += 48

        # Explanatory Sensory Drive Subtitle
        obs_dict = getattr(env, "_get_obs", lambda: {})()
        tdx = float(obs_dict.get("target_dx", [0.0])[0]) if "target_dx" in obs_dict else 0.0
        if abs(tdx) < 0.08:
            explain = f"Visual Input: Target aligned (dx={tdx:+.2f}) → Bilateral DNs fire!"
        elif tdx < 0:
            explain = f"Visual Input: Target on LEFT (dx={tdx:+.2f}) → DNa02 Left Steering"
        else:
            explain = f"Visual Input: Target on RIGHT (dx={tdx:+.2f}) → DNa02 Right Steering"
        self.screen.blit(self.font_small.render(explain, True, (160, 175, 190)), (rx, ry))
        ry += 24

        pygame.draw.line(self.screen, (48, 54, 61), (rx, ry), (rx + rw, ry), 1)
        ry += 12

        # 3. Group Firing Rates (Hz)
        self.screen.blit(self.font_med.render("DESCENDING NEURON FIRING RATES (Hz)", True, (201, 209, 217)), (rx, ry))
        ry += 22

        rates = rates or {"move_left": 0.0, "move_right": 0.0, "fire": 0.0}
        max_bar_rate = 30.0

        for group, label, color in [
            ("move_left", "DNa02 Left Steering (DN)", (0, 255, 136)),
            ("move_right", "DNa02 Right Steering (DN)", (0, 200, 255)),
            ("fire", "Trigger Cannon (DN/GF)", (255, 80, 50)),
        ]:
            rate_val = rates.get(group, 0.0)
            self.screen.blit(self.font_small.render(f"{label:26s} {rate_val:5.1f} Hz", True, (170, 180, 195)), (rx, ry))

            bar_x = rx + 240
            bar_w = rw - 250
            bar_h = 13
            fill_w = int(np.clip(rate_val / max_bar_rate, 0.0, 1.0) * bar_w)

            pygame.draw.rect(self.screen, (22, 27, 34), (bar_x, ry, bar_w, bar_h))
            if fill_w > 0:
                pygame.draw.rect(self.screen, color, (bar_x, ry, fill_w, bar_h))
            pygame.draw.rect(self.screen, (48, 54, 61), (bar_x, ry, bar_w, bar_h), 1)
            ry += 22

        ry += 8
        pygame.draw.line(self.screen, (48, 54, 61), (rx, ry), (rx + rw, ry), 1)
        ry += 12

        # 4. Rolling Spike Raster Plot
        self.screen.blit(self.font_med.render("SUBCIRCUIT SPIKE RASTER (LAST 30 STEPS)", True, (201, 209, 217)), (rx, ry))
        ry += 22

        raster_rect = pygame.Rect(rx, ry, rw, 140)
        pygame.draw.rect(self.screen, (10, 14, 20), raster_rect)
        pygame.draw.rect(self.screen, (48, 54, 61), raster_rect, 1)

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

        ry += 150
        # Legend
        self.screen.blit(self.font_small.render("● Input VPNs", True, (255, 220, 0)), (rx, ry))
        self.screen.blit(self.font_small.render("● Connectome Interneurons", True, (88, 166, 255)), (rx + 120, ry))
        self.screen.blit(self.font_small.render("● Motor DNs", True, (255, 68, 85)), (rx + 330, ry))
        ry += 22

        # 5. Keybindings Footer
        pygame.draw.line(self.screen, (48, 54, 61), (rx, ry), (rx + rw, ry), 1)
        ry += 8
        keys_help = [
            "[SPACE] Pause/Play  |  [R] Reset Seed  |  [ESC] Quit",
            "[1] Random [2] Heuristic [3] Brain [4] Shuffled [5] VNC [6] Trained",
        ]
        for line in keys_help:
            self.screen.blit(self.font_small.render(line, True, (139, 148, 158)), (rx, ry))
            ry += 15

    def close(self) -> None:
        """Close pygame window."""
        pygame.quit()
