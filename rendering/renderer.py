"""
rendering/renderer.py
=====================
Pygame 2D Tactical Renderer -- MDMARD Phase 1

Renders the simulation world with a radar-style top-down 2D view.
Completely decoupled from physics, reward, and RL logic.

The renderer receives only what it needs to draw:
  - AircraftState  (position, pitch, speed)
  - World          (waypoint, obstacles, bounds)
  - dict           (reward breakdown for HUD)
  - dict           (episode info for HUD)

Design
------
- Dark tactical theme (#0a0f1a background, green waypoint, blue aircraft)
- Aircraft drawn as a rotated triangle, nose pointing in heading direction
- Velocity vector arrow shows current speed and direction
- Waypoint shows capture-radius ring with pulse animation
- Coordinate convention: Pygame y-axis is INVERTED (y=0 at top)
  so we flip: screen_y = screen_height - sim_y
- Fully optional: pass render_mode=None to TacticalFlightEnv for headless

Top-down view is chosen (over side-view) because:
  1. Consistent across Air, Ground, Sea domains (Phase 2+)
  2. Matches radar/tactical display conventions
  3. Easier to visualize multi-agent scenarios

Colors (tactical palette)
-------------------------
  Background  : #0a0f1a  (near-black navy)
  Grid        : #1a2540  (dark blue grid)
  Aircraft    : #4fc3f7  (tactical blue)
  Waypoint    : #00e676  (green target)
  Danger zone : #ff5252  (red boundary)
  HUD text    : #e0e0e0  (light grey)
  Speed vec   : #ffeb3b  (amber)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from envs.physics import AircraftState
from envs.world import World


# ─────────────────────────────────────────────────────────────────────────────
# Color Palette
# ─────────────────────────────────────────────────────────────────────────────

class Colors:
    BG          = (10,  15,  26)    # #0a0f1a
    GRID        = (26,  37,  64)    # #1a2540
    AIRCRAFT    = (79,  195, 247)   # #4fc3f7  tactical blue
    AIRCRAFT_FILL = (30, 100, 160)  # darker fill
    WAYPOINT    = (0,   230, 118)   # #00e676  green target
    WAYPOINT_RING = (0, 180, 90)    # darker ring
    FLOOR       = (255, 82,  82)    # #ff5252  danger red
    CEILING     = (255, 145, 0)     # #ff9100  warning orange
    SPEED_VEC   = (255, 235, 59)    # #ffeb3b  amber
    TRAIL       = (79,  195, 247, 80)  # transparent blue
    HUD_BG      = (10,  20,  40,  200)
    HUD_TEXT    = (224, 224, 224)   # #e0e0e0
    HUD_ACCENT  = (79,  195, 247)   # tactical blue
    WHITE       = (255, 255, 255)
    OBSTACLE    = (255, 82,  82)    # red obstacles


# ─────────────────────────────────────────────────────────────────────────────
# Renderer
# ─────────────────────────────────────────────────────────────────────────────

class Renderer:
    """
    Pygame-based 2D tactical renderer for TacticalFlightEnv.

    Parameters
    ----------
    world_width : float
        Simulation world width (matches WorldConfig.width).
    world_height : float
        Simulation world height (matches WorldConfig.height).
    screen_w : int
        Pygame window width in pixels. Default 1200.
    screen_h : int
        Pygame window height in pixels (env area only). Default 800.
    fps : int
        Target frames per second. Default 60.

    Example
    -------
    >>> renderer = Renderer(world_width=1200, world_height=800)
    >>> renderer.init()
    >>> renderer.draw(state, world, reward_breakdown, info)
    >>> renderer.tick()
    >>> renderer.close()
    """

    # HUD panel width in pixels (right side panel)
    HUD_WIDTH = 280

    def __init__(
        self,
        world_width:  float = 1200.0,
        world_height: float = 800.0,
        screen_w:     int   = 1200,
        screen_h:     int   = 800,
        fps:          int   = 60,
    ) -> None:
        self.world_w  = world_width
        self.world_h  = world_height
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.fps      = fps

        # Simulation viewport (left of HUD panel)
        self.sim_w = screen_w - self.HUD_WIDTH
        self.sim_h = screen_h

        # Scale factors: simulation units → pixels
        self.scale_x = self.sim_w / world_width
        self.scale_y = self.sim_h / world_height

        # Pygame objects (created on init())
        self._screen  = None
        self._clock   = None
        self._fonts   = {}
        self._initialized = False

        # Trail buffer (last N positions)
        self._trail: list[tuple[int, int]] = []
        self._trail_max = 80

        # Waypoint pulse animation
        self._pulse_t: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def init(self) -> None:
        """
        Initialize Pygame and create the window.

        Must be called before draw(). Safe to call multiple times.
        """
        if self._initialized:
            return

        import pygame
        pygame.init()
        self._screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption(
            "MDMARD — Tactical Flight Simulation  |  Phase 1"
        )
        self._clock = pygame.time.Clock()

        # Load fonts
        self._fonts = {
            "title":  pygame.font.SysFont("consolas", 15, bold=True),
            "hud":    pygame.font.SysFont("consolas", 13),
            "small":  pygame.font.SysFont("consolas", 11),
            "large":  pygame.font.SysFont("consolas", 18, bold=True),
        }
        self._initialized = True

    def close(self) -> None:
        """Shut down Pygame cleanly."""
        if self._initialized:
            import pygame
            pygame.quit()
            self._initialized = False

    def tick(self) -> None:
        """Cap framerate and update display. Call once per step."""
        import pygame
        pygame.display.flip()
        self._clock.tick(self.fps)

    def handle_events(self) -> list:
        """
        Process Pygame event queue. Returns list of KEYDOWN events.

        Call this every frame in manual_control.py to get keyboard input.
        """
        import pygame
        events = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return [None]   # signal to close
            if event.type == pygame.KEYDOWN:
                events.append(event)
        return events

    # ── Main Draw ──────────────────────────────────────────────────────────────

    def draw(
        self,
        state: AircraftState,
        world: World,
        reward_breakdown: dict[str, float],
        info: dict,
    ) -> None:
        """
        Render one frame.

        Parameters
        ----------
        state : AircraftState
            Current aircraft kinematic state.
        world : World
            World state (waypoint, obstacles, bounds).
        reward_breakdown : dict[str, float]
            Per-component reward values for HUD bars.
        info : dict
            Episode info dict (step, speed, fuel, episode_reward, etc.)
        """
        import pygame

        self._pulse_t += 0.05

        # ── Background ─────────────────────────────────────────────────────
        self._screen.fill(Colors.BG)

        # ── Grid ───────────────────────────────────────────────────────────
        self._draw_grid()

        # ── Danger zones (floor / ceiling) ─────────────────────────────────
        self._draw_danger_zones(world)

        # ── Obstacles ──────────────────────────────────────────────────────
        self._draw_obstacles(world)

        # ── Waypoint ───────────────────────────────────────────────────────
        if world.waypoint is not None:
            self._draw_waypoint(world)

        # ── Trail ──────────────────────────────────────────────────────────
        sx, sy = self._to_screen(state.x, state.y)
        self._trail.append((sx, sy))
        if len(self._trail) > self._trail_max:
            self._trail.pop(0)
        self._draw_trail()

        # ── Aircraft ───────────────────────────────────────────────────────
        self._draw_aircraft(state)

        # ── Velocity vector ────────────────────────────────────────────────
        self._draw_velocity_vector(state)

        # ── HUD panel ──────────────────────────────────────────────────────
        self._draw_hud(state, world, reward_breakdown, info)

        # ── Divider line between sim and HUD ───────────────────────────────
        pygame.draw.line(
            self._screen, Colors.GRID,
            (self.sim_w, 0), (self.sim_w, self.screen_h), 1
        )

    # ── Private Draw Methods ──────────────────────────────────────────────────

    def _to_screen(self, sim_x: float, sim_y: float) -> tuple[int, int]:
        """
        Convert simulation coordinates to screen pixels.

        Pygame has y=0 at top; simulation has y=0 at bottom.
        We flip the y-axis.
        """
        px = int(sim_x * self.scale_x)
        py = int(self.sim_h - sim_y * self.scale_y)
        return px, py

    def _draw_grid(self) -> None:
        """Draw background tactical grid."""
        import pygame
        step_x = int(self.sim_w / 10)
        step_y = int(self.sim_h / 8)
        for x in range(0, self.sim_w, step_x):
            pygame.draw.line(self._screen, Colors.GRID, (x, 0), (x, self.sim_h), 1)
        for y in range(0, self.sim_h, step_y):
            pygame.draw.line(self._screen, Colors.GRID, (0, y), (self.sim_w, y), 1)

    def _draw_danger_zones(self, world: World) -> None:
        """Draw floor and ceiling danger bands."""
        import pygame
        floor_y  = self.sim_h - int(world.cfg.floor   * self.scale_y)
        ceil_y   = self.sim_h - int(world.cfg.ceiling * self.scale_y)

        # Floor band (bottom)
        pygame.draw.line(self._screen, Colors.FLOOR,
                         (0, floor_y), (self.sim_w, floor_y), 2)
        font = self._fonts["small"]
        label = font.render("FLOOR", True, Colors.FLOOR)
        self._screen.blit(label, (4, floor_y - 14))

        # Ceiling band (top)
        pygame.draw.line(self._screen, Colors.CEILING,
                         (0, ceil_y), (self.sim_w, ceil_y), 2)
        label = font.render("CEILING", True, Colors.CEILING)
        self._screen.blit(label, (4, ceil_y + 2))

    def _draw_obstacles(self, world: World) -> None:
        """Draw circular obstacles."""
        import pygame
        for obs in world.obstacles:
            cx, cy = self._to_screen(obs.x, obs.y)
            r = int(obs.radius * self.scale_x)
            pygame.draw.circle(self._screen, Colors.OBSTACLE, (cx, cy), r, 2)

    def _draw_waypoint(self, world: World) -> None:
        """Draw waypoint target with pulsing capture-radius ring."""
        import pygame
        wp = world.waypoint
        cx, cy = self._to_screen(wp.x, wp.y)
        r  = max(4, int(wp.radius * self.scale_x))

        # Pulsing outer ring
        pulse = math.sin(self._pulse_t) * 0.3 + 1.0
        pulse_r = int(r * pulse)
        pygame.draw.circle(self._screen, Colors.WAYPOINT_RING,
                           (cx, cy), pulse_r, 1)

        # Capture radius circle
        pygame.draw.circle(self._screen, Colors.WAYPOINT, (cx, cy), r, 2)

        # Centre crosshair
        size = 8
        pygame.draw.line(self._screen, Colors.WAYPOINT,
                         (cx - size, cy), (cx + size, cy), 2)
        pygame.draw.line(self._screen, Colors.WAYPOINT,
                         (cx, cy - size), (cx, cy + size), 2)

        # Label
        font = self._fonts["small"]
        label = font.render("WP", True, Colors.WAYPOINT)
        self._screen.blit(label, (cx + r + 4, cy - 8))

    def _draw_trail(self) -> None:
        """Draw fading position trail."""
        import pygame
        n = len(self._trail)
        for i, (px, py) in enumerate(self._trail):
            alpha = int(255 * (i / n) * 0.5)
            r_size = max(1, int(2 * (i / n)))
            color  = (*Colors.AIRCRAFT[:3], alpha)
            pygame.draw.circle(self._screen, Colors.AIRCRAFT[:3],
                               (px, py), r_size)

    def _draw_aircraft(self, state: AircraftState) -> None:
        """
        Draw aircraft as a rotated triangle.

        The nose points in the direction of theta (pitch angle).
        For top-down view, we use heading = atan2(vy, vx) for direction.
        """
        import pygame

        cx, cy = self._to_screen(state.x, state.y)

        # Use velocity direction for top-down heading
        heading = math.atan2(state.vy, state.vx) if state.speed > 0.5 else state.theta

        # Triangle: nose at front, two wingtips at back
        size  = 10
        wing  = 7
        nose  = (cx + int(size * math.cos(heading)),
                 cy - int(size * math.sin(heading)))
        left  = (cx + int(wing * math.cos(heading + 2.4)),
                 cy - int(wing * math.sin(heading + 2.4)))
        right = (cx + int(wing * math.cos(heading - 2.4)),
                 cy - int(wing * math.sin(heading - 2.4)))
        tail  = (cx + int(4  * math.cos(heading + math.pi)),
                 cy - int(4  * math.sin(heading + math.pi)))

        # Filled triangle
        pygame.draw.polygon(self._screen, Colors.AIRCRAFT_FILL,
                            [nose, left, tail, right])
        # Outline
        pygame.draw.polygon(self._screen, Colors.AIRCRAFT,
                            [nose, left, tail, right], 2)

        # Centre dot
        pygame.draw.circle(self._screen, Colors.WHITE, (cx, cy), 3)

    def _draw_velocity_vector(self, state: AircraftState) -> None:
        """Draw amber arrow showing velocity direction and magnitude."""
        import pygame

        if state.speed < 0.5:
            return

        cx, cy = self._to_screen(state.x, state.y)
        scale  = 1.5
        end_x  = cx + int(state.vx * scale * self.scale_x / self.world_w * self.sim_w * 0.1)
        end_y  = cy - int(state.vy * scale * self.scale_y / self.world_h * self.sim_h * 0.1)

        pygame.draw.line(self._screen, Colors.SPEED_VEC,
                         (cx, cy), (end_x, end_y), 2)
        # Arrowhead
        angle = math.atan2(cy - end_y, end_x - cx)
        for da in [0.5, -0.5]:
            ax = end_x - int(8 * math.cos(angle + da))
            ay = end_y + int(8 * math.sin(angle + da))
            pygame.draw.line(self._screen, Colors.SPEED_VEC,
                             (end_x, end_y), (ax, ay), 2)

    def _draw_hud(
        self,
        state: AircraftState,
        world: World,
        reward_breakdown: dict[str, float],
        info: dict,
    ) -> None:
        """Draw the full tactical HUD on the right panel."""
        import pygame

        panel_x = self.sim_w + 8
        y       = 10
        line_h  = 18
        bar_w   = self.HUD_WIDTH - 24

        title_font = self._fonts["title"]
        hud_font   = self._fonts["hud"]
        small_font = self._fonts["small"]

        def text(label: str, value: str, color=Colors.HUD_TEXT, y_off: int = 0):
            nonlocal y
            surf = hud_font.render(f"{label:<12} {value}", True, color)
            self._screen.blit(surf, (panel_x, y + y_off))
            y += line_h

        def divider():
            nonlocal y
            pygame.draw.line(self._screen, Colors.GRID,
                             (self.sim_w + 4, y), (self.screen_w - 4, y), 1)
            y += 6

        def bar(label: str, value: float, min_v: float, max_v: float,
                color_pos=(0, 230, 118), color_neg=(255, 82, 82)):
            nonlocal y
            # Label
            lbl = small_font.render(f"  {label}", True, Colors.HUD_TEXT)
            self._screen.blit(lbl, (panel_x, y))
            # Value text
            val_str = f"{value:+.2f}"
            vsurf = small_font.render(val_str, True, Colors.HUD_ACCENT)
            self._screen.blit(vsurf, (panel_x + bar_w - 45, y))
            y += 14
            # Bar background
            pygame.draw.rect(self._screen, Colors.GRID,
                             (panel_x, y, bar_w, 8), border_radius=2)
            # Bar fill
            span = max_v - min_v
            if span > 0:
                frac = max(0.0, min(1.0, (value - min_v) / span))
                fill = max(2, int(bar_w * frac))
                color = color_pos if value >= 0 else color_neg
                pygame.draw.rect(self._screen, color,
                                 (panel_x, y, fill, 8), border_radius=2)
            y += 14

        # ── Title ─────────────────────────────────────────────────────────
        title = title_font.render("TACTICAL HUD", True, Colors.HUD_ACCENT)
        self._screen.blit(title, (panel_x, y))
        y += 24
        divider()

        # ── State info ────────────────────────────────────────────────────
        text("STEP",    str(info.get("step", 0)))
        text("SPEED",   f"{state.speed:6.1f} u/s")
        text("ALT",     f"{state.y:6.1f}")
        text("PITCH",   f"{math.degrees(state.theta):+6.1f}°")
        text("OMEGA",   f"{state.omega:+6.2f} r/s")
        dist = info.get("dist_to_waypoint", 0)
        text("DIST WP", f"{dist:6.1f}",
             color=(Colors.WAYPOINT if dist < 100 else Colors.HUD_TEXT))
        text("FUEL",    f"{state.fuel:6.0f}")
        text("EP RWRD", f"{info.get('episode_reward', 0.0):+7.1f}", color=Colors.HUD_ACCENT)
        divider()

        # ── Action ────────────────────────────────────────────────────────
        action_name = info.get("action_name", "—")
        text("ACTION", action_name, color=Colors.SPEED_VEC)
        divider()

        # ── Reward Breakdown ──────────────────────────────────────────────
        rbd_title = title_font.render("REWARD BREAKDOWN", True, Colors.HUD_ACCENT)
        self._screen.blit(rbd_title, (panel_x, y))
        y += 22

        component_ranges = {
            "approach":  (-30,  30),
            "waypoint":  (0,   200),
            "crash":     (-100,  0),
            "alive":     (0,   0.2),
            "smooth":    (-0.2,  0),
            "altitude":  (-0.5,  0),
            "fuel":      (-0.01, 0),
        }
        for key, val in reward_breakdown.items():
            min_v, max_v = component_ranges.get(key, (-10, 10))
            bar(key, val, min_v, max_v)

        divider()

        # ── Termination status ────────────────────────────────────────────
        term = info.get("termination", "")
        if term == "goal":
            status = "MISSION SUCCESS"
            color  = Colors.WAYPOINT
        elif term in ("oob", "stall", "obstacle", "fuel"):
            status = f"MISSION FAILED ({term.upper()})"
            color  = Colors.FLOOR
        elif term == "timeout":
            status = "TIMEOUT"
            color  = Colors.CEILING
        else:
            status = "ACTIVE"
            color  = Colors.HUD_TEXT

        stat_surf = title_font.render(status, True, color)
        self._screen.blit(stat_surf, (panel_x, y))
        y += 22

        # ── Controls reminder ─────────────────────────────────────────────
        divider()
        controls = [
            "W/↑  : Thrust",
            "↑↓   : Pitch",
            "A/D  : Pitch ±",
            "SPACE: Glide",
            "R    : Reset",
            "Q    : Quit",
        ]
        for ctrl in controls:
            s = small_font.render(ctrl, True, Colors.GRID)
            self._screen.blit(s, (panel_x, y))
            y += 13
