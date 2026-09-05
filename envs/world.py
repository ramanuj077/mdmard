"""
envs/world.py
=============
Airspace World Model -- MDMARD Phase 1

Defines the simulation world: boundaries, waypoint placement,
terrain floor, restricted zones, and obstacle layouts.

The World is the shared environment state. It is owned by FlightEnv
and passed to the ObservationBuilder and RewardEngine each step.

Design Principles
-----------------
- World knows about geometry, not about agents or rewards
- Randomization is seeded via a numpy RNG for reproducibility
- All spatial data is in simulation units (pixels = units in Phase 1)
- Structured for multi-agent extension: world.agents dict in Phase 2

Phase 1 Scope
-------------
- Rectangular airspace with hard boundaries
- Single randomized waypoint per episode
- Terrain floor (no terrain height variation yet -- that's data/terrain/)
- Obstacle stubs (populated by CurriculumScheduler in Stage 13)

References
----------
  [1] OpenAI Gym / Gymnasium environment design guidelines
  [2] LunarLander-v2 source -- waypoint + boundary handling pattern
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Waypoint:
    """
    A mission waypoint (target zone) the aircraft must reach.

    Attributes
    ----------
    x : float
        Horizontal position of waypoint center.
    y : float
        Vertical position of waypoint center.
    radius : float
        Capture radius. Episode succeeds when aircraft enters this zone.
    captured : bool
        True once the aircraft has entered the capture radius.
    """
    x: float
    y: float
    radius: float = 30.0
    captured: bool = False

    @property
    def position(self) -> Tuple[float, float]:
        """(x, y) tuple."""
        return (self.x, self.y)

    def distance_to(self, x: float, y: float) -> float:
        """
        Euclidean distance from point (x, y) to waypoint center.

        Parameters
        ----------
        x, y : float
            Query point coordinates.

        Returns
        -------
        float
            Distance in simulation units.
        """
        return math.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)

    def check_capture(self, x: float, y: float) -> bool:
        """
        Test if point (x, y) is within the capture radius.
        Sets self.captured = True if yes.

        Parameters
        ----------
        x, y : float
            Aircraft position.

        Returns
        -------
        bool
            True if waypoint is now captured.
        """
        if self.distance_to(x, y) <= self.radius:
            self.captured = True
        return self.captured


@dataclass
class Obstacle:
    """
    A circular obstacle in the airspace.

    Used from Stage 13 (curriculum stage 1+). In Stage 1-12,
    the obstacle list is empty.

    Attributes
    ----------
    x, y : float
        Center position.
    radius : float
        Collision radius.
    obstacle_type : str
        "static" | "restricted_zone" | "threat" (future MARL)
    """
    x: float
    y: float
    radius: float
    obstacle_type: str = "static"

    def distance_to(self, x: float, y: float) -> float:
        """Distance from point to obstacle center."""
        return math.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)

    def contains(self, x: float, y: float) -> bool:
        """True if point is inside the obstacle."""
        return self.distance_to(x, y) <= self.radius


@dataclass
class WorldConfig:
    """
    Configuration for the simulation world geometry.

    Loaded from the 'world' section of configs/env_phase1.yaml.

    Attributes
    ----------
    width : float
        Horizontal extent of the airspace.
    height : float
        Vertical extent of the airspace.
    floor : float
        Minimum y (terrain floor / sea level). Aircraft below = crash.
    ceiling : float
        Maximum y. Aircraft above = out of bounds.
    goal_radius : float
        Default waypoint capture radius.
    """
    width: float = 1200.0
    height: float = 800.0
    floor: float = 50.0
    ceiling: float = 750.0
    goal_radius: float = 30.0

    @classmethod
    def from_yaml(cls, cfg: dict) -> "WorldConfig":
        """
        Construct from the 'world' sub-dict of a loaded YAML config.

        Parameters
        ----------
        cfg : dict
            The 'world' section of env_phase1.yaml.

        Returns
        -------
        WorldConfig
        """
        return cls(
            width=cfg.get("width", 1200.0),
            height=cfg.get("height", 800.0),
            floor=cfg.get("floor", 50.0),
            ceiling=cfg.get("ceiling", 750.0),
            goal_radius=cfg.get("goal_radius", 30.0),
        )

    @property
    def max_distance(self) -> float:
        """Diagonal of the world -- maximum possible distance between two points."""
        return math.sqrt(self.width ** 2 + self.height ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Aircraft Spawn Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpawnConfig:
    """
    Defines how the aircraft is spawned at episode reset.

    Attributes
    ----------
    x_range : Tuple[float, float]
        Min/max spawn x range (left portion of world for challenge).
    y_range : Tuple[float, float]
        Min/max spawn y range (mid-altitude band).
    vx_range : Tuple[float, float]
        Initial horizontal velocity range (positive = moving right).
    vy_range : Tuple[float, float]
        Initial vertical velocity range.
    theta_range : Tuple[float, float]
        Initial pitch range (radians). 0 = level.
    """
    x_range: Tuple[float, float] = (50.0,  250.0)
    y_range: Tuple[float, float] = (200.0, 600.0)
    vx_range: Tuple[float, float] = (8.0,  20.0)
    vy_range: Tuple[float, float] = (-3.0,  3.0)
    theta_range: Tuple[float, float] = (-0.2, 0.2)


@dataclass
class WaypointSpawnConfig:
    """
    Defines where waypoints are placed at episode reset.

    Waypoints spawn in the right half of the world to ensure
    the aircraft always has to fly toward them.

    Attributes
    ----------
    x_range : Tuple[float, float]
        Min/max waypoint x range.
    y_range : Tuple[float, float]
        Min/max waypoint y range.
    min_distance : float
        Minimum spawn distance from aircraft start position.
        Prevents trivially-close waypoints.
    """
    x_range: Tuple[float, float] = (700.0, 1100.0)
    y_range: Tuple[float, float] = (150.0,  650.0)
    min_distance: float = 300.0


# ─────────────────────────────────────────────────────────────────────────────
# World Class
# ─────────────────────────────────────────────────────────────────────────────

class World:
    """
    Simulation world: airspace geometry, waypoints, and obstacles.

    Owned by FlightEnv. Reset each episode. Passed by reference to
    ObservationBuilder and RewardEngine.

    Parameters
    ----------
    config : WorldConfig
        Airspace geometry constants.
    spawn : SpawnConfig
        Aircraft spawn configuration.
    wp_spawn : WaypointSpawnConfig
        Waypoint spawn configuration.
    obstacles : List[Obstacle]
        Static obstacle list. Empty in Stage 1-12.
        Populated by CurriculumScheduler from Stage 13.
    rng : np.random.Generator
        Seeded random number generator (passed from FlightEnv).

    Example
    -------
    >>> rng = np.random.default_rng(seed=42)
    >>> world = World(WorldConfig(), SpawnConfig(), WaypointSpawnConfig(),
    ...               obstacles=[], rng=rng)
    >>> spawn_pos, waypoint = world.reset()
    """

    def __init__(
        self,
        config: Optional[WorldConfig] = None,
        spawn: Optional[SpawnConfig] = None,
        wp_spawn: Optional[WaypointSpawnConfig] = None,
        obstacles: Optional[List[Obstacle]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.cfg = config if config is not None else WorldConfig()
        self.spawn = spawn if spawn is not None else SpawnConfig()
        self.wp_spawn = wp_spawn if wp_spawn is not None else WaypointSpawnConfig()
        self.obstacles: List[Obstacle] = obstacles or []
        self.rng = rng if rng is not None else np.random.default_rng()

        # Set by reset()
        self.waypoint: Optional[Waypoint] = None
        self._spawn_x: float = 0.0
        self._spawn_y: float = 0.0

    # ── Public API ───────────────────────────────────────────────────────────

    def reset(self) -> Tuple[Tuple[float, float, float, float, float], Waypoint]:
        """
        Reset the world for a new episode.

        Randomizes aircraft spawn position and waypoint location.
        Ensures minimum distance between spawn and waypoint.

        Returns
        -------
        spawn_state : Tuple[float, float, float, float, float]
            (x, y, vx, vy, theta) — initial aircraft kinematic state.
        waypoint : Waypoint
            Newly placed waypoint for this episode.
        """
        # ── Spawn aircraft ────────────────────────────────────────────────
        x  = self.rng.uniform(*self.spawn.x_range)
        y  = self.rng.uniform(*self.spawn.y_range)
        vx = self.rng.uniform(*self.spawn.vx_range)
        vy = self.rng.uniform(*self.spawn.vy_range)
        th = self.rng.uniform(*self.spawn.theta_range)

        self._spawn_x = x
        self._spawn_y = y

        # ── Place waypoint (with min-distance guarantee) ───────────────────
        max_attempts = 100
        for _ in range(max_attempts):
            wp_x = self.rng.uniform(*self.wp_spawn.x_range)
            wp_y = self.rng.uniform(*self.wp_spawn.y_range)
            dist = math.sqrt((wp_x - x) ** 2 + (wp_y - y) ** 2)
            if dist >= self.wp_spawn.min_distance:
                break
        # Fallback: place waypoint at far right-center if all attempts fail
        else:
            wp_x = self.cfg.width - 100.0
            wp_y = self.cfg.height / 2.0

        self.waypoint = Waypoint(
            x=wp_x,
            y=wp_y,
            radius=self.cfg.goal_radius,
        )

        return (x, y, vx, vy, th), self.waypoint

    def is_out_of_bounds(self, x: float, y: float) -> bool:
        """
        Check if position is outside world boundaries.

        Parameters
        ----------
        x, y : float
            Aircraft position.

        Returns
        -------
        bool
            True if aircraft has left the airspace.
        """
        return (
            x < 0.0
            or x > self.cfg.width
            or y < self.cfg.floor
            or y > self.cfg.ceiling
        )

    def is_in_obstacle(self, x: float, y: float) -> bool:
        """
        Check if position overlaps any obstacle.

        Parameters
        ----------
        x, y : float
            Aircraft position.

        Returns
        -------
        bool
            True if inside any obstacle. Always False in Stage 1-12.
        """
        return any(obs.contains(x, y) for obs in self.obstacles)

    def distance_to_waypoint(self, x: float, y: float) -> float:
        """
        Distance from (x, y) to the current waypoint center.

        Parameters
        ----------
        x, y : float
            Aircraft position.

        Returns
        -------
        float
            Distance in simulation units. 0.0 if no waypoint placed.
        """
        if self.waypoint is None:
            return 0.0
        return self.waypoint.distance_to(x, y)

    def bearing_to_waypoint(self, x: float, y: float) -> float:
        """
        Bearing angle from aircraft to waypoint, in radians.

        Returns the angle in [-π, π] between the x-axis and the
        vector pointing from (x, y) to the waypoint.

        Parameters
        ----------
        x, y : float
            Aircraft position.

        Returns
        -------
        float
            Angle in radians [-π, π]. 0 = waypoint is directly to the right.
        """
        if self.waypoint is None:
            return 0.0
        dx = self.waypoint.x - x
        dy = self.waypoint.y - y
        return math.atan2(dy, dx)

    def altitude_margin(self, y: float) -> float:
        """
        Normalized proximity to floor or ceiling, in [0, 1].

        0.0  = aircraft is at floor or ceiling (danger zone)
        1.0  = aircraft is at maximum safe distance from both boundaries

        Used by reward function's altitude penalty component.

        Parameters
        ----------
        y : float
            Aircraft altitude.

        Returns
        -------
        float
            Margin value in [0, 1].
        """
        floor_margin   = (y - self.cfg.floor)
        ceiling_margin = (self.cfg.ceiling - y)
        safe_band      = (self.cfg.ceiling - self.cfg.floor) / 2.0
        margin = min(floor_margin, ceiling_margin) / safe_band
        return float(np.clip(margin, 0.0, 1.0))

    def set_obstacles(self, obstacles: List[Obstacle]) -> None:
        """
        Replace obstacle list (called by CurriculumScheduler).

        Parameters
        ----------
        obstacles : List[Obstacle]
            New obstacle layout for the current curriculum stage.
        """
        self.obstacles = obstacles

    # ── Boundary Properties (for renderer) ───────────────────────────────────

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """(x_min, y_min, x_max, y_max) world boundary tuple."""
        return (0.0, self.cfg.floor, self.cfg.width, self.cfg.ceiling)

    @property
    def center(self) -> Tuple[float, float]:
        """World center point."""
        return (self.cfg.width / 2.0, self.cfg.height / 2.0)

    def __repr__(self) -> str:
        wp = self.waypoint
        return (
            f"World({self.cfg.width}×{self.cfg.height}, "
            f"floor={self.cfg.floor}, ceiling={self.cfg.ceiling}, "
            f"waypoint={wp}, obstacles={len(self.obstacles)})"
        )
