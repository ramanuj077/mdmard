"""
observation/builder.py
======================
Observation Builder -- MDMARD Phase 1

Constructs the normalized 10-dimensional state vector that the RL agent
observes each step. Completely decoupled from physics and reward logic.

Observation Vector Layout
--------------------------
  Index  Variable              Formula                  Range
  ─────────────────────────────────────────────────────────────
  0      x position            x / world.width          [0, 1]
  1      y position (alt)      y / world.height         [0, 1]
  2      velocity x            vx / vel_max             [-1, 1]
  3      velocity y            vy / vel_max             [-1, 1]
  4      pitch angle           theta / pi               [-1, 1]
  5      pitch rate            omega / omega_max        [-1, 1]
  6      distance to waypoint  d / d_max                [0, 1]
  7      bearing to waypoint   bearing / pi             [-1, 1]
  8      fuel remaining        fuel / fuel_max          [0, 1]
  9      time remaining        t_remaining / T_max      [0, 1]
  ─────────────────────────────────────────────────────────────

Design Notes
------------
- All components are clipped to [-1, 1] after normalization so the
  agent never sees out-of-range values (e.g. during overspeed).
- Normalization divisors come from ObservationConfig (YAML-driven).
- The builder is stateless: same inputs always produce same output.
- MARL Extension: in Phase 2, relative positions of other agents
  are appended as additional components via extend_with_agents().

References
----------
  [1] Mnih et al. -- "Human-level control through deep RL" (2015)
      (state normalization for stable gradient flow)
  [2] Schulman et al. -- "Proximal Policy Optimization" (2017)
      (observation normalization best practices)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import gymnasium as gym

from envs.physics import AircraftState
from envs.world import World


# ─────────────────────────────────────────────────────────────────────────────
# Observation Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ObservationConfig:
    """
    Normalization constants for the observation vector.

    Loaded from the 'observation' section of configs/env_phase1.yaml.

    Attributes
    ----------
    vel_max : float
        Velocity normalization divisor (same as physics.vel_max).
    omega_max : float
        Pitch rate normalization divisor.
    dist_max : float
        Distance normalization divisor (world diagonal).
    fuel_max : float
        Maximum fuel for normalization.
    t_max : float
        Maximum episode steps for time normalization.
    """
    vel_max:  float = 80.0
    omega_max: float = 2.0
    dist_max:  float = 1442.0   # sqrt(1200² + 800²)
    fuel_max:  float = 1000.0
    t_max:     float = 1000.0

    # Observation vector size -- update if you add components
    OBS_DIM: int = 10

    @classmethod
    def from_yaml(cls, cfg: dict) -> "ObservationConfig":
        """
        Construct from the 'observation' sub-dict of a loaded YAML config.

        Parameters
        ----------
        cfg : dict
            The 'observation' section of env_phase1.yaml.

        Returns
        -------
        ObservationConfig
        """
        return cls(
            vel_max=cfg.get("vel_max_norm",   80.0),
            omega_max=cfg.get("omega_max_norm", 2.0),
            dist_max=cfg.get("dist_max_norm",  1442.0),
            fuel_max=cfg.get("fuel_max",       1000.0),
            t_max=cfg.get("t_max",             1000.0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Observation Builder
# ─────────────────────────────────────────────────────────────────────────────

class ObservationBuilder:
    """
    Constructs the normalized observation vector for the RL agent.

    This class is the single source of truth for what the agent sees.
    Any change to the state representation happens here only.

    Parameters
    ----------
    obs_config : ObservationConfig
        Normalization constants.
    world : World
        The simulation world (provides waypoint bearing/distance).

    Example
    -------
    >>> builder = ObservationBuilder(ObservationConfig(), world)
    >>> state = AircraftState(x=100, y=400, vx=20, vy=0, theta=0.1, omega=0.0)
    >>> obs = builder.build(state, steps_remaining=900)
    >>> obs.shape
    (10,)
    >>> np.all((obs >= -1.0) & (obs <= 1.0))
    True
    """

    def __init__(self, obs_config: ObservationConfig, world: World) -> None:
        self.cfg   = obs_config
        self.world = world

    # ── Public API ───────────────────────────────────────────────────────────

    def build(
        self,
        state: AircraftState,
        steps_remaining: int,
    ) -> np.ndarray:
        """
        Build the 10-dimensional normalized observation vector.

        Parameters
        ----------
        state : AircraftState
            Current aircraft kinematic state.
        steps_remaining : int
            Episode steps remaining (T_max - current_step).

        Returns
        -------
        np.ndarray
            Shape (10,), dtype float32, all values clipped to [-1, 1].
        """
        cfg   = self.cfg
        world = self.world

        # ── [0] x position ────────────────────────────────────────────────
        obs_x = state.x / world.cfg.width

        # ── [1] y position (altitude) ──────────────────────────────────────
        obs_y = state.y / world.cfg.height

        # ── [2] velocity x ─────────────────────────────────────────────────
        obs_vx = state.vx / cfg.vel_max

        # ── [3] velocity y ─────────────────────────────────────────────────
        obs_vy = state.vy / cfg.vel_max

        # ── [4] pitch angle ────────────────────────────────────────────────
        obs_theta = state.theta / math.pi

        # ── [5] pitch rate ─────────────────────────────────────────────────
        obs_omega = state.omega / cfg.omega_max

        # ── [6] distance to waypoint ───────────────────────────────────────
        dist = world.distance_to_waypoint(state.x, state.y)
        obs_dist = dist / cfg.dist_max

        # ── [7] bearing to waypoint ────────────────────────────────────────
        bearing = world.bearing_to_waypoint(state.x, state.y)
        obs_bearing = bearing / math.pi           # atan2 already in [-π, π]

        # ── [8] fuel remaining ─────────────────────────────────────────────
        obs_fuel = state.fuel / cfg.fuel_max

        # ── [9] time remaining ─────────────────────────────────────────────
        obs_time = float(steps_remaining) / cfg.t_max

        # ── Assemble and clip ──────────────────────────────────────────────
        obs = np.array([
            obs_x,
            obs_y,
            obs_vx,
            obs_vy,
            obs_theta,
            obs_omega,
            obs_dist,
            obs_bearing,
            obs_fuel,
            obs_time,
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0)

    def breakdown(
        self,
        state: AircraftState,
        steps_remaining: int,
    ) -> dict[str, float]:
        """
        Build the observation and return it as a labeled dict.

        Useful for debugging, HUD display, and logging.

        Parameters
        ----------
        state : AircraftState
        steps_remaining : int

        Returns
        -------
        dict[str, float]
            Component name → normalized value.
        """
        obs = self.build(state, steps_remaining)
        labels = [
            "x_pos", "y_pos", "vel_x", "vel_y",
            "pitch", "pitch_rate",
            "dist_to_wp", "bearing_to_wp",
            "fuel", "time_remaining",
        ]
        return dict(zip(labels, obs.tolist()))

    # ── Gymnasium Space ───────────────────────────────────────────────────────

    @staticmethod
    def observation_space(obs_dim: int = 10) -> gym.spaces.Box:
        """
        Return the Gymnasium observation space for this builder.

        All components are clipped to [-1, 1], so we use a Box
        with low=-1, high=1 throughout.

        Parameters
        ----------
        obs_dim : int
            Observation vector dimension. Default 10.

        Returns
        -------
        gymnasium.spaces.Box
            Shape (obs_dim,), dtype float32.
        """
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
