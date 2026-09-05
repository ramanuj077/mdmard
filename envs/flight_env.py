"""
envs/flight_env.py
==================
TacticalFlightEnv -- MDMARD Phase 1

The main Gymnasium environment. Wires together:
  - PhysicsEngine  (dynamics)
  - World          (airspace, waypoint, obstacles)
  - ObservationBuilder (state vector)
  - Reward         (stub in Stage 3 -- full reward in Stage 4)
  - Termination    (stub in Stage 3 -- full logic in Stage 4)

Gymnasium API Compliance
------------------------
Fully implements the Gymnasium 0.26+ API:
  reset(seed, options) → (obs, info)
  step(action)         → (obs, reward, terminated, truncated, info)
  render()             → None | np.ndarray
  close()              → None

All spaces are declared before any step is taken (required by check_env).
`terminated` vs `truncated` are correctly separated:
  terminated = agent caused episode end (crash, goal, fuel out)
  truncated  = environment cut episode short (max steps exceeded)

Design Notes
------------
- The env owns the RNG, passes it to World for reproducibility.
- Physics state (AircraftState) is private; only observations are public.
- Reward and termination are computed inline here (Stage 3 stubs),
  and will be refactored into reward/ and termination/ in Stage 4.
- Rendering is a stub that prints a warning; full renderer in Stage 5.
- The info dict is always populated (required by SB3 VecEnv wrappers).

References
----------
  [1] Gymnasium documentation -- Env API
      https://gymnasium.farama.org/api/env/
  [2] SB3 check_env source for compliance requirements
  [3] Brockman et al. -- "OpenAI Gym" (2016)
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.physics import (
    AircraftState,
    PhysicsConfig,
    PhysicsEngine,
    ACTION_NAMES,
)
from envs.world import (
    World,
    WorldConfig,
    SpawnConfig,
    WaypointSpawnConfig,
    Obstacle,
)
from observation.builder import ObservationBuilder, ObservationConfig


# ─────────────────────────────────────────────────────────────────────────────
# Environment Constants (defaults; overridden by YAML in Stage 10)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_MAX_STEPS:  int   = 1000
_DEFAULT_GOAL_BONUS: float = 200.0
_DEFAULT_CRASH_PEN:  float = -100.0


# ─────────────────────────────────────────────────────────────────────────────
# TacticalFlightEnv
# ─────────────────────────────────────────────────────────────────────────────

class TacticalFlightEnv(gym.Env):
    """
    2D tactical flight environment for single-agent RL -- MDMARD Phase 1.

    An aircraft must navigate to a randomly placed waypoint within a
    bounded airspace, subject to aerodynamic forces and a step budget.

    Observation Space
    -----------------
    Box(-1, 1, shape=(10,), dtype=float32)
    [x, y, vx, vy, theta, omega, dist_to_wp, bearing_to_wp, fuel, time]

    Action Space (Phase 1A -- discrete)
    ------------------------------------
    Discrete(6):
      0: Thrust       1: Pitch Up    2: Pitch Down
      3: Thrust+Up    4: Thrust+Down 5: Glide (no-op)

    Parameters
    ----------
    physics_config : PhysicsConfig, optional
        Aircraft dynamics constants. Defaults to PhysicsConfig().
    world_config : WorldConfig, optional
        Airspace geometry. Defaults to WorldConfig().
    obs_config : ObservationConfig, optional
        Observation normalization constants. Defaults to ObservationConfig().
    max_steps : int, optional
        Maximum steps per episode before truncation. Default 1000.
    render_mode : str, optional
        "human"  -- Pygame window (available from Stage 5).
        "rgb_array" -- Return RGB array (available from Stage 5).
        None     -- No rendering (default; fastest for training).

    Attributes
    ----------
    observation_space : spaces.Box
    action_space : spaces.Discrete
    render_mode : str | None
    np_random : np.random.Generator
        Gymnasium-managed RNG (set by reset(seed=...)).

    Example
    -------
    >>> env = TacticalFlightEnv()
    >>> obs, info = env.reset(seed=42)
    >>> obs.shape
    (10,)
    >>> obs, reward, terminated, truncated, info = env.step(0)
    """

    # Gymnasium metadata -- required by check_env
    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        physics_config: Optional[PhysicsConfig] = None,
        world_config: Optional[WorldConfig] = None,
        obs_config: Optional[ObservationConfig] = None,
        max_steps: int = _DEFAULT_MAX_STEPS,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        # ── Validate render mode ────────────────────────────────────────────
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Invalid render_mode='{render_mode}'. "
                f"Must be one of {self.metadata['render_modes']} or None."
            )
        self.render_mode = render_mode

        # ── Configs ─────────────────────────────────────────────────────────
        self._phys_cfg = physics_config or PhysicsConfig()
        self._world_cfg = world_config or WorldConfig()
        self._obs_cfg = obs_config or ObservationConfig()
        self._max_steps = max_steps

        # ── Spaces (declared before any reset -- required by check_env) ─────
        self.observation_space: spaces.Box = ObservationBuilder.observation_space(
            obs_dim=self._obs_cfg.OBS_DIM
        )
        self.action_space: spaces.Discrete = spaces.Discrete(6)

        # ── Sub-components (initialized properly on reset) ──────────────────
        self._engine: PhysicsEngine = PhysicsEngine(self._phys_cfg)

        # World and builder are created here but require RNG from reset()
        # They are rebuilt in reset() with the seeded RNG.
        self._world: Optional[World] = None
        self._builder: Optional[ObservationBuilder] = None

        # ── Episode state ───────────────────────────────────────────────────
        self._state: Optional[AircraftState] = None
        self._step_count: int = 0
        self._prev_dist: float = 0.0     # for potential-based shaping
        self._episode_reward: float = 0.0

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment for a new episode.

        Randomizes aircraft spawn position and waypoint location.
        Must be called before the first step().

        Parameters
        ----------
        seed : int, optional
            Random seed for full reproducibility. When None, inherits
            the last seed (or uses system entropy on first call).
        options : dict, optional
            Reserved for future use (curriculum stage injection, etc.).

        Returns
        -------
        obs : np.ndarray
            Initial observation, shape (10,), dtype float32.
        info : dict
            Diagnostic information for logging and debugging.
        """
        # ── Seed the Gymnasium-managed RNG ──────────────────────────────────
        super().reset(seed=seed)
        rng = self.np_random  # gymnasium provides this after super().reset()

        # ── Rebuild world with seeded RNG ───────────────────────────────────
        self._world = World(
            config=self._world_cfg,
            spawn=SpawnConfig(),
            wp_spawn=WaypointSpawnConfig(),
            obstacles=[],
            rng=rng,
        )
        self._builder = ObservationBuilder(self._obs_cfg, self._world)

        # ── Randomize spawn and waypoint ────────────────────────────────────
        (x, y, vx, vy, theta), waypoint = self._world.reset()

        # ── Initialize aircraft state ───────────────────────────────────────
        self._state = AircraftState(
            x=x, y=y, vx=vx, vy=vy,
            theta=theta, omega=0.0,
            fuel=self._phys_cfg.fuel_burn_rate and 1000.0,
            step=0,
        )

        # ── Episode bookkeeping ─────────────────────────────────────────────
        self._step_count = 0
        self._episode_reward = 0.0
        self._prev_dist = self._world.distance_to_waypoint(x, y)
        self._stall_counter: int = 0

        # ── Build initial observation ───────────────────────────────────────
        obs = self._get_obs()

        info = self._make_info(
            action=None,
            reward=0.0,
            reward_breakdown={},
            terminated=False,
            truncated=False,
        )

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Advance the environment by one timestep.

        Parameters
        ----------
        action : int
            Discrete action index in [0, 5].

        Returns
        -------
        obs : np.ndarray
            New observation after the step.
        reward : float
            Scalar reward signal.
        terminated : bool
            True if episode ended due to agent action (crash or goal).
        truncated : bool
            True if episode ended due to step budget exhaustion.
        info : dict
            Diagnostic dict with reward breakdown and episode stats.

        Raises
        ------
        RuntimeError
            If step() is called before reset().
        """
        if self._state is None or self._world is None:
            raise RuntimeError(
                "step() called before reset(). "
                "Call env.reset() before env.step()."
            )

        # ── Physics step ────────────────────────────────────────────────────
        self._state = self._engine.step(self._state, action_id=int(action))
        self._step_count += 1

        x, y = self._state.x, self._state.y

        # ── Termination checks ──────────────────────────────────────────────
        terminated, term_reason = self._check_termination(x, y)
        truncated = (self._step_count >= self._max_steps) and not terminated

        # ── Reward computation ──────────────────────────────────────────────
        reward, reward_breakdown = self._compute_reward(
            x=x, y=y,
            terminated=terminated,
            term_reason=term_reason,
        )
        self._episode_reward += reward

        # ── Observation ─────────────────────────────────────────────────────
        obs = self._get_obs()

        # ── Info ────────────────────────────────────────────────────────────
        info = self._make_info(
            action=action,
            reward=reward,
            reward_breakdown=reward_breakdown,
            terminated=terminated,
            truncated=truncated,
            term_reason=term_reason,
        )

        # ── Render if requested ─────────────────────────────────────────────
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        Stage 3 stub -- full Pygame renderer is implemented in Stage 5.
        Returns None for "human" mode, np.ndarray for "rgb_array".

        Returns
        -------
        None or np.ndarray
        """
        if self.render_mode is None:
            gym.logger.warn(
                "render() called but render_mode=None. "
                "Pass render_mode='human' to TacticalFlightEnv() to enable."
            )
            return None

        if self.render_mode in ("human", "rgb_array"):
            warnings.warn(
                "Full renderer not yet implemented (Stage 5). "
                "Returning None.",
                stacklevel=2,
            )
        return None

    def close(self) -> None:
        """
        Clean up resources.

        Stage 3 stub -- will close Pygame window in Stage 5.
        """
        pass

    # ── Private: Observation ──────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Build the current observation vector."""
        steps_remaining = max(0, self._max_steps - self._step_count)
        return self._builder.build(self._state, steps_remaining)  # type: ignore[arg-type]

    # ── Private: Termination ──────────────────────────────────────────────────

    def _check_termination(
        self, x: float, y: float
    ) -> Tuple[bool, str]:
        """
        Check all termination conditions.

        Returns
        -------
        terminated : bool
        reason : str
            Human-readable reason code for logging / info dict.
            One of: "goal" | "oob" | "stall" | "fuel" | ""
        """
        # ── Goal reached ────────────────────────────────────────────────────
        assert self._world is not None
        if self._world.waypoint is not None:
            if self._world.waypoint.check_capture(x, y):
                return True, "goal"

        # ── Out of bounds ────────────────────────────────────────────────────
        if self._world.is_out_of_bounds(x, y):
            return True, "oob"

        # ── Stall ────────────────────────────────────────────────────────────
        assert self._state is not None
        if self._state.speed < self._phys_cfg.vel_stall:
            self._stall_counter += 1
            if self._stall_counter >= self._phys_cfg.stall_steps:
                return True, "stall"
        else:
            self._stall_counter = 0

        # ── Fuel exhausted ────────────────────────────────────────────────────
        if self._state.fuel <= 0.0 and self._phys_cfg.fuel_burn_rate > 0:
            return True, "fuel"

        return False, ""

    # ── Private: Reward ───────────────────────────────────────────────────────

    def _compute_reward(
        self,
        x: float,
        y: float,
        terminated: bool,
        term_reason: str,
    ) -> Tuple[float, dict]:
        """
        Compute scalar reward and per-component breakdown.

        Stage 3 implementation: potential-based approach shaping +
        sparse goal/crash bonuses. Full weighted reward in Stage 4.

        Components
        ----------
        approach  : dense  -- negative delta-distance to waypoint
        goal      : sparse -- +200 on capture
        crash     : sparse -- -100 on OOB / stall / fuel
        alive     : dense  -- +0.1 per step survived
        smooth    : dense  -- penalty for high pitch rate

        Parameters
        ----------
        x, y : float
            Current aircraft position.
        terminated : bool
        term_reason : str

        Returns
        -------
        total_reward : float
        breakdown : dict[str, float]
        """
        assert self._world is not None
        assert self._state is not None

        breakdown: dict[str, float] = {}

        # ── Approach reward (potential-based shaping) ────────────────────────
        # r_approach = -(d_new - d_prev) = d_prev - d_new
        # Positive when getting closer, negative when moving away.
        curr_dist = self._world.distance_to_waypoint(x, y)
        r_approach = self._prev_dist - curr_dist
        self._prev_dist = curr_dist
        breakdown["approach"] = round(r_approach, 4)

        # ── Survival bonus ────────────────────────────────────────────────────
        r_alive = 0.1
        breakdown["alive"] = r_alive

        # ── Smooth flight (pitch rate penalty) ────────────────────────────────
        r_smooth = -0.05 * abs(self._state.omega)
        breakdown["smooth"] = round(r_smooth, 4)

        # ── Altitude safety bonus ─────────────────────────────────────────────
        margin = self._world.altitude_margin(y)
        r_altitude = -0.5 * (1.0 - margin)     # 0 at center, -0.5 at floor/ceil
        breakdown["altitude"] = round(r_altitude, 4)

        # ── Sparse terminal rewards ───────────────────────────────────────────
        r_goal  = 0.0
        r_crash = 0.0
        if terminated:
            if term_reason == "goal":
                r_goal = _DEFAULT_GOAL_BONUS
            elif term_reason in ("oob", "stall", "fuel"):
                r_crash = _DEFAULT_CRASH_PEN
        breakdown["goal"]  = r_goal
        breakdown["crash"] = r_crash

        total = r_approach + r_alive + r_smooth + r_altitude + r_goal + r_crash

        return float(total), breakdown

    # ── Private: Info dict ────────────────────────────────────────────────────

    def _make_info(
        self,
        action: Optional[int],
        reward: float,
        reward_breakdown: dict,
        terminated: bool,
        truncated: bool,
        term_reason: str = "",
    ) -> dict:
        """
        Build the info dictionary returned with every step.

        SB3 and monitoring wrappers read specific keys from info.
        We follow the convention used by SB3's Monitor wrapper.

        Returns
        -------
        dict with keys:
            action_name       : str  -- human label of action taken
            step              : int  -- current episode step
            episode_reward    : float -- cumulative reward this episode
            dist_to_waypoint  : float -- current distance to goal
            speed             : float -- current aircraft speed
            pitch_deg         : float -- current pitch in degrees
            fuel              : float -- remaining fuel
            reward_breakdown  : dict  -- per-component reward values
            termination       : str  -- reason for termination (or "")
            terminated        : bool
            truncated         : bool
        """
        assert self._state is not None
        assert self._world is not None

        return {
            "action_name": ACTION_NAMES.get(action, "none") if action is not None else "none",
            "step": self._step_count,
            "episode_reward": round(self._episode_reward, 4),
            "dist_to_waypoint": round(
                self._world.distance_to_waypoint(self._state.x, self._state.y), 2
            ),
            "speed": round(self._state.speed, 2),
            "pitch_deg": round(math.degrees(self._state.theta), 2),
            "fuel": round(self._state.fuel, 1),
            "reward_breakdown": reward_breakdown,
            "termination": term_reason,
            "terminated": terminated,
            "truncated": truncated,
        }
