"""
envs/termination.py
===================
TerminationChecker -- MDMARD Phase 1

Encapsulates all episode termination logic in one class.
Decoupled from physics, reward, and observation.

Termination vs Truncation (Gymnasium 0.26+ API)
------------------------------------------------
  terminated = True  : Agent caused episode to end (crash, goal, fuel out)
                        These are MDPS terminal states.
  truncated  = True  : Environment cut the episode (max steps)
                        The episode is NOT at a terminal state.
  Both False         : Episode continues normally.
  Both True          : NEVER allowed (violates Gymnasium spec).

Termination Conditions
----------------------
  "goal"   : Aircraft entered waypoint capture radius.  → SUCCESS
  "oob"    : Aircraft exited airspace boundaries.       → FAILURE
  "stall"  : Speed < v_stall for N consecutive steps.  → FAILURE
  "fuel"   : Fuel exhausted (only when fuel enabled).  → FAILURE
  ""       : No termination this step.

References
----------
  [1] Gymnasium API -- https://gymnasium.farama.org/api/env/
  [2] Sutton & Barto (2018) Ch. 3 -- Finite MDPs and terminal states
"""

from __future__ import annotations

from dataclasses import dataclass

from envs.physics import AircraftState, PhysicsConfig
from envs.world import World


# ─────────────────────────────────────────────────────────────────────────────
# Termination Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TerminationResult:
    """
    Result of a termination check.

    Attributes
    ----------
    terminated : bool
        True if the episode has reached a terminal state.
    truncated : bool
        True if the episode was cut short by the step budget.
    reason : str
        Human-readable reason: "goal" | "oob" | "stall" | "fuel" | "timeout" | ""
    is_success : bool
        True only when termination reason is "goal".
    """
    terminated: bool = False
    truncated:  bool = False
    reason:     str  = ""

    @property
    def is_success(self) -> bool:
        """True if the episode ended with goal capture."""
        return self.reason == "goal"

    @property
    def is_done(self) -> bool:
        """True if episode is over for any reason."""
        return self.terminated or self.truncated

    def __repr__(self) -> str:
        return (
            f"TerminationResult(terminated={self.terminated}, "
            f"truncated={self.truncated}, reason='{self.reason}')"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Termination Checker
# ─────────────────────────────────────────────────────────────────────────────

class TerminationChecker:
    """
    Checks all episode termination conditions each step.

    Owned by FlightEnv. Called after every physics step.
    State is reset at the start of each episode via reset().

    Parameters
    ----------
    physics_config : PhysicsConfig
        Used for vel_stall and stall_steps thresholds.
    max_steps : int
        Episode step budget. Exhausting it gives truncated=True.

    Example
    -------
    >>> checker = TerminationChecker(PhysicsConfig(), max_steps=1000)
    >>> checker.reset()
    >>> result = checker.check(state, world, step=1)
    >>> result.terminated, result.reason
    (False, '')
    """

    def __init__(self, physics_config: PhysicsConfig, max_steps: int) -> None:
        self._cfg       = physics_config
        self._max_steps = max_steps
        self._stall_counter: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Reset episode-level state.

        Must be called at the start of each episode (in env.reset()).
        Clears the stall counter so it doesn't carry across episodes.
        """
        self._stall_counter = 0

    def check(
        self,
        state: AircraftState,
        world: World,
        step: int,
    ) -> TerminationResult:
        """
        Check all termination and truncation conditions.

        Conditions are checked in priority order:
          1. Goal (highest priority -- success always wins)
          2. Out of bounds
          3. Stall
          4. Fuel exhausted
          5. Timeout (truncation, lowest priority)

        Parameters
        ----------
        state : AircraftState
            Current aircraft state (after physics step).
        world : World
            Current world state (has waypoint reference).
        step : int
            Current episode step number (1-indexed).

        Returns
        -------
        TerminationResult
            terminated, truncated, reason, is_success flags.
        """
        x, y = state.x, state.y

        # ── 1. Goal reached ───────────────────────────────────────────────────
        if world.waypoint is not None:
            if world.waypoint.check_capture(x, y):
                return TerminationResult(terminated=True, reason="goal")

        # ── 2. Out of bounds ──────────────────────────────────────────────────
        if world.is_out_of_bounds(x, y):
            return TerminationResult(terminated=True, reason="oob")

        # ── 3. Obstacle collision ─────────────────────────────────────────────
        if world.is_in_obstacle(x, y):
            return TerminationResult(terminated=True, reason="obstacle")

        # ── 4. Stall ──────────────────────────────────────────────────────────
        if state.speed < self._cfg.vel_stall:
            self._stall_counter += 1
            if self._stall_counter >= self._cfg.stall_steps:
                return TerminationResult(terminated=True, reason="stall")
        else:
            self._stall_counter = 0

        # ── 5. Fuel exhausted ─────────────────────────────────────────────────
        if state.fuel <= 0.0 and self._cfg.fuel_burn_rate > 0:
            return TerminationResult(terminated=True, reason="fuel")

        # ── 6. Timeout (truncation) ───────────────────────────────────────────
        if step >= self._max_steps:
            return TerminationResult(truncated=True, reason="timeout")

        # ── No termination ────────────────────────────────────────────────────
        return TerminationResult()

    @property
    def stall_counter(self) -> int:
        """Current consecutive stall step count. Useful for HUD display."""
        return self._stall_counter
