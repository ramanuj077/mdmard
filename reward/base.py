"""
reward/base.py
==============
BaseReward Abstract Base Class -- MDMARD Phase 1

All reward functions inherit from BaseReward.
This makes the reward pluggable: swap the class in config
without touching the environment code.

Design follows the Strategy Pattern:
  FlightEnv holds a reference to a BaseReward instance.
  Calling reward.compute(...) dispatches to the concrete class.

Potential-Based Shaping
-----------------------
Dense components follow the theorem from:
  Ng, Harada & Russell (1999) -- "Policy Invariance Under Reward Transformations"

A shaping function F(s, s') = γΦ(s') - Φ(s) does not change the
optimal policy. We use Φ(s) = -distance_to_goal so:
  F(s, s') = γ(-d') - (-d) = d - γd'
  ≈ d - d' when γ ≈ 1 (finite horizon)

This ensures approach reward is provably safe to add to any base reward.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from envs.physics import AircraftState
from envs.world import World


# ─────────────────────────────────────────────────────────────────────────────
# Reward Step Context
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RewardContext:
    """
    All information available to a reward function at one step.

    Passed by FlightEnv to BaseReward.compute() each timestep.
    Decouples the reward from the env internals -- the reward
    only sees what it needs, nothing else.

    Attributes
    ----------
    state : AircraftState
        Current aircraft state after physics step.
    prev_state : AircraftState
        Aircraft state before the physics step.
    world : World
        Simulation world (waypoint, obstacles, bounds).
    action : int
        Discrete action taken this step.
    prev_dist : float
        Distance to waypoint before this step (for approach shaping).
    curr_dist : float
        Distance to waypoint after this step.
    terminated : bool
        Whether the episode ended this step.
    term_reason : str
        Termination reason: "goal" | "oob" | "stall" | "fuel" | ""
    step : int
        Current episode step number.
    """
    state:       AircraftState
    prev_state:  AircraftState
    world:       World
    action:      int | np.ndarray
    prev_dist:   float
    curr_dist:   float
    terminated:  bool
    term_reason: str
    step:        int


# ─────────────────────────────────────────────────────────────────────────────
# Reward Output
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RewardOutput:
    """
    Output of a reward computation.

    Attributes
    ----------
    total : float
        Scalar reward to return to the RL agent.
    breakdown : dict[str, float]
        Per-component contributions for logging, HUD, and ablation.
    """
    total: float
    breakdown: dict[str, float] = field(default_factory=dict)

    def __add__(self, other: "RewardOutput") -> "RewardOutput":
        """Combine two RewardOutputs (used internally by CompositeReward)."""
        merged = {**self.breakdown, **other.breakdown}
        return RewardOutput(total=self.total + other.total, breakdown=merged)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Base Class
# ─────────────────────────────────────────────────────────────────────────────

class BaseReward(ABC):
    """
    Abstract base class for all reward functions.

    Subclasses implement compute() to return a RewardOutput.
    The FlightEnv calls reward.compute(ctx) every step.

    Parameters
    ----------
    config : dict
        Reward weights from YAML config. Subclasses read their
        specific keys from this dict.

    Example
    -------
    >>> class MyReward(BaseReward):
    ...     def compute(self, ctx: RewardContext) -> RewardOutput:
    ...         r = -ctx.curr_dist / 1000.0
    ...         return RewardOutput(total=r, breakdown={"my_component": r})
    """

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def compute(self, ctx: RewardContext) -> RewardOutput:
        """
        Compute the reward for one environment step.

        Parameters
        ----------
        ctx : RewardContext
            Full step context (state, world, action, distances, etc.)

        Returns
        -------
        RewardOutput
            total: scalar reward, breakdown: per-component dict
        """
        ...

    def reset(self) -> None:
        """
        Called at the start of each episode.

        Override if your reward has episode-level state (e.g. counters).
        Default implementation does nothing.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
