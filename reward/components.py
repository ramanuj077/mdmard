"""
reward/components.py
====================
Individual Reward Components -- MDMARD Phase 1

Each component is an independent class inheriting from BaseReward.
This makes ablation studies trivial: remove a class from the composite.

Components
----------
  ApproachReward    : Dense shaping -- reward for closing distance to goal
  WaypointReward    : Sparse bonus  -- large reward on goal capture
  CrashPenalty      : Sparse penalty -- on OOB / stall / obstacle / fuel
  AliveReward       : Dense bonus   -- small per-step survival reward
  SmoothReward      : Dense penalty -- penalises erratic pitch rate
  AltitudeReward    : Dense penalty -- penalises floor/ceiling proximity
  FuelReward        : Dense penalty -- penalises fuel consumption rate

Design
------
Each component reads ONLY what it needs from RewardContext.
Components have no shared state (except where necessary, e.g. counters).
Weights are applied in CompositeReward, not here -- components always
return un-weighted values so the breakdown is interpretable.

References
----------
  [1] Ng, Harada & Russell (1999) -- "Policy Invariance Under Reward
      Transformations: Theory and Application to Reward Shaping"
  [2] Randlov & Alstrom (1998) -- reward shaping for bicycle control
"""

from __future__ import annotations

import math

from reward.base import BaseReward, RewardContext, RewardOutput


# ─────────────────────────────────────────────────────────────────────────────
# ApproachReward
# ─────────────────────────────────────────────────────────────────────────────

class ApproachReward(BaseReward):
    """
    Potential-based approach shaping reward.

    r = prev_dist - curr_dist

    Positive when the aircraft moves closer to the waypoint.
    Negative when moving away.

    This is provably safe under the reward shaping theorem:
    it does not change the optimal policy.

    Parameters
    ----------
    config : dict
        No specific keys required.
    """

    def compute(self, ctx: RewardContext) -> RewardOutput:
        r = ctx.prev_dist - ctx.curr_dist
        return RewardOutput(total=r, breakdown={"approach": round(r, 4)})


# ─────────────────────────────────────────────────────────────────────────────
# WaypointReward
# ─────────────────────────────────────────────────────────────────────────────

class WaypointReward(BaseReward):
    """
    Sparse goal-capture bonus.

    r = +bonus when term_reason == "goal", else 0.

    Parameters
    ----------
    config : dict
        bonus : float  -- reward on goal capture. Default 200.0
    """

    def compute(self, ctx: RewardContext) -> RewardOutput:
        bonus = float(self.config.get("bonus", 200.0))
        r = bonus if (ctx.terminated and ctx.term_reason == "goal") else 0.0
        return RewardOutput(total=r, breakdown={"waypoint": r})


# ─────────────────────────────────────────────────────────────────────────────
# CrashPenalty
# ─────────────────────────────────────────────────────────────────────────────

class CrashPenalty(BaseReward):
    """
    Sparse crash / failure penalty.

    r = -penalty when terminated for any failure reason (oob, stall,
    obstacle, fuel). No penalty on goal or timeout.

    Parameters
    ----------
    config : dict
        penalty : float  -- penalty magnitude. Default 100.0 (applied as -)
    """

    FAILURE_REASONS = {"oob", "stall", "obstacle", "fuel"}

    def compute(self, ctx: RewardContext) -> RewardOutput:
        penalty = float(self.config.get("penalty", 100.0))
        r = -penalty if (ctx.terminated and ctx.term_reason in self.FAILURE_REASONS) else 0.0
        return RewardOutput(total=r, breakdown={"crash": r})


# ─────────────────────────────────────────────────────────────────────────────
# AliveReward
# ─────────────────────────────────────────────────────────────────────────────

class AliveReward(BaseReward):
    """
    Small per-step survival bonus.

    Encourages the agent to stay alive longer (explore more).
    Also ensures the agent prefers long successful episodes
    over short crashing ones.

    r = +bonus per step (only when episode is not done).

    Parameters
    ----------
    config : dict
        bonus : float  -- per-step bonus. Default 0.1
    """

    def compute(self, ctx: RewardContext) -> RewardOutput:
        bonus = float(self.config.get("bonus", 0.1))
        # Only reward alive steps (not terminal steps)
        r = bonus if not ctx.terminated else 0.0
        return RewardOutput(total=r, breakdown={"alive": r})


# ─────────────────────────────────────────────────────────────────────────────
# SmoothReward
# ─────────────────────────────────────────────────────────────────────────────

class SmoothReward(BaseReward):
    """
    Pitch rate smoothness penalty.

    Penalises erratic maneuvering by charging for large angular
    velocity. Encourages tactically realistic, smooth flight.

    r = -|omega|   (always <= 0)

    Parameters
    ----------
    config : dict
        scale : float -- penalty multiplier. Default 0.05
    """

    def compute(self, ctx: RewardContext) -> RewardOutput:
        scale = float(self.config.get("scale", 0.05))
        r = -scale * abs(ctx.state.omega)
        return RewardOutput(total=r, breakdown={"smooth": round(r, 4)})


# ─────────────────────────────────────────────────────────────────────────────
# AltitudeReward
# ─────────────────────────────────────────────────────────────────────────────

class AltitudeReward(BaseReward):
    """
    Altitude safety penalty.

    Penalises flying near the terrain floor or ceiling.
    Uses World.altitude_margin() which returns 0 at boundaries
    and 1 at vertical center.

    r = -scale * (1 - margin)   (always <= 0)

    Parameters
    ----------
    config : dict
        scale : float -- penalty multiplier. Default 0.5
    """

    def compute(self, ctx: RewardContext) -> RewardOutput:
        scale = float(self.config.get("scale", 0.5))
        margin = ctx.world.altitude_margin(ctx.state.y)
        r = -scale * (1.0 - margin)
        return RewardOutput(total=r, breakdown={"altitude": round(r, 4)})


# ─────────────────────────────────────────────────────────────────────────────
# FuelReward
# ─────────────────────────────────────────────────────────────────────────────

class FuelReward(BaseReward):
    """
    Fuel consumption penalty.

    Penalises burning fuel, encouraging fuel-efficient flight.
    Inactive by default (weight=0 in config) until Stage 13.

    r = -scale * fuel_burned_this_step   (always <= 0)

    The fuel burned this step = prev_fuel - curr_fuel.

    Parameters
    ----------
    config : dict
        scale : float -- penalty multiplier. Default 0.001
    """

    def compute(self, ctx: RewardContext) -> RewardOutput:
        scale = float(self.config.get("scale", 0.001))
        fuel_burned = max(0.0, ctx.prev_state.fuel - ctx.state.fuel)
        r = -scale * fuel_burned
        return RewardOutput(total=r, breakdown={"fuel": round(r, 6)})
