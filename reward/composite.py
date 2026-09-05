"""
reward/composite.py
===================
CompositeReward -- MDMARD Phase 1

Combines multiple BaseReward components with per-component weights.
This is the reward function used by TacticalFlightEnv.

Usage Pattern
-------------
  1. Define components and weights in configs/env_phase1.yaml
  2. CompositeReward.from_config(yaml_dict) builds it automatically
  3. Ablation: set a component's weight to 0 in YAML

Weighted Total
--------------
  total = sum(weight_i * component_i.compute(ctx).total)

The breakdown dict contains WEIGHTED values -- this is what the
HUD, TensorBoard, and research report display.

Example Config (YAML)
---------------------
  reward:
    w_approach: 1.0
    w_waypoint: 200.0
    w_crash: 1.0
    w_alive: 0.1
    w_smooth: 0.05
    w_altitude: 0.5
    w_fuel: 0.0    # disabled until Stage 13
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from reward.base import BaseReward, RewardContext, RewardOutput
from reward.components import (
    ApproachReward,
    WaypointReward,
    CrashPenalty,
    AliveReward,
    SmoothReward,
    AltitudeReward,
    FuelReward,
)


# ─────────────────────────────────────────────────────────────────────────────
# CompositeReward
# ─────────────────────────────────────────────────────────────────────────────

class CompositeReward(BaseReward):
    """
    Weighted sum of multiple reward components.

    Parameters
    ----------
    components : List[Tuple[BaseReward, float]]
        List of (component_instance, weight) pairs.
        Components are evaluated in order; weights applied to totals.

    Example
    -------
    >>> reward_fn = CompositeReward(components=[
    ...     (ApproachReward(), 1.0),
    ...     (WaypointReward({"bonus": 200.0}), 1.0),
    ...     (CrashPenalty({"penalty": 100.0}), 1.0),
    ...     (AliveReward({"bonus": 0.1}), 1.0),
    ...     (SmoothReward({"scale": 0.05}), 1.0),
    ...     (AltitudeReward({"scale": 0.5}), 1.0),
    ... ])
    >>> output = reward_fn.compute(ctx)
    >>> output.total, output.breakdown
    """

    def __init__(
        self,
        components: List[Tuple[BaseReward, float]],
    ) -> None:
        super().__init__()
        self._components = components

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, reward_cfg: dict) -> "CompositeReward":
        """
        Build a CompositeReward from the 'reward' section of env_phase1.yaml.

        Parameters
        ----------
        reward_cfg : dict
            The 'reward' sub-dict. Keys: w_approach, w_waypoint, etc.

        Returns
        -------
        CompositeReward

        Notes
        -----
        Any weight set to 0.0 effectively disables that component
        (it's still computed but contributes 0 to total).
        For research efficiency, we skip components with weight=0.
        """
        components = [
            (ApproachReward(),                                    reward_cfg.get("w_approach", 1.0)),
            (WaypointReward({"bonus":   reward_cfg.get("w_waypoint", 200.0)}), 1.0),
            (CrashPenalty({"penalty":   abs(reward_cfg.get("w_crash", 100.0))}), 1.0),
            (AliveReward({"bonus":      reward_cfg.get("w_alive", 0.1)}),       1.0),
            (SmoothReward({"scale":     abs(reward_cfg.get("w_smooth", 0.05))}),1.0),
            (AltitudeReward({"scale":   abs(reward_cfg.get("w_altitude", 0.5))}),1.0),
            (FuelReward({"scale":       abs(reward_cfg.get("w_fuel", 0.0))}),   1.0),
        ]
        return cls(components=components)

    @classmethod
    def default(cls) -> "CompositeReward":
        """
        Build with Phase 1 default weights.
        Useful for unit tests without loading YAML.
        """
        return cls.from_config({})

    # ── BaseReward Interface ───────────────────────────────────────────────────

    def compute(self, ctx: RewardContext) -> RewardOutput:
        """
        Compute total reward as weighted sum of all components.

        Parameters
        ----------
        ctx : RewardContext

        Returns
        -------
        RewardOutput
            total: weighted sum, breakdown: {name: weighted_value}
        """
        total = 0.0
        breakdown: dict[str, float] = {}

        for component, weight in self._components:
            out = component.compute(ctx)
            weighted_total = weight * out.total
            total += weighted_total
            # Store WEIGHTED values in breakdown for interpretability
            for key, val in out.breakdown.items():
                breakdown[key] = round(weight * val, 4)

        return RewardOutput(total=float(total), breakdown=breakdown)

    def reset(self) -> None:
        """Reset all component states (called each episode)."""
        for component, _ in self._components:
            component.reset()

    # ── Introspection ─────────────────────────────────────────────────────────

    def component_names(self) -> list[str]:
        """Return list of component class names (for logging)."""
        return [type(c).__name__ for c, _ in self._components]

    def weights(self) -> list[float]:
        """Return list of component weights."""
        return [w for _, w in self._components]

    def summary(self) -> str:
        """Human-readable summary of components and weights."""
        lines = ["CompositeReward:"]
        for comp, w in self._components:
            lines.append(f"  {type(comp).__name__:<20}  weight={w}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        parts = [f"{type(c).__name__}×{w}" for c, w in self._components]
        return f"CompositeReward([{', '.join(parts)}])"
