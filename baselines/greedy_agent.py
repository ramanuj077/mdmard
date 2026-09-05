"""
baselines/greedy_agent.py
=========================
Greedy Heuristic Baseline Agent -- MDMARD Phase 1

A deterministic rule-based heuristic controller:
  1. Computes angular error between current pitch (theta) and target bearing.
  2. Enforces altitude safety envelopes (avoids ground crash and ceiling exit).
  3. Commands appropriate pitch corrections and engine thrust towards the goal.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from baselines.base_agent import BaseAgent


class GreedyAgent(BaseAgent):
    """Heuristic rule-based agent for tactical waypoint navigation."""

    # Action mappings:
    # 0: Thrust, 1: Pitch Up, 2: Pitch Down, 3: Thrust+Up, 4: Thrust+Down, 5: Glide
    ACTION_THRUST = 0
    ACTION_PITCH_UP = 1
    ACTION_PITCH_DOWN = 2
    ACTION_THRUST_UP = 3
    ACTION_THRUST_DOWN = 4
    ACTION_GLIDE = 5

    def __init__(
        self,
        heading_deadband: float = 0.08,        # ~4.5 degrees
        altitude_floor_threshold: float = 0.15, # 15% of world height
        altitude_ceil_threshold: float = 0.85,  # 85% of world height
        speed_stall_threshold: float = 0.25,    # 25% of vel_max
        name: str = "Greedy",
    ) -> None:
        super().__init__(name=name)
        self.heading_deadband = heading_deadband
        self.altitude_floor_threshold = altitude_floor_threshold
        self.altitude_ceil_threshold = altitude_ceil_threshold
        self.speed_stall_threshold = speed_stall_threshold

    def act(
        self,
        observation: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Compute greedy tactical flight action based on observation vector."""
        # Unpack normalized observation vector
        # [0:x, 1:y, 2:vx, 3:vy, 4:theta, 5:omega, 6:dist, 7:bearing, 8:fuel, 9:time]
        alt_norm = float(observation[1])
        vx_norm = float(observation[2])
        vy_norm = float(observation[3])
        theta = float(observation[4]) * math.pi
        bearing = float(observation[7]) * math.pi
        speed_norm = math.hypot(vx_norm, vy_norm)

        # 1. Critical Ground Safety: Override to climb if dangerously low
        if alt_norm < self.altitude_floor_threshold:
            return self.ACTION_THRUST_UP

        # 2. Critical Ceiling Safety: Override to descend if dangerously high
        if alt_norm > self.altitude_ceil_threshold:
            return self.ACTION_THRUST_DOWN

        # 3. Stall Recovery: If speed is very low, apply thrust
        if speed_norm < self.speed_stall_threshold:
            return self.ACTION_THRUST

        # 4. Heading error wrapped to [-pi, pi]
        delta_theta = (bearing - theta + math.pi) % (2.0 * math.pi) - math.pi

        # 5. Steering commands
        if delta_theta > self.heading_deadband:
            # Need to pitch up
            return self.ACTION_THRUST_UP
        elif delta_theta < -self.heading_deadband:
            # Need to pitch down
            return self.ACTION_THRUST_DOWN
        else:
            # Aligned: pure forward thrust to close distance
            return self.ACTION_THRUST
