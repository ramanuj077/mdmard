"""
baselines/pid_agent.py
======================
PID Flight Controller Baseline Agent -- MDMARD Phase 1

Implements a Proportional-Integral-Derivative (PID) tactical flight controller:
  - Heading error regulation towards waypoint bearing
  - Anti-windup integral accumulation
  - Derivative damping against angular rate overshoot
  - Safety overrides for ground clearance and ceiling bounds
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from baselines.base_agent import BaseAgent


class PIDAgent(BaseAgent):
    """Classical PID heading and tactical flight controller."""

    # Action mappings:
    ACTION_THRUST = 0
    ACTION_PITCH_UP = 1
    ACTION_PITCH_DOWN = 2
    ACTION_THRUST_UP = 3
    ACTION_THRUST_DOWN = 4
    ACTION_GLIDE = 5

    def __init__(
        self,
        kp: float = 2.5,
        ki: float = 0.05,
        kd: float = 0.8,
        integral_max: float = 1.0,
        control_threshold: float = 0.15,
        altitude_floor_threshold: float = 0.15,
        altitude_ceil_threshold: float = 0.85,
        name: str = "PID",
    ) -> None:
        super().__init__(name=name)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_max = integral_max
        self.control_threshold = control_threshold
        self.altitude_floor_threshold = altitude_floor_threshold
        self.altitude_ceil_threshold = altitude_ceil_threshold

        # Internal state
        self._integral_error: float = 0.0
        self._prev_error: Optional[float] = None

    def reset(self) -> None:
        """Clear PID controller state."""
        self._integral_error = 0.0
        self._prev_error = None

    def act(
        self,
        observation: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Compute PID control output and map to discrete action."""
        # Unpack observation:
        # [0:x, 1:y, 2:vx, 3:vy, 4:theta, 5:omega, 6:dist, 7:bearing, 8:fuel, 9:time]
        alt_norm = float(observation[1])
        theta = float(observation[4]) * math.pi
        bearing = float(observation[7]) * math.pi
        omega_norm = float(observation[5])

        # 1. Critical Boundary Safety Overrides
        if alt_norm < self.altitude_floor_threshold:
            self._integral_error = 0.0
            return self.ACTION_THRUST_UP
        if alt_norm > self.altitude_ceil_threshold:
            self._integral_error = 0.0
            return self.ACTION_THRUST_DOWN

        # 2. Heading Error computation [-pi, pi]
        error = (bearing - theta + math.pi) % (2.0 * math.pi) - math.pi

        # 3. PID terms
        # P-term
        p_term = self.kp * error

        # I-term with anti-windup clamping
        self._integral_error += error
        self._integral_error = max(-self.integral_max, min(self.integral_max, self._integral_error))
        i_term = self.ki * self._integral_error

        # D-term: use error derivative or negative omega for damping
        if self._prev_error is not None:
            d_error = error - self._prev_error
        else:
            d_error = -omega_norm
        self._prev_error = error
        d_term = self.kd * d_error

        # Total control effort
        u = p_term + i_term + d_term

        # 4. Map continuous control effort u to discrete actions
        if u > self.control_threshold:
            return self.ACTION_THRUST_UP
        elif u < -self.control_threshold:
            return self.ACTION_THRUST_DOWN
        else:
            return self.ACTION_THRUST
