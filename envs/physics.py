"""
envs/physics.py
===============
Aircraft Physics Engine — MDMARD Phase 1

Implements point-mass + pitch dynamics for a 2D tactical air agent.
Uses 4th-order Runge-Kutta (RK4) integration for numerical stability.

Coordinate Frame
----------------
  x : horizontal position (right is positive)
  y : vertical position   (up is positive)
  θ : pitch angle in radians (0 = level flight, + = nose up)
  ω : pitch rate (rad/s, + = nose pitching up)

Forces
------
  Thrust  : along heading (cos θ, sin θ), scaled by throttle ∈ [0, 1]
  Drag    : opposes velocity vector, proportional to v²
  Lift    : perpendicular to velocity, proportional to v² × sin(θ)
  Gravity : constant downward (0, -mg)

Numerical Method
----------------
  RK4 is used instead of Euler because aerodynamic forces (drag ∝ v²,
  lift ∝ v²) make Euler unstable at moderate dt. RK4 gives 4th-order
  accuracy with only 4 force evaluations per step.

References
----------
  [1] Stevens, Lewis & Johnson — "Aircraft Simulation and Control", 2016
  [2] Sutton & Barto — "Reinforcement Learning: An Introduction", 2018
      (Ch. 10 — continuous state spaces)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AircraftState:
    """
    Full kinematic state of the aircraft at a single timestep.

    All units are SI (meters, seconds, radians) unless noted.
    The renderer maps simulation units 1:1 to screen pixels.

    Attributes
    ----------
    x : float
        Horizontal position.
    y : float
        Vertical position (altitude). y=0 is ground level.
    vx : float
        Horizontal velocity component.
    vy : float
        Vertical velocity component.
    theta : float
        Pitch angle in radians. 0 = level, +π/2 = straight up.
    omega : float
        Pitch rate in rad/s. Positive = nose pitching up.
    fuel : float
        Remaining fuel units. Not consumed in Phase 1 Stage 1–8.
    step : int
        Current simulation step (incremented by PhysicsEngine).
    """
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    theta: float = 0.0
    omega: float = 0.0
    fuel: float = 1000.0
    step: int = 0

    # ── Derived convenience properties ──────────────────────────────────────

    @property
    def speed(self) -> float:
        """Scalar speed: ‖v‖ = sqrt(vx² + vy²)."""
        return math.sqrt(self.vx ** 2 + self.vy ** 2)

    @property
    def position(self) -> Tuple[float, float]:
        """(x, y) position tuple."""
        return (self.x, self.y)

    @property
    def velocity(self) -> Tuple[float, float]:
        """(vx, vy) velocity tuple."""
        return (self.vx, self.vy)

    def copy(self) -> "AircraftState":
        """Return a shallow copy (all fields are scalars)."""
        return AircraftState(
            x=self.x, y=self.y,
            vx=self.vx, vy=self.vy,
            theta=self.theta, omega=self.omega,
            fuel=self.fuel, step=self.step,
        )

    def as_array(self) -> np.ndarray:
        """Return state as numpy array for logging / debugging."""
        return np.array([self.x, self.y, self.vx, self.vy,
                         self.theta, self.omega, self.fuel], dtype=np.float32)

    def __repr__(self) -> str:
        return (
            f"AircraftState(pos=({self.x:.1f},{self.y:.1f}), "
            f"vel=({self.vx:.2f},{self.vy:.2f}), "
            f"θ={math.degrees(self.theta):.1f}°, "
            f"ω={self.omega:.2f}rad/s, "
            f"speed={self.speed:.2f}, fuel={self.fuel:.1f})"
        )


@dataclass
class PhysicsConfig:
    """
    Physical constants for the aircraft simulation.

    All values are loaded from configs/env_phase1.yaml via the
    config loader (utils/config.py). Direct instantiation is used
    in unit tests for isolation.

    Attributes
    ----------
    mass : float
        Aircraft mass (kg). Affects all translational dynamics.
    inertia : float
        Rotational inertia (kg·m²). Affects pitch dynamics.
    thrust_max : float
        Maximum thrust force (N).
    drag_coeff : float
        Translational drag coefficient (Cd). Drag = Cd × v².
    lift_coeff : float
        Lift coefficient (Cl). Lift = Cl × v² × sin(θ).
    gravity : float
        Gravitational acceleration (m/s²).
    vel_max : float
        Maximum allowable speed (m/s). Hard clamp applied.
    vel_stall : float
        Minimum speed for controlled flight (m/s).
    pitch_max : float
        Maximum pitch angle magnitude (rad). Clamped each step.
    pitch_rate_max : float
        Maximum pitch rate magnitude (rad/s).
    pitch_torque : float
        Torque magnitude applied per unit elevator input.
    dt : float
        Simulation timestep (seconds).
    fuel_burn_rate : float
        Fuel consumed per thrust unit per second.
    """
    mass: float = 1.0
    inertia: float = 0.1
    thrust_max: float = 15.0
    drag_coeff: float = 0.02
    lift_coeff: float = 0.08
    gravity: float = 9.81
    vel_max: float = 80.0
    vel_stall: float = 5.0
    pitch_max: float = math.pi / 2
    pitch_rate_max: float = 2.0
    pitch_torque: float = 5.0
    dt: float = 0.05
    fuel_burn_rate: float = 0.5
    stall_steps: int = 15        # consecutive steps below v_stall before termination

    @classmethod
    def from_yaml(cls, cfg: dict) -> "PhysicsConfig":
        """
        Construct from the 'physics' sub-dict of a loaded YAML config.

        Parameters
        ----------
        cfg : dict
            The 'physics' section of env_phase1.yaml.

        Returns
        -------
        PhysicsConfig
        """
        return cls(
            mass=cfg.get("mass", 1.0),
            inertia=cfg.get("inertia", 0.1),
            thrust_max=cfg.get("thrust_max", 15.0),
            drag_coeff=cfg.get("drag_coeff", 0.02),
            lift_coeff=cfg.get("lift_coeff", 0.08),
            gravity=cfg.get("gravity", 9.81),
            vel_max=cfg.get("vel_max", 80.0),
            vel_stall=cfg.get("vel_stall", 5.0),
            pitch_max=cfg.get("pitch_max", math.pi / 2),
            pitch_rate_max=cfg.get("pitch_rate_max", 2.0),
            pitch_torque=cfg.get("pitch_torque", 5.0),
            dt=cfg.get("dt", 0.05),
            fuel_burn_rate=cfg.get("burn_rate", 0.5),
            stall_steps=cfg.get("stall_steps", 15),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Action Decoder
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ActionComponents:
    """
    Decoded action as continuous control inputs.

    The discrete action set is mapped to these components before
    physics integration, so the PhysicsEngine is action-space-agnostic.
    Switching to continuous actions (Phase 1B) only changes the decoder.

    Attributes
    ----------
    throttle : float
        Thrust fraction ∈ [0.0, 1.0]. 1.0 = full thrust.
    elevator : float
        Elevator deflection ∈ [-1.0, 1.0].
        +1.0 = full nose-up, -1.0 = full nose-down.
    """
    throttle: float = 0.0
    elevator: float = 0.0


# Discrete action map: action_id → (throttle, elevator)
DISCRETE_ACTION_MAP: dict[int, ActionComponents] = {
    0: ActionComponents(throttle=1.0,  elevator=0.0),   # Thrust full
    1: ActionComponents(throttle=0.0,  elevator=1.0),   # Pitch up
    2: ActionComponents(throttle=0.0,  elevator=-1.0),  # Pitch down
    3: ActionComponents(throttle=1.0,  elevator=1.0),   # Thrust + Pitch up
    4: ActionComponents(throttle=1.0,  elevator=-1.0),  # Thrust + Pitch down
    5: ActionComponents(throttle=0.0,  elevator=0.0),   # No-op (glide)
}

ACTION_NAMES: dict[int, str] = {
    0: "Thrust",
    1: "Pitch Up",
    2: "Pitch Down",
    3: "Thrust + Pitch Up",
    4: "Thrust + Pitch Down",
    5: "Glide",
}


def decode_discrete_action(action_id: int) -> ActionComponents:
    """
    Map a discrete action index to continuous control components.

    Parameters
    ----------
    action_id : int
        Integer in [0, 5].

    Returns
    -------
    ActionComponents

    Raises
    ------
    ValueError
        If action_id is out of range.
    """
    if action_id not in DISCRETE_ACTION_MAP:
        raise ValueError(
            f"Invalid action_id={action_id}. "
            f"Valid range: {list(DISCRETE_ACTION_MAP.keys())}"
        )
    return DISCRETE_ACTION_MAP[action_id]


def decode_continuous_action(action: np.ndarray) -> ActionComponents:
    """
    Map a continuous action vector to control components.

    Used in Phase 1B (SAC). Action is expected to be in [-1, 1]².

    Parameters
    ----------
    action : np.ndarray
        Shape (2,) — [throttle_raw, elevator_raw] ∈ [-1, 1].

    Returns
    -------
    ActionComponents
        throttle ∈ [0, 1] (action[0] shifted from [-1,1] → [0,1])
        elevator ∈ [-1, 1] (action[1] passed through directly)
    """
    throttle = float(np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0))
    elevator = float(np.clip(action[1], -1.0, 1.0))
    return ActionComponents(throttle=throttle, elevator=elevator)


# ─────────────────────────────────────────────────────────────────────────────
# Physics Engine
# ─────────────────────────────────────────────────────────────────────────────

class PhysicsEngine:
    """
    Simulates 2D aircraft dynamics using RK4 numerical integration.

    The engine is stateless: it takes a state and action, returns a new state.
    The environment (FlightEnv) owns the state and calls step() each tick.

    This design means:
    - PhysicsEngine can be unit-tested independently
    - Multiple agents can share one engine (MARL Phase 2)
    - Rollout replay is trivially reproducible (deterministic given same seed)

    Parameters
    ----------
    config : PhysicsConfig
        Physical constants for the simulation.

    Example
    -------
    >>> cfg = PhysicsConfig()
    >>> engine = PhysicsEngine(cfg)
    >>> state = AircraftState(x=100, y=400, vx=10, vy=0, theta=0)
    >>> new_state = engine.step(state, action_id=0)
    >>> print(new_state)
    """

    def __init__(self, config: PhysicsConfig) -> None:
        self.cfg = config

    # ── Public API ───────────────────────────────────────────────────────────

    def step(
        self,
        state: AircraftState,
        action_id: int | None = None,
        action_continuous: np.ndarray | None = None,
    ) -> AircraftState:
        """
        Advance physics by one timestep (dt) using RK4 integration.

        Exactly one of action_id or action_continuous must be provided.

        Parameters
        ----------
        state : AircraftState
            Current state of the aircraft.
        action_id : int, optional
            Discrete action index ∈ [0, 5]. Used in Phase 1A.
        action_continuous : np.ndarray, optional
            Continuous action vector shape (2,). Used in Phase 1B.

        Returns
        -------
        AircraftState
            New state after one simulation step.

        Raises
        ------
        ValueError
            If neither or both action types are provided.
        """
        if action_id is not None and action_continuous is not None:
            raise ValueError("Provide exactly one of action_id or action_continuous.")
        if action_id is None and action_continuous is None:
            raise ValueError("Provide at least one of action_id or action_continuous.")

        # Decode action to control components
        if action_id is not None:
            ctrl = decode_discrete_action(action_id)
        else:
            ctrl = decode_continuous_action(action_continuous)  # type: ignore[arg-type]

        # Pack state into a raw vector for RK4
        # raw = [x, y, vx, vy, theta, omega]
        raw = np.array([state.x, state.y, state.vx, state.vy,
                        state.theta, state.omega], dtype=np.float64)

        # RK4 integration
        integrated = self._rk4(raw, ctrl)

        # Apply constraints
        new_state = self._apply_constraints(integrated, state, ctrl)
        new_state.step = state.step + 1

        return new_state

    def compute_forces(
        self,
        vx: float,
        vy: float,
        theta: float,
        ctrl: ActionComponents,
    ) -> Tuple[float, float]:
        """
        Compute net translational force components (Fx, Fy).

        Used internally by RK4 and also exposed for unit testing
        (verify each force component independently).

        Parameters
        ----------
        vx, vy : float
            Velocity components.
        theta : float
            Pitch angle (rad).
        ctrl : ActionComponents
            Decoded control inputs.

        Returns
        -------
        Tuple[float, float]
            (Fx, Fy) — net force in x and y directions.
        """
        cfg = self.cfg
        speed = math.sqrt(vx ** 2 + vy ** 2)

        # ── Thrust (along heading vector) ────────────────────────────────────
        thrust = cfg.thrust_max * ctrl.throttle
        fx_thrust = thrust * math.cos(theta)
        fy_thrust = thrust * math.sin(theta)

        # ── Drag (opposes velocity) ──────────────────────────────────────────
        # F_drag = -Cd * speed * v  (linear in v for each component, ∝ v²)
        fx_drag = -cfg.drag_coeff * speed * vx
        fy_drag = -cfg.drag_coeff * speed * vy

        # ── Lift (perpendicular to velocity, ∝ v² × sin θ) ──────────────────
        # Simplified: lift acts vertically (ignores velocity direction in Ph1)
        lift_magnitude = cfg.lift_coeff * speed ** 2 * math.sin(theta)
        # Perpendicular to velocity: rotate velocity 90° CCW
        if speed > 1e-6:
            fx_lift = -lift_magnitude * (vy / speed)
            fy_lift =  lift_magnitude * (vx / speed)
        else:
            fx_lift, fy_lift = 0.0, 0.0

        # ── Gravity ──────────────────────────────────────────────────────────
        fy_gravity = -cfg.mass * cfg.gravity

        # ── Net force ────────────────────────────────────────────────────────
        fx = fx_thrust + fx_drag + fx_lift
        fy = fy_thrust + fy_drag + fy_lift + fy_gravity

        return fx, fy

    # ── Private: RK4 Integration ─────────────────────────────────────────────

    def _derivatives(
        self, raw: np.ndarray, ctrl: ActionComponents
    ) -> np.ndarray:
        """
        Compute time-derivatives of the state vector.

        State raw = [x, y, vx, vy, theta, omega]
        Derivatives: [vx, vy, ax, ay, omega, alpha]

        Parameters
        ----------
        raw : np.ndarray
            Current raw state (6,).
        ctrl : ActionComponents
            Control inputs.

        Returns
        -------
        np.ndarray
            Derivatives (6,).
        """
        x, y, vx, vy, theta, omega = raw
        cfg = self.cfg

        # Translational acceleration
        fx, fy = self.compute_forces(vx, vy, theta, ctrl)
        ax = fx / cfg.mass
        ay = fy / cfg.mass

        # Rotational acceleration (pitch)
        # Torque = pitch_torque × elevator_input
        # alpha = torque / inertia
        torque = cfg.pitch_torque * ctrl.elevator
        alpha = torque / cfg.inertia

        return np.array([vx, vy, ax, ay, omega, alpha], dtype=np.float64)

    def _rk4(self, raw: np.ndarray, ctrl: ActionComponents) -> np.ndarray:
        """
        4th-order Runge-Kutta integration.

        Computes the state at t+dt given state at t.

        RK4 Formula:
            k1 = f(t, y)
            k2 = f(t + dt/2, y + dt/2 × k1)
            k3 = f(t + dt/2, y + dt/2 × k2)
            k4 = f(t + dt,   y + dt × k3)
            y(t+dt) = y(t) + (dt/6) × (k1 + 2k2 + 2k3 + k4)

        Parameters
        ----------
        raw : np.ndarray
            State at time t, shape (6,).
        ctrl : ActionComponents
            Control inputs (constant over this step).

        Returns
        -------
        np.ndarray
            State at time t+dt, shape (6,).
        """
        dt = self.cfg.dt

        k1 = self._derivatives(raw,              ctrl)
        k2 = self._derivatives(raw + dt/2 * k1, ctrl)
        k3 = self._derivatives(raw + dt/2 * k2, ctrl)
        k4 = self._derivatives(raw + dt   * k3, ctrl)

        return raw + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def _apply_constraints(
        self,
        raw: np.ndarray,
        prev_state: AircraftState,
        ctrl: ActionComponents,
    ) -> AircraftState:
        """
        Apply physical constraints and pack raw array back to AircraftState.

        Constraints applied:
        - Speed clamped to [0, vel_max]
        - Pitch angle clamped to [-pitch_max, +pitch_max]
        - Pitch rate clamped to [-pitch_rate_max, +pitch_rate_max]
        - Fuel decremented (when enabled)

        Parameters
        ----------
        raw : np.ndarray
            Integrated raw state (6,).
        prev_state : AircraftState
            Previous state (used for fuel, step count).
        ctrl : ActionComponents
            Control inputs (for fuel burn calculation).

        Returns
        -------
        AircraftState
        """
        cfg = self.cfg
        x, y, vx, vy, theta, omega = raw

        # ── Speed clamp ──────────────────────────────────────────────────────
        speed = math.sqrt(vx**2 + vy**2)
        if speed > cfg.vel_max:
            scale = cfg.vel_max / speed
            vx *= scale
            vy *= scale

        # ── Pitch clamp ──────────────────────────────────────────────────────
        theta = float(np.clip(theta, -cfg.pitch_max, cfg.pitch_max))
        omega = float(np.clip(omega, -cfg.pitch_rate_max, cfg.pitch_rate_max))

        # ── Fuel consumption ─────────────────────────────────────────────────
        fuel_burned = ctrl.throttle * cfg.fuel_burn_rate * cfg.dt
        new_fuel = max(0.0, prev_state.fuel - fuel_burned)

        return AircraftState(
            x=float(x),
            y=float(y),
            vx=float(vx),
            vy=float(vy),
            theta=float(theta),
            omega=float(omega),
            fuel=new_fuel,
            step=prev_state.step,  # step incremented by caller
        )

    # ── Utility ──────────────────────────────────────────────────────────────

    def kinetic_energy(self, state: AircraftState) -> float:
        """
        Compute kinetic energy: KE = 0.5 × m × v².

        Used in unit tests to verify energy conservation (no forces → KE const).

        Parameters
        ----------
        state : AircraftState

        Returns
        -------
        float
            Kinetic energy in Joules.
        """
        return 0.5 * self.cfg.mass * state.speed ** 2

    def potential_energy(self, state: AircraftState) -> float:
        """
        Compute gravitational potential energy: PE = m × g × y.

        Parameters
        ----------
        state : AircraftState

        Returns
        -------
        float
            Potential energy in Joules.
        """
        return self.cfg.mass * self.cfg.gravity * state.y

    def mechanical_energy(self, state: AircraftState) -> float:
        """
        Total mechanical energy: E = KE + PE.

        In the absence of thrust and drag, mechanical energy is
        approximately conserved (RK4 numerical dissipation is very small).

        Parameters
        ----------
        state : AircraftState

        Returns
        -------
        float
            Total mechanical energy in Joules.
        """
        return self.kinetic_energy(state) + self.potential_energy(state)
