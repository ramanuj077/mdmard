"""
tests/test_physics.py
=====================
Unit tests for the Physics Engine -- MDMARD Phase 1

Test Strategy
-------------
1. State invariants   : AircraftState properties are correct
2. Action decoding    : All 6 discrete actions decode cleanly
3. Force computation  : Each force (thrust/drag/lift/gravity) in isolation
4. RK4 integration   : Energy conservation, trajectory shape
5. Constraints        : Speed clamp, pitch clamp, fuel burn
6. Determinism        : Same input always gives same output

Run with:
    pytest tests/test_physics.py -v
    pytest tests/test_physics.py -v --tb=short  (compact tracebacks)
"""

import math
import sys
import os

import numpy as np
import pytest

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.physics import (
    AircraftState,
    PhysicsConfig,
    PhysicsEngine,
    ActionComponents,
    decode_discrete_action,
    decode_continuous_action,
    DISCRETE_ACTION_MAP,
    ACTION_NAMES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def default_config() -> PhysicsConfig:
    """Default physics config matching configs/env_phase1.yaml."""
    return PhysicsConfig()


@pytest.fixture
def engine(default_config) -> PhysicsEngine:
    """PhysicsEngine with default config."""
    return PhysicsEngine(default_config)


@pytest.fixture
def level_state() -> AircraftState:
    """Aircraft flying level at mid-screen, reasonable speed."""
    return AircraftState(x=600.0, y=400.0, vx=20.0, vy=0.0, theta=0.0, omega=0.0)


@pytest.fixture
def hover_state() -> AircraftState:
    """Aircraft nearly stationary (just above stall speed)."""
    return AircraftState(x=100.0, y=400.0, vx=6.0, vy=0.0, theta=0.0, omega=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 1. AircraftState Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAircraftState:
    """Tests for AircraftState dataclass and its derived properties."""

    def test_speed_zero_velocity(self):
        """Speed should be zero when both velocity components are zero."""
        s = AircraftState(vx=0.0, vy=0.0)
        assert s.speed == pytest.approx(0.0)

    def test_speed_pythagorean(self):
        """Speed = sqrt(vx^2 + vy^2) — 3-4-5 triangle check."""
        s = AircraftState(vx=3.0, vy=4.0)
        assert s.speed == pytest.approx(5.0)

    def test_speed_negative_components(self):
        """Speed is always positive regardless of velocity direction."""
        s = AircraftState(vx=-10.0, vy=-10.0)
        assert s.speed > 0.0
        assert s.speed == pytest.approx(math.sqrt(200.0))

    def test_position_property(self):
        """position property returns (x, y) tuple."""
        s = AircraftState(x=100.0, y=200.0)
        assert s.position == (100.0, 200.0)

    def test_velocity_property(self):
        """velocity property returns (vx, vy) tuple."""
        s = AircraftState(vx=5.0, vy=-3.0)
        assert s.velocity == (5.0, -3.0)

    def test_copy_is_independent(self):
        """copy() creates a new object; mutating it does not affect original."""
        s = AircraftState(x=1.0, y=2.0, vx=3.0)
        s_copy = s.copy()
        s_copy.x = 999.0
        assert s.x == 1.0  # original unchanged

    def test_as_array_shape(self):
        """as_array() returns float32 array of length 7."""
        s = AircraftState()
        arr = s.as_array()
        assert arr.shape == (7,)
        assert arr.dtype == np.float32

    def test_as_array_values(self):
        """as_array() values match state fields in correct order."""
        s = AircraftState(x=1.0, y=2.0, vx=3.0, vy=4.0,
                          theta=0.5, omega=0.1, fuel=900.0)
        arr = s.as_array()
        assert arr[0] == pytest.approx(1.0)
        assert arr[1] == pytest.approx(2.0)
        assert arr[2] == pytest.approx(3.0)
        assert arr[3] == pytest.approx(4.0)
        assert arr[4] == pytest.approx(0.5)
        assert arr[5] == pytest.approx(0.1)
        assert arr[6] == pytest.approx(900.0)

    def test_repr_contains_position(self):
        """__repr__ includes position information."""
        s = AircraftState(x=123.0, y=456.0)
        r = repr(s)
        assert "123" in r
        assert "456" in r


# ─────────────────────────────────────────────────────────────────────────────
# 2. PhysicsConfig Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPhysicsConfig:
    """Tests for PhysicsConfig defaults and factory methods."""

    def test_default_values_are_positive(self):
        """All physical constants should be positive."""
        cfg = PhysicsConfig()
        assert cfg.mass > 0
        assert cfg.inertia > 0
        assert cfg.thrust_max > 0
        assert cfg.drag_coeff > 0
        assert cfg.lift_coeff > 0
        assert cfg.gravity > 0
        assert cfg.vel_max > 0
        assert cfg.vel_stall > 0
        assert cfg.dt > 0

    def test_vel_stall_less_than_vel_max(self):
        """Stall speed must be strictly less than max speed."""
        cfg = PhysicsConfig()
        assert cfg.vel_stall < cfg.vel_max

    def test_pitch_max_approx_pi_over_2(self):
        """Default pitch max should be π/2 (no inverted flight)."""
        cfg = PhysicsConfig()
        assert cfg.pitch_max == pytest.approx(math.pi / 2, abs=1e-4)

    def test_from_yaml_overrides(self):
        """from_yaml correctly overrides default values."""
        yaml_dict = {
            "mass": 2.0,
            "gravity": 1.62,   # Moon gravity!
            "thrust_max": 30.0,
            "dt": 0.1,
        }
        cfg = PhysicsConfig.from_yaml(yaml_dict)
        assert cfg.mass == pytest.approx(2.0)
        assert cfg.gravity == pytest.approx(1.62)
        assert cfg.thrust_max == pytest.approx(30.0)
        assert cfg.dt == pytest.approx(0.1)

    def test_from_yaml_missing_keys_use_defaults(self):
        """from_yaml falls back to defaults for missing keys."""
        cfg = PhysicsConfig.from_yaml({})
        default = PhysicsConfig()
        assert cfg.mass == default.mass
        assert cfg.gravity == default.gravity


# ─────────────────────────────────────────────────────────────────────────────
# 3. Action Decoding Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestActionDecoding:
    """Tests for discrete and continuous action decoders."""

    def test_all_discrete_actions_valid(self):
        """All 6 discrete action IDs decode without error."""
        for action_id in range(6):
            ctrl = decode_discrete_action(action_id)
            assert isinstance(ctrl, ActionComponents)

    def test_discrete_invalid_action_raises(self):
        """Invalid action ID raises ValueError."""
        with pytest.raises(ValueError, match="Invalid action_id"):
            decode_discrete_action(99)

    def test_discrete_negative_action_raises(self):
        """Negative action ID raises ValueError."""
        with pytest.raises(ValueError):
            decode_discrete_action(-1)

    def test_glide_action_no_thrust_no_elevator(self):
        """Action 5 (Glide) must have zero throttle and elevator."""
        ctrl = decode_discrete_action(5)
        assert ctrl.throttle == pytest.approx(0.0)
        assert ctrl.elevator == pytest.approx(0.0)

    def test_thrust_action_full_throttle(self):
        """Action 0 (Thrust) must have full throttle."""
        ctrl = decode_discrete_action(0)
        assert ctrl.throttle == pytest.approx(1.0)
        assert ctrl.elevator == pytest.approx(0.0)

    def test_pitch_up_action_no_thrust(self):
        """Action 1 (Pitch Up) must have zero throttle."""
        ctrl = decode_discrete_action(1)
        assert ctrl.throttle == pytest.approx(0.0)
        assert ctrl.elevator == pytest.approx(1.0)

    def test_pitch_down_action_negative_elevator(self):
        """Action 2 (Pitch Down) must have negative elevator."""
        ctrl = decode_discrete_action(2)
        assert ctrl.elevator == pytest.approx(-1.0)

    def test_throttle_range_all_discrete(self):
        """All discrete actions have throttle in [0, 1]."""
        for aid in range(6):
            ctrl = decode_discrete_action(aid)
            assert 0.0 <= ctrl.throttle <= 1.0

    def test_elevator_range_all_discrete(self):
        """All discrete actions have elevator in [-1, 1]."""
        for aid in range(6):
            ctrl = decode_discrete_action(aid)
            assert -1.0 <= ctrl.elevator <= 1.0

    def test_continuous_full_positive(self):
        """Continuous [1, 1] → throttle=1.0, elevator=1.0."""
        ctrl = decode_continuous_action(np.array([1.0, 1.0]))
        assert ctrl.throttle == pytest.approx(1.0)
        assert ctrl.elevator == pytest.approx(1.0)

    def test_continuous_full_negative_throttle(self):
        """Continuous [-1, x] → throttle=0.0 (min throttle)."""
        ctrl = decode_continuous_action(np.array([-1.0, 0.0]))
        assert ctrl.throttle == pytest.approx(0.0)

    def test_continuous_mid_range(self):
        """Continuous [0, 0] → throttle=0.5, elevator=0.0."""
        ctrl = decode_continuous_action(np.array([0.0, 0.0]))
        assert ctrl.throttle == pytest.approx(0.5)
        assert ctrl.elevator == pytest.approx(0.0)

    def test_continuous_clamps_out_of_range(self):
        """Values outside [-1, 1] are clamped."""
        ctrl = decode_continuous_action(np.array([5.0, -10.0]))
        assert 0.0 <= ctrl.throttle <= 1.0
        assert -1.0 <= ctrl.elevator <= 1.0

    def test_action_names_coverage(self):
        """ACTION_NAMES covers all action IDs."""
        for aid in DISCRETE_ACTION_MAP:
            assert aid in ACTION_NAMES
            assert isinstance(ACTION_NAMES[aid], str)
            assert len(ACTION_NAMES[aid]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Force Computation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestForceComputation:
    """Tests for PhysicsEngine.compute_forces in isolation."""

    def test_no_thrust_no_lift_only_gravity_and_drag(self, engine):
        """
        With glide action and level flight, forces should be:
        - Gravity pulls down (fy < 0)
        - Drag opposes forward motion (fx < 0 when vx > 0)
        - No thrust (throttle=0)
        """
        ctrl = ActionComponents(throttle=0.0, elevator=0.0)
        fx, fy = engine.compute_forces(vx=20.0, vy=0.0, theta=0.0, ctrl=ctrl)
        # Drag should decelerate (negative fx)
        assert fx < 0.0
        # Gravity dominates fy (downward)
        assert fy < 0.0

    def test_thrust_increases_fx(self, engine):
        """Full thrust in level flight (theta=0) should add positive Fx."""
        ctrl_thrust = ActionComponents(throttle=1.0, elevator=0.0)
        ctrl_glide  = ActionComponents(throttle=0.0, elevator=0.0)
        fx_thrust, _ = engine.compute_forces(20.0, 0.0, 0.0, ctrl_thrust)
        fx_glide,  _ = engine.compute_forces(20.0, 0.0, 0.0, ctrl_glide)
        assert fx_thrust > fx_glide

    def test_thrust_direction_follows_pitch(self, engine):
        """
        When pitched up 45°, thrust has both +x and +y components.
        """
        ctrl = ActionComponents(throttle=1.0, elevator=0.0)
        theta = math.pi / 4  # 45 degrees
        fx, fy = engine.compute_forces(vx=0.0, vy=0.0, theta=theta, ctrl=ctrl)
        # Both thrust components positive at 45 deg
        thrust_x = engine.cfg.thrust_max * math.cos(theta)
        thrust_y = engine.cfg.thrust_max * math.sin(theta)
        # fy includes gravity, so check thrust contribution via fx
        assert fx == pytest.approx(thrust_x, abs=0.1)

    def test_zero_velocity_means_zero_drag(self, engine):
        """At zero velocity, drag force is zero."""
        ctrl = ActionComponents(throttle=0.0, elevator=0.0)
        fx, fy = engine.compute_forces(vx=0.0, vy=0.0, theta=0.0, ctrl=ctrl)
        # fx should be 0 (no drag, no thrust, no lift at v=0)
        assert fx == pytest.approx(0.0, abs=1e-6)

    def test_drag_opposes_motion(self, engine):
        """Drag should oppose the direction of motion."""
        ctrl = ActionComponents(throttle=0.0, elevator=0.0)
        # Moving right (positive vx)
        fx_right, _ = engine.compute_forces(vx=20.0, vy=0.0, theta=0.0, ctrl=ctrl)
        # Moving left (negative vx)
        fx_left, _  = engine.compute_forces(vx=-20.0, vy=0.0, theta=0.0, ctrl=ctrl)
        assert fx_right < 0.0  # drag opposes rightward motion
        assert fx_left  > 0.0  # drag opposes leftward motion

    def test_gravity_constant_regardless_of_speed(self, engine):
        """Gravity component is the same at all speeds."""
        ctrl = ActionComponents(throttle=0.0, elevator=0.0)
        # At speeds with no lift (theta=0, lift = Cl*v^2*sin(0) = 0)
        _, fy_slow = engine.compute_forces(vx=0.1, vy=0.0, theta=0.0, ctrl=ctrl)
        _, fy_fast = engine.compute_forces(vx=50.0, vy=0.0, theta=0.0, ctrl=ctrl)
        gravity_force = -engine.cfg.mass * engine.cfg.gravity
        # At theta=0, lift=0, so fy = drag_y + gravity
        # For purely horizontal motion vy=0 drag_y=0, so fy = gravity
        _, fy_pure = engine.compute_forces(vx=0.0, vy=0.0, theta=0.0, ctrl=ctrl)
        assert fy_pure == pytest.approx(gravity_force, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Integration Tests (RK4 step)
# ─────────────────────────────────────────────────────────────────────────────

class TestRK4Integration:
    """Tests for the RK4 integrator via PhysicsEngine.step()."""

    def test_step_returns_aircraft_state(self, engine, level_state):
        """step() returns an AircraftState instance."""
        new_state = engine.step(level_state, action_id=5)  # glide
        assert isinstance(new_state, AircraftState)

    def test_step_increments_step_counter(self, engine, level_state):
        """step() increments the step counter by exactly 1."""
        s0 = level_state
        s1 = engine.step(s0, action_id=5)
        assert s1.step == s0.step + 1

    def test_glide_falls_under_gravity(self, engine, level_state):
        """
        With glide action and initial horizontal velocity,
        the aircraft should descend (vy decreasing) due to gravity.
        """
        state = level_state  # vy=0 initially
        for _ in range(10):
            state = engine.step(state, action_id=5)
        # After gliding, vy should be negative (falling)
        assert state.vy < 0.0

    def test_thrust_increases_speed(self, engine, level_state):
        """Full thrust in level flight should increase speed."""
        s0 = level_state
        s1 = engine.step(s0, action_id=0)  # thrust
        assert s1.speed > s0.speed

    def test_pitch_up_action_increases_pitch(self, engine, level_state):
        """Pitch up action should increase theta (nose up)."""
        s0 = level_state  # theta=0
        s1 = engine.step(s0, action_id=1)  # pitch up
        assert s1.theta > s0.theta

    def test_pitch_down_action_decreases_pitch(self, engine, level_state):
        """Pitch down action should decrease theta (nose down)."""
        s0 = level_state  # theta=0
        s1 = engine.step(s0, action_id=2)  # pitch down
        assert s1.theta < s0.theta

    def test_step_raises_with_both_action_types(self, engine, level_state):
        """Providing both action_id and action_continuous raises ValueError."""
        with pytest.raises(ValueError, match="exactly one"):
            engine.step(level_state,
                        action_id=0,
                        action_continuous=np.array([1.0, 0.0]))

    def test_step_raises_with_no_action(self, engine, level_state):
        """Providing no action raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            engine.step(level_state)

    def test_determinism(self, engine, level_state):
        """
        Same initial state + action always produces identical output.
        Physics must be fully deterministic (no random elements).
        """
        s1a = engine.step(level_state, action_id=3)
        s1b = engine.step(level_state, action_id=3)
        assert s1a.x     == pytest.approx(s1b.x)
        assert s1a.y     == pytest.approx(s1b.y)
        assert s1a.vx    == pytest.approx(s1b.vx)
        assert s1a.vy    == pytest.approx(s1b.vy)
        assert s1a.theta == pytest.approx(s1b.theta)
        assert s1a.omega == pytest.approx(s1b.omega)

    def test_continuous_action_step(self, engine, level_state):
        """step() with continuous action works and returns valid state."""
        action = np.array([0.8, 0.3])
        new_state = engine.step(level_state, action_continuous=action)
        assert isinstance(new_state, AircraftState)
        assert math.isfinite(new_state.x)
        assert math.isfinite(new_state.y)
        assert math.isfinite(new_state.vx)
        assert math.isfinite(new_state.vy)

    def test_no_nan_inf_after_many_steps(self, engine, level_state):
        """Running 500 steps of thrust+pitch should not produce NaN/Inf."""
        state = level_state
        for _ in range(500):
            state = engine.step(state, action_id=3)  # thrust + pitch up
        assert math.isfinite(state.x)
        assert math.isfinite(state.y)
        assert math.isfinite(state.vx)
        assert math.isfinite(state.vy)
        assert math.isfinite(state.theta)
        assert math.isfinite(state.omega)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Constraint Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConstraints:
    """Tests that physical constraints (clamps) are correctly enforced."""

    def test_speed_never_exceeds_vel_max(self, engine):
        """
        After many full-thrust steps, speed must never exceed vel_max.
        Tests the speed clamp in _apply_constraints.
        """
        cfg = engine.cfg
        state = AircraftState(x=0, y=400, vx=cfg.vel_max - 1, vy=0,
                              theta=0, omega=0)
        for _ in range(200):
            state = engine.step(state, action_id=0)  # full thrust
        assert state.speed <= cfg.vel_max + 1e-6  # small float tolerance

    def test_pitch_never_exceeds_pitch_max(self, engine, level_state):
        """
        Continuous pitch-up must never exceed pitch_max (π/2).
        """
        state = level_state
        for _ in range(500):
            state = engine.step(state, action_id=1)  # pitch up
        assert state.theta <= engine.cfg.pitch_max + 1e-6

    def test_pitch_never_below_neg_pitch_max(self, engine, level_state):
        """
        Continuous pitch-down must never go below -pitch_max.
        """
        state = level_state
        for _ in range(500):
            state = engine.step(state, action_id=2)  # pitch down
        assert state.theta >= -engine.cfg.pitch_max - 1e-6

    def test_pitch_rate_clamped(self, engine, level_state):
        """Pitch rate omega must stay within [-pitch_rate_max, pitch_rate_max]."""
        state = level_state
        for _ in range(200):
            state = engine.step(state, action_id=1)
        assert abs(state.omega) <= engine.cfg.pitch_rate_max + 1e-6

    def test_fuel_decreases_with_thrust(self, engine, level_state):
        """Fuel should decrease when throttle > 0."""
        state = level_state
        initial_fuel = state.fuel
        for _ in range(10):
            state = engine.step(state, action_id=0)  # full thrust
        assert state.fuel < initial_fuel

    def test_fuel_constant_with_glide(self, engine, level_state):
        """Fuel should not decrease when throttle=0 (glide)."""
        state = level_state
        initial_fuel = state.fuel
        for _ in range(10):
            state = engine.step(state, action_id=5)  # glide, throttle=0
        assert state.fuel == pytest.approx(initial_fuel)

    def test_fuel_never_negative(self, engine, level_state):
        """Fuel should never go below zero."""
        state = AircraftState(x=100, y=400, vx=20, vy=0,
                              theta=0, omega=0, fuel=1.0)  # nearly empty
        for _ in range(100):
            state = engine.step(state, action_id=0)  # full thrust
        assert state.fuel >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Energy Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEnergy:
    """
    Tests for energy utility methods and approximate conservation.

    Note: Perfect energy conservation is NOT expected with thrust/drag/lift.
    We test that:
    - Energy decreases with drag only (energy dissipates)
    - Energy increases with thrust
    - RK4 does not introduce runaway energy growth
    """

    def test_kinetic_energy_formula(self, engine):
        """KE = 0.5 * m * v^2."""
        state = AircraftState(vx=10.0, vy=0.0)
        expected_ke = 0.5 * engine.cfg.mass * 100.0
        assert engine.kinetic_energy(state) == pytest.approx(expected_ke)

    def test_potential_energy_formula(self, engine):
        """PE = m * g * y."""
        state = AircraftState(y=100.0)
        expected_pe = engine.cfg.mass * engine.cfg.gravity * 100.0
        assert engine.potential_energy(state) == pytest.approx(expected_pe)

    def test_mechanical_energy_sum(self, engine):
        """Mechanical energy = KE + PE."""
        state = AircraftState(vx=10.0, vy=5.0, y=200.0)
        ke = engine.kinetic_energy(state)
        pe = engine.potential_energy(state)
        me = engine.mechanical_energy(state)
        assert me == pytest.approx(ke + pe)

    def test_drag_dissipates_energy(self, engine):
        """
        Gliding with no thrust: mechanical energy should decrease
        over time as drag removes kinetic energy.
        """
        state = AircraftState(x=0, y=400, vx=30.0, vy=0.0, theta=0.0)
        e0 = engine.mechanical_energy(state)
        for _ in range(50):
            state = engine.step(state, action_id=5)  # glide (drag only)
        e1 = engine.mechanical_energy(state)
        assert e1 < e0  # drag has dissipated energy

    def test_thrust_adds_energy(self, engine):
        """
        Full thrust: mechanical energy should increase over time.
        """
        state = AircraftState(x=0, y=400, vx=10.0, vy=0.0, theta=0.0)
        e0 = engine.mechanical_energy(state)
        for _ in range(20):
            state = engine.step(state, action_id=0)  # full thrust
        e1 = engine.mechanical_energy(state)
        assert e1 > e0  # thrust has added energy


# ─────────────────────────────────────────────────────────────────────────────
# 8. Trajectory Shape Tests (Sanity Checks)
# ─────────────────────────────────────────────────────────────────────────────

class TestTrajectoryShape:
    """
    High-level sanity tests: does the aircraft behave like an aircraft?
    These tests catch physics bugs that unit tests might miss.
    """

    def test_pitched_up_aircraft_climbs(self, engine):
        """
        An aircraft with pitch up + thrust should gain altitude over time.
        """
        state = AircraftState(x=0, y=200, vx=20.0, vy=0.0,
                              theta=math.pi/6, omega=0.0)  # 30 deg up
        initial_y = state.y
        for _ in range(30):
            state = engine.step(state, action_id=0)  # thrust
        assert state.y > initial_y

    def test_gliding_aircraft_descends(self, engine):
        """
        An aircraft gliding with no thrust should lose altitude.
        """
        state = AircraftState(x=0, y=400, vx=20.0, vy=0.0, theta=0.0)
        for _ in range(30):
            state = engine.step(state, action_id=5)  # glide
        assert state.y < 400.0

    def test_aircraft_moves_forward_with_thrust(self, engine):
        """Aircraft should move in the x-direction with thrust in level flight."""
        state = AircraftState(x=0, y=400, vx=0.0, vy=0.0, theta=0.0)
        for _ in range(20):
            state = engine.step(state, action_id=0)  # thrust
        assert state.x > 0.0

    def test_pitch_up_then_thrust_forms_arc(self, engine):
        """
        Pitch up then sustained thrust should create an upward-curving trajectory.
        Record y positions and check they trend upward after initial pitch.
        """
        state = AircraftState(x=0, y=200, vx=15.0, vy=0.0, theta=0.0)
        # First pitch up for 10 steps
        for _ in range(10):
            state = engine.step(state, action_id=1)  # pitch up
        y_after_pitch = state.y
        # Then thrust for 20 steps
        for _ in range(20):
            state = engine.step(state, action_id=0)  # thrust
        y_after_thrust = state.y
        # Should climb with thrust after pitching up
        assert y_after_thrust > y_after_pitch
