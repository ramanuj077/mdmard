"""
tests/test_observation.py
=========================
Unit tests for World and ObservationBuilder -- MDMARD Phase 1

Test Strategy
-------------
1. WorldConfig       : Defaults, YAML override, derived properties
2. Waypoint          : Distance, capture, position property
3. World.reset()     : Spawn positions in range, waypoint min-distance
4. World boundaries  : OOB detection for all edges
5. World utilities   : Bearing, altitude margin, obstacle check
6. ObservationConfig : Defaults, YAML override
7. ObservationBuilder: Shape, dtype, all-values-in-bounds
8. Observation components: each of the 10 components is correct
9. Breakdown dict    : Labels and values match build()
10. Gymnasium space  : Shape and dtype match builder output

Run with:
    pytest tests/test_observation.py -v
"""

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.physics import AircraftState
from envs.world import (
    World, WorldConfig, SpawnConfig, WaypointSpawnConfig,
    Waypoint, Obstacle,
)
from observation.builder import ObservationBuilder, ObservationConfig


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def world_cfg() -> WorldConfig:
    return WorldConfig()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


@pytest.fixture
def world(world_cfg, rng) -> World:
    """Standard world with no obstacles, seeded RNG."""
    return World(world_cfg, SpawnConfig(), WaypointSpawnConfig(),
                 obstacles=[], rng=rng)


@pytest.fixture
def reset_world(world) -> tuple:
    """World after reset -- returns (spawn, waypoint, world)."""
    spawn, wp = world.reset()
    return spawn, wp, world


@pytest.fixture
def obs_cfg() -> ObservationConfig:
    return ObservationConfig()


@pytest.fixture
def builder(obs_cfg, reset_world) -> ObservationBuilder:
    _, _, w = reset_world
    return ObservationBuilder(obs_cfg, w)


@pytest.fixture
def mid_state(reset_world) -> AircraftState:
    """Aircraft at world center, flying right at moderate speed."""
    spawn, wp, w = reset_world
    return AircraftState(
        x=w.cfg.width / 2,
        y=w.cfg.height / 2,
        vx=20.0, vy=0.0, theta=0.0, omega=0.0,
        fuel=1000.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. WorldConfig Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWorldConfig:

    def test_default_floor_less_than_ceiling(self):
        cfg = WorldConfig()
        assert cfg.floor < cfg.ceiling

    def test_default_floor_above_zero(self):
        """Floor must be above ground (y=0) for meaningful airspace."""
        cfg = WorldConfig()
        assert cfg.floor > 0.0

    def test_max_distance_equals_diagonal(self):
        cfg = WorldConfig(width=1200.0, height=800.0)
        expected = math.sqrt(1200.0**2 + 800.0**2)
        assert cfg.max_distance == pytest.approx(expected)

    def test_from_yaml_overrides(self):
        d = {"width": 800.0, "height": 600.0, "floor": 30.0, "ceiling": 570.0}
        cfg = WorldConfig.from_yaml(d)
        assert cfg.width == 800.0
        assert cfg.height == 600.0
        assert cfg.floor == 30.0
        assert cfg.ceiling == 570.0

    def test_from_yaml_missing_uses_defaults(self):
        cfg = WorldConfig.from_yaml({})
        default = WorldConfig()
        assert cfg.width == default.width
        assert cfg.floor == default.floor


# ─────────────────────────────────────────────────────────────────────────────
# 2. Waypoint Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWaypoint:

    def test_distance_to_self_is_zero(self):
        wp = Waypoint(x=500.0, y=300.0)
        assert wp.distance_to(500.0, 300.0) == pytest.approx(0.0)

    def test_distance_pythagorean(self):
        wp = Waypoint(x=0.0, y=0.0)
        assert wp.distance_to(3.0, 4.0) == pytest.approx(5.0)

    def test_capture_inside_radius(self):
        wp = Waypoint(x=500.0, y=300.0, radius=30.0)
        captured = wp.check_capture(510.0, 305.0)  # well within radius
        assert captured is True
        assert wp.captured is True

    def test_no_capture_outside_radius(self):
        wp = Waypoint(x=500.0, y=300.0, radius=30.0)
        captured = wp.check_capture(600.0, 400.0)  # far away
        assert captured is False
        assert wp.captured is False

    def test_capture_exactly_on_edge(self):
        """Point exactly on radius boundary should be captured."""
        wp = Waypoint(x=0.0, y=0.0, radius=30.0)
        captured = wp.check_capture(30.0, 0.0)
        assert captured is True

    def test_position_property(self):
        wp = Waypoint(x=123.0, y=456.0)
        assert wp.position == (123.0, 456.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. World Reset Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWorldReset:

    def test_reset_returns_tuple_and_waypoint(self, world):
        spawn, wp = world.reset()
        assert isinstance(spawn, tuple)
        assert len(spawn) == 5
        assert isinstance(wp, Waypoint)

    def test_spawn_x_in_range(self, world):
        spawn_cfg = world.spawn
        for _ in range(20):
            (x, *_), _ = world.reset()
            assert spawn_cfg.x_range[0] <= x <= spawn_cfg.x_range[1]

    def test_spawn_y_in_range(self, world):
        spawn_cfg = world.spawn
        for _ in range(20):
            (_, y, *_), _ = world.reset()
            assert spawn_cfg.y_range[0] <= y <= spawn_cfg.y_range[1]

    def test_spawn_vx_positive(self, world):
        """Aircraft spawns moving forward (vx > 0)."""
        for _ in range(20):
            (_, _, vx, *_), _ = world.reset()
            assert vx >= 0.0

    def test_waypoint_x_in_range(self, world):
        wp_cfg = world.wp_spawn
        for _ in range(20):
            _, wp = world.reset()
            assert wp_cfg.x_range[0] <= wp.x <= wp_cfg.x_range[1]

    def test_waypoint_y_in_range(self, world):
        wp_cfg = world.wp_spawn
        for _ in range(20):
            _, wp = world.reset()
            assert wp_cfg.y_range[0] <= wp.y <= wp_cfg.y_range[1]

    def test_waypoint_min_distance_satisfied(self, world):
        """Waypoint must be at least min_distance away from spawn."""
        min_dist = world.wp_spawn.min_distance
        for _ in range(30):
            (x, y, *_), wp = world.reset()
            dist = math.sqrt((wp.x - x)**2 + (wp.y - y)**2)
            assert dist >= min_dist - 1e-6  # small tolerance for float

    def test_reset_creates_fresh_waypoint(self, world):
        """Each reset creates a new waypoint (captured flag cleared)."""
        _, wp1 = world.reset()
        wp1.captured = True
        _, wp2 = world.reset()
        assert wp2.captured is False

    def test_world_waypoint_attribute_set(self, world):
        """After reset, world.waypoint is populated."""
        world.reset()
        assert world.waypoint is not None

    def test_seeded_rng_is_reproducible(self, world_cfg):
        """Same seed produces identical spawn/waypoint positions."""
        def run(seed):
            w = World(world_cfg, SpawnConfig(), WaypointSpawnConfig(),
                      rng=np.random.default_rng(seed))
            return w.reset()

        (s1, wp1) = run(42)
        (s2, wp2) = run(42)
        assert s1 == s2
        assert wp1.x == wp2.x
        assert wp1.y == wp2.y


# ─────────────────────────────────────────────────────────────────────────────
# 4. World Boundary Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWorldBoundaries:

    def test_center_is_not_oob(self, reset_world):
        _, _, w = reset_world
        cx, cy = w.center
        assert w.is_out_of_bounds(cx, cy) is False

    def test_left_wall_is_oob(self, reset_world):
        _, _, w = reset_world
        assert w.is_out_of_bounds(-1.0, w.cfg.height / 2) is True

    def test_right_wall_is_oob(self, reset_world):
        _, _, w = reset_world
        assert w.is_out_of_bounds(w.cfg.width + 1.0, w.cfg.height / 2) is True

    def test_below_floor_is_oob(self, reset_world):
        _, _, w = reset_world
        assert w.is_out_of_bounds(w.cfg.width / 2, w.cfg.floor - 1.0) is True

    def test_above_ceiling_is_oob(self, reset_world):
        _, _, w = reset_world
        assert w.is_out_of_bounds(w.cfg.width / 2, w.cfg.ceiling + 1.0) is True

    def test_exactly_on_boundary_is_not_oob(self, reset_world):
        """Being exactly on the boundary (not past it) is still in bounds."""
        _, _, w = reset_world
        # x=0, y=floor: right on the edge
        assert w.is_out_of_bounds(0.0, w.cfg.floor) is False
        assert w.is_out_of_bounds(w.cfg.width, w.cfg.ceiling) is False


# ─────────────────────────────────────────────────────────────────────────────
# 5. World Utility Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWorldUtilities:

    def test_distance_to_waypoint_decreases_when_approaching(self, reset_world):
        _, wp, w = reset_world
        d1 = w.distance_to_waypoint(wp.x + 100, wp.y)
        d2 = w.distance_to_waypoint(wp.x + 50,  wp.y)
        d3 = w.distance_to_waypoint(wp.x,        wp.y)
        assert d1 > d2 > d3
        assert d3 == pytest.approx(0.0)

    def test_bearing_to_waypoint_right(self, reset_world):
        """Waypoint directly to the right → bearing = 0."""
        _, wp, w = reset_world
        bearing = w.bearing_to_waypoint(wp.x - 100, wp.y)
        assert bearing == pytest.approx(0.0, abs=1e-6)

    def test_bearing_to_waypoint_above(self, reset_world):
        """Waypoint directly above → bearing = π/2."""
        _, wp, w = reset_world
        bearing = w.bearing_to_waypoint(wp.x, wp.y - 100)
        assert bearing == pytest.approx(math.pi / 2, abs=1e-6)

    def test_altitude_margin_at_center_is_one(self, reset_world):
        """Aircraft at vertical center has maximum altitude margin."""
        _, _, w = reset_world
        cy = (w.cfg.floor + w.cfg.ceiling) / 2.0
        margin = w.altitude_margin(cy)
        assert margin == pytest.approx(1.0, abs=1e-6)

    def test_altitude_margin_at_floor_is_zero(self, reset_world):
        """Aircraft at floor has zero altitude margin."""
        _, _, w = reset_world
        margin = w.altitude_margin(w.cfg.floor)
        assert margin == pytest.approx(0.0, abs=1e-6)

    def test_altitude_margin_at_ceiling_is_zero(self, reset_world):
        """Aircraft at ceiling has zero altitude margin."""
        _, _, w = reset_world
        margin = w.altitude_margin(w.cfg.ceiling)
        assert margin == pytest.approx(0.0, abs=1e-6)

    def test_altitude_margin_always_in_0_1(self, reset_world):
        _, _, w = reset_world
        for y in np.linspace(w.cfg.floor, w.cfg.ceiling, 100):
            m = w.altitude_margin(float(y))
            assert 0.0 <= m <= 1.0

    def test_no_obstacles_by_default(self, reset_world):
        """Aircraft is never in an obstacle when obstacle list is empty."""
        _, _, w = reset_world
        assert w.is_in_obstacle(600.0, 400.0) is False

    def test_obstacle_detection(self, reset_world):
        """Aircraft inside obstacle radius is detected."""
        _, _, w = reset_world
        w.set_obstacles([Obstacle(x=500.0, y=300.0, radius=50.0)])
        assert w.is_in_obstacle(500.0, 300.0) is True
        assert w.is_in_obstacle(600.0, 400.0) is False

    def test_bounds_property(self, reset_world):
        _, _, w = reset_world
        x_min, y_min, x_max, y_max = w.bounds
        assert x_min == 0.0
        assert y_min == w.cfg.floor
        assert x_max == w.cfg.width
        assert y_max == w.cfg.ceiling


# ─────────────────────────────────────────────────────────────────────────────
# 6. ObservationConfig Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestObservationConfig:

    def test_obs_dim_is_ten(self):
        assert ObservationConfig.OBS_DIM == 10

    def test_all_divisors_positive(self):
        cfg = ObservationConfig()
        assert cfg.vel_max > 0
        assert cfg.omega_max > 0
        assert cfg.dist_max > 0
        assert cfg.fuel_max > 0
        assert cfg.t_max > 0

    def test_from_yaml_overrides(self):
        d = {
            "vel_max_norm":   60.0,
            "omega_max_norm":  3.0,
            "dist_max_norm": 1000.0,
        }
        cfg = ObservationConfig.from_yaml(d)
        assert cfg.vel_max   == 60.0
        assert cfg.omega_max == 3.0
        assert cfg.dist_max  == 1000.0

    def test_from_yaml_missing_uses_defaults(self):
        cfg = ObservationConfig.from_yaml({})
        default = ObservationConfig()
        assert cfg.vel_max  == default.vel_max
        assert cfg.t_max    == default.t_max


# ─────────────────────────────────────────────────────────────────────────────
# 7. ObservationBuilder — Shape & Bounds Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestObservationBuilderBasic:

    def test_output_shape(self, builder, mid_state):
        obs = builder.build(mid_state, steps_remaining=500)
        assert obs.shape == (10,)

    def test_output_dtype_float32(self, builder, mid_state):
        obs = builder.build(mid_state, steps_remaining=500)
        assert obs.dtype == np.float32

    def test_all_values_in_minus1_to_1(self, builder, mid_state):
        obs = builder.build(mid_state, steps_remaining=500)
        assert np.all(obs >= -1.0), f"Values below -1: {obs[obs < -1.0]}"
        assert np.all(obs <= 1.0),  f"Values above +1: {obs[obs > 1.0]}"

    def test_extreme_state_still_bounded(self, builder, reset_world):
        """
        Even with extreme values (max speed, max altitude, etc.),
        all obs components must remain in [-1, 1].
        """
        _, _, w = reset_world
        extreme = AircraftState(
            x=w.cfg.width,        # far right
            y=w.cfg.ceiling,      # at ceiling
            vx=200.0,             # over max speed (should be clamped)
            vy=-200.0,
            theta=math.pi,        # over pitch max (should be clamped)
            omega=10.0,           # over omega max (should be clamped)
            fuel=0.0,             # empty
        )
        obs = builder.build(extreme, steps_remaining=0)
        assert np.all(obs >= -1.0)
        assert np.all(obs <= 1.0)

    def test_deterministic(self, builder, mid_state):
        """Same inputs always produce exactly the same observation."""
        obs1 = builder.build(mid_state, steps_remaining=500)
        obs2 = builder.build(mid_state, steps_remaining=500)
        np.testing.assert_array_equal(obs1, obs2)

    def test_no_nan_in_output(self, builder, mid_state):
        obs = builder.build(mid_state, steps_remaining=500)
        assert not np.any(np.isnan(obs))

    def test_no_inf_in_output(self, builder, mid_state):
        obs = builder.build(mid_state, steps_remaining=500)
        assert not np.any(np.isinf(obs))


# ─────────────────────────────────────────────────────────────────────────────
# 8. Individual Observation Component Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestObservationComponents:
    """
    Verify each of the 10 observation components individually.
    We construct controlled states and check exact expected values.
    """

    def test_x_pos_normalized(self, builder, reset_world):
        """obs[0] = x / width."""
        _, _, w = reset_world
        state = AircraftState(x=w.cfg.width / 2, y=400)
        obs = builder.build(state, steps_remaining=500)
        assert obs[0] == pytest.approx(0.5, abs=1e-5)

    def test_x_pos_at_zero(self, builder, reset_world):
        _, _, w = reset_world
        state = AircraftState(x=0.0, y=400)
        obs = builder.build(state, steps_remaining=500)
        assert obs[0] == pytest.approx(0.0, abs=1e-5)

    def test_y_pos_normalized(self, builder, reset_world):
        """obs[1] = y / height."""
        _, _, w = reset_world
        state = AircraftState(x=600, y=w.cfg.height / 2)
        obs = builder.build(state, steps_remaining=500)
        assert obs[1] == pytest.approx(0.5, abs=1e-5)

    def test_vx_zero(self, builder, reset_world):
        """obs[2] = 0 when vx = 0."""
        state = AircraftState(x=600, y=400, vx=0.0, vy=0.0)
        obs = builder.build(state, steps_remaining=500)
        assert obs[2] == pytest.approx(0.0, abs=1e-5)

    def test_vx_positive(self, builder, reset_world):
        """obs[2] > 0 when vx > 0."""
        state = AircraftState(x=600, y=400, vx=20.0, vy=0.0)
        obs = builder.build(state, steps_remaining=500)
        assert obs[2] > 0.0

    def test_vy_negative_maps_negative(self, builder, reset_world):
        """obs[3] < 0 when vy < 0 (descending)."""
        state = AircraftState(x=600, y=400, vx=0.0, vy=-10.0)
        obs = builder.build(state, steps_remaining=500)
        assert obs[3] < 0.0

    def test_theta_zero_maps_zero(self, builder, reset_world):
        """obs[4] = 0 when theta = 0 (level flight)."""
        state = AircraftState(x=600, y=400, theta=0.0)
        obs = builder.build(state, steps_remaining=500)
        assert obs[4] == pytest.approx(0.0, abs=1e-5)

    def test_omega_zero_maps_zero(self, builder, reset_world):
        """obs[5] = 0 when omega = 0."""
        state = AircraftState(x=600, y=400, omega=0.0)
        obs = builder.build(state, steps_remaining=500)
        assert obs[5] == pytest.approx(0.0, abs=1e-5)

    def test_dist_to_waypoint_decreases(self, builder, reset_world):
        """obs[6] decreases as aircraft approaches waypoint."""
        _, wp, w = reset_world
        far   = AircraftState(x=wp.x - 300, y=wp.y)
        close = AircraftState(x=wp.x - 50,  y=wp.y)
        obs_far   = builder.build(far,   steps_remaining=500)
        obs_close = builder.build(close, steps_remaining=500)
        assert obs_close[6] < obs_far[6]

    def test_dist_at_waypoint_near_zero(self, builder, reset_world):
        """obs[6] ≈ 0 when aircraft is at waypoint center."""
        _, wp, w = reset_world
        at_wp = AircraftState(x=wp.x, y=wp.y)
        obs = builder.build(at_wp, steps_remaining=500)
        assert obs[6] == pytest.approx(0.0, abs=1e-4)

    def test_fuel_full_maps_to_one(self, builder, reset_world):
        """obs[8] = 1.0 when fuel = fuel_max."""
        state = AircraftState(x=600, y=400, fuel=1000.0)
        obs = builder.build(state, steps_remaining=500)
        assert obs[8] == pytest.approx(1.0, abs=1e-5)

    def test_fuel_empty_maps_to_zero(self, builder, reset_world):
        """obs[8] = 0.0 when fuel = 0."""
        state = AircraftState(x=600, y=400, fuel=0.0)
        obs = builder.build(state, steps_remaining=500)
        assert obs[8] == pytest.approx(0.0, abs=1e-5)

    def test_time_full_maps_to_one(self, builder, reset_world):
        """obs[9] = 1.0 at episode start (steps_remaining = T_max)."""
        state = AircraftState(x=600, y=400)
        obs = builder.build(state, steps_remaining=1000)
        assert obs[9] == pytest.approx(1.0, abs=1e-5)

    def test_time_zero_maps_to_zero(self, builder, reset_world):
        """obs[9] = 0.0 at episode end (steps_remaining = 0)."""
        state = AircraftState(x=600, y=400)
        obs = builder.build(state, steps_remaining=0)
        assert obs[9] == pytest.approx(0.0, abs=1e-5)

    def test_time_half_maps_to_half(self, builder, reset_world):
        """obs[9] = 0.5 at midpoint (steps_remaining = T_max/2)."""
        state = AircraftState(x=600, y=400)
        obs = builder.build(state, steps_remaining=500)
        assert obs[9] == pytest.approx(0.5, abs=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Breakdown Dict Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBreakdownDict:

    def test_breakdown_has_ten_keys(self, builder, mid_state):
        bd = builder.breakdown(mid_state, steps_remaining=500)
        assert len(bd) == 10

    def test_breakdown_keys_are_strings(self, builder, mid_state):
        bd = builder.breakdown(mid_state, steps_remaining=500)
        assert all(isinstance(k, str) for k in bd.keys())

    def test_breakdown_values_match_build(self, builder, mid_state):
        """breakdown() values must exactly match build() array."""
        obs = builder.build(mid_state, steps_remaining=500)
        bd  = builder.breakdown(mid_state, steps_remaining=500)
        for i, val in enumerate(bd.values()):
            assert val == pytest.approx(float(obs[i]), abs=1e-6)

    def test_breakdown_expected_keys_present(self, builder, mid_state):
        bd = builder.breakdown(mid_state, steps_remaining=500)
        expected = {"x_pos", "y_pos", "vel_x", "vel_y",
                    "pitch", "pitch_rate", "dist_to_wp",
                    "bearing_to_wp", "fuel", "time_remaining"}
        assert set(bd.keys()) == expected


# ─────────────────────────────────────────────────────────────────────────────
# 10. Gymnasium Space Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestObservationSpace:

    def test_space_shape(self):
        space = ObservationBuilder.observation_space()
        assert space.shape == (10,)

    def test_space_dtype(self):
        space = ObservationBuilder.observation_space()
        assert space.dtype == np.float32

    def test_space_bounds(self):
        space = ObservationBuilder.observation_space()
        assert np.all(space.low  == -1.0)
        assert np.all(space.high ==  1.0)

    def test_obs_contained_in_space(self, builder, mid_state):
        """Every observation produced by builder.build() is in the Gym space."""
        space = ObservationBuilder.observation_space()
        obs   = builder.build(mid_state, steps_remaining=500)
        assert space.contains(obs), f"Observation not in space: {obs}"

    def test_custom_dim_space(self):
        space = ObservationBuilder.observation_space(obs_dim=15)
        assert space.shape == (15,)
