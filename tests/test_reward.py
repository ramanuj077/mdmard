"""
tests/test_reward.py
====================
Unit tests for Reward and Termination modules -- MDMARD Phase 1

Test Strategy
-------------
1. RewardContext      : Builds correctly from state + world
2. RewardOutput       : Addition, breakdown keys
3. BaseReward subclass: Interface contract
4. ApproachReward     : Positive when closing, negative when receding
5. WaypointReward     : Fires only on goal, correct magnitude
6. CrashPenalty       : Fires only on failure reasons, not goal/timeout
7. AliveReward        : +bonus every non-terminal step
8. SmoothReward       : 0 at omega=0, scales with |omega|
9. AltitudeReward     : 0 at center, -scale at boundaries
10. FuelReward        : Scales with fuel consumed per step
11. CompositeReward   : Weighted sum correct, from_config factory
12. TerminationResult : Flags and properties
13. TerminationChecker: All conditions trigger, mutex, stall counter

Run with:
    pytest tests/test_reward.py -v
"""

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.physics import AircraftState, PhysicsConfig
from envs.world import World, WorldConfig, SpawnConfig, WaypointSpawnConfig, Waypoint
from envs.termination import TerminationChecker, TerminationResult
from reward.base import BaseReward, RewardContext, RewardOutput
from reward.components import (
    ApproachReward, WaypointReward, CrashPenalty,
    AliveReward, SmoothReward, AltitudeReward, FuelReward,
)
from reward.composite import CompositeReward


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def world() -> World:
    rng = np.random.default_rng(seed=0)
    w = World(WorldConfig(), SpawnConfig(), WaypointSpawnConfig(), rng=rng)
    w.reset()
    return w


@pytest.fixture
def mid_state() -> AircraftState:
    return AircraftState(x=600.0, y=400.0, vx=20.0, vy=0.0,
                         theta=0.0, omega=0.0, fuel=1000.0)


@pytest.fixture
def far_ctx(world, mid_state) -> RewardContext:
    """Aircraft far from waypoint, not terminated."""
    return RewardContext(
        state=mid_state,
        prev_state=mid_state.copy(),
        world=world,
        action=0,
        prev_dist=500.0,
        curr_dist=490.0,    # moved 10 units closer
        terminated=False,
        term_reason="",
        step=1,
    )


@pytest.fixture
def goal_ctx(world, mid_state) -> RewardContext:
    """Aircraft reached goal."""
    return RewardContext(
        state=mid_state,
        prev_state=mid_state.copy(),
        world=world,
        action=0,
        prev_dist=5.0,
        curr_dist=0.0,
        terminated=True,
        term_reason="goal",
        step=100,
    )


@pytest.fixture
def crash_ctx(world, mid_state) -> RewardContext:
    """Aircraft crashed (OOB)."""
    return RewardContext(
        state=mid_state,
        prev_state=mid_state.copy(),
        world=world,
        action=0,
        prev_dist=300.0,
        curr_dist=310.0,
        terminated=True,
        term_reason="oob",
        step=50,
    )


@pytest.fixture
def phys_cfg() -> PhysicsConfig:
    return PhysicsConfig()


# ─────────────────────────────────────────────────────────────────────────────
# 1. RewardOutput Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRewardOutput:

    def test_addition_sums_totals(self):
        a = RewardOutput(total=5.0, breakdown={"a": 5.0})
        b = RewardOutput(total=3.0, breakdown={"b": 3.0})
        c = a + b
        assert c.total == pytest.approx(8.0)

    def test_addition_merges_breakdowns(self):
        a = RewardOutput(total=1.0, breakdown={"a": 1.0})
        b = RewardOutput(total=2.0, breakdown={"b": 2.0})
        c = a + b
        assert "a" in c.breakdown
        assert "b" in c.breakdown

    def test_default_breakdown_is_empty(self):
        r = RewardOutput(total=0.0)
        assert r.breakdown == {}


# ─────────────────────────────────────────────────────────────────────────────
# 2. BaseReward ABC
# ─────────────────────────────────────────────────────────────────────────────

class TestBaseReward:

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseReward()

    def test_concrete_subclass_works(self, far_ctx):
        class MyReward(BaseReward):
            def compute(self, ctx):
                return RewardOutput(total=42.0, breakdown={"test": 42.0})

        r = MyReward()
        out = r.compute(far_ctx)
        assert out.total == pytest.approx(42.0)

    def test_reset_default_does_nothing(self, far_ctx):
        class MyReward(BaseReward):
            def compute(self, ctx):
                return RewardOutput(total=0.0)
        r = MyReward()
        r.reset()   # should not raise


# ─────────────────────────────────────────────────────────────────────────────
# 3. ApproachReward
# ─────────────────────────────────────────────────────────────────────────────

class TestApproachReward:

    def test_positive_when_closing(self, far_ctx):
        """prev_dist=500, curr_dist=490 → approach = +10."""
        out = ApproachReward().compute(far_ctx)
        assert out.total == pytest.approx(10.0)

    def test_negative_when_receding(self, world, mid_state):
        ctx = RewardContext(
            state=mid_state, prev_state=mid_state.copy(), world=world,
            action=0, prev_dist=490.0, curr_dist=500.0,   # moved away
            terminated=False, term_reason="", step=1,
        )
        out = ApproachReward().compute(ctx)
        assert out.total == pytest.approx(-10.0)

    def test_zero_when_stationary(self, world, mid_state):
        ctx = RewardContext(
            state=mid_state, prev_state=mid_state.copy(), world=world,
            action=0, prev_dist=300.0, curr_dist=300.0,
            terminated=False, term_reason="", step=1,
        )
        out = ApproachReward().compute(ctx)
        assert out.total == pytest.approx(0.0)

    def test_breakdown_key_present(self, far_ctx):
        out = ApproachReward().compute(far_ctx)
        assert "approach" in out.breakdown

    def test_breakdown_matches_total(self, far_ctx):
        out = ApproachReward().compute(far_ctx)
        assert out.breakdown["approach"] == pytest.approx(out.total, abs=0.001)


# ─────────────────────────────────────────────────────────────────────────────
# 4. WaypointReward
# ─────────────────────────────────────────────────────────────────────────────

class TestWaypointReward:

    def test_fires_on_goal(self, goal_ctx):
        out = WaypointReward({"bonus": 200.0}).compute(goal_ctx)
        assert out.total == pytest.approx(200.0)

    def test_zero_on_normal_step(self, far_ctx):
        out = WaypointReward({"bonus": 200.0}).compute(far_ctx)
        assert out.total == pytest.approx(0.0)

    def test_zero_on_crash(self, crash_ctx):
        out = WaypointReward({"bonus": 200.0}).compute(crash_ctx)
        assert out.total == pytest.approx(0.0)

    def test_custom_bonus(self, goal_ctx):
        out = WaypointReward({"bonus": 500.0}).compute(goal_ctx)
        assert out.total == pytest.approx(500.0)

    def test_default_bonus_200(self, goal_ctx):
        out = WaypointReward().compute(goal_ctx)
        assert out.total == pytest.approx(200.0)

    def test_breakdown_key(self, goal_ctx):
        out = WaypointReward().compute(goal_ctx)
        assert "waypoint" in out.breakdown


# ─────────────────────────────────────────────────────────────────────────────
# 5. CrashPenalty
# ─────────────────────────────────────────────────────────────────────────────

class TestCrashPenalty:

    def test_fires_on_oob(self, crash_ctx):
        out = CrashPenalty({"penalty": 100.0}).compute(crash_ctx)
        assert out.total == pytest.approx(-100.0)

    @pytest.mark.parametrize("reason", ["oob", "stall", "obstacle", "fuel"])
    def test_fires_on_all_failure_reasons(self, world, mid_state, reason):
        ctx = RewardContext(
            state=mid_state, prev_state=mid_state.copy(), world=world,
            action=0, prev_dist=100.0, curr_dist=100.0,
            terminated=True, term_reason=reason, step=10,
        )
        out = CrashPenalty({"penalty": 100.0}).compute(ctx)
        assert out.total == pytest.approx(-100.0)

    def test_zero_on_goal(self, goal_ctx):
        """Goal is not a crash -- penalty must be 0."""
        out = CrashPenalty({"penalty": 100.0}).compute(goal_ctx)
        assert out.total == pytest.approx(0.0)

    def test_zero_on_normal_step(self, far_ctx):
        out = CrashPenalty({"penalty": 100.0}).compute(far_ctx)
        assert out.total == pytest.approx(0.0)

    def test_penalty_is_negative(self, crash_ctx):
        out = CrashPenalty({"penalty": 100.0}).compute(crash_ctx)
        assert out.total < 0.0

    def test_breakdown_key(self, crash_ctx):
        out = CrashPenalty().compute(crash_ctx)
        assert "crash" in out.breakdown


# ─────────────────────────────────────────────────────────────────────────────
# 6. AliveReward
# ─────────────────────────────────────────────────────────────────────────────

class TestAliveReward:

    def test_positive_on_normal_step(self, far_ctx):
        out = AliveReward({"bonus": 0.1}).compute(far_ctx)
        assert out.total == pytest.approx(0.1)

    def test_zero_on_terminal_step(self, goal_ctx):
        """No alive bonus on the terminal step."""
        out = AliveReward({"bonus": 0.1}).compute(goal_ctx)
        assert out.total == pytest.approx(0.0)

    def test_zero_on_crash(self, crash_ctx):
        out = AliveReward({"bonus": 0.1}).compute(crash_ctx)
        assert out.total == pytest.approx(0.0)

    def test_default_bonus(self, far_ctx):
        out = AliveReward().compute(far_ctx)
        assert out.total == pytest.approx(0.1)

    def test_breakdown_key(self, far_ctx):
        out = AliveReward().compute(far_ctx)
        assert "alive" in out.breakdown


# ─────────────────────────────────────────────────────────────────────────────
# 7. SmoothReward
# ─────────────────────────────────────────────────────────────────────────────

class TestSmoothReward:

    def test_zero_at_zero_omega(self, world, far_ctx):
        state = AircraftState(x=600, y=400, omega=0.0)
        ctx = RewardContext(
            state=state, prev_state=state.copy(), world=world,
            action=0, prev_dist=300.0, curr_dist=290.0,
            terminated=False, term_reason="", step=1,
        )
        out = SmoothReward({"scale": 0.05}).compute(ctx)
        assert out.total == pytest.approx(0.0)

    def test_negative_at_nonzero_omega(self, world):
        state = AircraftState(x=600, y=400, omega=1.0)
        ctx = RewardContext(
            state=state, prev_state=state.copy(), world=world,
            action=1, prev_dist=300.0, curr_dist=300.0,
            terminated=False, term_reason="", step=1,
        )
        out = SmoothReward({"scale": 0.05}).compute(ctx)
        assert out.total < 0.0

    def test_scales_with_omega(self, world):
        state_slow = AircraftState(omega=0.5)
        state_fast = AircraftState(omega=2.0)
        def make_ctx(s):
            return RewardContext(
                state=s, prev_state=s.copy(), world=world,
                action=0, prev_dist=100.0, curr_dist=100.0,
                terminated=False, term_reason="", step=1,
            )
        out_slow = SmoothReward({"scale": 0.05}).compute(make_ctx(state_slow))
        out_fast = SmoothReward({"scale": 0.05}).compute(make_ctx(state_fast))
        assert out_slow.total > out_fast.total   # less penalty for slower pitch

    def test_symmetric_positive_negative_omega(self, world):
        """Penalty same magnitude for +omega and -omega."""
        s_pos = AircraftState(omega=1.0)
        s_neg = AircraftState(omega=-1.0)
        def make_ctx(s):
            return RewardContext(
                state=s, prev_state=s.copy(), world=world,
                action=0, prev_dist=100.0, curr_dist=100.0,
                terminated=False, term_reason="", step=1,
            )
        out_pos = SmoothReward({"scale": 0.05}).compute(make_ctx(s_pos))
        out_neg = SmoothReward({"scale": 0.05}).compute(make_ctx(s_neg))
        assert abs(out_pos.total) == pytest.approx(abs(out_neg.total))

    def test_breakdown_key(self, far_ctx):
        out = SmoothReward().compute(far_ctx)
        assert "smooth" in out.breakdown


# ─────────────────────────────────────────────────────────────────────────────
# 8. AltitudeReward
# ─────────────────────────────────────────────────────────────────────────────

class TestAltitudeReward:

    def _make_ctx(self, world, y):
        s = AircraftState(x=600, y=y)
        return RewardContext(
            state=s, prev_state=s.copy(), world=world,
            action=0, prev_dist=300.0, curr_dist=300.0,
            terminated=False, term_reason="", step=1,
        )

    def test_zero_penalty_at_center(self, world):
        cy = (world.cfg.floor + world.cfg.ceiling) / 2.0
        ctx = self._make_ctx(world, cy)
        out = AltitudeReward({"scale": 0.5}).compute(ctx)
        assert out.total == pytest.approx(0.0, abs=1e-5)

    def test_max_penalty_at_floor(self, world):
        ctx = self._make_ctx(world, world.cfg.floor)
        out = AltitudeReward({"scale": 0.5}).compute(ctx)
        assert out.total == pytest.approx(-0.5, abs=1e-5)

    def test_max_penalty_at_ceiling(self, world):
        ctx = self._make_ctx(world, world.cfg.ceiling)
        out = AltitudeReward({"scale": 0.5}).compute(ctx)
        assert out.total == pytest.approx(-0.5, abs=1e-5)

    def test_always_nonpositive(self, world):
        for y in np.linspace(world.cfg.floor, world.cfg.ceiling, 50):
            ctx = self._make_ctx(world, float(y))
            out = AltitudeReward({"scale": 0.5}).compute(ctx)
            assert out.total <= 0.0 + 1e-9

    def test_breakdown_key(self, far_ctx):
        out = AltitudeReward().compute(far_ctx)
        assert "altitude" in out.breakdown


# ─────────────────────────────────────────────────────────────────────────────
# 9. FuelReward
# ─────────────────────────────────────────────────────────────────────────────

class TestFuelReward:

    def test_zero_when_no_fuel_burned(self, world):
        state = AircraftState(fuel=1000.0)
        prev  = AircraftState(fuel=1000.0)
        ctx = RewardContext(
            state=state, prev_state=prev, world=world,
            action=5, prev_dist=300.0, curr_dist=300.0,
            terminated=False, term_reason="", step=1,
        )
        out = FuelReward({"scale": 0.001}).compute(ctx)
        assert out.total == pytest.approx(0.0)

    def test_negative_when_fuel_burned(self, world):
        state = AircraftState(fuel=995.0)   # burned 5 units
        prev  = AircraftState(fuel=1000.0)
        ctx = RewardContext(
            state=state, prev_state=prev, world=world,
            action=0, prev_dist=300.0, curr_dist=290.0,
            terminated=False, term_reason="", step=1,
        )
        out = FuelReward({"scale": 0.001}).compute(ctx)
        assert out.total < 0.0

    def test_scales_with_consumption(self, world):
        def make_ctx(burned):
            s = AircraftState(fuel=1000.0 - burned)
            p = AircraftState(fuel=1000.0)
            return RewardContext(
                state=s, prev_state=p, world=world,
                action=0, prev_dist=100.0, curr_dist=95.0,
                terminated=False, term_reason="", step=1,
            )
        out_small = FuelReward({"scale": 0.001}).compute(make_ctx(1.0))
        out_large = FuelReward({"scale": 0.001}).compute(make_ctx(10.0))
        assert out_large.total < out_small.total


# ─────────────────────────────────────────────────────────────────────────────
# 10. CompositeReward
# ─────────────────────────────────────────────────────────────────────────────

class TestCompositeReward:

    def test_default_creates_without_error(self):
        r = CompositeReward.default()
        assert r is not None

    def test_from_config_creates_without_error(self):
        cfg = {"w_approach": 1.0, "w_waypoint": 200.0, "w_crash": 100.0}
        r = CompositeReward.from_config(cfg)
        assert r is not None

    def test_compute_returns_reward_output(self, far_ctx):
        r = CompositeReward.default()
        out = r.compute(far_ctx)
        assert isinstance(out, RewardOutput)
        assert isinstance(out.total, float)
        assert isinstance(out.breakdown, dict)

    def test_breakdown_has_all_component_keys(self, far_ctx):
        r = CompositeReward.default()
        out = r.compute(far_ctx)
        expected = {"approach", "waypoint", "crash", "alive", "smooth", "altitude", "fuel"}
        assert set(out.breakdown.keys()) == expected

    def test_goal_gives_large_positive_reward(self, goal_ctx):
        r = CompositeReward.default()
        out = r.compute(goal_ctx)
        assert out.total > 100.0   # waypoint bonus dominates

    def test_crash_gives_large_negative_reward(self, crash_ctx):
        r = CompositeReward.default()
        out = r.compute(crash_ctx)
        assert out.total < -50.0   # crash penalty dominates

    def test_weighted_sum_correct(self, far_ctx):
        """
        With approach weight=2.0, approach contribution should be doubled.
        """
        # approach = prev_dist - curr_dist = 500 - 490 = 10
        # with weight=2.0 → approach contributes 20 to total
        r_double = CompositeReward(components=[
            (ApproachReward(), 2.0),
        ])
        r_single = CompositeReward(components=[
            (ApproachReward(), 1.0),
        ])
        out_double = r_double.compute(far_ctx)
        out_single = r_single.compute(far_ctx)
        assert out_double.total == pytest.approx(out_single.total * 2.0)

    def test_zero_weight_disables_component(self, goal_ctx):
        """Setting waypoint weight=0 should suppress goal bonus."""
        r = CompositeReward(components=[
            (WaypointReward({"bonus": 200.0}), 0.0),
        ])
        out = r.compute(goal_ctx)
        assert out.total == pytest.approx(0.0)

    def test_reset_called_without_error(self):
        r = CompositeReward.default()
        r.reset()   # should not raise

    def test_component_names_list(self):
        r = CompositeReward.default()
        names = r.component_names()
        assert "ApproachReward" in names
        assert "WaypointReward" in names

    def test_weights_list(self):
        r = CompositeReward.default()
        weights = r.weights()
        assert all(isinstance(w, float) for w in weights)


# ─────────────────────────────────────────────────────────────────────────────
# 11. TerminationResult
# ─────────────────────────────────────────────────────────────────────────────

class TestTerminationResult:

    def test_default_is_not_done(self):
        r = TerminationResult()
        assert r.terminated is False
        assert r.truncated is False
        assert r.is_done is False

    def test_is_done_terminated(self):
        r = TerminationResult(terminated=True, reason="oob")
        assert r.is_done is True

    def test_is_done_truncated(self):
        r = TerminationResult(truncated=True, reason="timeout")
        assert r.is_done is True

    def test_is_success_goal(self):
        r = TerminationResult(terminated=True, reason="goal")
        assert r.is_success is True

    def test_is_success_false_for_crash(self):
        r = TerminationResult(terminated=True, reason="oob")
        assert r.is_success is False

    def test_is_success_false_for_truncation(self):
        r = TerminationResult(truncated=True, reason="timeout")
        assert r.is_success is False


# ─────────────────────────────────────────────────────────────────────────────
# 12. TerminationChecker
# ─────────────────────────────────────────────────────────────────────────────

class TestTerminationChecker:

    def _make_checker(self, max_steps=1000):
        return TerminationChecker(PhysicsConfig(), max_steps=max_steps)

    def _mid_state(self, speed=20.0, fuel=1000.0, y=400.0):
        return AircraftState(x=600.0, y=y, vx=speed, vy=0.0, fuel=fuel)

    def test_no_termination_normal_step(self, world):
        checker = self._make_checker()
        checker.reset()
        state = self._mid_state()
        result = checker.check(state, world, step=1)
        assert result.terminated is False
        assert result.truncated is False
        assert result.reason == ""

    def test_oob_terminates(self, world):
        checker = self._make_checker()
        checker.reset()
        state = AircraftState(x=-100.0, y=400.0, vx=20.0)  # left wall
        result = checker.check(state, world, step=1)
        assert result.terminated is True
        assert result.reason == "oob"

    def test_goal_terminates(self, world):
        checker = self._make_checker()
        checker.reset()
        wp = world.waypoint
        state = AircraftState(x=wp.x, y=wp.y, vx=20.0)
        result = checker.check(state, world, step=1)
        assert result.terminated is True
        assert result.reason == "goal"
        assert result.is_success is True

    def test_timeout_truncates(self, world):
        checker = self._make_checker(max_steps=10)
        checker.reset()
        state = self._mid_state()   # safe state, won't crash
        result = checker.check(state, world, step=10)
        assert result.truncated is True
        assert result.reason == "timeout"

    def test_terminated_not_truncated_on_crash(self, world):
        checker = self._make_checker(max_steps=10)
        checker.reset()
        state = AircraftState(x=-100.0, y=400.0, vx=20.0)
        result = checker.check(state, world, step=10)
        assert result.terminated is True
        assert result.truncated is False   # crash wins over timeout

    def test_stall_counter_increments(self, world):
        checker = self._make_checker()
        checker.reset()
        slow = AircraftState(x=600.0, y=400.0, vx=1.0)   # below stall
        assert checker.stall_counter == 0
        checker.check(slow, world, step=1)
        assert checker.stall_counter == 1
        checker.check(slow, world, step=2)
        assert checker.stall_counter == 2

    def test_stall_terminates_after_n_steps(self, world):
        cfg = PhysicsConfig(stall_steps=3, vel_stall=5.0)
        checker = TerminationChecker(cfg, max_steps=1000)
        checker.reset()
        slow = AircraftState(x=600.0, y=400.0, vx=1.0)
        for step in range(1, 3):
            result = checker.check(slow, world, step=step)
            assert result.terminated is False
        result = checker.check(slow, world, step=3)
        assert result.terminated is True
        assert result.reason == "stall"

    def test_stall_counter_resets_on_speed_recovery(self, world):
        checker = self._make_checker()
        checker.reset()
        slow  = AircraftState(x=600.0, y=400.0, vx=1.0)
        fast  = AircraftState(x=600.0, y=400.0, vx=30.0)
        checker.check(slow, world, step=1)
        checker.check(slow, world, step=2)
        assert checker.stall_counter == 2
        checker.check(fast, world, step=3)
        assert checker.stall_counter == 0

    def test_reset_clears_stall_counter(self, world):
        checker = self._make_checker()
        checker.reset()
        slow = AircraftState(x=600.0, y=400.0, vx=1.0)
        for i in range(5):
            checker.check(slow, world, step=i+1)
        assert checker.stall_counter > 0
        checker.reset()
        assert checker.stall_counter == 0

    def test_terminated_and_truncated_never_both_true(self, world):
        """
        The checker must never set both terminated and truncated.
        Gymnasium invariant: they are mutually exclusive.
        """
        checker = self._make_checker(max_steps=5)
        checker.reset()
        # Crash at step 5 (same step as timeout)
        oob = AircraftState(x=-100.0, y=400.0, vx=20.0)
        result = checker.check(oob, world, step=5)
        assert not (result.terminated and result.truncated), \
            "Both terminated and truncated are True -- Gymnasium violation!"
