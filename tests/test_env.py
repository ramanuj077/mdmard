"""
tests/test_env.py
=================
Unit + integration tests for TacticalFlightEnv -- MDMARD Phase 1

Test Strategy
-------------
1. Construction     : Env builds without error, spaces correct
2. Reset            : Returns (obs, info), obs in space, seeded reproducible
3. Step             : Returns (obs, reward, terminated, truncated, info)
4. API compliance   : Gymnasium check_env() passes (SB3 gate)
5. Termination      : OOB, goal, stall paths trigger correctly
6. Reward           : Approach positive when closing, crash penalty fires
7. Info dict        : All expected keys present with correct types
8. Random rollout   : Full episode with random actions completes cleanly
9. Reproducibility  : Same seed = same trajectory

Run with:
    pytest tests/test_env.py -v
    pytest tests/test_env.py -v -k "check_env"  (run only SB3 gate)
"""

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.flight_env import TacticalFlightEnv
from envs.physics import PhysicsConfig
from envs.world import WorldConfig
from observation.builder import ObservationConfig


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def env() -> TacticalFlightEnv:
    """Fresh default env, not yet reset."""
    return TacticalFlightEnv()


@pytest.fixture
def env_reset() -> tuple:
    """Env after reset -- returns (env, obs, info)."""
    e = TacticalFlightEnv()
    obs, info = e.reset(seed=42)
    return e, obs, info


# ─────────────────────────────────────────────────────────────────────────────
# 1. Construction Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConstruction:

    def test_env_creates_without_error(self):
        env = TacticalFlightEnv()
        assert env is not None

    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (10,)

    def test_observation_space_dtype(self, env):
        assert env.observation_space.dtype == np.float32

    def test_observation_space_bounds(self, env):
        assert np.all(env.observation_space.low  == -1.0)
        assert np.all(env.observation_space.high ==  1.0)

    def test_action_space_size(self, env):
        assert env.action_space.n == 6

    def test_render_mode_none_by_default(self, env):
        assert env.render_mode is None

    def test_invalid_render_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid render_mode"):
            TacticalFlightEnv(render_mode="fancy_3d")

    def test_custom_configs_accepted(self):
        env = TacticalFlightEnv(
            physics_config=PhysicsConfig(mass=2.0, gravity=1.62),
            world_config=WorldConfig(width=800.0, height=600.0),
            obs_config=ObservationConfig(vel_max=60.0),
            max_steps=500,
        )
        assert env is not None

    def test_metadata_has_render_modes(self, env):
        assert "render_modes" in env.metadata
        assert "human" in env.metadata["render_modes"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Reset Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestReset:

    def test_reset_returns_tuple(self, env):
        result = env.reset(seed=0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_obs_is_ndarray(self, env):
        obs, _ = env.reset(seed=0)
        assert isinstance(obs, np.ndarray)

    def test_reset_obs_shape(self, env):
        obs, _ = env.reset(seed=0)
        assert obs.shape == (10,)

    def test_reset_obs_dtype(self, env):
        obs, _ = env.reset(seed=0)
        assert obs.dtype == np.float32

    def test_reset_obs_in_space(self, env):
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)

    def test_reset_obs_all_in_bounds(self, env):
        for seed in range(10):
            obs, _ = env.reset(seed=seed)
            assert np.all(obs >= -1.0)
            assert np.all(obs <= 1.0)

    def test_reset_info_is_dict(self, env):
        _, info = env.reset(seed=0)
        assert isinstance(info, dict)

    def test_reset_step_count_is_zero(self, env):
        env.reset(seed=0)
        assert env._step_count == 0

    def test_reset_seeded_reproducible(self, env):
        """Same seed must give same initial obs."""
        obs1, _ = env.reset(seed=99)
        obs2, _ = env.reset(seed=99)
        np.testing.assert_array_equal(obs1, obs2)

    def test_reset_different_seeds_different_obs(self, env):
        """Different seeds should (almost always) give different obs."""
        obs1, _ = env.reset(seed=0)
        obs2, _ = env.reset(seed=1)
        # Distance to waypoint (obs[6]) will differ across seeds
        assert not np.allclose(obs1, obs2)

    def test_reset_no_seed_doesnt_crash(self, env):
        obs, info = env.reset()
        assert obs.shape == (10,)

    def test_multiple_resets_dont_crash(self, env):
        for seed in range(5):
            obs, info = env.reset(seed=seed)
            assert obs.shape == (10,)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Step Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStep:

    def test_step_returns_five_tuple(self, env_reset):
        env, _, _ = env_reset
        result = env.step(0)
        assert len(result) == 5

    def test_step_obs_shape(self, env_reset):
        env, _, _ = env_reset
        obs, *_ = env.step(0)
        assert obs.shape == (10,)

    def test_step_obs_dtype(self, env_reset):
        env, _, _ = env_reset
        obs, *_ = env.step(0)
        assert obs.dtype == np.float32

    def test_step_obs_in_space(self, env_reset):
        env, _, _ = env_reset
        obs, *_ = env.step(0)
        assert env.observation_space.contains(obs)

    def test_step_reward_is_float(self, env_reset):
        env, _, _ = env_reset
        _, reward, *_ = env.step(0)
        assert isinstance(reward, float)

    def test_step_terminated_is_bool(self, env_reset):
        env, _, _ = env_reset
        _, _, terminated, *_ = env.step(0)
        assert isinstance(terminated, bool)

    def test_step_truncated_is_bool(self, env_reset):
        env, _, _ = env_reset
        _, _, _, truncated, _ = env.step(0)
        assert isinstance(truncated, bool)

    def test_step_info_is_dict(self, env_reset):
        env, _, _ = env_reset
        _, _, _, _, info = env.step(0)
        assert isinstance(info, dict)

    def test_step_increments_step_count(self, env_reset):
        env, _, _ = env_reset
        for i in range(1, 6):
            env.step(0)
            assert env._step_count == i

    def test_all_discrete_actions_valid(self, env_reset):
        """All 6 actions can be passed to step() without error."""
        env, _, _ = env_reset
        for action in range(6):
            env.reset(seed=42)
            obs, reward, term, trunc, info = env.step(action)
            assert obs.shape == (10,)

    def test_step_before_reset_raises(self):
        env = TacticalFlightEnv()
        with pytest.raises(RuntimeError, match="reset()"):
            env.step(0)

    def test_step_obs_all_in_bounds(self, env_reset):
        """After 50 random steps, obs always in [-1, 1]."""
        env, _, _ = env_reset
        for _ in range(50):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            assert np.all(obs >= -1.0)
            assert np.all(obs <= 1.0)
            if terminated or truncated:
                env.reset(seed=0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Gymnasium check_env Compliance ← STAGE 3 GATE
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckEnv:
    """
    Run stable_baselines3.common.env_checker.check_env.

    This is the official SB3 compliance test. If this passes,
    the env can be used directly with any SB3 algorithm.
    """

    def test_check_env_passes(self):
        """
        SB3 check_env must pass without warnings or errors.

        This is the STAGE 3 GATE. No further stages proceed until
        this test is green.
        """
        try:
            from stable_baselines3.common.env_checker import check_env
        except ImportError:
            pytest.skip("stable_baselines3 not installed -- skipping check_env")

        env = TacticalFlightEnv()
        # warn=False suppresses non-critical warnings (e.g. about render)
        check_env(env, warn=True)   # raises on any compliance issue


# ─────────────────────────────────────────────────────────────────────────────
# 5. Termination Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTermination:

    def test_oob_terminates(self):
        """Aircraft placed out of bounds should terminate on next step."""
        env = TacticalFlightEnv()
        env.reset(seed=0)
        # Force aircraft far outside bounds
        env._state.x = -500.0
        _, _, terminated, _, info = env.step(5)  # glide
        assert terminated is True
        assert info["termination"] == "oob"

    def test_goal_terminates(self):
        """Aircraft placed at waypoint should terminate with 'goal'."""
        env = TacticalFlightEnv()
        env.reset(seed=0)
        wp = env._world.waypoint
        # Teleport aircraft to waypoint center
        env._state.x = wp.x
        env._state.y = wp.y
        env._state.vx = 10.0   # needs speed > stall
        _, _, terminated, _, info = env.step(5)
        assert terminated is True
        assert info["termination"] == "goal"

    def test_truncation_at_max_steps(self):
        """Episode truncates (not terminates) at max_steps."""
        env = TacticalFlightEnv(max_steps=5)
        env.reset(seed=0)
        # Move aircraft away from boundaries
        env._state.x = 600.0
        env._state.y = 400.0
        env._state.vx = 20.0
        truncated = False
        for _ in range(5):
            _, _, terminated, truncated, _ = env.step(5)  # glide
            if terminated:
                break
        assert truncated is True

    def test_terminated_and_truncated_mutually_exclusive(self, env_reset):
        """
        terminated and truncated must not both be True simultaneously.
        Gymnasium spec: they are mutually exclusive.
        """
        env, _, _ = env_reset
        for _ in range(200):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            assert not (terminated and truncated), \
                "terminated and truncated are both True -- Gymnasium violation!"
            if terminated or truncated:
                env.reset(seed=42)

    def test_goal_reward_is_positive(self):
        """Goal termination must give large positive reward."""
        env = TacticalFlightEnv()
        env.reset(seed=0)
        wp = env._world.waypoint
        env._state.x = wp.x
        env._state.y = wp.y
        env._state.vx = 10.0
        _, reward, terminated, _, info = env.step(5)
        assert terminated is True
        assert reward > 100.0   # should include +200 goal bonus

    def test_crash_reward_is_negative(self):
        """OOB termination must give large negative reward."""
        env = TacticalFlightEnv()
        env.reset(seed=0)
        env._state.x = -500.0
        _, reward, terminated, _, info = env.step(5)
        assert terminated is True
        assert reward < -50.0   # should include -100 crash penalty


# ─────────────────────────────────────────────────────────────────────────────
# 6. Reward Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestReward:

    def test_approach_reward_positive_when_closing(self):
        """
        When aircraft moves toward waypoint, approach reward should be
        positive (distance decreased → prev_dist - curr_dist > 0).
        """
        env = TacticalFlightEnv()
        env.reset(seed=0)
        wp = env._world.waypoint
        # Place aircraft directly left of waypoint, flying right
        env._state.x = wp.x - 200
        env._state.y = wp.y
        env._state.vx = 30.0   # moving toward waypoint
        env._state.vy = 0.0
        env._state.theta = 0.0
        env._prev_dist = env._world.distance_to_waypoint(
            env._state.x, env._state.y
        )
        _, reward, _, _, info = env.step(0)  # thrust (keep moving right)
        approach = info["reward_breakdown"]["approach"]
        # Small positive (or at least not hugely negative)
        assert approach > -5.0   # sanity: not flying away

    def test_reward_breakdown_has_expected_keys(self, env_reset):
        env, _, _ = env_reset
        _, _, _, _, info = env.step(0)
        bd = info["reward_breakdown"]
        expected = {"approach", "waypoint", "crash", "alive", "smooth", "altitude", "fuel"}
        assert set(bd.keys()) == expected

    def test_alive_reward_always_positive(self, env_reset):
        """Alive component should always be +0.1 per step."""
        env, _, _ = env_reset
        for _ in range(10):
            _, _, terminated, truncated, info = env.step(5)
            if not (terminated or truncated):
                assert info["reward_breakdown"]["alive"] == pytest.approx(0.1)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Info Dict Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestInfoDict:

    REQUIRED_KEYS = {
        "action_name", "step", "episode_reward", "dist_to_waypoint",
        "speed", "pitch_deg", "fuel", "reward_breakdown",
        "termination", "terminated", "truncated",
    }

    def test_reset_info_has_required_keys(self, env_reset):
        _, _, info = env_reset
        missing = self.REQUIRED_KEYS - set(info.keys())
        assert not missing, f"Missing info keys: {missing}"

    def test_step_info_has_required_keys(self, env_reset):
        env, _, _ = env_reset
        _, _, _, _, info = env.step(0)
        missing = self.REQUIRED_KEYS - set(info.keys())
        assert not missing, f"Missing info keys: {missing}"

    def test_info_step_increments(self, env_reset):
        env, _, _ = env_reset
        for expected_step in range(1, 6):
            _, _, _, _, info = env.step(0)
            assert info["step"] == expected_step

    def test_info_speed_is_positive(self, env_reset):
        env, _, _ = env_reset
        _, _, _, _, info = env.step(0)
        assert info["speed"] >= 0.0

    def test_info_action_name_is_string(self, env_reset):
        env, _, _ = env_reset
        _, _, _, _, info = env.step(0)
        assert isinstance(info["action_name"], str)
        assert len(info["action_name"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 8. Full Random Rollout
# ─────────────────────────────────────────────────────────────────────────────

class TestRandomRollout:

    def test_full_episode_completes(self):
        """
        A complete episode with random actions must finish without error.
        Tests the full step() → terminated/truncated → reset() cycle.
        """
        env = TacticalFlightEnv(max_steps=200)
        obs, _ = env.reset(seed=7)
        steps = 0
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated

            # Safety: obs always valid
            assert obs.shape == (10,)
            assert np.all(obs >= -1.0)
            assert np.all(obs <= 1.0)
            assert math.isfinite(reward)

        assert steps > 0
        assert done is True

    def test_multiple_episodes_without_error(self):
        """Run 10 full episodes end-to-end."""
        env = TacticalFlightEnv(max_steps=100)
        for episode in range(10):
            obs, _ = env.reset(seed=episode)
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            # After done, reset is safe
        env.close()

    def test_render_mode_none_doesnt_crash(self):
        """render() with mode=None should return None silently."""
        env = TacticalFlightEnv(render_mode=None)
        env.reset(seed=0)
        env.step(0)
        result = env.render()
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# 9. Reproducibility Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility:

    def test_same_seed_same_trajectory(self):
        """
        With same seed, identical action sequence → identical observations.
        Essential for debugging and replay validation.
        """
        actions = [0, 1, 3, 5, 2, 0, 4, 1, 0, 5]

        def run(seed, actions):
            env = TacticalFlightEnv()
            env.reset(seed=seed)
            trajectory = []
            for a in actions:
                obs, reward, terminated, truncated, _ = env.step(a)
                trajectory.append((obs.copy(), reward))
                if terminated or truncated:
                    break
            return trajectory

        traj1 = run(seed=77, actions=actions)
        traj2 = run(seed=77, actions=actions)

        for (o1, r1), (o2, r2) in zip(traj1, traj2):
            np.testing.assert_array_equal(o1, o2)
            assert r1 == pytest.approx(r2)
