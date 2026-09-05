"""
Unit tests for Baseline Agents (Stage 9).

Verifies RandomAgent, GreedyAgent, PIDAgent behaviors, safety overrides,
control responses, reset functionality, and integration evaluation with TacticalFlightEnv.
"""

import math
import numpy as np
import pytest

from baselines.base_agent import BaseAgent
from baselines.random_agent import RandomAgent
from baselines.greedy_agent import GreedyAgent
from baselines.pid_agent import PIDAgent
from envs.flight_env import TacticalFlightEnv
from metrics.tracker import MetricsTracker


@pytest.fixture
def base_obs():
    # [0:x, 1:y, 2:vx, 3:vy, 4:theta, 5:omega, 6:dist, 7:bearing, 8:fuel, 9:time]
    # Mid-altitude (0.5), speed (0.5), theta = 0, bearing = 0
    return np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.3, 0.0, 1.0, 1.0], dtype=np.float32)


def test_random_agent():
    agent = RandomAgent(num_actions=6, seed=123)
    actions = [agent.act(np.zeros(10)) for _ in range(100)]
    assert all(0 <= a < 6 for a in actions)
    assert len(set(actions)) > 1

    # Biased random
    biased_agent = RandomAgent(num_actions=6, action_probabilities=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    biased_actions = [biased_agent.act(np.zeros(10)) for _ in range(20)]
    assert all(a == 3 for a in biased_actions)


def test_greedy_agent_heading_response(base_obs):
    agent = GreedyAgent(heading_deadband=0.05)

    # 1. Target above (bearing > 0) -> Action 3 (Thrust + Pitch Up)
    obs_up = base_obs.copy()
    obs_up[7] = 0.3  # bearing = +0.3 * pi
    assert agent.act(obs_up) == GreedyAgent.ACTION_THRUST_UP

    # 2. Target below (bearing < 0) -> Action 4 (Thrust + Pitch Down)
    obs_down = base_obs.copy()
    obs_down[7] = -0.3  # bearing = -0.3 * pi
    assert agent.act(obs_down) == GreedyAgent.ACTION_THRUST_DOWN

    # 3. Target straight ahead -> Action 0 (Thrust)
    obs_ahead = base_obs.copy()
    obs_ahead[7] = 0.01  # within deadband
    assert agent.act(obs_ahead) == GreedyAgent.ACTION_THRUST


def test_greedy_agent_safety_overrides(base_obs):
    agent = GreedyAgent(altitude_floor_threshold=0.15, altitude_ceil_threshold=0.85)

    # Low altitude override -> Pitch Up
    obs_low = base_obs.copy()
    obs_low[1] = 0.05  # 5% altitude
    obs_low[7] = -0.5  # target is below, but ground danger takes priority
    assert agent.act(obs_low) == GreedyAgent.ACTION_THRUST_UP

    # High altitude override -> Pitch Down
    obs_high = base_obs.copy()
    obs_high[1] = 0.95  # 95% altitude
    obs_high[7] = 0.5   # target is above, but ceiling danger takes priority
    assert agent.act(obs_high) == GreedyAgent.ACTION_THRUST_DOWN


def test_pid_agent_heading_and_integral(base_obs):
    agent = PIDAgent(kp=2.0, ki=0.1, kd=0.5, control_threshold=0.1)

    # Positive heading error
    obs_up = base_obs.copy()
    obs_up[7] = 0.25
    act1 = agent.act(obs_up)
    assert act1 == PIDAgent.ACTION_THRUST_UP

    # Negative heading error
    agent.reset()
    obs_down = base_obs.copy()
    obs_down[7] = -0.25
    act2 = agent.act(obs_down)
    assert act2 == PIDAgent.ACTION_THRUST_DOWN

    # Reset clears internal state
    agent.reset()
    assert agent._integral_error == 0.0
    assert agent._prev_error is None


def test_pid_agent_safety_overrides(base_obs):
    agent = PIDAgent(altitude_floor_threshold=0.15, altitude_ceil_threshold=0.85)

    # Ground proximity
    obs_ground = base_obs.copy()
    obs_ground[1] = 0.08
    obs_ground[7] = -0.4
    assert agent.act(obs_ground) == PIDAgent.ACTION_THRUST_UP

    # Ceiling proximity
    obs_ceil = base_obs.copy()
    obs_ceil[1] = 0.92
    obs_ceil[7] = 0.4
    assert agent.act(obs_ceil) == PIDAgent.ACTION_THRUST_DOWN


def test_evaluate_in_flight_env():
    env = TacticalFlightEnv()
    tracker = MetricsTracker()

    greedy = GreedyAgent()
    stats = greedy.evaluate(env, num_episodes=3, seed=100, metrics_tracker=tracker)

    assert stats["num_episodes"] == 3
    assert "mean_reward" in stats
    assert "success_rate" in stats
    assert "mean_steps" in stats
    assert tracker.num_episodes == 3
    env.close()
