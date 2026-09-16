"""
tests/test_agents.py
====================
Unit tests for RL Agent Wrappers (Stage 11-14) -- MDMARD Phase 1

Tests verify:
  - PPOAgent, DQNAgent, SACAgent instantiate correctly
  - from_config() factory works
  - act() raises RuntimeError before training
  - save/load round-trip produces consistent actions
  - num_timesteps property works
"""

import pytest
import numpy as np
import tempfile
import os


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def sample_obs():
    """A valid 10-dim observation vector."""
    rng = np.random.default_rng(0)
    return rng.uniform(-1.0, 1.0, size=(10,)).astype(np.float32)


@pytest.fixture
def dummy_env():
    """Minimal TacticalFlightEnv for training tests."""
    from envs.flight_env import TacticalFlightEnv
    env = TacticalFlightEnv()
    env.reset(seed=0)
    return env


# ---------------------------------------------------------------------
# PPO Agent tests
# ---------------------------------------------------------------------

class TestPPOAgent:
    def test_instantiation_defaults(self):
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent()
        assert agent.name == "PPO"
        assert agent.learning_rate == pytest.approx(3e-4)
        assert agent.n_steps == 2048
        assert agent.net_arch == [128, 128]
        assert agent.model is None
        assert agent.num_timesteps == 0

    def test_instantiation_custom(self):
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(learning_rate=1e-3, n_steps=512, net_arch=[64, 64], name="PPO_test")
        assert agent.name == "PPO_test"
        assert agent.learning_rate == pytest.approx(1e-3)
        assert agent.n_steps == 512
        assert agent.net_arch == [64, 64]

    def test_act_raises_before_training(self, sample_obs):
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent()
        with pytest.raises(RuntimeError, match="train"):
            agent.act(sample_obs)

    def test_reset_noop(self):
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent()
        agent.reset()  # Should not raise

    def test_from_config(self):
        from agents.ppo_agent import PPOAgent
        cfg = {
            "learning_rate": 5e-4,
            "n_steps": 1024,
            "batch_size": 32,
            "gamma": 0.95,
            "net_arch": [64, 64],
            "seed": 7,
        }
        agent = PPOAgent.from_config(cfg, name="TestPPO")
        assert agent.learning_rate == pytest.approx(5e-4)
        assert agent.n_steps == 1024
        assert agent.batch_size == 32
        assert agent.gamma == pytest.approx(0.95)
        assert agent.name == "TestPPO"

    def test_repr(self):
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent()
        r = repr(agent)
        assert "PPOAgent" in r
        assert "untrained" in r

    def test_train_short(self, dummy_env):
        """Smoke test: 512-step train should not crash."""
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(n_steps=64, batch_size=32, verbose=0)
        agent.train(env=dummy_env, total_timesteps=128)
        assert agent.num_timesteps >= 64
        dummy_env.close()

    def test_save_load(self, dummy_env, sample_obs):
        """Save and load should produce the same deterministic action."""
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(n_steps=64, batch_size=32, verbose=0, seed=0)
        agent.train(env=dummy_env, total_timesteps=128)

        action_before = agent.act(sample_obs)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ppo_test")
            agent.save(path)

            agent2 = PPOAgent(verbose=0)
            agent2.load(path + ".zip")
            action_after = agent2.act(sample_obs)

        assert action_before == action_after
        dummy_env.close()


# ---------------------------------------------------------------------
# DQN Agent tests
# ---------------------------------------------------------------------

class TestDQNAgent:
    def test_instantiation(self):
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent()
        assert agent.name == "DQN"
        assert agent.learning_rate == pytest.approx(1e-4)
        assert agent.buffer_size == 100_000
        assert agent.model is None

    def test_act_raises_before_training(self, sample_obs):
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent()
        with pytest.raises(RuntimeError):
            agent.act(sample_obs)

    def test_from_config(self):
        from agents.dqn_agent import DQNAgent
        cfg = {"learning_rate": 1e-3, "buffer_size": 50_000, "batch_size": 32}
        agent = DQNAgent.from_config(cfg, name="DQN_cfg")
        assert agent.learning_rate == pytest.approx(1e-3)
        assert agent.buffer_size == 50_000
        assert agent.name == "DQN_cfg"

    def test_train_short(self, dummy_env):
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent(learning_starts=50, buffer_size=500, verbose=0)
        agent.train(env=dummy_env, total_timesteps=100)
        assert agent.num_timesteps >= 100
        dummy_env.close()

    def test_repr(self):
        from agents.dqn_agent import DQNAgent
        r = repr(DQNAgent())
        assert "DQNAgent" in r
        assert "untrained" in r


# ---------------------------------------------------------------------
# SAC Agent tests
# ---------------------------------------------------------------------

class TestSACAgent:
    def test_instantiation(self):
        from agents.sac_agent import SACAgent
        agent = SACAgent()
        assert agent.name == "SAC"
        assert agent.learning_rate == pytest.approx(3e-4)
        assert agent.net_arch == [256, 256]
        assert agent.model is None

    def test_act_raises_before_training(self, sample_obs):
        from agents.sac_agent import SACAgent
        agent = SACAgent()
        with pytest.raises(RuntimeError):
            agent.act(sample_obs)

    def test_from_config(self):
        from agents.sac_agent import SACAgent
        cfg = {"learning_rate": 1e-3, "buffer_size": 10_000, "net_arch": [128, 128]}
        agent = SACAgent.from_config(cfg, name="SAC_cfg")
        assert agent.learning_rate == pytest.approx(1e-3)
        assert agent.name == "SAC_cfg"

    def test_repr(self):
        from agents.sac_agent import SACAgent
        r = repr(SACAgent())
        assert "SACAgent" in r
