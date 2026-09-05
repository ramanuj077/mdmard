"""
Unit tests for Config System & Utilities (Stage 10).

Verifies typed config dataclasses, YAML loading, merging with dotted keys,
seeding determinism, geometry calculations, and integrity of all YAML configs in configs/.
"""

import math
from pathlib import Path
import numpy as np
import pytest

from utils.config import (
    EnvFullConfig,
    PPOConfig,
    DQNConfig,
    SACConfig,
    load_config,
    save_config,
    merge_configs,
)
from utils.seeding import seed_everything
from utils.geometry import (
    wrap_to_pi,
    euclidean_distance,
    bearing_to_target,
    clamp,
    point_in_circle,
)
from utils.io import read_yaml, write_yaml, read_json, write_json


def test_geometry_helpers():
    # wrap_to_pi
    assert math.isclose(wrap_to_pi(0.0), 0.0)
    assert math.isclose(abs(wrap_to_pi(3.0 * math.pi)), math.pi, abs_tol=1e-5)
    assert math.isclose(abs(wrap_to_pi(-3.0 * math.pi)), math.pi, abs_tol=1e-5)
    assert math.isclose(wrap_to_pi(2.5 * math.pi), 0.5 * math.pi, abs_tol=1e-5)
    assert math.isclose(wrap_to_pi(math.pi / 2), math.pi / 2)

    # euclidean_distance
    assert math.isclose(euclidean_distance(0, 0, 3, 4), 5.0)

    # bearing_to_target
    assert math.isclose(bearing_to_target(0, 0, 10, 0), 0.0)
    assert math.isclose(bearing_to_target(0, 0, 0, 10), math.pi / 2)

    # clamp
    assert clamp(5.0, 0.0, 10.0) == 5.0
    assert clamp(-2.0, 0.0, 10.0) == 0.0
    assert clamp(15.0, 0.0, 10.0) == 10.0

    # point_in_circle
    assert point_in_circle(2, 2, 0, 0, 3.0) is True
    assert point_in_circle(5, 5, 0, 0, 3.0) is False


def test_seeding():
    s1 = seed_everything(42)
    val1 = np.random.rand()

    s2 = seed_everything(42)
    val2 = np.random.rand()

    assert s1 == s2 == 42
    assert val1 == val2


def test_io_helpers(tmp_path):
    data = {"a": 1, "b": [1, 2, 3], "nested": {"k": "v"}}

    # YAML
    y_path = tmp_path / "test.yaml"
    write_yaml(data, y_path)
    loaded_y = read_yaml(y_path)
    assert loaded_y == data

    # JSON
    j_path = tmp_path / "test.json"
    write_json(data, j_path)
    loaded_j = read_json(j_path)
    assert loaded_j == data


def test_merge_configs_nested_and_dotted():
    base = {
        "physics": {"vel_max": 80.0, "mass": 1.0},
        "world": {"width": 1200},
        "learning_rate": 0.001,
    }

    # Standard nested merge + dotted key override
    overrides = {
        "physics.vel_max": 95.0,
        "world": {"height": 900},
        "learning_rate": 0.0003,
    }

    merged = merge_configs(base, overrides)
    assert merged["physics"]["vel_max"] == 95.0
    assert merged["physics"]["mass"] == 1.0
    assert merged["world"]["width"] == 1200
    assert merged["world"]["height"] == 900
    assert merged["learning_rate"] == 0.0003


def test_typed_configs():
    ppo = PPOConfig(learning_rate=1e-4, total_timesteps=100_000)
    d = ppo.to_dict()
    assert d["learning_rate"] == 1e-4
    assert d["n_steps"] == 2048

    dqn = DQNConfig(batch_size=32)
    assert dqn.batch_size == 32
    assert dqn.gamma == 0.99

    sac = SACConfig(tau=0.01)
    assert sac.tau == 0.01

    env_full = EnvFullConfig()
    assert env_full.world.width == 1200.0
    assert env_full.physics.mass == 1.0


def test_all_configs_validity():
    config_dir = Path("configs")

    # env_phase1.yaml
    env_cfg = load_config(config_dir / "env_phase1.yaml")
    assert "world" in env_cfg
    assert "physics" in env_cfg
    assert "reward" in env_cfg
    assert "observation" in env_cfg
    parsed_env = EnvFullConfig.from_dict(env_cfg)
    assert parsed_env.world.width == 1200.0

    # ppo_phase1.yaml
    ppo_cfg = load_config(config_dir / "ppo_phase1.yaml")
    assert ppo_cfg["algorithm"] == "PPO"
    assert ppo_cfg["learning_rate"] == 0.0003

    # dqn_phase1.yaml
    dqn_cfg = load_config(config_dir / "dqn_phase1.yaml")
    assert dqn_cfg["algorithm"] == "DQN"
    assert dqn_cfg["buffer_size"] == 100000

    # sac_phase1.yaml
    sac_cfg = load_config(config_dir / "sac_phase1.yaml")
    assert sac_cfg["algorithm"] == "SAC"

    # curriculum_phase1.yaml
    curr_cfg = load_config(config_dir / "curriculum_phase1.yaml")
    assert "curriculum" in curr_cfg
    assert len(curr_cfg["curriculum"]["stages"]) == 4
