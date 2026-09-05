"""
utils/config.py
===============
Typed Configuration System & YAML Loader -- MDMARD Phase 1

Provides strongly typed configuration dataclasses for:
  - Environment & Physics settings
  - PPO, DQN, SAC algorithm hyperparameters
  - Curriculum progression stages
  - Deep configuration merging and dotted-key overrides (e.g. 'physics.vel_max'=90.0)
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from utils.io import read_yaml, write_yaml


@dataclass
class WorldCfg:
    width: float = 1200.0
    height: float = 800.0
    floor: float = 50.0
    ceiling: float = 750.0
    goal_radius: float = 30.0


@dataclass
class PhysicsCfg:
    mass: float = 1.0
    inertia: float = 0.1
    thrust_max: float = 15.0
    drag_coeff: float = 0.02
    lift_coeff: float = 0.08
    gravity: float = 9.81
    vel_max: float = 80.0
    vel_stall: float = 5.0
    stall_steps: int = 15
    pitch_max: float = 1.5708
    pitch_rate_max: float = 2.0
    pitch_torque: float = 5.0
    dt: float = 0.05


@dataclass
class RewardCfg:
    w_approach: float = 1.0
    w_waypoint: float = 200.0
    w_fuel: float = 0.0
    w_alive: float = 0.1
    w_crash: float = -100.0
    w_smooth: float = -0.05
    w_altitude: float = -0.5


@dataclass
class ObsCfg:
    vel_max_norm: float = 80.0
    omega_max_norm: float = 2.0
    dist_max_norm: float = 1442.0
    fuel_max: float = 1000.0
    t_max: float = 1000.0


@dataclass
class EnvFullConfig:
    world: WorldCfg = field(default_factory=WorldCfg)
    physics: PhysicsCfg = field(default_factory=PhysicsCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    observation: ObsCfg = field(default_factory=ObsCfg)
    max_steps: int = 1000

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EnvFullConfig:
        w = WorldCfg(**data.get("world", {}))
        p = PhysicsCfg(**data.get("physics", {}))
        r = RewardCfg(**data.get("reward", {}))
        o = ObsCfg(**data.get("observation", {}))
        max_s = data.get("episode", {}).get("max_steps", data.get("max_steps", 1000))
        return cls(world=w, physics=p, reward=r, observation=o, max_steps=max_s)


@dataclass
class PPOConfig:
    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 500_000
    net_arch: List[int] = field(default_factory=lambda: [128, 128])
    eval_freq: int = 10_000
    save_freq: int = 25_000

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DQNConfig:
    policy: str = "MlpPolicy"
    learning_rate: float = 1e-4
    buffer_size: int = 100_000
    learning_starts: int = 1_000
    batch_size: int = 64
    tau: float = 1.0
    gamma: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 1_000
    exploration_fraction: float = 0.2
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    total_timesteps: int = 500_000
    net_arch: List[int] = field(default_factory=lambda: [128, 128])
    eval_freq: int = 10_000
    save_freq: int = 25_000

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SACConfig:
    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    buffer_size: int = 200_000
    learning_starts: int = 1_000
    batch_size: int = 128
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: Union[str, float] = "auto"
    target_update_interval: int = 1
    total_timesteps: int = 500_000
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    eval_freq: int = 10_000
    save_freq: int = 25_000

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def merge_configs(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries or apply dotted-key overrides."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if "." in key:
            parts = key.split(".")
            curr = merged
            for p in parts[:-1]:
                if p not in curr or not isinstance(curr[p], dict):
                    curr[p] = {}
                curr = curr[p]
            curr[parts[-1]] = value
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(
    path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load configuration from a YAML file with optional parameter overrides."""
    cfg = read_yaml(path)
    if overrides:
        cfg = merge_configs(cfg, overrides)
    return cfg


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> Path:
    """Save configuration dictionary to a YAML file."""
    return write_yaml(config, path)
