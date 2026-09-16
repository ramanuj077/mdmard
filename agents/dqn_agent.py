"""
agents/dqn_agent.py
===================
DQN Agent -- MDMARD Phase 1 (Stable-Baselines3 Wrapper)

Wraps SB3 DQN into the project BaseAgent interface.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

import numpy as np

from baselines.base_agent import BaseAgent


class DQNAgent(BaseAgent):
    """Deep Q-Network agent (SB3 backend)."""

    def __init__(
        self,
        policy: str = "MlpPolicy",
        learning_rate: float = 1e-4,
        buffer_size: int = 100_000,
        learning_starts: int = 1_000,
        batch_size: int = 64,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int = 4,
        gradient_steps: int = 1,
        target_update_interval: int = 1_000,
        exploration_fraction: float = 0.2,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        net_arch: Optional[List[int]] = None,
        seed: Optional[int] = 42,
        device: str = "auto",
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
        name: str = "DQN",
    ) -> None:
        super().__init__(name=name)
        self.policy = policy
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.net_arch = net_arch if net_arch is not None else [128, 128]
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.tensorboard_log = tensorboard_log
        self._model = None

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        if self._model is None:
            raise RuntimeError("DQNAgent: call train() or load() before act().")
        action, _ = self._model.predict(observation, deterministic=True)
        return int(action)

    def reset(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        env: Any,
        total_timesteps: int = 500_000,
        eval_env: Optional[Any] = None,
        eval_freq: int = 10_000,
        eval_episodes: int = 20,
        save_freq: int = 25_000,
        save_path: Optional[str] = None,
        experiment_manager: Optional[Any] = None,
        metrics_tracker: Optional[Any] = None,
        callbacks: Optional[List[Any]] = None,
        reset_num_timesteps: bool = True,
    ) -> None:
        """Train the DQN agent with optional eval / checkpoint callbacks."""
        try:
            from stable_baselines3 import DQN
            from stable_baselines3.common.callbacks import (
                CallbackList, CheckpointCallback, EvalCallback,
            )
        except ImportError as e:
            raise ImportError("pip install stable-baselines3") from e

        tb_log = self.tensorboard_log
        if experiment_manager is not None and tb_log is None:
            tb_log = str(experiment_manager.run_dir / "tensorboard")

        policy_kwargs = {"net_arch": self.net_arch}

        if self._model is None:
            self._model = DQN(
                policy=self.policy,
                env=env,
                learning_rate=self.learning_rate,
                buffer_size=self.buffer_size,
                learning_starts=self.learning_starts,
                batch_size=self.batch_size,
                tau=self.tau,
                gamma=self.gamma,
                train_freq=self.train_freq,
                gradient_steps=self.gradient_steps,
                target_update_interval=self.target_update_interval,
                exploration_fraction=self.exploration_fraction,
                exploration_initial_eps=self.exploration_initial_eps,
                exploration_final_eps=self.exploration_final_eps,
                policy_kwargs=policy_kwargs,
                seed=self.seed,
                device=self.device,
                verbose=self.verbose,
                tensorboard_log=tb_log,
            )
        else:
            self._model.set_env(env)

        cb_list: List[Any] = list(callbacks or [])

        ckpt_dir = save_path
        if ckpt_dir is None and experiment_manager is not None:
            ckpt_dir = str(experiment_manager.checkpoints_dir)
        if ckpt_dir is not None:
            cb_list.append(
                CheckpointCallback(
                    save_freq=save_freq,
                    save_path=ckpt_dir,
                    name_prefix="dqn_ckpt",
                    verbose=0,
                )
            )

        if eval_env is not None:
            best_dir = ckpt_dir or "/tmp/dqn_best"
            cb_list.append(
                EvalCallback(
                    eval_env,
                    best_model_save_path=best_dir,
                    log_path=best_dir,
                    eval_freq=eval_freq,
                    n_eval_episodes=eval_episodes,
                    deterministic=True,
                    render=False,
                    verbose=0,
                )
            )

        combined_cb = CallbackList(cb_list) if cb_list else None

        if experiment_manager is not None:
            experiment_manager.start()

        t0 = time.time()
        self._model.learn(
            total_timesteps=total_timesteps,
            callback=combined_cb,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=(self.verbose > 0),
        )
        elapsed = time.time() - t0

        if experiment_manager is not None:
            experiment_manager.log_metrics(
                {"total_timesteps": total_timesteps, "training_time_sec": elapsed, "agent": self.name},
                step=total_timesteps,
            )
            experiment_manager.close()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("No model to save.")
        self._model.save(path)

    def load(self, path: str, env: Optional[Any] = None) -> None:
        try:
            from stable_baselines3 import DQN
        except ImportError as e:
            raise ImportError("pip install stable-baselines3") from e
        self._model = DQN.load(path, env=env, device=self.device)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def model(self):
        return self._model

    @property
    def num_timesteps(self) -> int:
        return getattr(self._model, "num_timesteps", 0) if self._model else 0

    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> "DQNAgent":
        merged = {**config, **kwargs}
        valid = {
            "policy", "learning_rate", "buffer_size", "learning_starts", "batch_size",
            "tau", "gamma", "train_freq", "gradient_steps", "target_update_interval",
            "exploration_fraction", "exploration_initial_eps", "exploration_final_eps",
            "net_arch", "seed", "device", "verbose", "tensorboard_log", "name",
        }
        return cls(**{k: v for k, v in merged.items() if k in valid})

    def __repr__(self) -> str:
        state = "trained" if self._model else "untrained"
        return f"DQNAgent(lr={self.learning_rate}, buf={self.buffer_size}, arch={self.net_arch}, {state})"
