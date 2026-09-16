"""
agents/ppo_agent.py
===================
PPO Agent -- MDMARD Phase 1 (Stable-Baselines3 Wrapper)

Wraps SB3 PPO into the project BaseAgent interface with:
  - Full ExperimentManager / MetricsTracker integration
  - Checkpoint + EvalCallback hooks
  - from_config() factory method
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from baselines.base_agent import BaseAgent


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization agent (SB3 backend)."""

    def __init__(
        self,
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        net_arch: Union[List[int], Dict[str, List[int]], None] = None,
        seed: Optional[int] = 42,
        device: str = "auto",
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
        name: str = "PPO",
    ) -> None:
        super().__init__(name=name)
        self.policy = policy
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.net_arch = net_arch if net_arch is not None else [128, 128]
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.tensorboard_log = tensorboard_log
        self._model = None  # lazy-init

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        if self._model is None:
            raise RuntimeError("PPOAgent: call train() or load() before act().")
        action, _ = self._model.predict(observation, deterministic=True)
        return int(action)

    def reset(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _policy_kwargs(self) -> Dict[str, Any]:
        if isinstance(self.net_arch, dict):
            return {"net_arch": self.net_arch}
        return {"net_arch": self.net_arch}

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
        """Train the PPO agent with optional eval / checkpoint callbacks."""
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import (
                CallbackList, CheckpointCallback, EvalCallback,
            )
        except ImportError as e:
            raise ImportError("pip install stable-baselines3") from e

        tb_log = self.tensorboard_log
        if experiment_manager is not None and tb_log is None:
            tb_log = str(experiment_manager.run_dir / "tensorboard")

        if self._model is None:
            self._model = PPO(
                policy=self.policy,
                env=env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                max_grad_norm=self.max_grad_norm,
                policy_kwargs=self._policy_kwargs(),
                seed=self.seed,
                device=self.device,
                verbose=self.verbose,
                tensorboard_log=tb_log,
            )
        else:
            self._model.set_env(env)

        cb_list: List[Any] = list(callbacks or [])

        # Checkpoint callback
        ckpt_dir = save_path
        if ckpt_dir is None and experiment_manager is not None:
            ckpt_dir = str(experiment_manager.checkpoints_dir)
        if ckpt_dir is not None:
            cb_list.append(
                CheckpointCallback(
                    save_freq=save_freq,
                    save_path=ckpt_dir,
                    name_prefix="ppo_ckpt",
                    verbose=0,
                )
            )

        # Eval callback
        if eval_env is not None:
            best_dir = ckpt_dir or "/tmp/ppo_best"
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
            from stable_baselines3 import PPO
        except ImportError as e:
            raise ImportError("pip install stable-baselines3") from e
        self._model = PPO.load(path, env=env, device=self.device)

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
    def from_config(cls, config: Dict[str, Any], **kwargs) -> "PPOAgent":
        """Build PPOAgent from a config dict (e.g., loaded from YAML)."""
        merged = {**config, **kwargs}
        valid = {
            "policy", "learning_rate", "n_steps", "batch_size", "n_epochs",
            "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef",
            "max_grad_norm", "net_arch", "seed", "device", "verbose",
            "tensorboard_log", "name",
        }
        return cls(**{k: v for k, v in merged.items() if k in valid})

    def __repr__(self) -> str:
        state = "trained" if self._model else "untrained"
        return f"PPOAgent(lr={self.learning_rate}, n_steps={self.n_steps}, arch={self.net_arch}, {state})"
