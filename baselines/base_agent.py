"""
baselines/base_agent.py
=======================
Base Agent Abstract Interface -- MDMARD Phase 1

Provides standard interface for all non-RL and RL baseline agents:
  - act(obs, info) -> action
  - reset() -> clears agent internal states / memories
  - evaluate(env, num_episodes, seed, metrics_tracker, render) -> evaluation stats
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


class BaseAgent(ABC):
    """Abstract base class for all policy agents (Random, Greedy, PID, RL)."""

    def __init__(self, name: str = "BaseAgent") -> None:
        self.name = name

    @abstractmethod
    def act(
        self,
        observation: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Select an action given current observation vector.

        Parameters
        ----------
        observation : np.ndarray
            10-dimensional normalized observation vector.
        info : dict, optional
            Optional environment info dictionary.

        Returns
        -------
        int
            Discrete action index (0..5).
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset agent internal state (PID integral buffers, random state, etc.)."""
        pass

    def evaluate(
        self,
        env: Any,
        num_episodes: int = 20,
        seed: Optional[int] = 42,
        metrics_tracker: Optional[Any] = None,
        render: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate agent over multiple episodes in the given environment.

        Parameters
        ----------
        env : Gymnasium Environment
            The simulation environment instance.
        num_episodes : int
            Number of benchmark episodes.
        seed : int, optional
            Starting random seed.
        metrics_tracker : MetricsTracker, optional
            Optional tracker to record telemetry.
        render : bool
            Whether to render frames during evaluation.

        Returns
        -------
        dict
            Summary performance statistics.
        """
        episode_rewards = []
        episode_lengths = []
        successes = []
        crashes = []
        fuel_consumed_list = []

        for ep_i in range(num_episodes):
            ep_seed = (seed + ep_i) if seed is not None else None
            obs, info = env.reset(seed=ep_seed)
            self.reset()

            if metrics_tracker is not None and hasattr(env, "state") and hasattr(env, "world"):
                wp = env.world.current_waypoint
                target_pos = (wp.x, wp.y) if wp else None
                metrics_tracker.start_episode(
                    episode_id=ep_i,
                    initial_state=env.state,
                    target_pos=target_pos,
                )

            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                if render:
                    env.render()

                action = self.act(obs, info)
                next_obs, reward, terminated, truncated, step_info = env.step(action)

                total_reward += reward
                steps += 1

                if metrics_tracker is not None and hasattr(env, "state"):
                    breakdown = step_info.get("reward_breakdown", {})
                    metrics_tracker.step(
                        state=env.state,
                        action=action,
                        reward=reward,
                        reward_breakdown=breakdown,
                        info=step_info,
                    )

                obs = next_obs
                info = step_info

            term_reason = info.get("termination_reason", "unknown")
            is_success = (term_reason in ("goal", "waypoint_reached", "success"))
            is_crash = (term_reason in ("oob", "stall", "obstacle", "fuel", "crash"))

            if metrics_tracker is not None:
                ep_metrics = metrics_tracker.end_episode(
                    termination_reason=term_reason,
                    info=info,
                )
                fuel_consumed_list.append(ep_metrics.fuel_consumed)
            else:
                fuel_consumed_list.append(0.0)

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            successes.append(1.0 if is_success else 0.0)
            crashes.append(1.0 if is_crash else 0.0)

        results = {
            "agent_name": self.name,
            "num_episodes": num_episodes,
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "min_reward": float(np.min(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
            "success_rate": float(np.mean(successes)),
            "crash_rate": float(np.mean(crashes)),
            "mean_steps": float(np.mean(episode_lengths)),
            "mean_fuel_consumed": float(np.mean(fuel_consumed_list)),
        }
        return results
