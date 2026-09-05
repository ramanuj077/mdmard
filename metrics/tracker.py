"""
metrics/tracker.py
==================
Research Metrics Tracker -- MDMARD Phase 1

Tracks, aggregates, and computes statistical and flight-mechanics metrics:
  - Episode outcomes: success rate, crash rate, timeout rate, termination reasons
  - Flight telemetry: path efficiency, min approach distance, speed profile, fuel efficiency
  - Control smoothness: angular velocity penalties / jerk metrics
  - Reward component breakdowns across training / evaluation
  - Sliding window statistics (rolling mean, std, quantiles) and global summaries
  - Export utilities: JSON Lines, CSV, Pandas DataFrame
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class EpisodeMetrics:
    """Detailed telemetry and performance metrics for a single episode."""
    episode_id: int
    total_reward: float
    steps: int
    duration_sec: float
    termination_reason: str
    success: bool
    crashed: bool
    reward_breakdown: Dict[str, float] = field(default_factory=dict)
    initial_distance: float = 0.0
    final_distance: float = 0.0
    min_distance: float = 0.0
    distance_closed: float = 0.0
    avg_speed: float = 0.0
    max_speed: float = 0.0
    min_speed: float = 0.0
    avg_altitude: float = 0.0
    fuel_consumed: float = 0.0
    fuel_efficiency: float = 0.0
    smoothness: float = 0.0
    path_length: float = 0.0
    path_efficiency: float = 0.0
    action_counts: Dict[int, int] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EpisodeMetrics:
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


class MetricsTracker:
    """Aggregates step-level and episode-level metrics across an experiment."""

    def __init__(self, window_sizes: Tuple[int, ...] = (10, 50, 100)) -> None:
        self.window_sizes = sorted(window_sizes)
        self.episodes: List[EpisodeMetrics] = []
        self.total_steps: int = 0

        # In-flight episode state tracker
        self._active_episode_id: Optional[int] = None
        self._ep_start_time: float = 0.0
        self._ep_rewards: List[float] = []
        self._ep_breakdowns: List[Dict[str, float]] = []
        self._ep_positions: List[Tuple[float, float]] = []
        self._ep_speeds: List[float] = []
        self._ep_altitudes: List[float] = []
        self._ep_omegas: List[float] = []
        self._ep_fuels: List[float] = []
        self._ep_actions: List[int] = []
        self._target_pos: Optional[Tuple[float, float]] = None
        self._initial_dist: float = 0.0

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def start_episode(
        self,
        episode_id: int,
        initial_state: Any,
        target_pos: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Start tracking a new in-progress episode."""
        self._active_episode_id = episode_id
        self._ep_start_time = time.time()
        self._ep_rewards = []
        self._ep_breakdowns = []
        self._ep_positions = [(float(initial_state.x), float(initial_state.y))]
        init_speed = math.hypot(float(initial_state.vx), float(initial_state.vy))
        self._ep_speeds = [init_speed]
        self._ep_altitudes = [float(initial_state.y)]
        self._ep_omegas = [float(getattr(initial_state, "omega", 0.0))]
        self._ep_fuels = [float(getattr(initial_state, "fuel", 100.0))]
        self._ep_actions = []

        if target_pos is not None:
            self._target_pos = (float(target_pos[0]), float(target_pos[1]))
            self._initial_dist = math.hypot(
                float(initial_state.x) - self._target_pos[0],
                float(initial_state.y) - self._target_pos[1],
            )
        else:
            self._target_pos = None
            self._initial_dist = 0.0

    def step(
        self,
        state: Any,
        action: int,
        reward: float,
        reward_breakdown: Optional[Dict[str, float]] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record telemetry from a single environment step."""
        self.total_steps += 1
        self._ep_rewards.append(float(reward))
        self._ep_breakdowns.append(dict(reward_breakdown) if reward_breakdown else {})
        self._ep_actions.append(int(action))

        x, y = float(state.x), float(state.y)
        self._ep_positions.append((x, y))
        speed = math.hypot(float(state.vx), float(state.vy))
        self._ep_speeds.append(speed)
        self._ep_altitudes.append(y)
        self._ep_omegas.append(float(getattr(state, "omega", 0.0)))
        self._ep_fuels.append(float(getattr(state, "fuel", 0.0)))

    def end_episode(
        self,
        termination_reason: str,
        info: Optional[Dict[str, Any]] = None,
    ) -> EpisodeMetrics:
        """Finalize the current episode, compute telemetry metrics, and record it."""
        duration = max(0.001, time.time() - self._ep_start_time)
        steps = len(self._ep_rewards)
        total_reward = sum(self._ep_rewards)

        # Success & Crash logic
        success = (termination_reason in ("goal", "waypoint_reached", "success"))
        crashed = (termination_reason in ("oob", "stall", "obstacle", "fuel", "crash"))

        # Reward breakdown cumulative sum
        cumulative_breakdown: Dict[str, float] = {}
        for b in self._ep_breakdowns:
            for k, v in b.items():
                cumulative_breakdown[k] = cumulative_breakdown.get(k, 0.0) + float(v)

        # Distances
        if self._target_pos is not None and self._ep_positions:
            tx, ty = self._target_pos
            distances = [math.hypot(px - tx, py - ty) for px, py in self._ep_positions]
            init_dist = distances[0]
            final_dist = distances[-1]
            min_dist = min(distances)
            dist_closed = max(0.0, init_dist - final_dist)
        else:
            init_dist = final_dist = min_dist = dist_closed = 0.0

        # Speeds & Altitudes
        avg_speed = float(np.mean(self._ep_speeds)) if self._ep_speeds else 0.0
        max_speed = float(np.max(self._ep_speeds)) if self._ep_speeds else 0.0
        min_speed = float(np.min(self._ep_speeds)) if self._ep_speeds else 0.0
        avg_altitude = float(np.mean(self._ep_altitudes)) if self._ep_altitudes else 0.0

        # Fuel
        if len(self._ep_fuels) >= 2:
            fuel_consumed = max(0.0, self._ep_fuels[0] - self._ep_fuels[-1])
        else:
            fuel_consumed = 0.0
        fuel_efficiency = dist_closed / fuel_consumed if fuel_consumed > 1e-4 else 0.0

        # Smoothness (mean absolute omega or omega^2 integral)
        smoothness = float(np.mean([abs(w) for w in self._ep_omegas])) if self._ep_omegas else 0.0

        # Path Length & Efficiency
        path_length = 0.0
        for i in range(1, len(self._ep_positions)):
            p0, p1 = self._ep_positions[i - 1], self._ep_positions[i]
            path_length += math.hypot(p1[0] - p0[0], p1[1] - p0[1])

        if path_length > 1e-4 and self._target_pos is not None:
            # Straight-line distance from start to target
            straight_line = self._initial_dist
            path_efficiency = min(1.0, straight_line / path_length) if success else (dist_closed / path_length)
        else:
            path_efficiency = 0.0

        # Action counts
        action_counts: Dict[int, int] = {}
        for a in self._ep_actions:
            action_counts[a] = action_counts.get(a, 0) + 1

        ep_id = self._active_episode_id if self._active_episode_id is not None else len(self.episodes)

        metrics = EpisodeMetrics(
            episode_id=ep_id,
            total_reward=total_reward,
            steps=steps,
            duration_sec=duration,
            termination_reason=termination_reason,
            success=success,
            crashed=crashed,
            reward_breakdown=cumulative_breakdown,
            initial_distance=init_dist,
            final_distance=final_dist,
            min_distance=min_dist,
            distance_closed=dist_closed,
            avg_speed=avg_speed,
            max_speed=max_speed,
            min_speed=min_speed,
            avg_altitude=avg_altitude,
            fuel_consumed=fuel_consumed,
            fuel_efficiency=fuel_efficiency,
            smoothness=smoothness,
            path_length=path_length,
            path_efficiency=path_efficiency,
            action_counts=action_counts,
        )

        self.record_episode(metrics)
        self._active_episode_id = None
        return metrics

    def record_episode(self, metrics: EpisodeMetrics) -> None:
        """Directly append a pre-computed EpisodeMetrics object."""
        self.episodes.append(metrics)

    def get_window_stats(self, window_size: int = 100) -> Dict[str, Any]:
        """Compute summary statistics over the last `window_size` episodes."""
        if not self.episodes:
            return {
                "window_size": 0,
                "episodes_count": 0,
                "success_rate": 0.0,
                "crash_rate": 0.0,
                "timeout_rate": 0.0,
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
                "mean_steps": 0.0,
                "mean_fuel_consumed": 0.0,
                "mean_min_distance": 0.0,
                "mean_path_efficiency": 0.0,
                "mean_smoothness": 0.0,
                "component_means": {},
                "termination_counts": {},
            }

        window = self.episodes[-window_size:]
        n = len(window)

        rewards = [ep.total_reward for ep in window]
        steps = [ep.steps for ep in window]
        successes = [1.0 if ep.success else 0.0 for ep in window]
        crashes = [1.0 if ep.crashed else 0.0 for ep in window]
        timeouts = [1.0 if ep.termination_reason == "timeout" else 0.0 for ep in window]
        fuels = [ep.fuel_consumed for ep in window]
        min_dists = [ep.min_distance for ep in window]
        path_effs = [ep.path_efficiency for ep in window]
        smoothness_vals = [ep.smoothness for ep in window]

        # Component means
        component_sums: Dict[str, float] = {}
        for ep in window:
            for k, v in ep.reward_breakdown.items():
                component_sums[k] = component_sums.get(k, 0.0) + v
        component_means = {k: v / n for k, v in component_sums.items()}

        # Termination counts
        term_counts: Dict[str, int] = {}
        for ep in window:
            r = ep.termination_reason
            term_counts[r] = term_counts.get(r, 0) + 1

        return {
            "window_size": window_size,
            "episodes_count": n,
            "success_rate": float(np.mean(successes)),
            "crash_rate": float(np.mean(crashes)),
            "timeout_rate": float(np.mean(timeouts)),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "median_reward": float(np.median(rewards)),
            "mean_steps": float(np.mean(steps)),
            "mean_fuel_consumed": float(np.mean(fuels)),
            "mean_min_distance": float(np.mean(min_dists)),
            "mean_path_efficiency": float(np.mean(path_effs)),
            "mean_smoothness": float(np.mean(smoothness_vals)),
            "component_means": component_means,
            "termination_counts": term_counts,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Compute comprehensive lifetime summary across all recorded episodes."""
        return {
            "total_episodes": len(self.episodes),
            "total_steps": self.total_steps,
            "all_time": self.get_window_stats(window_size=max(1, len(self.episodes))),
            "last_10": self.get_window_stats(window_size=10),
            "last_50": self.get_window_stats(window_size=50),
            "last_100": self.get_window_stats(window_size=100),
        }

    def to_list(self) -> List[Dict[str, Any]]:
        """Return all recorded episodes as list of dictionaries."""
        return [ep.to_dict() for ep in self.episodes]

    def to_dataframe(self) -> Any:
        """Convert recorded episodes to a pandas DataFrame if pandas is installed."""
        try:
            import pandas as pd
            return pd.DataFrame(self.to_list())
        except ImportError:
            return self.to_list()

    def export_jsonl(self, filepath: Union[str, Path]) -> Path:
        """Export all episodes to a JSON Lines file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for ep in self.episodes:
                f.write(json.dumps(ep.to_dict()) + "\n")
        return p

    def export_csv(self, filepath: Union[str, Path]) -> Path:
        """Export flat summary of episodes to CSV."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)

        if not self.episodes:
            p.touch()
            return p

        # Extract flat headers (flattening reward breakdown keys)
        all_breakdown_keys = set()
        for ep in self.episodes:
            all_breakdown_keys.update(ep.reward_breakdown.keys())
        sorted_breakdown_keys = sorted(all_breakdown_keys)

        fieldnames = [
            "episode_id", "total_reward", "steps", "duration_sec", "termination_reason",
            "success", "crashed", "initial_distance", "final_distance", "min_distance",
            "distance_closed", "avg_speed", "max_speed", "min_speed", "avg_altitude",
            "fuel_consumed", "fuel_efficiency", "smoothness", "path_length", "path_efficiency",
        ] + [f"rew_{k}" for k in sorted_breakdown_keys]

        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for ep in self.episodes:
                row = {
                    "episode_id": ep.episode_id,
                    "total_reward": ep.total_reward,
                    "steps": ep.steps,
                    "duration_sec": ep.duration_sec,
                    "termination_reason": ep.termination_reason,
                    "success": ep.success,
                    "crashed": ep.crashed,
                    "initial_distance": ep.initial_distance,
                    "final_distance": ep.final_distance,
                    "min_distance": ep.min_distance,
                    "distance_closed": ep.distance_closed,
                    "avg_speed": ep.avg_speed,
                    "max_speed": ep.max_speed,
                    "min_speed": ep.min_speed,
                    "avg_altitude": ep.avg_altitude,
                    "fuel_consumed": ep.fuel_consumed,
                    "fuel_efficiency": ep.fuel_efficiency,
                    "smoothness": ep.smoothness,
                    "path_length": ep.path_length,
                    "path_efficiency": ep.path_efficiency,
                }
                for k in sorted_breakdown_keys:
                    row[f"rew_{k}"] = ep.reward_breakdown.get(k, 0.0)
                writer.writerow(row)

        return p
