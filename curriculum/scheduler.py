"""
curriculum/scheduler.py
=======================
Curriculum Learning Scheduler -- MDMARD Phase 1

Manages progressive task difficulty scheduling:
  - Loads stage definitions from YAML
  - Tracks rolling success rate per stage
  - Advances to the next stage when threshold + min_episodes are met
  - Provides env configuration overrides per stage
  - Integrates with ExperimentManager for stage-transition logging
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Union

import yaml

from curriculum.stages import CurriculumStage


class CurriculumScheduler:
    """Progressive curriculum scheduler driven by rolling success rate.

    Parameters
    ----------
    config_path : str or Path
        Path to curriculum YAML configuration file.
    experiment_manager : optional
        ExperimentManager instance for logging stage transitions.
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        experiment_manager: Optional[Any] = None,
    ) -> None:
        self._config_path = Path(config_path)
        self.experiment_manager = experiment_manager
        self.stages: List[CurriculumStage] = []
        self._current_idx: int = 0
        self._episode_outcomes: Deque[bool] = deque()
        self._episodes_at_stage: int = 0
        self._global_success_threshold: float = 0.80
        self._global_eval_episodes: int = 25

        self._load_config(self._config_path)

    # ------------------------------------------------------------------
    # Config Loading
    # ------------------------------------------------------------------

    def _load_config(self, path: Path) -> None:
        """Parse YAML and build CurriculumStage list."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        curriculum = raw.get("curriculum", raw)
        self._global_success_threshold = float(
            curriculum.get("success_rate_threshold", 0.80)
        )
        self._global_eval_episodes = int(curriculum.get("eval_episodes", 25))

        stage_dicts = curriculum.get("stages", [])
        for sd in stage_dicts:
            # Fill in global threshold / min_episodes if not overridden per-stage
            if "success_threshold" not in sd:
                sd["success_threshold"] = self._global_success_threshold
            if "min_episodes" not in sd:
                sd["min_episodes"] = self._global_eval_episodes

            self.stages.append(CurriculumStage.from_dict(sd))

        if not self.stages:
            raise ValueError(f"No stages found in curriculum config: {path}")

    # ------------------------------------------------------------------
    # Current Stage Access
    # ------------------------------------------------------------------

    @property
    def current_stage(self) -> CurriculumStage:
        """Active curriculum stage."""
        return self.stages[self._current_idx]

    @property
    def current_stage_index(self) -> int:
        """Zero-based index of the active stage."""
        return self._current_idx

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @property
    def is_final_stage(self) -> bool:
        return self._current_idx >= len(self.stages) - 1

    def get_stage(self, idx: int) -> CurriculumStage:
        """Return stage by zero-based index."""
        return self.stages[idx]

    # ------------------------------------------------------------------
    # Episode Outcome Tracking
    # ------------------------------------------------------------------

    def record_episode(self, success: bool) -> None:
        """Record the outcome of one episode.

        Parameters
        ----------
        success : bool
            True if the episode was a goal-reaching success.
        """
        self._episode_outcomes.append(success)
        self._episodes_at_stage += 1

    def get_success_rate(self) -> float:
        """Rolling success rate over all recorded outcomes at current stage."""
        if not self._episode_outcomes:
            return 0.0
        return float(sum(self._episode_outcomes)) / len(self._episode_outcomes)

    def update(self, success_rate: Optional[float] = None) -> None:
        """Convenience: update the tracked success rate directly.

        Use this when computing success rate externally (e.g. from MetricsTracker).

        Parameters
        ----------
        success_rate : float, optional
            Externally computed rolling success rate [0, 1].
            If None, re-computed from internal episode_outcomes buffer.
        """
        if success_rate is not None:
            # Reconstruct buffer from rate (approximation)
            n = max(1, self._episodes_at_stage)
            successes = round(success_rate * n)
            self._episode_outcomes = deque(
                [True] * successes + [False] * (n - successes)
            )

    # ------------------------------------------------------------------
    # Stage Advancement
    # ------------------------------------------------------------------

    def should_advance(self) -> bool:
        """Return True if conditions are met to advance to the next stage."""
        if self.is_final_stage:
            return False

        stage = self.current_stage
        min_eps = getattr(stage, "min_episodes", self._global_eval_episodes)
        threshold = getattr(stage, "success_threshold", self._global_success_threshold)

        met_min_episodes = self._episodes_at_stage >= min_eps
        met_success_rate = self.get_success_rate() >= threshold

        return met_min_episodes and met_success_rate

    def advance(self) -> CurriculumStage:
        """Advance to the next curriculum stage and reset tracking counters.

        Returns
        -------
        CurriculumStage
            The new (advanced-to) stage.

        Raises
        ------
        StopIteration
            If already at the final stage.
        """
        if self.is_final_stage:
            raise StopIteration("Already at the final curriculum stage.")

        prev = self.current_stage
        self._current_idx += 1
        self._episode_outcomes.clear()
        self._episodes_at_stage = 0
        new_stage = self.current_stage

        if self.experiment_manager is not None:
            self.experiment_manager.log_metrics(
                {
                    "curriculum_stage": new_stage.stage_id,
                    "prev_stage_success_rate": self.get_success_rate(),
                },
                step=0,
            )

        return new_stage

    def try_advance(self) -> bool:
        """Advance if conditions are met. Returns True if advanced."""
        if self.should_advance():
            self.advance()
            return True
        return False

    # ------------------------------------------------------------------
    # Environment Configuration Overlay
    # ------------------------------------------------------------------

    def get_env_overrides(self, stage: Optional[CurriculumStage] = None) -> Dict[str, Any]:
        """Return environment config overrides for the given stage.

        Callers can apply these to their environment or config dict.

        Parameters
        ----------
        stage : CurriculumStage, optional
            Defaults to the current active stage.

        Returns
        -------
        dict
            Keys matching TacticalFlightEnv / World constructor parameters.
        """
        s = stage or self.current_stage
        overrides: Dict[str, Any] = {
            "goal_radius": s.goal_radius,
            "min_spawn_distance": s.min_distance,
            "max_spawn_distance": s.max_distance,
            "fuel_enabled": s.fuel_enabled,
            "obstacles_count": s.obstacles_count,
            "wind_speed_max": s.wind_speed_max,
        }
        # Include any extra stage-level keys
        overrides.update(s.extra)
        return overrides

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_path": str(self._config_path),
            "current_stage_index": self._current_idx,
            "episodes_at_stage": self._episodes_at_stage,
            "success_rate": self.get_success_rate(),
            "stages": [s.to_dict() for s in self.stages],
        }

    def save_state(self, path: Union[str, Path]) -> Path:
        """Save scheduler state to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return p

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a one-line human-readable status string."""
        s = self.current_stage
        return (
            f"Stage {self._current_idx + 1}/{len(self.stages)} "
            f"[{s.name}] | "
            f"eps@stage={self._episodes_at_stage} | "
            f"success_rate={self.get_success_rate():.2%} | "
            f"threshold={s.success_threshold:.2%}"
        )

    def __repr__(self) -> str:
        return (
            f"CurriculumScheduler("
            f"stage={self._current_idx + 1}/{len(self.stages)}, "
            f"success_rate={self.get_success_rate():.2%})"
        )
