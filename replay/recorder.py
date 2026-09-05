"""
EpisodeRecorder: Records full simulation trajectory step-by-step.

Captures state, action, reward, reward breakdown, observation, and termination reason
during episode execution for later replay, analysis, and debugging.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from envs.physics import AircraftState
from envs.world import World
from replay.formats import RecordedEpisode, RecordedStep, save_episode


class EpisodeRecorder:
    """
    Records trajectory steps during environment execution.
    """

    def __init__(self, output_dir: Union[str, Path] = "logs/replays") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._active: bool = False
        self._episode_id: str = ""
        self._timestamp: str = ""
        self._world_config: Dict[str, Any] = {}
        self._steps: List[RecordedStep] = []
        self._total_reward: float = 0.0
        self._step_counter: int = 0

    @property
    def is_active(self) -> bool:
        return self._active

    def start_episode(
        self,
        world: World,
        episode_id: Optional[str] = None,
    ) -> None:
        """
        Initialize recording for a new episode.

        Args:
            world: Current World instance (to capture spawn/waypoint config).
            episode_id: Optional custom episode identifier.
        """
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._episode_id = episode_id or f"ep_{self._timestamp}"
        self._world_config = {
            "waypoint": {
                "x": float(world.waypoint.x),
                "y": float(world.waypoint.y),
                "radius": float(world.waypoint.radius),
            },
            "obstacles": [
                {"x": float(o.x), "y": float(o.y), "radius": float(o.radius)}
                for o in world.obstacles
            ],
            "width": float(world.cfg.width),
            "height": float(world.cfg.height),
        }
        self._steps = []
        self._total_reward = 0.0
        self._step_counter = 0
        self._active = True

    def record_step(
        self,
        state: AircraftState,
        action: int,
        reward: float,
        reward_breakdown: Dict[str, float],
        obs: Union[np.ndarray, List[float]],
        terminated: bool,
        truncated: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a single simulation step.
        """
        if not self._active:
            raise RuntimeError("Cannot record step: call start_episode() first.")

        self._step_counter += 1
        self._total_reward += reward

        obs_list = obs.tolist() if isinstance(obs, np.ndarray) else list(obs)

        step_data = RecordedStep(
            step=self._step_counter,
            state_dict={
                "x": float(state.x),
                "y": float(state.y),
                "vx": float(state.vx),
                "vy": float(state.vy),
                "theta": float(state.theta),
                "omega": float(state.omega),
                "fuel": float(state.fuel),
                "step": int(state.step),
            },
            action=int(action),
            reward=float(reward),
            reward_breakdown={k: float(v) for k, v in reward_breakdown.items()},
            obs=obs_list,
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info or {}),
        )
        self._steps.append(step_data)

    def end_episode(self, termination_reason: str = "unknown") -> RecordedEpisode:
        """
        Finalize recording and build RecordedEpisode object.

        Args:
            termination_reason: Description of termination (e.g., 'goal', 'crash').

        Returns:
            RecordedEpisode instance.
        """
        if not self._active:
            raise RuntimeError("Cannot end episode: no active episode recording.")

        episode = RecordedEpisode(
            episode_id=self._episode_id,
            timestamp=self._timestamp,
            total_reward=self._total_reward,
            total_steps=self._step_counter,
            termination_reason=termination_reason,
            world_config=self._world_config,
            steps=list(self._steps),
        )
        self._active = False
        return episode

    def save(
        self,
        episode: Optional[RecordedEpisode] = None,
        filepath: Optional[Union[str, Path]] = None,
        compress: bool = True,
    ) -> Path:
        """
        Save episode to file.

        Args:
            episode: RecordedEpisode instance. If None, uses last recorded episode.
            filepath: Optional explicit path. Defaults to output_dir/<episode_id>.json[.gz]
            compress: If True, saves as .json.gz

        Returns:
            Path to saved file.
        """
        if episode is None:
            if self._active:
                raise RuntimeError("Episode is still active. Call end_episode() first.")
            raise ValueError("No episode provided to save.")

        if filepath is None:
            ext = ".json.gz" if compress else ".json"
            filepath = self.output_dir / f"{episode.episode_id}{ext}"

        return save_episode(episode, filepath)
