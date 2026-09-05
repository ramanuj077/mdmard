"""
Serialization formats and schema definitions for the Replay System.

Provides Episode/Step dataclasses and JSON/Gzip serialization routines
to record, store, and reload simulation trajectories.
"""

import gzip
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union


@dataclass
class RecordedStep:
    """Single step trajectory data."""

    step: int
    state_dict: Dict[str, float]
    action: int
    reward: float
    reward_breakdown: Dict[str, float]
    obs: List[float]
    terminated: bool
    truncated: bool
    info: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecordedStep":
        return cls(
            step=int(data["step"]),
            state_dict={k: float(v) for k, v in data["state_dict"].items()},
            action=int(data["action"]),
            reward=float(data["reward"]),
            reward_breakdown={k: float(v) for k, v in data["reward_breakdown"].items()},
            obs=[float(x) for x in data["obs"]],
            terminated=bool(data["terminated"]),
            truncated=bool(data["truncated"]),
            info=dict(data.get("info", {})),
        )


@dataclass
class RecordedEpisode:
    """Full episode trajectory and metadata container."""

    episode_id: str
    timestamp: str
    total_reward: float
    total_steps: int
    termination_reason: str
    world_config: Dict[str, Any]
    steps: List[RecordedStep] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "total_reward": self.total_reward,
            "total_steps": self.total_steps,
            "termination_reason": self.termination_reason,
            "world_config": self.world_config,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecordedEpisode":
        return cls(
            episode_id=str(data["episode_id"]),
            timestamp=str(data["timestamp"]),
            total_reward=float(data["total_reward"]),
            total_steps=int(data["total_steps"]),
            termination_reason=str(data["termination_reason"]),
            world_config=dict(data["world_config"]),
            steps=[RecordedStep.from_dict(s) for s in data.get("steps", [])],
        )


def save_episode(episode: RecordedEpisode, filepath: Union[str, Path]) -> Path:
    """
    Save RecordedEpisode to disk (.json or .json.gz).

    Args:
        episode: RecordedEpisode instance.
        filepath: Target save path. If ends with .gz, compresses with gzip.

    Returns:
        Path object pointing to saved file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    data_dict = episode.to_dict()

    if path.suffix == ".gz" or str(path).endswith(".json.gz"):
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=2)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=2)

    return path


def load_episode(filepath: Union[str, Path]) -> RecordedEpisode:
    """
    Load RecordedEpisode from disk (.json or .json.gz).

    Args:
        filepath: Path to saved episode file.

    Returns:
        RecordedEpisode instance.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Replay file not found: {path}")

    if path.suffix == ".gz" or str(path).endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data_dict = json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data_dict = json.load(f)

    return RecordedEpisode.from_dict(data_dict)
