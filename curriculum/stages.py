"""
curriculum/stages.py
====================
Curriculum Stage Definitions -- MDMARD Phase 1

Defines the CurriculumStage dataclass representing one stage of
progressive training difficulty, loadable from YAML configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


@dataclass
class CurriculumStage:
    """A single stage in the progressive curriculum schedule.

    Attributes
    ----------
    stage_id : int
        Unique identifier (1-indexed).
    name : str
        Human-readable stage name.
    description : str
        Optional description for logs / reports.
    goal_radius : float
        Waypoint capture radius in world units.
    min_distance : float
        Minimum initial distance to waypoint.
    max_distance : float
        Maximum initial distance to waypoint.
    fuel_enabled : bool
        Whether fuel depletion is active.
    obstacles_count : int
        Number of static obstacle zones.
    wind_speed_max : float
        Maximum wind gust speed (0.0 = no wind).
    success_threshold : float
        Rolling success rate required to advance to the next stage.
    min_episodes : int
        Minimum episodes at this stage before advancement is allowed.
    """

    stage_id: int
    name: str = ""
    description: str = ""
    goal_radius: float = 30.0
    min_distance: float = 200.0
    max_distance: float = 800.0
    fuel_enabled: bool = False
    obstacles_count: int = 0
    wind_speed_max: float = 0.0
    success_threshold: float = 0.80
    min_episodes: int = 100
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CurriculumStage":
        valid = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)

    def __repr__(self) -> str:
        return (
            f"CurriculumStage(id={self.stage_id}, name={self.name!r}, "
            f"goal_r={self.goal_radius}, fuel={self.fuel_enabled}, "
            f"obs={self.obstacles_count}, wind={self.wind_speed_max})"
        )
