"""
experiments package -- MDMARD Phase 1
======================================
Provides experiment lifecycle, run folder management, checkpointing,
metadata tracking, and artifact persistence.
"""

from experiments.manager import (
    ExperimentManager,
    ExperimentMetadata,
    ExperimentStatus,
)

__all__ = [
    "ExperimentManager",
    "ExperimentMetadata",
    "ExperimentStatus",
]
