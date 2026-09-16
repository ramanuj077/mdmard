"""
metrics package -- MDMARD Phase 1
=================================
Research metrics tracking, flight telemetry computation,
console summaries, publication-ready reports, and visualization.
"""

from metrics.tracker import (
    EpisodeMetrics,
    MetricsTracker,
)
from metrics.reporter import (
    MetricsReporter,
)
from metrics.comparison import (
    AgentResult,
    BaselineComparison,
)

__all__ = [
    "EpisodeMetrics",
    "MetricsTracker",
    "MetricsReporter",
    "AgentResult",
    "BaselineComparison",
]
