"""
utils/geometry.py
=================
2D Tactical Geometry Utilities -- MDMARD Phase 1

Pure mathematical and geometric helpers for angles, distances, headings,
and spatial containment checks in 2D simulation space.
"""

from __future__ import annotations

import math
from typing import Tuple, Union


def wrap_to_pi(angle_rad: float) -> float:
    """Normalize an angle in radians to the range [-pi, pi]."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Compute 2D Euclidean distance between two points."""
    return math.hypot(x2 - x1, y2 - y1)


def bearing_to_target(from_x: float, from_y: float, to_x: float, to_y: float) -> float:
    """Compute bearing angle from source point to target point in radians [-pi, pi]."""
    return math.atan2(to_y - from_y, to_x - from_x)


def clamp(val: float, min_val: float, max_val: float) -> float:
    """Clamp a scalar value within [min_val, max_val]."""
    return max(min_val, min(max_val, val))


def point_in_circle(px: float, py: float, cx: float, cy: float, radius: float) -> bool:
    """Check if point (px, py) is inside a circle at (cx, cy) with radius."""
    return math.hypot(px - cx, py - cy) <= radius
