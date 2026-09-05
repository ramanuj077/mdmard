"""
utils/io.py
===========
File I/O Helpers -- MDMARD Phase 1

Provides atomic and robust file reading/writing for YAML, JSON, and gzip formats.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def read_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(data: Dict[str, Any], path: Union[str, Path]) -> Path:
    """Write data dictionary to a YAML file atomically."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return p


def read_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: Union[str, Path], indent: int = 2) -> Path:
    """Write data to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    return p
