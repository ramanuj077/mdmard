"""
utils/seeding.py
================
Global Deterministic Seeding Utility -- MDMARD Phase 1

Ensures 100% reproducible training and evaluation runs by seeding:
  - Python built-in `random`
  - NumPy `np.random`
  - PyTorch `torch.manual_seed` & `torch.cuda.manual_seed_all` (if available)
  - Optional Gymnasium environment
"""

from __future__ import annotations

import os
import random
from typing import Any, Optional

import numpy as np


def seed_everything(seed: int = 42, env: Optional[Any] = None) -> int:
    """Seed all pseudo-random number generators across the workspace.

    Parameters
    ----------
    seed : int
        The integer seed to apply.
    env : Optional[gym.Env]
        Optional Gymnasium environment to reset/seed.

    Returns
    -------
    int
        The applied seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import sys
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        torch.manual_seed(seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if env is not None:
        try:
            env.reset(seed=seed)
            if hasattr(env.action_space, "seed"):
                env.action_space.seed(seed)
            if hasattr(env.observation_space, "seed"):
                env.observation_space.seed(seed)
        except Exception:
            pass

    return seed
