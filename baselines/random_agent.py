"""
baselines/random_agent.py
=========================
Random Baseline Agent -- MDMARD Phase 1

Selects uniformly random actions from the action space (or biased random if configured).
Provides minimum performance floor for RL benchmarking.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from baselines.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Uniform random baseline agent."""

    def __init__(
        self,
        num_actions: int = 6,
        action_probabilities: Optional[Sequence[float]] = None,
        seed: Optional[int] = None,
        name: str = "Random",
    ) -> None:
        super().__init__(name=name)
        self.num_actions = num_actions
        self.probs = None
        if action_probabilities is not None:
            p = np.array(action_probabilities, dtype=np.float64)
            self.probs = p / np.sum(p)
        self.rng = np.random.RandomState(seed)

    def seed(self, seed: Optional[int]) -> None:
        """Seed agent RNG."""
        self.rng = np.random.RandomState(seed)

    def act(
        self,
        observation: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Select action randomly."""
        if self.probs is not None:
            return int(self.rng.choice(self.num_actions, p=self.probs))
        return int(self.rng.randint(0, self.num_actions))
