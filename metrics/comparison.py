"""
metrics/comparison.py
=====================
Cross-Agent Comparison Engine -- MDMARD Phase 1

Aggregates evaluation results from multiple agents and produces:
  - Paper-ready comparison tables (ASCII + Markdown + LaTeX)
  - Matplotlib bar/radar charts comparing all agents
  - JSON export for programmatic consumption
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


# -----------------------------------------------------------------------------
# Data structure
# -----------------------------------------------------------------------------

class AgentResult:
    """Container for a single agent's evaluation summary."""

    COLUMNS = [
        "agent_name", "num_episodes",
        "success_rate", "crash_rate",
        "mean_reward", "std_reward",
        "mean_steps", "mean_fuel_consumed",
    ]

    def __init__(self, data: Dict[str, Any]) -> None:
        self.data = dict(data)
        self.agent_name: str = str(data.get("agent_name", "Unknown"))
        self.num_episodes: int = int(data.get("num_episodes", 0))
        self.success_rate: float = float(data.get("success_rate", 0.0))
        self.crash_rate: float = float(data.get("crash_rate", 0.0))
        self.mean_reward: float = float(data.get("mean_reward", 0.0))
        self.std_reward: float = float(data.get("std_reward", 0.0))
        self.mean_steps: float = float(data.get("mean_steps", 0.0))
        self.mean_fuel_consumed: float = float(data.get("mean_fuel_consumed", 0.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "num_episodes": self.num_episodes,
            "success_rate": self.success_rate,
            "crash_rate": self.crash_rate,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "mean_steps": self.mean_steps,
            "mean_fuel_consumed": self.mean_fuel_consumed,
        }

    def __repr__(self) -> str:
        return (
            f"AgentResult({self.agent_name}: "
            f"success={self.success_rate:.1%}, reward={self.mean_reward:.1f})"
        )


# -----------------------------------------------------------------------------
# Comparison engine
# -----------------------------------------------------------------------------

class BaselineComparison:
    """Aggregates multiple AgentResult objects and generates comparison outputs.

    Usage
    -----
    comp = BaselineComparison()
    comp.add_result(random_results)
    comp.add_result(pid_results)
    comp.add_result(ppo_results)

    print(comp.ascii_table())
    comp.save_markdown("reports/comparison.md")
    comp.plot_bar_chart("reports/comparison_bar.png")
    """

    def __init__(self) -> None:
        self._results: List[AgentResult] = []

    def add_result(self, result: Union[AgentResult, Dict[str, Any]]) -> None:
        """Add an agent result.

        Parameters
        ----------
        result : AgentResult or dict
            Evaluation statistics for one agent (output of BaseAgent.evaluate()).
        """
        if isinstance(result, dict):
            result = AgentResult(result)
        self._results.append(result)

    def add_results(self, results: List[Union[AgentResult, Dict[str, Any]]]) -> None:
        for r in results:
            self.add_result(r)

    def clear(self) -> None:
        self._results.clear()

    @property
    def results(self) -> List[AgentResult]:
        return list(self._results)

    @property
    def num_agents(self) -> int:
        return len(self._results)

    # ------------------------------------------------------------------
    # ASCII table
    # ------------------------------------------------------------------

    def ascii_table(self) -> str:
        """Render a bordered ASCII comparison table."""
        if not self._results:
            return "(No results to display)"

        headers = ["Agent", "Episodes", "Success%", "Crash%", "MeanRew", "StdRew", "Steps", "Fuel"]
        rows = []
        for r in self._results:
            rows.append([
                r.agent_name,
                str(r.num_episodes),
                f"{r.success_rate * 100:.1f}%",
                f"{r.crash_rate * 100:.1f}%",
                f"{r.mean_reward:.1f}",
                f"{r.std_reward:.1f}",
                f"{r.mean_steps:.0f}",
                f"{r.mean_fuel_consumed:.1f}",
            ])

        # Column widths
        col_w = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]

        sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
        header_row = "|" + "|".join(f" {h:<{w}} " for h, w in zip(headers, col_w)) + "|"

        lines = [sep, header_row, sep]
        for row in rows:
            lines.append("|" + "|".join(f" {c:<{w}} " for c, w in zip(row, col_w)) + "|")
        lines.append(sep)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Markdown table
    # ------------------------------------------------------------------

    def markdown_table(self) -> str:
        """Render a GitHub-Flavored Markdown comparison table."""
        if not self._results:
            return "_No results_"

        headers = ["Agent", "Episodes", "Success%", "Crash%", "MeanRew +/- Std", "Avg Steps", "Avg Fuel"]
        rows = []
        for r in self._results:
            rows.append([
                f"**{r.agent_name}**",
                str(r.num_episodes),
                f"{r.success_rate * 100:.1f}%",
                f"{r.crash_rate * 100:.1f}%",
                f"{r.mean_reward:.1f} +/- {r.std_reward:.1f}",
                f"{r.mean_steps:.0f}",
                f"{r.mean_fuel_consumed:.1f}",
            ])

        header_str = "| " + " | ".join(headers) + " |"
        sep_str = "|" + "|".join("---" for _ in headers) + "|"
        row_strs = ["| " + " | ".join(row) + " |" for row in rows]

        return "\n".join([header_str, sep_str] + row_strs)

    # ------------------------------------------------------------------
    # LaTeX table
    # ------------------------------------------------------------------

    def latex_table(self, caption: str = "Agent Comparison", label: str = "tab:comparison") -> str:
        """Render a LaTeX tabular comparison table."""
        if not self._results:
            return "% No results"

        lines = [
            r"\begin{table}[h!]",
            r"  \centering",
            r"  \begin{tabular}{lrrrrrrr}",
            r"    \hline",
            r"    Agent & Episodes & Success\% & Crash\% & Mean Rew & Std Rew & Steps & Fuel \\",
            r"    \hline",
        ]
        for r in self._results:
            lines.append(
                f"    {r.agent_name} & {r.num_episodes} & "
                f"{r.success_rate * 100:.1f} & {r.crash_rate * 100:.1f} & "
                f"{r.mean_reward:.1f} & {r.std_reward:.1f} & "
                f"{r.mean_steps:.0f} & {r.mean_fuel_consumed:.1f} \\\\"
            )
        lines += [
            r"    \hline",
            r"  \end{tabular}",
            f"  \\caption{{{caption}}}",
            f"  \\label{{{label}}}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_agents": self.num_agents,
            "results": [r.to_dict() for r in self._results],
        }

    def save_json(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return p

    def save_markdown(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        content = "# Agent Comparison Report\n\n" + self.markdown_table() + "\n"
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return p

    def save_latex(self, path: Union[str, Path], **kwargs) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.latex_table(**kwargs))
        return p

    # ------------------------------------------------------------------
    # Matplotlib plots
    # ------------------------------------------------------------------

    def plot_bar_chart(
        self,
        path: Union[str, Path],
        metric: str = "success_rate",
        title: Optional[str] = None,
        figsize=(10, 5),
    ) -> Path:
        """Save a grouped bar chart comparing agents on one metric.

        Parameters
        ----------
        path : str or Path
            Output file path (.png).
        metric : str
            One of: 'success_rate', 'crash_rate', 'mean_reward', 'mean_steps', 'mean_fuel_consumed'.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self._results:
            return Path(path)

        agents = [r.agent_name for r in self._results]
        values = [getattr(r, metric, 0.0) for r in self._results]

        colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db", "#9b59b6", "#1abc9c"]
        bar_colors = [colors[i % len(colors)] for i in range(len(agents))]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(agents, values, color=bar_colors, edgecolor="white", linewidth=1.2)

        for bar, val in zip(bars, values):
            label = f"{val:.1%}" if "rate" in metric else f"{val:.1f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                label,
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

        ax.set_title(title or f"Agent Comparison - {metric.replace('_', ' ').title()}", fontsize=14)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_ylim(0, max(values) * 1.15 if values else 1.0)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_facecolor("#f8f8f8")
        fig.patch.set_facecolor("#ffffff")

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return p

    def plot_all_metrics(
        self,
        output_dir: Union[str, Path],
        metrics: Optional[List[str]] = None,
    ) -> List[Path]:
        """Save bar charts for multiple metrics.

        Parameters
        ----------
        output_dir : str or Path
            Directory to save charts into.
        metrics : list of str, optional
            Defaults to ['success_rate', 'crash_rate', 'mean_reward', 'mean_steps', 'mean_fuel_consumed'].
        """
        if metrics is None:
            metrics = ["success_rate", "crash_rate", "mean_reward", "mean_steps", "mean_fuel_consumed"]

        out = Path(output_dir)
        saved = []
        for m in metrics:
            p = self.plot_bar_chart(out / f"comparison_{m}.png", metric=m)
            saved.append(p)
        return saved

    def plot_radar_chart(
        self,
        path: Union[str, Path],
        figsize=(7, 7),
    ) -> Path:
        """Radar / spider chart comparing normalised agent metrics.

        Metrics shown: success_rate, (1-crash_rate), mean_reward_norm, steps_efficiency, fuel_efficiency.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self._results:
            return Path(path)

        labels = ["Success Rate", "Safety\n(1-Crash)", "Reward", "Step Efficiency", "Fuel Efficiency"]
        num_vars = len(labels)
        angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
        angles += angles[:1]  # close polygon

        # Normalise metrics to [0, 1]
        def _norm(vals):
            mn, mx = min(vals), max(vals)
            if mx == mn:
                return [0.5] * len(vals)
            return [(v - mn) / (mx - mn) for v in vals]

        sr = [r.success_rate for r in self._results]
        safety = [1.0 - r.crash_rate for r in self._results]
        rew = [r.mean_reward for r in self._results]
        # Step efficiency: fewer steps = more efficient
        steps_inv = [-r.mean_steps for r in self._results]
        fuel_inv = [-r.mean_fuel_consumed for r in self._results]

        sr_n = _norm(sr)
        safe_n = _norm(safety)
        rew_n = _norm(rew)
        step_n = _norm(steps_inv)
        fuel_n = _norm(fuel_inv)

        colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db", "#9b59b6", "#1abc9c"]

        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})
        for i, r in enumerate(self._results):
            values_n = [sr_n[i], safe_n[i], rew_n[i], step_n[i], fuel_n[i]]
            values_n += values_n[:1]
            color = colors[i % len(colors)]
            ax.plot(angles, values_n, color=color, linewidth=2, label=r.agent_name)
            ax.fill(angles, values_n, color=color, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_yticklabels([])
        ax.set_title("Agent Performance Radar", size=14, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return p

    # ------------------------------------------------------------------
    # Best agent selection
    # ------------------------------------------------------------------

    def best_by(self, metric: str = "success_rate", maximize: bool = True) -> Optional[AgentResult]:
        """Return the best-performing agent on a given metric."""
        if not self._results:
            return None
        key = lambda r: getattr(r, metric, 0.0)
        return max(self._results, key=key) if maximize else min(self._results, key=key)

    def __repr__(self) -> str:
        agents = [r.agent_name for r in self._results]
        return f"BaselineComparison(agents={agents})"
