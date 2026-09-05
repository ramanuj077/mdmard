"""
metrics/reporter.py
===================
Research Metrics Reporter & Visualization -- MDMARD Phase 1

Generates publication-ready reports, console summaries, LaTeX comparison tables,
and multi-panel training plots for baseline comparison and experiment analysis.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from metrics.tracker import EpisodeMetrics, MetricsTracker


class MetricsReporter:
    """Formats and visualizes metrics collected by MetricsTracker."""

    @staticmethod
    def format_episode_banner(ep: EpisodeMetrics) -> str:
        """Return a single-line or compact banner summarizing an episode outcome."""
        status_icon = "🎯 SUCCESS" if ep.success else ("💥 CRASH" if ep.crashed else "⏱ TIMEOUT")
        return (
            f"[{status_icon}] Ep {ep.episode_id:04d} | "
            f"Reward: {ep.total_reward:+7.1f} | "
            f"Steps: {ep.steps:4d} | "
            f"Reason: {ep.termination_reason:<12} | "
            f"MinDist: {ep.min_distance:5.1f}m | "
            f"Fuel: {ep.fuel_consumed:4.1f}% | "
            f"PathEff: {ep.path_efficiency * 100:4.1f}%"
        )

    @staticmethod
    def format_window_table(
        stats: Dict[str, Any],
        ep_num: int,
        total_steps: int,
    ) -> str:
        """Generate a formatted console table summarizing rolling window performance."""
        win_size = stats.get("window_size", 100)
        succ = stats.get("success_rate", 0.0) * 100
        crash = stats.get("crash_rate", 0.0) * 100
        tout = stats.get("timeout_rate", 0.0) * 100
        m_rew = stats.get("mean_reward", 0.0)
        s_rew = stats.get("std_reward", 0.0)
        m_steps = stats.get("mean_steps", 0.0)
        m_fuel = stats.get("mean_fuel_consumed", 0.0)
        m_dist = stats.get("mean_min_distance", 0.0)
        m_eff = stats.get("mean_path_efficiency", 0.0) * 100

        lines = [
            "================================================================================",
            f"  EXPERIMENT TELEMETRY | Episode {ep_num:04d} | Total Steps: {total_steps:7,d} | Window: Last {win_size}",
            "--------------------------------------------------------------------------------",
            f"  Reward (Mean ± Std) : {m_rew:+8.2f} ± {s_rew:6.2f}   | Success Rate : {succ:5.1f}%",
            f"  Episode Steps (Mean): {m_steps:8.1f}            | Crash Rate   : {crash:5.1f}%",
            f"  Min Distance (Mean) : {m_dist:8.1f} m          | Timeout Rate : {tout:5.1f}%",
            f"  Fuel Consumed (Mean): {m_fuel:8.1f} %          | Path Effic.  : {m_eff:5.1f}%",
            "--------------------------------------------------------------------------------",
        ]

        # Reward components breakdown if present
        comp_means = stats.get("component_means", {})
        if comp_means:
            comp_str = " | ".join(f"{k}: {v:+6.2f}" for k, v in sorted(comp_means.items()))
            lines.append(f"  Rewards Breakdown   : {comp_str}")
            lines.append("--------------------------------------------------------------------------------")

        lines.append("================================================================================")
        return "\n".join(lines)

    @staticmethod
    def format_comparison_table(
        results_by_agent: Dict[str, Dict[str, float]],
        metrics_to_show: Optional[List[str]] = None,
    ) -> str:
        """Format an ASCII comparison table across multiple agents/baselines."""
        if not results_by_agent:
            return "No results to display."

        default_metrics = [
            ("mean_reward", "Reward (Mean)"),
            ("success_rate", "Success (%)"),
            ("crash_rate", "Crash (%)"),
            ("mean_steps", "Steps (Mean)"),
            ("mean_fuel_consumed", "Fuel (%)"),
            ("mean_path_efficiency", "Path Eff (%)"),
        ]

        if metrics_to_show is None:
            active_metrics = default_metrics
        else:
            active_metrics = [(m, m.replace("_", " ").title()) for m in metrics_to_show]

        headers = ["Agent / Policy"] + [label for _, label in active_metrics]
        rows = []

        for agent_name, agent_metrics in results_by_agent.items():
            row = [agent_name]
            for key, label in active_metrics:
                val = agent_metrics.get(key, 0.0)
                if "%" in label or "rate" in key.lower() or "efficiency" in key.lower():
                    disp_val = val * 100 if val <= 1.0 and val >= 0.0 else val
                    row.append(f"{disp_val:6.1f}%")
                elif "reward" in key.lower():
                    row.append(f"{val:+7.2f}")
                else:
                    row.append(f"{val:7.2f}")
            rows.append(row)

        # Formatting with column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        def make_row(cells: List[str]) -> str:
            return " | ".join(cell.ljust(w) for cell, w in zip(cells, col_widths))

        sep_line = "-+-".join("-" * w for w in col_widths)

        table_lines = [
            make_row(headers),
            sep_line,
        ]
        for r in rows:
            table_lines.append(make_row(r))

        return "\n".join(table_lines)

    @staticmethod
    def generate_latex_table(
        results_by_agent: Dict[str, Dict[str, float]],
        caption: str = "Performance Benchmark Across Tactical Flight Baselines",
        label: str = "tab:flight_baselines",
    ) -> str:
        """Generate LaTeX table code for scientific publication."""
        metrics_keys = [
            ("mean_reward", "Mean Reward"),
            ("success_rate", "Success Rate (\\%)"),
            ("crash_rate", "Crash Rate (\\%)"),
            ("mean_steps", "Steps"),
            ("mean_fuel_consumed", "Fuel (\\%)"),
            ("mean_path_efficiency", "Path Eff. (\\%)"),
        ]

        num_cols = len(metrics_keys) + 1
        col_align = "l" + "r" * len(metrics_keys)

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{col_align}}}",
            "\\toprule",
            "Policy & " + " & ".join(label for _, label in metrics_keys) + " \\\\",
            "\\midrule",
        ]

        for agent_name, agent_metrics in results_by_agent.items():
            row_items = [f"\\textbf{{{agent_name}}}"]
            for key, _ in metrics_keys:
                val = agent_metrics.get(key, 0.0)
                if "rate" in key or "efficiency" in key or "fuel" in key:
                    disp_val = val * 100 if val <= 1.0 and val >= 0.0 else val
                    row_items.append(f"{disp_val:.1f}")
                elif "reward" in key:
                    row_items.append(f"{val:+.1f}")
                else:
                    row_items.append(f"{val:.1f}")
            lines.append(" & ".join(row_items) + " \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])
        return "\n".join(lines)

    @staticmethod
    def generate_markdown_report(
        tracker: MetricsTracker,
        title: str = "MDMARD Phase 1 — Flight Simulation Report",
        experiment_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a complete GitHub Markdown evaluation / training report."""
        summary = tracker.get_summary()
        all_time = summary["all_time"]
        last_100 = summary["last_100"]

        lines = [
            f"# {title}",
            "",
            "> Automated RL Research Telemetry & Flight Evaluation Report",
            "",
            "## 1. Executive Summary",
            "",
            f"- **Total Episodes**: `{tracker.num_episodes}`",
            f"- **Total Environment Steps**: `{tracker.total_steps:,}`",
            f"- **Overall Success Rate**: `{all_time.get('success_rate', 0.0) * 100:.1f}%` (Last 100: `{last_100.get('success_rate', 0.0) * 100:.1f}%`)",
            f"- **Overall Crash Rate**: `{all_time.get('crash_rate', 0.0) * 100:.1f}%` (Last 100: `{last_100.get('crash_rate', 0.0) * 100:.1f}%`)",
            f"- **Mean Episode Reward**: `{all_time.get('mean_reward', 0.0):+.2f} ± {all_time.get('std_reward', 0.0):.2f}`",
            f"- **Mean Path Efficiency**: `{all_time.get('mean_path_efficiency', 0.0) * 100:.1f}%`",
            "",
            "## 2. Window Performance Breakdown",
            "",
            "| Window | Success Rate | Crash Rate | Mean Reward | Min Dist (m) | Fuel Consumed (%) | Path Efficiency |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for win_name in ["last_10", "last_50", "last_100", "all_time"]:
            w_stats = summary[win_name]
            if w_stats.get("episodes_count", 0) > 0:
                lines.append(
                    f"| **{win_name.replace('_', ' ').title()}** | "
                    f"`{w_stats.get('success_rate', 0.0) * 100:.1f}%` | "
                    f"`{w_stats.get('crash_rate', 0.0) * 100:.1f}%` | "
                    f"`{w_stats.get('mean_reward', 0.0):+.1f}` | "
                    f"`{w_stats.get('mean_min_distance', 0.0):.1f}` | "
                    f"`{w_stats.get('mean_fuel_consumed', 0.0):.1f}%` | "
                    f"`{w_stats.get('mean_path_efficiency', 0.0) * 100:.1f}%` |"
                )

        lines.extend([
            "",
            "## 3. Termination Breakdown",
            "",
            "| Reason | Total Count | Frequency |",
            "| :--- | :--- | :--- |",
        ])

        term_counts = all_time.get("termination_counts", {})
        total_eps = max(1, tracker.num_episodes)
        for reason, count in sorted(term_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| `{reason}` | {count} | `{count / total_eps * 100:.1f}%` |")

        if experiment_meta:
            lines.extend([
                "",
                "## 4. Experiment Metadata",
                "",
                "```json",
                f"{json.dumps(experiment_meta, indent=2)}",
                "```",
            ])

        return "\n".join(lines)

    @staticmethod
    def plot_learning_curves(
        tracker: MetricsTracker,
        save_path: Union[str, Path],
        window_size: int = 50,
    ) -> Optional[Path]:
        """Generate a 4-panel publication-ready learning curves plot using Matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        if not tracker.episodes:
            return None

        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        episodes = [ep.episode_id for ep in tracker.episodes]
        rewards = [ep.total_reward for ep in tracker.episodes]
        successes = [1.0 if ep.success else 0.0 for ep in tracker.episodes]
        crashes = [1.0 if ep.crashed else 0.0 for ep in tracker.episodes]
        steps = [ep.steps for ep in tracker.episodes]
        fuels = [ep.fuel_consumed for ep in tracker.episodes]
        min_dists = [ep.min_distance for ep in tracker.episodes]
        path_effs = [ep.path_efficiency * 100 for ep in tracker.episodes]

        def moving_avg(data: List[float], w: int) -> np.ndarray:
            if len(data) < w:
                return np.array(data)
            ret = np.convolve(data, np.ones(w) / w, mode="valid")
            pad = np.full(w - 1, ret[0])
            return np.concatenate([pad, ret])

        fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
        plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

        # 1. Episode Reward
        axs[0, 0].plot(episodes, rewards, alpha=0.3, color="#4A90E2", label="Raw Episode Reward")
        axs[0, 0].plot(episodes, moving_avg(rewards, window_size), color="#0055D4", linewidth=2.0, label=f"MA ({window_size} eps)")
        axs[0, 0].set_title("Episode Reward Curve", fontsize=12, fontweight="bold")
        axs[0, 0].set_xlabel("Episode")
        axs[0, 0].set_ylabel("Total Reward")
        axs[0, 0].legend(loc="upper left")
        axs[0, 0].grid(True, linestyle="--", alpha=0.6)

        # 2. Success & Crash Rates
        succ_ma = moving_avg(successes, window_size) * 100
        crash_ma = moving_avg(crashes, window_size) * 100
        axs[0, 1].plot(episodes, succ_ma, color="#2ECC71", linewidth=2.0, label=f"Success Rate ({window_size} eps)")
        axs[0, 1].plot(episodes, crash_ma, color="#E74C3C", linewidth=2.0, label=f"Crash Rate ({window_size} eps)")
        axs[0, 1].set_title("Success & Crash Rates (%)", fontsize=12, fontweight="bold")
        axs[0, 1].set_xlabel("Episode")
        axs[0, 1].set_ylabel("Rate (%)")
        axs[0, 1].set_ylim([-5, 105])
        axs[0, 1].legend(loc="upper left")
        axs[0, 1].grid(True, linestyle="--", alpha=0.6)

        # 3. Steps & Fuel
        axs[1, 0].plot(episodes, moving_avg(steps, window_size), color="#E67E22", linewidth=2.0, label=f"Mean Steps ({window_size} eps)")
        ax1_twin = axs[1, 0].twinx()
        ax1_twin.plot(episodes, moving_avg(fuels, window_size), color="#9B59B6", linewidth=1.8, linestyle="--", label=f"Fuel Consumed (%)")
        axs[1, 0].set_title("Episode Steps & Fuel Consumption", fontsize=12, fontweight="bold")
        axs[1, 0].set_xlabel("Episode")
        axs[1, 0].set_ylabel("Steps", color="#E67E22")
        ax1_twin.set_ylabel("Fuel Consumed (%)", color="#9B59B6")
        axs[1, 0].grid(True, linestyle="--", alpha=0.6)

        # 4. Waypoint Approach & Path Efficiency
        axs[1, 1].plot(episodes, moving_avg(min_dists, window_size), color="#34495E", linewidth=2.0, label=f"Min Dist to Target (m)")
        ax2_twin = axs[1, 1].twinx()
        ax2_twin.plot(episodes, moving_avg(path_effs, window_size), color="#1ABC9C", linewidth=1.8, linestyle="--", label="Path Efficiency (%)")
        axs[1, 1].set_title("Target Proximity & Path Efficiency", fontsize=12, fontweight="bold")
        axs[1, 1].set_xlabel("Episode")
        axs[1, 1].set_ylabel("Min Distance (m)", color="#34495E")
        ax2_twin.set_ylabel("Path Efficiency (%)", color="#1ABC9C")
        axs[1, 1].grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path
