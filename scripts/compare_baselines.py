"""
scripts/compare_baselines.py
=============================
Baseline Comparison Runner -- MDMARD Phase 1

Runs Random, Greedy, and PID agents for N episodes each,
collects results, and dumps a comparison table + plots.

Usage:
  python scripts/compare_baselines.py --episodes 50 --output reports/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.flight_env import TacticalFlightEnv
from metrics.tracker import MetricsTracker
from metrics.comparison import BaselineComparison
from baselines.random_agent import RandomAgent
from baselines.greedy_agent import GreedyAgent
from baselines.pid_agent import PIDAgent


BASELINE_AGENTS = [
    ("Random", RandomAgent),
    ("Greedy", GreedyAgent),
    ("PID",    PIDAgent),
]


def run_comparison(args: argparse.Namespace) -> None:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison = BaselineComparison()

    for name, AgentClass in BASELINE_AGENTS:
        print(f"\n[compare] Evaluating {name} ({args.episodes} episodes) ...")
        env = TacticalFlightEnv()
        tracker = MetricsTracker()
        agent = AgentClass()

        results = agent.evaluate(
            env=env,
            num_episodes=args.episodes,
            seed=args.seed,
            metrics_tracker=tracker,
        )
        env.close()

        comparison.add_result(results)
        print(f"  success={results['success_rate']:.1%}  "
              f"reward={results['mean_reward']:.1f}  "
              f"crash={results['crash_rate']:.1%}")

    # ---- Print table ----
    print("\n" + comparison.ascii_table())

    # ---- Save outputs ----
    comparison.save_json(out_dir / "comparison.json")
    comparison.save_markdown(out_dir / "comparison.md")
    comparison.save_latex(out_dir / "comparison.tex")

    if args.plots:
        comparison.plot_all_metrics(out_dir)
        comparison.plot_radar_chart(out_dir / "radar.png")
        print(f"[compare] Plots saved to: {out_dir}")

    print(f"\n[compare] Reports saved to: {out_dir}")
    best = comparison.best_by("success_rate")
    if best:
        print(f"[compare] Best baseline: {best.agent_name}  "
              f"(success={best.success_rate:.1%})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MDMARD Phase 1 Baseline Comparison")
    p.add_argument("--episodes", type=int, default=50,
                   help="Evaluation episodes per agent (default: 50)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--output", default="reports",
                   help="Output directory for reports (default: reports/)")
    p.add_argument("--plots", action="store_true", help="Also generate plots")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_comparison(args)
