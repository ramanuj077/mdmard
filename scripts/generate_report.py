"""
scripts/generate_report.py
===========================
Research Report Generator -- MDMARD Phase 1

Reads MetricsTracker JSONL files from an ExperimentManager run directory
and auto-generates a Markdown + LaTeX research report with:
  - Reward curve plot
  - Success rate plot
  - Episode length plot
  - Reward breakdown (stacked bar)
  - Baseline comparison table

Usage:
  python scripts/generate_report.py --exp experiments/ppo_phase1_run/
  python scripts/generate_report.py --exp experiments/ppo_phase1_run/ --format latex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics.reporter import MetricsReporter
from metrics.tracker import EpisodeMetrics, MetricsTracker


def load_tracker_from_exp(exp_dir: Path) -> MetricsTracker:
    """Load a MetricsTracker from an experiment metrics/metrics.jsonl log."""
    jsonl_file = exp_dir / "metrics" / "metrics.jsonl"
    tracker = MetricsTracker()

    if not jsonl_file.exists():
        print(f"[report] No metrics.jsonl found at {jsonl_file}. Tracker will be empty.")
        return tracker

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Only load EpisodeMetrics rows (those with 'episode_id')
                if "episode_id" in data:
                    tracker.record_episode(EpisodeMetrics.from_dict(data))
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

    return tracker


def generate_report(args: argparse.Namespace) -> None:
    exp_dir = Path(args.exp)
    if not exp_dir.exists():
        print(f"[report] Experiment directory not found: {exp_dir}")
        sys.exit(1)

    out_dir = Path(args.output) if args.output else exp_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = load_tracker_from_exp(exp_dir)
    reporter = MetricsReporter(tracker)

    print(f"\n[report] Generating report for: {exp_dir}")
    print(f"[report] Episodes loaded: {tracker.num_episodes}")
    print(f"[report] Output directory: {out_dir}\n")

    # ---- ASCII summary to stdout ----
    print(reporter.ascii_banner())

    # ---- Markdown report ----
    md_path = reporter.save_markdown_report(out_dir / "report.md")
    print(f"[report] Markdown report: {md_path}")

    # ---- LaTeX tables ----
    if args.format in ("latex", "all"):
        latex_path = reporter.save_latex_report(out_dir / "report.tex")
        print(f"[report] LaTeX report: {latex_path}")

    # ---- Plots ----
    if tracker.num_episodes > 0:
        try:
            plot_paths = reporter.plot_all(out_dir)
            print(f"[report] Saved {len(plot_paths)} plots to {out_dir}")
        except Exception as e:
            print(f"[report] Plot generation skipped: {e}")

    print(f"\n[report] Done. Reports in: {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MDMARD Phase 1 Report Generator")
    p.add_argument("--exp", required=True,
                   help="Path to ExperimentManager run directory")
    p.add_argument("--output", default=None,
                   help="Output directory for reports (default: <exp>/reports/)")
    p.add_argument("--format", default="markdown", choices=["markdown", "latex", "all"],
                   help="Output format (default: markdown)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_report(args)
