"""
Unit tests for Research Metrics Tracker & Reporter (Stage 8).

Verifies EpisodeMetrics computation, rolling window aggregations,
CSV/JSONL export, summary formatting, LaTeX table creation, and Matplotlib plotting.
"""

import json
from pathlib import Path
import pytest
import numpy as np

from envs.physics import AircraftState
from metrics.tracker import EpisodeMetrics, MetricsTracker
from metrics.reporter import MetricsReporter


@pytest.fixture
def sample_state():
    return AircraftState(
        x=100.0,
        y=500.0,
        vx=40.0,
        vy=0.0,
        theta=0.0,
        omega=0.05,
        fuel=98.0,
        step=1,
    )


def test_episode_metrics_dataclass():
    ep = EpisodeMetrics(
        episode_id=1,
        total_reward=150.0,
        steps=25,
        duration_sec=0.5,
        termination_reason="goal",
        success=True,
        crashed=False,
        reward_breakdown={"approach": 50.0, "waypoint": 100.0},
        min_distance=12.5,
    )

    d = ep.to_dict()
    reconstructed = EpisodeMetrics.from_dict(d)
    assert reconstructed.episode_id == 1
    assert reconstructed.total_reward == 150.0
    assert reconstructed.success is True
    assert reconstructed.min_distance == 12.5
    assert reconstructed.reward_breakdown["approach"] == 50.0


def test_metrics_tracker_online_workflow(sample_state):
    tracker = MetricsTracker(window_sizes=(5, 10))
    assert tracker.num_episodes == 0

    # Start episode with target at (500, 500)
    tracker.start_episode(episode_id=0, initial_state=sample_state, target_pos=(500.0, 500.0))

    # Simulate 4 steps moving towards waypoint
    for step_i in range(1, 5):
        sample_state.x += 80.0
        sample_state.fuel -= 0.5
        tracker.step(
            state=sample_state,
            action=3,
            reward=25.0,
            reward_breakdown={"approach": 25.0},
        )

    metrics = tracker.end_episode(termination_reason="waypoint_reached")
    assert tracker.num_episodes == 1
    assert metrics.success is True
    assert metrics.crashed is False
    assert metrics.steps == 4
    assert metrics.total_reward == 100.0
    assert metrics.distance_closed > 0.0
    assert metrics.fuel_consumed > 0.0
    assert metrics.path_efficiency > 0.0
    assert metrics.action_counts[3] == 4


def test_metrics_tracker_window_and_summary():
    tracker = MetricsTracker(window_sizes=(10, 50))

    # Add 20 dummy episodes (15 successes, 5 crashes)
    for i in range(20):
        is_succ = (i % 4 != 0)
        ep = EpisodeMetrics(
            episode_id=i,
            total_reward=100.0 if is_succ else -50.0,
            steps=50,
            duration_sec=1.0,
            termination_reason="goal" if is_succ else "oob",
            success=is_succ,
            crashed=(not is_succ),
            reward_breakdown={"approach": 50.0 if is_succ else -10.0},
            fuel_consumed=10.0,
            min_distance=5.0 if is_succ else 200.0,
            path_efficiency=0.85 if is_succ else 0.2,
        )
        tracker.record_episode(ep)

    assert tracker.num_episodes == 20

    stats_10 = tracker.get_window_stats(window_size=10)
    assert stats_10["episodes_count"] == 10
    assert 0.0 <= stats_10["success_rate"] <= 1.0
    assert stats_10["mean_steps"] == 50.0

    summary = tracker.get_summary()
    assert summary["total_episodes"] == 20
    assert "all_time" in summary
    assert "last_10" in summary
    assert summary["all_time"]["success_rate"] == 0.75  # 15/20


def test_metrics_export_jsonl_csv_df(tmp_path):
    tracker = MetricsTracker()
    for i in range(5):
        tracker.record_episode(
            EpisodeMetrics(
                episode_id=i,
                total_reward=float(i * 10),
                steps=10 + i,
                duration_sec=0.1,
                termination_reason="goal",
                success=True,
                crashed=False,
                reward_breakdown={"alive": 5.0, "approach": float(i * 5)},
            )
        )

    # Export CSV
    csv_file = tmp_path / "metrics.csv"
    tracker.export_csv(csv_file)
    assert csv_file.exists()
    content = csv_file.read_text()
    assert "episode_id" in content
    assert "rew_approach" in content

    # Export JSONL
    jsonl_file = tmp_path / "metrics.jsonl"
    tracker.export_jsonl(jsonl_file)
    assert jsonl_file.exists()
    lines = jsonl_file.read_text().strip().split("\n")
    assert len(lines) == 5

    # DataFrame
    df = tracker.to_dataframe()
    assert len(df) == 5


def test_metrics_reporter_formatting():
    ep = EpisodeMetrics(
        episode_id=42,
        total_reward=245.5,
        steps=120,
        duration_sec=2.5,
        termination_reason="waypoint_reached",
        success=True,
        crashed=False,
        min_distance=4.2,
        fuel_consumed=18.5,
        path_efficiency=0.92,
    )

    banner = MetricsReporter.format_episode_banner(ep)
    assert "SUCCESS" in banner
    assert "Ep 0042" in banner
    assert "+245.5" in banner

    stats = {
        "window_size": 50,
        "success_rate": 0.85,
        "crash_rate": 0.10,
        "timeout_rate": 0.05,
        "mean_reward": 312.4,
        "std_reward": 45.2,
        "mean_steps": 150.0,
        "mean_fuel_consumed": 22.0,
        "mean_min_distance": 12.0,
        "mean_path_efficiency": 0.88,
        "component_means": {"approach": 110.0, "waypoint": 200.0},
    }

    table = MetricsReporter.format_window_table(stats, ep_num=50, total_steps=7500)
    assert "EXPERIMENT TELEMETRY" in table
    assert "85.0%" in table


def test_comparison_tables_and_markdown(tmp_path):
    tracker = MetricsTracker()
    for i in range(10):
        tracker.record_episode(
            EpisodeMetrics(
                episode_id=i,
                total_reward=100.0,
                steps=50,
                duration_sec=1.0,
                termination_reason="goal",
                success=True,
                crashed=False,
                reward_breakdown={"alive": 10.0, "waypoint": 90.0},
            )
        )

    # Markdown report
    report_md = MetricsReporter.generate_markdown_report(
        tracker,
        title="Benchmark Test Report",
        experiment_meta={"agent": "PPO", "seed": 42},
    )
    assert "# Benchmark Test Report" in report_md
    assert "Overall Success Rate" in report_md

    # Comparison ASCII table
    results = {
        "Random": {"mean_reward": -85.2, "success_rate": 0.05, "crash_rate": 0.95, "mean_steps": 45.0, "mean_fuel_consumed": 15.0, "mean_path_efficiency": 0.12},
        "PID": {"mean_reward": 180.5, "success_rate": 0.72, "crash_rate": 0.18, "mean_steps": 140.0, "mean_fuel_consumed": 25.0, "mean_path_efficiency": 0.78},
        "PPO": {"mean_reward": 350.0, "success_rate": 0.94, "crash_rate": 0.04, "mean_steps": 110.0, "mean_fuel_consumed": 18.0, "mean_path_efficiency": 0.91},
    }

    ascii_table = MetricsReporter.format_comparison_table(results)
    assert "Agent / Policy" in ascii_table
    assert "Random" in ascii_table
    assert "PPO" in ascii_table

    # LaTeX table
    latex_tbl = MetricsReporter.generate_latex_table(results)
    assert "\\begin{table}" in latex_tbl
    assert "\\textbf{PPO}" in latex_tbl

    # Plot learning curves
    plot_path = tmp_path / "learning_curves.png"
    saved_plot = MetricsReporter.plot_learning_curves(tracker, plot_path, window_size=2)
    assert saved_plot is not None
    assert saved_plot.exists()
