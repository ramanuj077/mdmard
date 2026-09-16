"""
tests/test_comparison.py
========================
Unit tests for BaselineComparison (Stage 15) -- MDMARD Phase 1
"""

import pytest
import tempfile
from pathlib import Path

from metrics.comparison import AgentResult, BaselineComparison


SAMPLE_RESULTS = [
    {"agent_name": "Random",  "num_episodes": 50, "success_rate": 0.08, "crash_rate": 0.92,
     "mean_reward": -45.3, "std_reward": 12.1, "mean_steps": 412.0, "mean_fuel_consumed": 87.3},
    {"agent_name": "Greedy",  "num_episodes": 50, "success_rate": 0.62, "crash_rate": 0.38,
     "mean_reward":  82.4, "std_reward":  9.3, "mean_steps": 287.0, "mean_fuel_consumed": 71.2},
    {"agent_name": "PID",     "num_episodes": 50, "success_rate": 0.74, "crash_rate": 0.26,
     "mean_reward":  91.7, "std_reward":  8.1, "mean_steps": 263.0, "mean_fuel_consumed": 68.4},
    {"agent_name": "PPO",     "num_episodes": 50, "success_rate": 0.87, "crash_rate": 0.13,
     "mean_reward": 143.8, "std_reward": 10.5, "mean_steps": 198.0, "mean_fuel_consumed": 54.2},
]


@pytest.fixture
def comparison():
    c = BaselineComparison()
    c.add_results(SAMPLE_RESULTS)
    return c


# ---------------------------------------------------------------------
# AgentResult
# ---------------------------------------------------------------------

class TestAgentResult:
    def test_from_dict(self):
        r = AgentResult(SAMPLE_RESULTS[0])
        assert r.agent_name == "Random"
        assert r.success_rate == pytest.approx(0.08)
        assert r.crash_rate == pytest.approx(0.92)

    def test_to_dict(self):
        r = AgentResult(SAMPLE_RESULTS[1])
        d = r.to_dict()
        assert d["agent_name"] == "Greedy"
        assert "success_rate" in d

    def test_repr(self):
        r = AgentResult(SAMPLE_RESULTS[0])
        assert "Random" in repr(r)


# ---------------------------------------------------------------------
# BaselineComparison
# ---------------------------------------------------------------------

class TestBaselineComparison:
    def test_add_results(self, comparison):
        assert comparison.num_agents == 4

    def test_add_dict_directly(self):
        c = BaselineComparison()
        c.add_result({"agent_name": "X", "num_episodes": 10, "success_rate": 0.5,
                      "crash_rate": 0.5, "mean_reward": 0.0, "std_reward": 0.0,
                      "mean_steps": 100.0, "mean_fuel_consumed": 50.0})
        assert c.num_agents == 1

    def test_clear(self, comparison):
        comparison.clear()
        assert comparison.num_agents == 0

    def test_ascii_table_contains_agents(self, comparison):
        t = comparison.ascii_table()
        for name in ["Random", "Greedy", "PID", "PPO"]:
            assert name in t

    def test_markdown_table(self, comparison):
        md = comparison.markdown_table()
        assert "Random" in md
        assert "PPO" in md
        assert "|" in md

    def test_latex_table(self, comparison):
        latex = comparison.latex_table()
        assert "\\begin{table}" in latex
        assert "Random" in latex

    def test_best_by_success_rate(self, comparison):
        best = comparison.best_by("success_rate")
        assert best.agent_name == "PPO"
        assert best.success_rate == pytest.approx(0.87)

    def test_best_by_minimize_steps(self, comparison):
        best = comparison.best_by("mean_steps", maximize=False)
        assert best.agent_name == "PPO"  # lowest steps

    def test_save_json(self, comparison, tmp_path):
        p = comparison.save_json(tmp_path / "comp.json")
        assert p.exists()
        import json
        data = json.loads(p.read_text())
        assert data["num_agents"] == 4
        assert len(data["results"]) == 4

    def test_save_markdown(self, comparison, tmp_path):
        p = comparison.save_markdown(tmp_path / "comp.md")
        assert p.exists()
        assert "Agent" in p.read_text()

    def test_save_latex(self, comparison, tmp_path):
        p = comparison.save_latex(tmp_path / "comp.tex")
        assert p.exists()
        assert "\\begin{table}" in p.read_text()

    def test_plot_bar_chart(self, comparison, tmp_path):
        p = comparison.plot_bar_chart(tmp_path / "bar.png", metric="success_rate")
        assert p.exists()

    def test_plot_all_metrics(self, comparison, tmp_path):
        paths = comparison.plot_all_metrics(tmp_path, metrics=["success_rate", "mean_reward"])
        assert len(paths) == 2
        for p in paths:
            assert p.exists()

    def test_plot_radar_chart(self, comparison, tmp_path):
        p = comparison.plot_radar_chart(tmp_path / "radar.png")
        assert p.exists()

    def test_to_dict(self, comparison):
        d = comparison.to_dict()
        assert d["num_agents"] == 4
        assert len(d["results"]) == 4

    def test_empty_ascii_table(self):
        c = BaselineComparison()
        t = c.ascii_table()
        assert "No results" in t

    def test_repr(self, comparison):
        r = repr(comparison)
        assert "BaselineComparison" in r
        assert "Random" in r
