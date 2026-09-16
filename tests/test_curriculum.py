"""
tests/test_curriculum.py
========================
Unit tests for CurriculumStage and CurriculumScheduler (Stage 13) -- MDMARD Phase 1
"""

import pytest
import tempfile
from pathlib import Path


CURRICULUM_YAML = """
curriculum:
  eval_episodes: 10
  success_rate_threshold: 0.80

  stages:
    - stage_id: 1
      name: "Stage One"
      goal_radius: 50.0
      min_distance: 100.0
      max_distance: 300.0
      fuel_enabled: false
      obstacles_count: 0
      wind_speed_max: 0.0
      success_threshold: 0.80
      min_episodes: 5

    - stage_id: 2
      name: "Stage Two"
      goal_radius: 30.0
      min_distance: 300.0
      max_distance: 600.0
      fuel_enabled: true
      obstacles_count: 1
      wind_speed_max: 2.0
      success_threshold: 0.75
      min_episodes: 5

    - stage_id: 3
      name: "Stage Three"
      goal_radius: 25.0
      min_distance: 400.0
      max_distance: 800.0
      fuel_enabled: true
      obstacles_count: 3
      wind_speed_max: 5.0
      success_threshold: 0.70
      min_episodes: 5
"""


@pytest.fixture
def curriculum_yaml_path(tmp_path):
    p = tmp_path / "curriculum_test.yaml"
    p.write_text(CURRICULUM_YAML, encoding="utf-8")
    return p


@pytest.fixture
def scheduler(curriculum_yaml_path):
    from curriculum.scheduler import CurriculumScheduler
    return CurriculumScheduler(curriculum_yaml_path)


# ---------------------------------------------------------------------
# CurriculumStage tests
# ---------------------------------------------------------------------

class TestCurriculumStage:
    def test_defaults(self):
        from curriculum.stages import CurriculumStage
        s = CurriculumStage(stage_id=1)
        assert s.stage_id == 1
        assert s.goal_radius == pytest.approx(30.0)
        assert s.fuel_enabled is False
        assert s.obstacles_count == 0

    def test_from_dict(self):
        from curriculum.stages import CurriculumStage
        data = {
            "stage_id": 2,
            "name": "Test",
            "goal_radius": 40.0,
            "fuel_enabled": True,
            "obstacles_count": 2,
            "wind_speed_max": 3.0,
        }
        s = CurriculumStage.from_dict(data)
        assert s.stage_id == 2
        assert s.goal_radius == pytest.approx(40.0)
        assert s.fuel_enabled is True
        assert s.obstacles_count == 2

    def test_to_dict_round_trip(self):
        from curriculum.stages import CurriculumStage
        s = CurriculumStage(stage_id=1, name="Round Trip", goal_radius=45.0, fuel_enabled=True)
        d = s.to_dict()
        s2 = CurriculumStage.from_dict(d)
        assert s2.stage_id == s.stage_id
        assert s2.name == s.name
        assert s2.goal_radius == pytest.approx(s.goal_radius)
        assert s2.fuel_enabled == s.fuel_enabled

    def test_repr(self):
        from curriculum.stages import CurriculumStage
        r = repr(CurriculumStage(stage_id=1, name="Alpha"))
        assert "CurriculumStage" in r
        assert "1" in r


# ---------------------------------------------------------------------
# CurriculumScheduler tests
# ---------------------------------------------------------------------

class TestCurriculumScheduler:
    def test_loads_stages(self, scheduler):
        assert scheduler.num_stages == 3

    def test_initial_state(self, scheduler):
        assert scheduler.current_stage_index == 0
        assert scheduler.current_stage.stage_id == 1
        assert scheduler.current_stage.name == "Stage One"
        assert not scheduler.is_final_stage

    def test_env_overrides(self, scheduler):
        overrides = scheduler.get_env_overrides()
        assert overrides["goal_radius"] == pytest.approx(50.0)
        assert overrides["fuel_enabled"] is False
        assert overrides["obstacles_count"] == 0

    def test_success_rate_empty(self, scheduler):
        assert scheduler.get_success_rate() == pytest.approx(0.0)

    def test_record_episodes_success_rate(self, scheduler):
        for _ in range(8):
            scheduler.record_episode(success=True)
        for _ in range(2):
            scheduler.record_episode(success=False)
        assert scheduler.get_success_rate() == pytest.approx(0.8, abs=1e-6)
        assert scheduler._episodes_at_stage == 10

    def test_should_advance_not_yet(self, scheduler):
        # Only 3 episodes � below min_episodes=5
        for _ in range(3):
            scheduler.record_episode(success=True)
        assert not scheduler.should_advance()

    def test_should_advance_true(self, scheduler):
        # 5 successes � meets threshold 0.80 and min_episodes=5
        for _ in range(5):
            scheduler.record_episode(success=True)
        assert scheduler.should_advance()

    def test_advance(self, scheduler):
        for _ in range(5):
            scheduler.record_episode(success=True)
        assert scheduler.should_advance()
        new_stage = scheduler.advance()
        assert new_stage.stage_id == 2
        assert scheduler.current_stage_index == 1
        # Counters reset
        assert scheduler._episodes_at_stage == 0
        assert scheduler.get_success_rate() == pytest.approx(0.0)

    def test_try_advance(self, scheduler):
        for _ in range(5):
            scheduler.record_episode(success=True)
        advanced = scheduler.try_advance()
        assert advanced is True
        assert scheduler.current_stage_index == 1

    def test_final_stage_does_not_advance(self, scheduler):
        # Advance to stage 3 (final)
        for _ in range(5):
            scheduler.record_episode(success=True)
        scheduler.advance()  # -> stage 2
        # Reset outcomes and advance again
        for _ in range(5):
            scheduler.record_episode(success=True)
        scheduler.advance()  # -> stage 3 (final)
        assert scheduler.is_final_stage
        assert not scheduler.should_advance()
        with pytest.raises(StopIteration):
            scheduler.advance()

    def test_summary(self, scheduler):
        s = scheduler.summary()
        assert "Stage" in s
        assert "success_rate" in s

    def test_save_state(self, scheduler, tmp_path):
        state_path = tmp_path / "sched_state.json"
        scheduler.save_state(state_path)
        assert state_path.exists()
        import json
        data = json.loads(state_path.read_text())
        assert "stages" in data
        assert len(data["stages"]) == 3

    def test_update_external_success_rate(self, scheduler):
        scheduler.update(success_rate=0.9)
        rate = scheduler.get_success_rate()
        assert rate > 0.0

    def test_repr(self, scheduler):
        r = repr(scheduler)
        assert "CurriculumScheduler" in r
