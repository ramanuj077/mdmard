"""
Unit tests for Replay System (Stage 6).

Verifies EpisodeRecorder, ReplayPlayer, and JSON/Gzip serialization routines.
"""

from pathlib import Path
import pytest
import numpy as np

from envs.physics import AircraftState, PhysicsConfig
from envs.world import World, WorldConfig
from replay.formats import RecordedEpisode, RecordedStep, load_episode, save_episode
from replay.recorder import EpisodeRecorder
from replay.player import ReplayPlayer


@pytest.fixture
def sample_state():
    return AircraftState(
        x=500.0,
        y=500.0,
        vx=50.0,
        vy=0.0,
        theta=0.0,
        omega=0.0,
        fuel=95.0,
        step=1,
    )


@pytest.fixture
def sample_world():
    w = World(WorldConfig(width=1000.0, height=1000.0))
    w.reset()
    return w


def test_recorded_step_serialization():
    step = RecordedStep(
        step=1,
        state_dict={"x": 100.0, "y": 200.0, "vx": 10.0, "vy": 0.0, "theta": 0.0, "omega": 0.0, "fuel": 99.0, "step": 1},
        action=3,
        reward=2.5,
        reward_breakdown={"approach": 1.5, "alive": 1.0},
        obs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        terminated=False,
        truncated=False,
        info={"test": "data"},
    )

    d = step.to_dict()
    reconstructed = RecordedStep.from_dict(d)

    assert reconstructed.step == 1
    assert reconstructed.action == 3
    assert reconstructed.reward == 2.5
    assert reconstructed.reward_breakdown["approach"] == 1.5
    assert len(reconstructed.obs) == 10
    assert reconstructed.info["test"] == "data"


def test_recorded_episode_serialization(tmp_path):
    step = RecordedStep(
        step=1,
        state_dict={"x": 100.0, "y": 200.0, "vx": 10.0, "vy": 0.0, "theta": 0.0, "omega": 0.0, "fuel": 99.0, "step": 1},
        action=0,
        reward=1.0,
        reward_breakdown={"alive": 1.0},
        obs=[0.0] * 10,
        terminated=False,
        truncated=False,
        info={},
    )
    episode = RecordedEpisode(
        episode_id="ep_test_01",
        timestamp="20260906_120000",
        total_reward=15.5,
        total_steps=10,
        termination_reason="waypoint_reached",
        world_config={"width": 1000.0, "height": 1000.0, "waypoint": {"x": 500.0, "y": 500.0, "radius": 30.0}},
        steps=[step],
    )

    # Test plain JSON
    json_path = tmp_path / "test.json"
    save_episode(episode, json_path)
    loaded_json = load_episode(json_path)

    assert loaded_json.episode_id == "ep_test_01"
    assert loaded_json.total_reward == 15.5
    assert len(loaded_json.steps) == 1

    # Test compressed Gzip JSON
    gz_path = tmp_path / "test.json.gz"
    save_episode(episode, gz_path)
    loaded_gz = load_episode(gz_path)

    assert loaded_gz.episode_id == "ep_test_01"
    assert loaded_gz.termination_reason == "waypoint_reached"


def test_episode_recorder_workflow(sample_world, sample_state, tmp_path):
    recorder = EpisodeRecorder(output_dir=tmp_path)
    assert not recorder.is_active

    recorder.start_episode(sample_world, episode_id="ep_custom_42")
    assert recorder.is_active

    # Record 3 steps
    obs = np.zeros(10, dtype=np.float32)
    for i in range(3):
        recorder.record_step(
            state=sample_state,
            action=i,
            reward=1.0 * (i + 1),
            reward_breakdown={"approach": 1.0 * (i + 1)},
            obs=obs,
            terminated=(i == 2),
            truncated=False,
            info={"step_num": i},
        )

    episode = recorder.end_episode(termination_reason="waypoint_reached")
    assert not recorder.is_active
    assert episode.total_steps == 3
    assert episode.total_reward == 6.0  # 1.0 + 2.0 + 3.0
    assert episode.termination_reason == "waypoint_reached"

    saved_path = recorder.save(episode, compress=True)
    assert saved_path.exists()
    assert saved_path.suffix == ".gz"

    reloaded = load_episode(saved_path)
    assert reloaded.total_steps == 3
    assert reloaded.total_reward == 6.0


def test_recorder_errors(sample_world, sample_state):
    recorder = EpisodeRecorder()

    # Step without starting episode
    with pytest.raises(RuntimeError, match="call start_episode"):
        recorder.record_step(
            state=sample_state,
            action=0,
            reward=0.0,
            reward_breakdown={},
            obs=[0.0] * 10,
            terminated=False,
            truncated=False,
        )

    # End episode without starting
    with pytest.raises(RuntimeError, match="no active episode"):
        recorder.end_episode()


def test_replay_player_iter(sample_world, sample_state, tmp_path):
    recorder = EpisodeRecorder(output_dir=tmp_path)
    recorder.start_episode(sample_world, episode_id="ep_player_test")

    obs = np.zeros(10, dtype=np.float32)
    for i in range(5):
        sample_state.x += 10.0
        recorder.record_step(
            state=sample_state,
            action=1,
            reward=0.5,
            reward_breakdown={"alive": 0.5},
            obs=obs,
            terminated=False,
            truncated=False,
        )

    episode = recorder.end_episode("timeout")
    saved_path = recorder.save(episode, compress=False)

    player = ReplayPlayer(saved_path)
    assert player.total_steps == 5
    assert player.world.cfg.width == sample_world.cfg.width

    steps_iterated = 0
    for state, step in player.iter_steps():
        steps_iterated += 1
        assert isinstance(state, AircraftState)
        assert isinstance(step, RecordedStep)

    assert steps_iterated == 5


def test_replay_player_out_of_bounds(tmp_path):
    episode = RecordedEpisode(
        episode_id="ep_empty",
        timestamp="20260906",
        total_reward=0.0,
        total_steps=0,
        termination_reason="test",
        world_config={"width": 1000.0, "height": 1000.0},
        steps=[],
    )
    player = ReplayPlayer(episode)
    with pytest.raises(IndexError):
        player.get_step(0)
