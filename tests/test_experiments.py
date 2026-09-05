"""
Unit tests for Experiment Manager (Stage 7).

Verifies directory creation, metadata tracking, config snapshots, metrics logging,
checkpoint saving/retrieval, replay artifact integration, and context manager lifecycle.
"""

import json
from pathlib import Path
import pytest
import yaml

from experiments.manager import (
    ExperimentManager,
    ExperimentMetadata,
    ExperimentStatus,
)
from replay.formats import RecordedEpisode, RecordedStep


@pytest.fixture
def sample_config():
    return {
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "env": {"id": "TacticalFlight-v0", "reward_scale": 1.0},
    }


def test_experiment_dir_structure(tmp_path, sample_config):
    exp = ExperimentManager(
        exp_name="test_ppo",
        base_dir=tmp_path,
        seed=42,
        algorithm="PPO",
        tags=["baseline", "phase1"],
        config=sample_config,
    )

    assert exp.run_dir.exists()
    assert exp.checkpoints_dir.exists()
    assert exp.logs_dir.exists()
    assert exp.replays_dir.exists()
    assert exp.metrics_dir.exists()

    assert (exp.run_dir / "metadata.json").exists()
    assert (exp.run_dir / "config.yaml").exists()
    assert (exp.run_dir / "config.json").exists()

    assert exp.metadata.experiment_name == "test_ppo"
    assert exp.metadata.algorithm == "PPO"
    assert exp.metadata.seed == 42
    assert "baseline" in exp.metadata.tags


def test_config_saving_and_loading(tmp_path, sample_config):
    exp = ExperimentManager(
        exp_name="test_config",
        base_dir=tmp_path,
        config=sample_config,
    )

    # Read back config YAML
    with open(exp.run_dir / "config.yaml", "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert loaded["learning_rate"] == 0.0003
    assert loaded["env"]["reward_scale"] == 1.0

    # Update config
    exp.set_config({"gamma": 0.995, "new_param": 123})
    assert exp.config["gamma"] == 0.995
    assert exp.config["new_param"] == 123

    # Verify updated on disk
    reloaded = exp.load_config()
    assert reloaded["gamma"] == 0.995
    assert reloaded["new_param"] == 123


def test_metrics_logging(tmp_path):
    exp = ExperimentManager(
        exp_name="test_metrics",
        base_dir=tmp_path,
    )

    exp.log_metric("loss", 0.45, step=100)
    exp.log_metrics({"loss": 0.32, "reward": 15.2, "success_rate": 0.8}, step=200)

    jsonl_file = exp.metrics_dir / "metrics.jsonl"
    assert jsonl_file.exists()

    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) == 2
    assert lines[0]["step"] == 100
    assert lines[0]["loss"] == 0.45
    assert lines[1]["step"] == 200
    assert lines[1]["reward"] == 15.2
    assert lines[1]["success_rate"] == 0.8
    assert exp.metadata.total_steps == 200


def test_checkpoint_saving_and_best(tmp_path):
    exp = ExperimentManager(
        exp_name="test_checkpoint",
        base_dir=tmp_path,
    )

    # Dummy model weights / dictionary
    weights_step_1000 = {"layer1.weight": [1.0, 2.0], "step": 1000}
    weights_step_2000 = {"layer1.weight": [1.5, 2.5], "step": 2000}

    p1 = exp.save_checkpoint(weights_step_1000, name_or_step=1000)
    assert p1.exists()
    assert "checkpoint_step_00001000.pt" in p1.name

    # Save best checkpoint
    p2 = exp.save_checkpoint(
        weights_step_2000,
        name_or_step=2000,
        is_best=True,
        metric_value=450.5,
        metric_name="eval/mean_reward",
    )
    assert p2.exists()

    best_p = exp.get_best_checkpoint()
    assert best_p is not None
    assert best_p.name == "best_model.pt"
    assert exp.metadata.best_metric_value == 450.5
    assert exp.metadata.best_metric_name == "eval/mean_reward"


def test_replay_saving(tmp_path):
    exp = ExperimentManager(
        exp_name="test_replays",
        base_dir=tmp_path,
    )

    episode = RecordedEpisode(
        episode_id="ep_exp_01",
        timestamp="20260906_000000",
        total_reward=100.0,
        total_steps=50,
        termination_reason="goal",
        world_config={"width": 1000.0, "height": 1000.0},
        steps=[],
    )

    # Compressed save
    saved_gz = exp.save_replay(episode, filename="eval_ep01.json.gz", compressed=True)
    assert saved_gz.exists()
    assert saved_gz.suffix == ".gz"
    assert saved_gz.parent == exp.replays_dir

    # Uncompressed save
    saved_json = exp.save_replay(episode, filename="eval_ep01.json", compressed=False)
    assert saved_json.exists()
    assert saved_json.suffix == ".json"


def test_context_manager_success(tmp_path):
    with ExperimentManager(exp_name="ctx_test", base_dir=tmp_path) as exp:
        assert exp.metadata.status == ExperimentStatus.RUNNING.value
        exp.log_metric("reward", 10.0, step=1)

    # After clean exit, should be COMPLETED
    assert exp.metadata.status == ExperimentStatus.COMPLETED.value


def test_context_manager_failure(tmp_path):
    try:
        with ExperimentManager(exp_name="ctx_fail", base_dir=tmp_path) as exp:
            assert exp.metadata.status == ExperimentStatus.RUNNING.value
            raise ValueError("Something exploded during training!")
    except ValueError:
        pass

    assert exp.metadata.status == ExperimentStatus.FAILED.value
    assert "Something exploded" in (exp.metadata.error_message or "")


def test_load_and_list_runs(tmp_path):
    exp1 = ExperimentManager(
        exp_name="run_alpha",
        base_dir=tmp_path,
        seed=1,
        algorithm="PPO",
        run_id="run_alpha_01",
    )
    exp1.save_checkpoint({"w": 1}, name_or_step="final", is_best=True, metric_value=200.0, metric_name="reward")
    exp1.close(ExperimentStatus.COMPLETED)

    exp2 = ExperimentManager(
        exp_name="run_beta",
        base_dir=tmp_path,
        seed=2,
        algorithm="DQN",
        run_id="run_beta_01",
    )
    exp2.save_checkpoint({"w": 2}, name_or_step="final", is_best=True, metric_value=350.0, metric_name="reward")
    exp2.close(ExperimentStatus.COMPLETED)

    # Test load()
    loaded_mgr = ExperimentManager.load(exp1.run_dir)
    assert loaded_mgr.exp_name == "run_alpha"
    assert loaded_mgr.seed == 1
    assert loaded_mgr.algorithm == "PPO"
    assert loaded_mgr.metadata.status == ExperimentStatus.COMPLETED.value

    # Test list_runs()
    runs = ExperimentManager.list_runs(tmp_path)
    assert len(runs) == 2
    run_ids = [r["run_id"] for r in runs]
    assert "run_alpha_01" in run_ids
    assert "run_beta_01" in run_ids

    # Test find_best_run()
    best = ExperimentManager.find_best_run(tmp_path, metric_name="reward", maximize=True)
    assert best is not None
    assert best["run_id"] == "run_beta_01"
    assert best["best_metric_value"] == 350.0

    worst = ExperimentManager.find_best_run(tmp_path, metric_name="reward", maximize=False)
    assert worst is not None
    assert worst["run_id"] == "run_alpha_01"
