"""
experiments/manager.py
======================
Experiment Lifecycle & Artifact Management -- MDMARD Phase 1

Provides a rigorous, reproducible experiment manager for tracking:
  - Structured per-run folders: checkpoints/, logs/, replays/, metrics/
  - Hyperparameter configs (YAML/JSON snapshot)
  - Experiment metadata (environment info, git hash, platform, timestamps, status)
  - Metrics logging (JSON Lines + summary)
  - Checkpoint tracking (regular checkpoints + best model tracking)
  - Replay artifact integration (gzip episode logs)
  - Resumption, loading, and cross-run comparison/search
"""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


class ExperimentStatus(str, Enum):
    """Execution status of an experiment run."""
    INITIALIZED = "INITIALIZED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


def _get_git_commit() -> Optional[str]:
    """Retrieve current git commit hash if running in a git repository."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        ).decode("utf-8").strip()
        return out
    except Exception:
        return None


@dataclass
class ExperimentMetadata:
    """Metadata describing a single experiment execution."""
    experiment_name: str
    run_id: str
    run_dir: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = ExperimentStatus.INITIALIZED.value
    seed: Optional[int] = None
    algorithm: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    git_commit: Optional[str] = field(default_factory=_get_git_commit)
    python_version: str = field(default_factory=lambda: platform.python_version())
    platform: str = field(default_factory=lambda: f"{platform.system()} {platform.release()}")
    total_steps: int = 0
    total_episodes: int = 0
    best_metric_value: Optional[float] = None
    best_metric_name: Optional[str] = None
    error_message: Optional[str] = None
    notes: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentMetadata:
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


class ExperimentManager:
    """Manages files, logs, metadata, checkpoints, and replays for an RL experiment."""

    def __init__(
        self,
        exp_name: str,
        base_dir: Union[str, Path] = "experiments",
        seed: Optional[int] = None,
        algorithm: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        notes: Optional[str] = None,
        auto_timestamp: bool = True,
    ) -> None:
        """Initialize experiment manager and directory structure.

        Args:
            exp_name: Name of the experiment (e.g., 'ppo_baseline', 'curriculum_stage1').
            base_dir: Root directory where experiment folders will be saved.
            seed: Random seed used for the run.
            algorithm: Algorithm name (e.g., 'PPO', 'DQN', 'PID', 'Greedy').
            tags: List of descriptive tags for filtering/categorization.
            config: Initial hyperparameter config dictionary.
            run_id: Optional custom run identifier. If None, generated automatically.
            notes: Optional human-readable notes.
            auto_timestamp: Whether to suffix run_id with timestamp for uniqueness.
        """
        self.exp_name = exp_name
        self.base_dir = Path(base_dir)
        self.seed = seed
        self.algorithm = algorithm
        self.tags = list(tags) if tags is not None else []
        self._config: Dict[str, Any] = dict(config) if config is not None else {}
        self._logger: Optional[logging.Logger] = None

        if run_id is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            seed_str = f"_s{seed}" if seed is not None else ""
            algo_str = f"_{algorithm.lower()}" if algorithm else ""
            if auto_timestamp:
                self.run_id = f"{exp_name}{algo_str}{seed_str}_{ts}"
            else:
                self.run_id = f"{exp_name}{algo_str}{seed_str}"
        else:
            self.run_id = run_id

        self.run_dir = self.base_dir / self.run_id
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.logs_dir = self.run_dir / "logs"
        self.replays_dir = self.run_dir / "replays"
        self.metrics_dir = self.run_dir / "metrics"

        self._create_directories()

        self.metadata = ExperimentMetadata(
            experiment_name=self.exp_name,
            run_id=self.run_id,
            run_dir=str(self.run_dir.resolve()),
            seed=self.seed,
            algorithm=self.algorithm,
            tags=self.tags,
            notes=notes,
        )

        if self._config:
            self.save_config(self._config)

        self.save_metadata()

    def _create_directories(self) -> None:
        """Create standard folder hierarchy for the experiment."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.replays_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    @property
    def config(self) -> Dict[str, Any]:
        """Current experiment configuration."""
        return self._config

    def set_config(self, config: Dict[str, Any]) -> None:
        """Update and save configuration."""
        self._config.update(config)
        self.save_config(self._config)

    def save_config(self, config: Optional[Dict[str, Any]] = None) -> Path:
        """Save configuration dictionary as YAML and JSON snapshots."""
        if config is not None:
            self._config = config

        yaml_path = self.run_dir / "config.yaml"
        json_path = self.run_dir / "config.json"

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)

        return yaml_path

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from config.yaml if present."""
        yaml_path = self.run_dir / "config.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
        return self._config

    def save_metadata(self) -> Path:
        """Write metadata to metadata.json in run_dir."""
        self.metadata.updated_at = datetime.now().isoformat()
        meta_path = self.run_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata.to_dict(), f, indent=2, ensure_ascii=False)
        return meta_path

    def set_status(
        self,
        status: Union[ExperimentStatus, str],
        error_message: Optional[str] = None,
    ) -> None:
        """Update execution status and save metadata."""
        stat_val = status.value if isinstance(status, ExperimentStatus) else str(status)
        self.metadata.status = stat_val
        if error_message:
            self.metadata.error_message = error_message
        self.save_metadata()

    def start(self) -> None:
        """Mark experiment as RUNNING."""
        self.set_status(ExperimentStatus.RUNNING)

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log a single metric at a given environment step."""
        self.log_metrics({name: value}, step=step)

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log multiple scalar metrics at a given step into metrics/metrics.jsonl."""
        self.metadata.total_steps = max(self.metadata.total_steps, step)
        entry = {
            "step": int(step),
            "timestamp": time.time(),
            "iso_time": datetime.now().isoformat(),
            **metrics,
        }

        jsonl_path = self.metrics_dir / "metrics.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def log_summary(self, summary_data: Dict[str, Any]) -> Path:
        """Save a final summary dictionary to summary.json."""
        summary_path = self.run_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        return summary_path

    def get_checkpoint_path(self, name_or_step: Union[str, int]) -> Path:
        """Return the standardized path for a checkpoint file."""
        if isinstance(name_or_step, int):
            filename = f"checkpoint_step_{name_or_step:08d}.pt"
        else:
            filename = name_or_step if "." in name_or_step else f"{name_or_step}.pt"
        return self.checkpoints_dir / filename

    def save_checkpoint(
        self,
        model_or_state: Any,
        name_or_step: Union[str, int],
        is_best: bool = False,
        metric_value: Optional[float] = None,
        metric_name: Optional[str] = None,
    ) -> Path:
        """Save a model checkpoint or state dictionary.

        Supports PyTorch, Stable-Baselines3, or arbitrary serializable objects.
        """
        ckpt_path = self.get_checkpoint_path(name_or_step)

        # 1. SB3 Model with .save() method
        if hasattr(model_or_state, "save") and callable(model_or_state.save):
            save_target = str(ckpt_path).removesuffix(".zip").removesuffix(".pt")
            model_or_state.save(save_target)
            if not ckpt_path.exists():
                sb3_zip = ckpt_path.parent / f"{ckpt_path.stem}.zip"
                if sb3_zip.exists():
                    ckpt_path = sb3_zip

        # 2. PyTorch object with torch.save
        elif "torch" in sys.modules and hasattr(sys.modules["torch"], "save"):
            torch = sys.modules["torch"]
            torch.save(model_or_state, ckpt_path)

        # 3. Fallback: JSON / Text / Pickling or Raw bytes
        elif isinstance(model_or_state, (dict, list, str, int, float, bool)):
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(model_or_state, f, indent=2)
        else:
            import pickle
            with open(ckpt_path, "wb") as f:
                pickle.dump(model_or_state, f)

        if is_best:
            best_path = self.checkpoints_dir / "best_model.pt"
            # Copy or save to best_model
            if ckpt_path.exists():
                import shutil
                shutil.copy2(ckpt_path, best_path)
            if metric_value is not None:
                self.metadata.best_metric_value = float(metric_value)
            if metric_name is not None:
                self.metadata.best_metric_name = metric_name
            self.save_metadata()

        return ckpt_path

    def get_best_checkpoint(self) -> Optional[Path]:
        """Return path to best_model if it exists."""
        best_pt = self.checkpoints_dir / "best_model.pt"
        if best_pt.exists():
            return best_pt
        best_zip = self.checkpoints_dir / "best_model.zip"
        if best_zip.exists():
            return best_zip
        return None

    def save_replay(
        self,
        recorder_or_episode: Any,
        filename: Optional[str] = None,
        compressed: bool = True,
    ) -> Path:
        """Save an EpisodeRecorder or RecordedEpisode into replays_dir."""
        if filename is None:
            ep_id = getattr(recorder_or_episode, "episode_id", 0)
            ext = ".json.gz" if compressed else ".json"
            filename = f"replay_ep_{ep_id:04d}{ext}"

        dest_path = self.replays_dir / filename

        # If it has save() method (e.g., EpisodeRecorder or RecordedEpisode)
        if hasattr(recorder_or_episode, "save") and callable(recorder_or_episode.save):
            return recorder_or_episode.save(dest_path, compressed=compressed)

        # Fallback dict serialization
        if hasattr(recorder_or_episode, "to_dict"):
            data = recorder_or_episode.to_dict()
        elif isinstance(recorder_or_episode, dict):
            data = recorder_or_episode
        else:
            raise TypeError(f"Cannot save replay of type {type(recorder_or_episode)}")

        if compressed:
            import gzip
            with gzip.open(dest_path, "wt", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        else:
            with open(dest_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        return dest_path

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """Configure and return a standard logger attached to run_dir/logs/run.log."""
        logger_name = name or f"exp_{self.run_id}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        # Avoid duplicate handlers
        if not logger.handlers:
            log_file = self.logs_dir / "run.log"
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            fh.setFormatter(formatter)
            logger.addHandler(fh)

            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        self._logger = logger
        return logger

    def close(self, status: ExperimentStatus = ExperimentStatus.COMPLETED) -> None:
        """Mark experiment as completed and close any attached loggers."""
        if self.metadata.status in (ExperimentStatus.INITIALIZED.value, ExperimentStatus.RUNNING.value):
            self.set_status(status)

        if self._logger:
            for handler in list(self._logger.handlers):
                handler.close()
                self._logger.removeHandler(handler)

    def __enter__(self) -> ExperimentManager:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            import traceback
            tb_str = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            self.set_status(ExperimentStatus.FAILED, error_message=f"{exc_val}\n{tb_str}")
            if self._logger:
                self._logger.error(f"Experiment failed with exception: {exc_val}\n{tb_str}")
        else:
            self.close(ExperimentStatus.COMPLETED)

    # -------------------------------------------------------------------------
    # Class Utilities
    # -------------------------------------------------------------------------

    @classmethod
    def load(cls, run_dir: Union[str, Path]) -> ExperimentManager:
        """Load an existing experiment run from its directory."""
        run_path = Path(run_dir)
        meta_file = run_path / "metadata.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"No metadata.json found in {run_path}")

        with open(meta_file, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)

        metadata = ExperimentMetadata.from_dict(meta_dict)

        mgr = cls.__new__(cls)
        mgr.exp_name = metadata.experiment_name
        mgr.base_dir = run_path.parent
        mgr.seed = metadata.seed
        mgr.algorithm = metadata.algorithm
        mgr.tags = list(metadata.tags)
        mgr.run_id = metadata.run_id
        mgr.run_dir = run_path
        mgr.checkpoints_dir = run_path / "checkpoints"
        mgr.logs_dir = run_path / "logs"
        mgr.replays_dir = run_path / "replays"
        mgr.metrics_dir = run_path / "metrics"
        mgr.metadata = metadata
        mgr._logger = None
        mgr._config = {}
        mgr.load_config()
        return mgr

    @classmethod
    def list_runs(cls, base_dir: Union[str, Path] = "experiments") -> List[Dict[str, Any]]:
        """List all experiment runs in base_dir with their metadata."""
        base_path = Path(base_dir)
        if not base_path.exists():
            return []

        runs = []
        for child in sorted(base_path.iterdir()):
            if child.is_dir():
                meta_file = child / "metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            runs.append(json.load(f))
                    except Exception:
                        pass
        return runs

    @classmethod
    def find_best_run(
        cls,
        base_dir: Union[str, Path] = "experiments",
        metric_name: Optional[str] = None,
        maximize: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Find the run with the best recorded metric in base_dir."""
        runs = cls.list_runs(base_dir)
        if not runs:
            return None

        candidates = [
            r for r in runs
            if r.get("best_metric_value") is not None and (
                metric_name is None or r.get("best_metric_name") == metric_name
            )
        ]

        if not candidates:
            return None

        key_fn = lambda r: float(r["best_metric_value"])
        return max(candidates, key=key_fn) if maximize else min(candidates, key=key_fn)
