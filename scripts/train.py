"""
scripts/train.py
================
Training Entry Point -- MDMARD Phase 1

Usage:
  python scripts/train.py --algo ppo --timesteps 500000 --seed 42
  python scripts/train.py --algo dqn --timesteps 200000 --eval
  python scripts/train.py --algo sac --config configs/sac_phase1.yaml

Trains the selected RL agent, saving checkpoints and metrics via
ExperimentManager. TensorBoard logs are written automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym

from envs.flight_env import TacticalFlightEnv
from experiments.manager import ExperimentManager
from metrics.tracker import MetricsTracker
from utils.config import load_config


def make_env(seed: int = 42) -> TacticalFlightEnv:
    """Construct a single TacticalFlightEnv instance."""
    env = TacticalFlightEnv()
    env.reset(seed=seed)
    return env


def train(args: argparse.Namespace) -> None:
    algo = args.algo.upper()
    seed = args.seed
    timesteps = args.timesteps

    # ---- Load optional YAML config ----
    cfg: dict = {}
    if args.config:
        cfg = load_config(args.config)

    # ---- Build agent ----
    if algo == "PPO":
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent.from_config(cfg, seed=seed, verbose=1)
    elif algo == "DQN":
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent.from_config(cfg, seed=seed, verbose=1)
    elif algo == "SAC":
        from agents.sac_agent import SACAgent
        agent = SACAgent.from_config(cfg, seed=seed, verbose=1)
    else:
        raise ValueError(f"Unknown algorithm: {algo}. Choose ppo | dqn | sac")

    # ---- Environments ----
    env = make_env(seed=seed)
    eval_env = make_env(seed=seed + 1000) if args.eval else None

    # ---- Experiment manager ----
    exp = ExperimentManager(
        exp_name=f"{algo.lower()}_phase1",
        base_dir="experiments",
        seed=seed,
        algorithm=algo,
        tags=["phase1", algo.lower()],
        config=cfg,
    )
    print(f"\n[train.py] Experiment: {exp.run_id}")
    print(f"[train.py] Run directory: {exp.run_dir}\n")

    # ---- Train ----
    agent.train(
        env=env,
        total_timesteps=timesteps,
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_freq=args.save_freq,
        experiment_manager=exp,
    )

    # ---- Save final model ----
    final_path = str(exp.checkpoints_dir / f"{algo.lower()}_final")
    agent.save(final_path)
    print(f"\n[train.py] Final model saved: {final_path}")
    print(f"[train.py] Total timesteps: {agent.num_timesteps}")

    if eval_env is not None:
        eval_env.close()
    env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MDMARD Phase 1 Training Script")
    p.add_argument("--algo", default="ppo", choices=["ppo", "dqn", "sac"],
                   help="RL algorithm to train (default: ppo)")
    p.add_argument("--timesteps", type=int, default=500_000,
                   help="Total training timesteps (default: 500000)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--config", default=None, help="Path to YAML config override file")
    p.add_argument("--eval", action="store_true", help="Enable periodic evaluation")
    p.add_argument("--eval-freq", type=int, default=10_000, dest="eval_freq",
                   help="Evaluation frequency in steps (default: 10000)")
    p.add_argument("--eval-episodes", type=int, default=20, dest="eval_episodes",
                   help="Episodes per evaluation cycle (default: 20)")
    p.add_argument("--save-freq", type=int, default=25_000, dest="save_freq",
                   help="Checkpoint save frequency in steps (default: 25000)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
