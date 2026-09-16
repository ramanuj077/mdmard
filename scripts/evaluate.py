"""
scripts/evaluate.py
===================
Evaluation Script -- MDMARD Phase 1

Usage:
  python scripts/evaluate.py --model experiments/ppo_run/checkpoints/ppo_final.zip --algo ppo
  python scripts/evaluate.py --agent pid --episodes 50
  python scripts/evaluate.py --agent greedy --render
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.flight_env import TacticalFlightEnv
from metrics.tracker import MetricsTracker


def build_agent(algo: str, model_path: str | None):
    """Instantiate any agent type."""
    algo = algo.lower()
    if algo == "random":
        from baselines.random_agent import RandomAgent
        return RandomAgent()
    elif algo == "greedy":
        from baselines.greedy_agent import GreedyAgent
        return GreedyAgent()
    elif algo == "pid":
        from baselines.pid_agent import PIDAgent
        return PIDAgent()
    elif algo == "ppo":
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(verbose=0)
        if model_path is None:
            raise ValueError("--model is required for PPO")
        agent.load(model_path)
        return agent
    elif algo == "dqn":
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent(verbose=0)
        if model_path is None:
            raise ValueError("--model is required for DQN")
        agent.load(model_path)
        return agent
    elif algo == "sac":
        from agents.sac_agent import SACAgent
        agent = SACAgent(verbose=0)
        if model_path is None:
            raise ValueError("--model is required for SAC")
        agent.load(model_path)
        return agent
    else:
        raise ValueError(f"Unknown agent: {algo}")


def evaluate(args: argparse.Namespace) -> None:
    env = TacticalFlightEnv()
    tracker = MetricsTracker()
    agent = build_agent(args.agent, args.model)

    print(f"\n[evaluate.py] Agent: {agent.name}")
    print(f"[evaluate.py] Episodes: {args.episodes}, Seed: {args.seed}\n")

    results = agent.evaluate(
        env=env,
        num_episodes=args.episodes,
        seed=args.seed,
        metrics_tracker=tracker,
        render=args.render,
    )

    env.close()

    # Print results
    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<25}: {v:.4f}")
        else:
            print(f"  {k:<25}: {v}")

    # Save JSON if requested
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[evaluate.py] Results saved to: {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MDMARD Phase 1 Evaluation Script")
    p.add_argument("--agent", default="pid", choices=["random", "greedy", "pid", "ppo", "dqn", "sac"],
                   help="Agent type to evaluate (default: pid)")
    p.add_argument("--model", default=None, help="Path to saved RL model (.zip)")
    p.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--render", action="store_true", help="Render episodes with Pygame")
    p.add_argument("--output", default=None, help="Save results JSON to path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
