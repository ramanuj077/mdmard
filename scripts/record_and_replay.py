"""
Script to test recording an episode and playing it back.

Usage:
    python scripts/record_and_replay.py --mode record
    python scripts/record_and_replay.py --mode play --file logs/replays/sample_replay.json.gz
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym

from envs.flight_env import TacticalFlightEnv
from replay.player import ReplayPlayer
from replay.recorder import EpisodeRecorder


def record_sample_episode(output_file: str = "logs/replays/sample_replay.json.gz") -> Path:
    """Run a sample episode with random/heuristic actions and record it."""
    env = TacticalFlightEnv()
    recorder = EpisodeRecorder(output_dir=Path(output_file).parent)

    obs, info = env.reset(seed=42)
    recorder.start_episode(world=env.world, episode_id="demo_episode")

    terminated = False
    truncated = False
    step_count = 0

    print("[RECORDING] Recording sample trajectory episode...")
    while not (terminated or truncated):
        # Action 3: Thrust + Pitch Up
        action = 3 if step_count < 100 else (1 if step_count < 200 else 0)
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        recorder.record_step(
            state=env.state,
            action=action,
            reward=reward,
            reward_breakdown=env.last_reward_breakdown,
            obs=obs,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    term_reason = info.get("termination_reason", "unknown")
    episode = recorder.end_episode(termination_reason=term_reason)
    saved_path = recorder.save(episode, filepath=output_file, compress=output_file.endswith(".gz"))

    env.close()
    print(f"[SAVED] Saved episode replay ({episode.total_steps} steps, reward: {episode.total_reward:.2f}) to {saved_path}")
    return saved_path


def main():
    parser = argparse.ArgumentParser(description="Replay System Demo")
    parser.add_argument("--mode", choices=["record", "play", "both"], default="both", help="Mode of operation")
    parser.add_argument("--file", type=str, default="logs/replays/sample_replay.json.gz", help="Replay file path")
    args = parser.parse_args()

    replay_path = Path(args.file)

    if args.mode in ("record", "both"):
        replay_path = record_sample_episode(args.file)

    if args.mode in ("play", "both"):
        print(f"[PLAY] Playing back recorded episode from {replay_path}...")
        player = ReplayPlayer(replay_path)
        print(f"Episode ID: {player.episode.episode_id}")
        print(f"Total Steps: {player.total_steps}")
        print(f"Total Reward: {player.episode.total_reward:.2f}")
        print(f"Termination Reason: {player.episode.termination_reason}")
        print("\nOpening Pygame visual playback window... (Press SPACE to Pause/Resume, Arrow keys to step frame, Q to quit)")
        player.play_visual(fps=30)


if __name__ == "__main__":
    main()
