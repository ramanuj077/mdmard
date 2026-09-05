"""
scripts/manual_control.py
==========================
Interactive Keyboard Control -- MDMARD Phase 1

Lets you fly the aircraft yourself using keyboard input.
This is the single most important debugging tool:
  "If you can't fly it yourself, the RL agent can't learn it."

Controls
--------
  W  or  UP    : Thrust (forward acceleration)
  A             : Gentle Pitch Up   (nose up, with some thrust)
  D             : Gentle Pitch Down (nose down, with some thrust)
  E             : Thrust + Full Pitch Up
  Q             : Thrust + Full Pitch Down
  S  or  DOWN   : Full Pitch Down (no thrust -- emergency dive)
  SPACE         : Glide (coast, no input)
  R             : Reset episode (new random spawn + waypoint)
  ESC           : Quit

Action Mapping (matches TacticalFlightEnv.action_space)
---------------------------------------------------------
  0: Thrust only     1: Pitch Up      2: Pitch Down
  3: Thrust+Up       4: Thrust+Down   5: Glide (no-op)

What to verify manually
-----------------------
  [ ] Aircraft spawns in visible airspace
  [ ] Throttle accelerates aircraft to the right
  [ ] Pitch up makes aircraft climb (vy > 0)
  [ ] Pitch down makes aircraft descend (vy < 0)
  [ ] Reaching waypoint shows "MISSION SUCCESS"
  [ ] Flying out of bounds shows "MISSION FAILED"
  [ ] Reward breakdown bars respond in real time
  [ ] Trail shows recent trajectory
  [ ] FPS stays above 30 (check Pygame title bar)

Usage
-----
    cd c:\\Users\\raman\\OneDrive\\Desktop\\mdmard
    python scripts/manual_control.py

    # Custom map size:
    python scripts/manual_control.py --width 800 --height 600

    # Faster physics (for testing):
    python scripts/manual_control.py --fps 30
"""

from __future__ import annotations

import argparse
import sys
import os
import math
from pathlib import Path

# ── Path setup (run from project root or any dir) ────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.flight_env import TacticalFlightEnv
from envs.physics import PhysicsConfig
from envs.world import WorldConfig
from rendering.renderer import Renderer


# ─────────────────────────────────────────────────────────────────────────────
# Keyboard → Action Mapping
# ─────────────────────────────────────────────────────────────────────────────

def get_held_action(keys) -> int:
    """
    Read held keys and return the matching action.

    Priority (highest first):
      E        → 3  (Thrust + Full Pitch Up)
      Q        → 4  (Thrust + Full Pitch Down)
      W+A      → 3  (Thrust + Pitch Up)
      W+D      → 4  (Thrust + Pitch Down)
      W / UP   → 0  (Thrust only)
      A        → 1  (Pitch Up -- but with half thrust for controllability)
      D / DOWN → 2  (Pitch Down -- with half thrust)
      S        → 2  (Full Pitch Down, no thrust -- emergency dive)
      nothing  → 5  (Glide)

    Design note: pure pitch-only actions (action 1, 2) apply maximum elevator
    with ZERO thrust. At 30 FPS with dt=0.05 and pitch_torque/inertia=50,
    this rotates the aircraft ~2.5 rad/s in one second -- extremely aggressive.
    For human control, A/D map to Thrust+Pitch (action 3/4) with softer feel.
    Pure pitch-only (action 1/2) is kept for the RL agent but removed from
    keyboard mapping since it's uncontrollable by humans.
    """
    import pygame

    # ── Explicit combo keys first ─────────────────────────────────────────
    if keys[pygame.K_e]:
        return 3   # Thrust + Full Pitch Up

    # NOTE: Q is NOT used for quit anymore (ESC only).
    # Q = Thrust + Pitch Down (action 4)
    if keys[pygame.K_q]:
        return 4   # Thrust + Full Pitch Down

    # ── Individual keys ───────────────────────────────────────────────────
    thrust     = keys[pygame.K_w] or keys[pygame.K_UP]
    pitch_up   = keys[pygame.K_a]
    pitch_down = keys[pygame.K_d] or keys[pygame.K_DOWN]
    dive       = keys[pygame.K_s]

    # Combined: thrust + pitch
    if thrust and pitch_up:
        return 3   # Thrust + Pitch Up
    if thrust and pitch_down:
        return 4   # Thrust + Pitch Down
    if thrust:
        return 0   # Thrust only

    # A/D alone: map to Thrust + Pitch (actions 3/4) for controllability.
    # Pure pitch-only (actions 1/2) bleeds speed too fast for human control.
    if pitch_up:
        return 3   # Thrust + Pitch Up (instead of pure pitch-only)
    if pitch_down:
        return 4   # Thrust + Pitch Down (instead of pure pitch-only)

    # S = emergency pure pitch down (no thrust)
    if dive:
        return 2   # Pitch Down only

    return 5       # Glide (no keys held)


def run(args) -> None:
    """Main game loop."""
    import pygame

    # ── Build environment ─────────────────────────────────────────────────
    world_cfg  = WorldConfig(width=args.width, height=args.height)
    phys_cfg   = PhysicsConfig()

    env = TacticalFlightEnv(
        physics_config=phys_cfg,
        world_config=world_cfg,
        max_steps=args.max_steps,
        render_mode=None,   # we render manually
    )

    # ── Build renderer ────────────────────────────────────────────────────
    renderer = Renderer(
        world_width=args.width,
        world_height=args.height,
        screen_w=args.width + Renderer.HUD_WIDTH,
        screen_h=args.height,
        fps=args.fps,
    )
    renderer.init()

    # ── Initial episode ───────────────────────────────────────────────────
    episode_seed = args.seed
    obs, info = env.reset(seed=episode_seed)
    reward_breakdown: dict = {}
    episode_num = 1
    done = False

    print("\n" + "=" * 60)
    print("  MDMARD Manual Control  |  Press ESC to quit, R to reset")
    print("=" * 60)
    print("  W / UP  : Thrust        A : Pitch Up    D : Pitch Down")
    print("  E       : Thrust+Up     Q : Thrust+Down")
    print("  S       : Emergency Dive (no thrust)")
    print("  SPACE   : Glide         R : Reset episode")
    print("  ESC     : Quit")
    print("=" * 60 + "\n")

    # ── Main loop ─────────────────────────────────────────────────────────
    running = True

    while running:
        # ── Event handling ────────────────────────────────────────────────
        events = renderer.handle_events()
        if events and events[0] is None:
            running = False
            break

        keys = pygame.key.get_pressed()

        # Quit -- ESC only (Q is now Thrust+PitchDown)
        if keys[pygame.K_ESCAPE]:
            running = False
            break

        # Reset
        if keys[pygame.K_r]:
            episode_seed = (episode_seed or 0) + 1
            obs, info = env.reset(seed=episode_seed)
            reward_breakdown = {}
            done = False
            episode_num += 1
            renderer._trail.clear()
            print(f"  Episode {episode_num} reset (seed={episode_seed})")

        # ── Physics step (only if episode not done) ────────────────────────
        if not done:
            action = get_held_action(keys)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_breakdown = info.get("reward_breakdown", {})
            done = terminated or truncated

            if done:
                reason = info.get("termination", "timeout")
                ep_reward = info.get("episode_reward", 0.0)
                print(f"  Episode {episode_num} ended | "
                      f"reason={reason:12s} | "
                      f"steps={info.get('step',0):4d} | "
                      f"reward={ep_reward:+8.1f}")

        # ── Render ────────────────────────────────────────────────────────
        renderer.draw(
            state=env._state,
            world=env._world,
            reward_breakdown=reward_breakdown,
            info=info,
        )

        # Update window title with FPS
        pygame.display.set_caption(
            f"MDMARD  |  Episode {episode_num}  |  "
            f"Step {info.get('step', 0)}  |  "
            f"Reward {info.get('episode_reward', 0.0):+.1f}"
        )

        renderer.tick()

    renderer.close()
    env.close()
    print("\n  Session ended. Goodbye.\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MDMARD Manual Flight Control")
    p.add_argument("--width",     type=int,   default=1200,
                   help="World width in simulation units (default: 1200)")
    p.add_argument("--height",    type=int,   default=800,
                   help="World height in simulation units (default: 800)")
    p.add_argument("--fps",       type=int,   default=30,
                   help="Target FPS (default: 30)")
    p.add_argument("--max-steps", type=int,   default=2000,
                   help="Max steps per episode (default: 2000)")
    p.add_argument("--seed",      type=int,   default=42,
                   help="Initial random seed (default: 42)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
