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

def get_held_action(keys, current_omega: float) -> np.ndarray:
    """
    Read held keys and return a continuous action [throttle, elevator].
    Uses the exact same key bindings as before, but with scaled-down 
    elevator deflection so humans can actually control it (instead of 
    spinning 90-180 degrees instantly).

    Priority (highest first):
      E        → Thrust + Pitch Up
      Q        → Thrust + Pitch Down
      W+A      → Thrust + Pitch Up
      W+D      → Thrust + Pitch Down
      W / UP   → Thrust only
      A        → Pitch Up
      D / DOWN → Pitch Down
      S        → Emergency dive (Full Pitch Down, no thrust)
      nothing  → Glide
    """
    import pygame
    import numpy as np

    throttle = 0.0
    elevator = 0.0

    # ── Explicit combo keys first ─────────────────────────────────────────
    if keys[pygame.K_e]:
        throttle, elevator = 1.0, 0.06
    elif keys[pygame.K_q]:
        throttle, elevator = 1.0, -0.06
    else:
        # ── Individual keys ───────────────────────────────────────────────────
        thrust_key = keys[pygame.K_w] or keys[pygame.K_UP]
        pitch_up_key = keys[pygame.K_a]
        pitch_down_key = keys[pygame.K_d] or keys[pygame.K_DOWN]
        dive_key = keys[pygame.K_s]

        if thrust_key:
            throttle = 1.0

        if dive_key:
            throttle = 0.0
            elevator = -0.06
        elif pitch_up_key:
            elevator = 0.06
            if not thrust_key:
                throttle = 0.5  # A/D alone give half thrust for controllability (same as old logic)
        elif pitch_down_key:
            elevator = -0.06
            if not thrust_key:
                throttle = 0.5

    # ── Mild Flight Assist / Damping ──────────────────────────────────────
    # If the user is not actively pitching, apply a mild counter-elevator 
    # to slowly stop the rotation. This prevents the "spins forever" problem.
    if elevator == 0.0 and abs(current_omega) > 0.05:
        # Apply a counter elevator proportional to current_omega
        # pitch_torque = 5.0, inertia = 0.1, so alpha = 50 * elevator.
        # elevator = -0.02 * omega gives alpha = -1.0 * omega (mild damping)
        elevator = -0.02 * current_omega

    # Map to SAC continuous action range [-1.0, 1.0]
    # throttle_raw: -1 is 0 thrust, +1 is full thrust.
    throttle_raw = np.clip(2.0 * throttle - 1.0, -1.0, 1.0)
    elevator_raw = np.clip(elevator, -1.0, 1.0)

    return np.array([throttle_raw, elevator_raw], dtype=np.float32)


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
            action = get_held_action(keys, env._state.omega)
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
