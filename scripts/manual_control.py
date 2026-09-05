"""
scripts/manual_control.py
==========================
Interactive Keyboard Control -- MDMARD Phase 1

Lets you fly the aircraft yourself using keyboard input.
This is the single most important debugging tool:
  "If you can't fly it yourself, the RL agent can't learn it."

Controls
--------
  W  or  UP    : Thrust + maintain pitch
  A             : Pitch Up   (nose up)
  D             : Pitch Down (nose down)
  SPACE         : Glide      (no thrust, no pitch change)
  S             : Pitch Down (aggressive)
  R             : Reset episode (new random spawn + waypoint)
  Q  or  ESC   : Quit

Action Mapping (matches TacticalFlightEnv.action_space)
---------------------------------------------------------
  0: Thrust only
  1: Pitch Up
  2: Pitch Down
  3: Thrust + Pitch Up
  4: Thrust + Pitch Down
  5: Glide (no-op)

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

# ── Path setup (run from project root or any dir) ────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.flight_env import TacticalFlightEnv
from envs.physics import PhysicsConfig
from envs.world import WorldConfig
from rendering.renderer import Renderer


# ─────────────────────────────────────────────────────────────────────────────
# Keyboard → Action Mapping
# ─────────────────────────────────────────────────────────────────────────────

def build_action_map():
    """Return a dict mapping Pygame key → discrete action id."""
    import pygame
    return {
        pygame.K_w:     0,   # Thrust
        pygame.K_UP:    0,   # Thrust (arrow)
        pygame.K_a:     1,   # Pitch Up
        pygame.K_d:     2,   # Pitch Down
        pygame.K_DOWN:  2,   # Pitch Down (arrow)
        pygame.K_e:     3,   # Thrust + Pitch Up
        pygame.K_q:     4,   # Thrust + Pitch Down
        pygame.K_SPACE: 5,   # Glide (no-op)
        pygame.K_s:     2,   # Pitch Down (alternative)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Episode loop
# ─────────────────────────────────────────────────────────────────────────────

def get_held_action(keys, action_map) -> int:
    """
    Read held keys and return the highest-priority matching action.

    Priority: Thrust > Pitch Up > Pitch Down > Glide
    """
    import pygame

    # Check combined: thrust + pitch
    thrust = keys[pygame.K_w] or keys[pygame.K_UP]
    pitch_up   = keys[pygame.K_a]
    pitch_down = keys[pygame.K_d] or keys[pygame.K_DOWN] or keys[pygame.K_s]

    if thrust and pitch_up:
        return 3   # Thrust + Pitch Up
    if thrust and pitch_down:
        return 4   # Thrust + Pitch Down
    if thrust:
        return 0   # Thrust only
    if pitch_up:
        return 1   # Pitch Up
    if pitch_down:
        return 2   # Pitch Down
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

    print("\n" + "═" * 60)
    print("  MDMARD Manual Control  |  Press Q to quit, R to reset")
    print("═" * 60)
    print("  W/↑  : Thrust      A : Pitch Up    D : Pitch Down")
    print("  E    : Thrust+Up   Q : Thrust+Down SPACE : Glide")
    print("  R    : Reset episode")
    print("═" * 60 + "\n")

    action_map = build_action_map()

    # ── Main loop ─────────────────────────────────────────────────────────
    running = True
    paused  = False

    while running:
        # ── Event handling ────────────────────────────────────────────────
        events = renderer.handle_events()
        if events and events[0] is None:
            running = False
            break

        keys = pygame.key.get_pressed()

        # Quit
        if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
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
            action = get_held_action(keys, action_map)
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
