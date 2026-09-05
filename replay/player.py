"""
ReplayPlayer: Replays recorded episodes visually or programmatically.

Allows playing back saved trajectory files step-by-step with HUD breakdown overlay,
interactive pause/resume, frame stepping, and speed adjustment.
"""

from pathlib import Path
from typing import Generator, Optional, Tuple, Union

from envs.physics import AircraftState
from envs.world import Obstacle, Waypoint, World, WorldConfig
from rendering.renderer import Renderer
from replay.formats import RecordedEpisode, RecordedStep, load_episode


class ReplayPlayer:
    """
    Player for simulation trajectory replay.
    """

    def __init__(self, episode_or_path: Union[RecordedEpisode, str, Path]) -> None:
        if isinstance(episode_or_path, RecordedEpisode):
            self.episode = episode_or_path
        else:
            self.episode = load_episode(episode_or_path)

        self._current_index: int = 0
        self._world = self._reconstruct_world()

    @property
    def total_steps(self) -> int:
        return len(self.episode.steps)

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def world(self) -> World:
        return self._world

    def _reconstruct_world(self) -> World:
        """
        Reconstruct World instance from recorded episode metadata.
        """
        cfg_dict = self.episode.world_config
        w_cfg = WorldConfig(
            width=float(cfg_dict.get("width", 1000.0)),
            height=float(cfg_dict.get("height", 1000.0)),
        )
        world = World(w_cfg)

        wp_dict = cfg_dict.get("waypoint", {"x": 800.0, "y": 800.0, "radius": 30.0})
        world.waypoint = Waypoint(
            x=float(wp_dict["x"]),
            y=float(wp_dict["y"]),
            radius=float(wp_dict["radius"]),
        )

        obstacles = []
        for o in cfg_dict.get("obstacles", []):
            obstacles.append(
                Obstacle(
                    x=float(o["x"]),
                    y=float(o["y"]),
                    radius=float(o["radius"]),
                )
            )
        world.obstacles = obstacles
        return world

    def get_step(self, step_idx: int) -> RecordedStep:
        """
        Retrieve step at given index.
        """
        if not (0 <= step_idx < self.total_steps):
            raise IndexError(f"Step index {step_idx} out of range [0, {self.total_steps - 1}].")
        return self.episode.steps[step_idx]

    def reconstruct_state(self, step_idx: int) -> AircraftState:
        """
        Reconstruct AircraftState object for step index.
        """
        step = self.get_step(step_idx)
        sd = step.state_dict
        return AircraftState(
            x=sd.get("x", 0.0),
            y=sd.get("y", 0.0),
            vx=sd.get("vx", 0.0),
            vy=sd.get("vy", 0.0),
            theta=sd.get("theta", 0.0),
            omega=sd.get("omega", 0.0),
            fuel=sd.get("fuel", 1000.0),
            step=int(sd.get("step", step_idx)),
        )

    def iter_steps(self) -> Generator[Tuple[AircraftState, RecordedStep], None, None]:
        """
        Generator over all steps in episode yielding (AircraftState, RecordedStep).
        """
        for i in range(self.total_steps):
            self._current_index = i
            state = self.reconstruct_state(i)
            step = self.episode.steps[i]
            yield state, step

    def play_visual(
        self,
        fps: int = 30,
        render_width: int = 1200,
        render_height: int = 800,
    ) -> None:
        """
        Play recorded episode in interactive Pygame visual window.

        Controls:
            SPACE       : Pause / Resume playback
            RIGHT ARROW : Step forward 1 frame (when paused or active)
            LEFT ARROW  : Step backward 1 frame
            R           : Restart playback from step 0
            ESC / Q     : Quit playback
        """
        import time
        import pygame

        renderer = Renderer(
            width=render_width,
            height=render_height,
            world_width=self.world.cfg.width,
            world_height=self.world.cfg.height,
            fps=fps,
        )

        clock = pygame.time.Clock()
        paused = False
        idx = 0
        running = True

        while running and idx < self.total_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_RIGHT:
                        idx = min(idx + 1, self.total_steps - 1)
                    elif event.key == pygame.K_LEFT:
                        idx = max(idx - 1, 0)
                    elif event.key == pygame.K_r:
                        idx = 0

            state = self.reconstruct_state(idx)
            step = self.get_step(idx)

            info_overlay = {
                "Step": f"{idx + 1} / {self.total_steps}",
                "Action": step.action,
                "Ep Total Reward": f"{self.episode.total_reward:.1f}",
                "Term Reason": self.episode.termination_reason,
                "Status": "PAUSED" if paused else "PLAYING",
            }
            if step.info:
                info_overlay.update(step.info)

            renderer.render(
                state=state,
                world=self.world,
                reward_breakdown=step.reward_breakdown,
                info=info_overlay,
            )

            clock.tick(fps)

            if not paused:
                idx += 1

        renderer.close()
