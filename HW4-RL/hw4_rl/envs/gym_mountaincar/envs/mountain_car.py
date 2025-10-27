"""
Mountain Car Environment Implementation.

This module implements the Mountain Car environment as described in Andrew Moore's PhD Thesis (1990).
Adapted for CSCI 5302 by Brad Hayes <bradley.hayes@colorado.edu>

Original source: http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding


class MountainCarEnv(gym.Env):
    """
    Mountain Car environment with continuous state space and discrete actions.
    
    Description:
        The agent (a car) is started at the bottom of a valley. For any given
        state the agent may choose to accelerate to the left, right or cease
        any acceleration.

    Source:
        The environment appeared first in Andrew Moore's PhD Thesis (1990).

    Observation:
        Type: Box(2)
        Num    Observation               Min            Max
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07

    Actions:
        Type: Discrete(3)
        Num    Action
        0      Accelerate to the Left
        1      Don't accelerate
        2      Accelerate to the Right

        Note: This does not affect the amount of velocity affected by the
        gravitational pull acting on the car.

    Reward:
         Reward of 0 is awarded if the agent reached the flag (position = 0.5)
         on top of the mountain.
         Reward of -1 is awarded if the position of the agent is less than 0.5.

    Starting State:
         The position of the car is assigned a uniform random value in
         [-0.6 , -0.4].
         The starting velocity of the car is always assigned to 0.

    Episode Termination:
         The car position is more than 0.5
         Episode length is greater than 200
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self) -> None:
        """Initialize the Mountain Car environment."""
        self.min_position: float = -1.2
        self.max_position: float = 0.6
        self.max_speed: float = 0.07
        self.goal_position: float = 0.5
        self.force: float = 0.001
        self.gravity: float = 0.0025

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.screen = None
        self.clock = None
        self.isopen: bool = True

        self.state: Optional[np.ndarray] = None
        self.np_random: Optional[np.random.Generator] = None
        self.seed()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.

        Args:
            action: The action to take (0: left, 1: no action, 2: right)

        Returns:
            A tuple containing:
            - observation (np.ndarray): The current state
            - reward (float): The reward for the action
            - terminated (bool): Whether the episode has ended
            - truncated (bool): Whether the episode was artificially terminated
            - info (dict): Additional information about the step
        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        done = bool(position >= self.goal_position)
        reward = -1.0

        self.state = np.array([position, velocity], dtype=np.float32)
        return self.state, reward, done, False, {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Args:
            seed: Optional seed for the random number generator
            options: Optional configuration arguments

        Returns:
            A tuple containing:
            - observation (np.ndarray): Initial state
            - info (dict): Additional information
        """
        super().reset(seed=seed)
        self.state = np.array(
            [self.np_random.uniform(low=-0.6, high=-0.4), 0], dtype=np.float32
        )
        return self.state, {}

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        Set the seed for the environment's random number generator.

        Args:
            seed: The seed value to use

        Returns:
            A list containing the seed used
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _height(self, xs: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the height of the mountain at given x-coordinates.

        Args:
            xs: X-coordinate(s) to calculate height for

        Returns:
            The height(s) of the mountain at the given x-coordinate(s)
        """
        return np.sin(3 * xs) * 0.45 + 0.55

    def render(self, mode: str = "human") -> Union[np.ndarray, bool]:
        """
        Render the environment.

        Args:
            mode: The mode to render with ('human' or 'rgb_array')

        Returns:
            For mode 'rgb_array': np.ndarray of the rendered frame
            For mode 'human': True if the window is still open

        Raises:
            DependencyNotInstalled: If pygame is not installed
        """
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        pos = self.state[0]

        # Track
        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        xys = list(zip((xs - self.min_position) * scale, ys * scale))

        pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

        clearance = 10

        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            coords.append(
                (
                    c[0] + (pos - self.min_position) * scale,
                    c[1] + clearance + self._height(pos) * scale,
                )
            )

        gfxdraw.aapolygon(self.surf, coords, (0, 0, 255))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 255))

        for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            wheel = (
                int(c[0] + (pos - self.min_position) * scale),
                int(c[1] + clearance + self._height(pos) * scale),
            )

            gfxdraw.aacircle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )
            gfxdraw.filled_circle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )

        self.screen.blit(self.surf, (0, 0))

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self) -> None:
        """Close the environment and clean up any resources."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def get_keys_to_action(self) -> Dict[Tuple[int, ...], int]:
        """
        Get the mapping of keyboard keys to actions.

        Returns:
            A dictionary mapping keyboard key tuples to action indices
        """
        # Control with left and right arrow keys.
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}
