"""
Gridworld Environment Implementation.

This module implements a customizable gridworld environment with support for:
- Multiple terrain types (walls, empty spaces, rewards)
- Stochastic transitions
- Custom map loading
- Visualization

The environment follows the Gymnasium interface and provides a 2D grid-based world
where an agent can move in four directions plus staying in place.
"""

import copy
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence, TypeVar, cast

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
from PIL import Image as Image

# Define colors for visualization
# 0: black (wall); 1: gray (empty); 2: blue (agent); 3: green (pos reward terminal)
# 4: red (neg reward terminal); 5: purple (small negative reward); 6: yellow
COLORS = {
    0: [0.0, 0.0, 0.0],  # Black - Wall
    1: [0.5, 0.5, 0.5],  # Gray - Empty
    2: [0.0, 0.0, 1.0],  # Blue - Agent
    3: [0.0, 1.0, 0.0],  # Green - Positive terminal
    4: [1.0, 0.0, 0.0],  # Red - Negative terminal
    5: [1.0, 0.0, 1.0],  # Purple - Small negative
    6: [1.0, 1.0, 0.0],  # Yellow
}

# Type aliases
State = Tuple[int, int]
StateList = List[State]
TransitionList = List[Tuple[float, State]]

class GridworldEnv(gym.Env):
    """
    A 2D gridworld environment with customizable maps and stochastic transitions.
    
    The environment allows for:
    - Loading custom maps from files
    - Stochastic transitions with configurable noise
    - Multiple terrain types with different rewards
    - Visualization of the environment state
    
    Attributes:
        metadata (Dict): Environment metadata for rendering
        num_env (int): Counter for number of environment instances
        actions (List[int]): Available actions (0=Stay, 1=Up, 2=Down, 3=Left, 4=Right)
        action_space (spaces.Discrete): Gymnasium action space
        observation_space (spaces.Box): Gymnasium observation space
        grid_map_shape (Tuple[int, int]): Shape of the grid map
        num_states (int): Total number of possible states
        num_actions (int): Total number of possible actions
        _transition_noise (float): Probability of random transition to orthogonal direction
    """

    metadata = {"render_modes": ["human"]}
    num_env = 0

    def __init__(self, map_file: str = "plan0.txt", transition_noise: float = 0) -> None:
        """
        Initialize the Gridworld environment.

        Args:
            map_file: Path to the map file relative to this file's location
            transition_noise: Probability of transitioning in an orthogonal direction
        """
        self._seed = 0
        self.actions = [0, 1, 2, 3, 4]  # Stay, Up, Down, Left, Right
        self.inv_actions = [0, 2, 1, 4, 3]
        self.action_space = spaces.Discrete(5)
        self.action_pos_dict = {0: [0, 0], 1: [-1, 0], 2: [1, 0], 3: [0, -1], 4: [0, 1]}
        self.finished = False

        # Initialize system state
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        self.grid_map_path = os.path.join(this_file_path, map_file)
        self.start_grid_map = self._read_grid_map(self.grid_map_path)
        self.current_grid_map = copy.deepcopy(self.start_grid_map)

        # Initialize agent state
        self.agent_start_state = self._get_agent_start_state(self.start_grid_map)
        self.start_grid_map[self.agent_start_state] = 1
        self.agent_state = copy.deepcopy(self.agent_start_state)

        # Set observation space
        self.grid_map_shape = self.start_grid_map.shape
        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array(self.grid_map_shape, dtype=np.float32),
            dtype=np.float32,
        )

        # Set rendering parameters
        self.render_shape = [128, 128, 3]
        self.render_init = False
        self.fig: Optional[plt.Figure] = None
        self.this_fig_num: int = 0

        # Set environment parameters
        self.num_states = np.prod(self.start_grid_map.shape)
        self.num_actions = len(self.actions)
        self._transition_noise = transition_noise
        self.restart_once_done = False
        self.verbose = False

        GridworldEnv.num_env += 1

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, bool]]:
        """
        Execute one time step within the environment.

        Args:
            action: The action to take (0=Stay, 1=Up, 2=Down, 3=Left, 4=Right)

        Returns:
            A tuple containing:
            - observation (np.ndarray): The current state
            - reward (float): The reward for the action
            - terminated (bool): Whether the episode has ended
            - truncated (bool): Whether the episode was artificially terminated
            - info (dict): Additional information about the step

        Raises:
            Exception: If no valid next state exists in the transition function
        """
        if self.finished:
            return (
                np.array(self.agent_state, dtype=np.float32),
                0,
                True,
                False,
                {"success": True},
            )

        action = int(action)
        info = {"success": True}
        cur_state = copy.deepcopy(self.agent_state)

        # Sample from transition function
        distribution = []
        total_prb = 0
        state_dist = self.T(self.agent_state, action)
        for entry in state_dist:
            prb = entry[0]
            state = entry[1]
            if prb > 0:
                total_prb += prb
                distribution.append((total_prb, state))

        random_number = random.random() * total_prb
        next_state = None
        for sample in distribution:
            if random_number < sample[0]:
                next_state = sample[1]
                break

        if next_state is None:
            info["success"] = False
            raise Exception("No valid next state in transition function")

        self.agent_state = copy.copy(next_state)
        reward = self.R(cur_state, action, next_state)
        self.finished = self.is_terminal(next_state)

        return (
            np.array(self.agent_state, dtype=np.float32),
            reward,
            self.finished,
            False,
            info,
        )

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
        self.finished = False
        self.agent_state = copy.deepcopy(self.agent_start_state)
        self.current_grid_map = copy.deepcopy(self.start_grid_map)
        self.observation = self._gridmap_to_observation(self.start_grid_map)
        return np.array(self.agent_state, dtype=np.float32), {}

    def render(self, mode: str = "human") -> None:
        """
        Render the environment.

        Args:
            mode: The mode to render with (only 'human' is supported)
        """
        if not self.render_init:
            self.this_fig_num = GridworldEnv.num_env
            self.fig = plt.figure(self.this_fig_num)
            plt.show(block=False)
            plt.axis("off")
            self.render_init = True

        img = self.observation
        if self.fig is not None:
            plt.figure(self.this_fig_num)
            plt.clf()
            plt.imshow(img)
            self.fig.canvas.draw()
            plt.pause(0.00001)

    def close(self) -> None:
        """Close the environment and clean up any resources."""
        plt.close()

    def get_state_ranges(self) -> np.ndarray:
        """
        Get the range of valid state values.

        Returns:
            Array of shape (2, 2) containing [min_row, max_row], [min_col, max_col]
        """
        return np.array(
            [[0, self.start_grid_map.shape[0]], [0, self.start_grid_map.shape[1]]]
        )

    def R(self, state: State, action: int, next_state: State) -> float:
        """
        Get the reward for a state-action-next_state triple.

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state

        Returns:
            The reward value
        """
        if action == 0:
            return 0
        
        if (self.start_grid_map[next_state[0], next_state[1]] == 3):
            if (self.start_grid_map[state[0], state[1]] == 3):
                return 0
            return 10
        elif (self.start_grid_map[next_state[0], next_state[1]] == 4):
            if (self.start_grid_map[state[0], state[1]] == 4):
                return 0
            return -10
        elif self.start_grid_map[next_state[0], next_state[1]] == 5:
            return -1
        return 0

    def can_move(self, state: State, action: int, next_state: State) -> bool:
        """
        Check if a move from state to next_state is valid.

        Args:
            state: Current state
            action: Action to take
            next_state: Target state

        Returns:
            True if the move is valid, False otherwise
        """
        if next_state[0] < 0 or next_state[0] >= self.grid_map_shape[0]:
            return False
        if next_state[1] < 0 or next_state[1] >= self.grid_map_shape[1]:
            return False
        if self.start_grid_map[next_state[0], next_state[1]] == 0:
            return False  # Can't move to walls
        if self.start_grid_map[state[0], state[1]] == 0:
            return False  # Can't move from walls
        if self.start_grid_map[state[0], state[1]] == 3:
            return False  # Can't move from terminal states
        if self.start_grid_map[state[0], state[1]] == 4:
            return False  # Can't move from terminal states
        return True

    def is_terminal(self, state: State) -> bool:
        """
        Check if a state is terminal.

        Args:
            state: State to check

        Returns:
            True if the state is terminal, False otherwise
        """
        return bool(
            self.start_grid_map[state[0], state[1]] == 3
            or self.start_grid_map[state[0], state[1]] == 4
        )

    def T(self, state: State, action: int, next_state: Optional[State] = None) -> TransitionList:
        """
        Get transition probabilities for a state-action pair.

        Args:
            state: Current state
            action: Action to take
            next_state: If provided, only return probability for this specific next state

        Returns:
            List of (probability, next_state) tuples
        """
        possible_next_states: TransitionList = []

        # Add probability of each non-zero likelihood state
        target_next_state = (
            state[0] + self.action_pos_dict[action][0],
            state[1] + self.action_pos_dict[action][1],
        )

        if self.can_move(state, action, target_next_state):
            possible_next_states = [(1.0 - self._transition_noise, target_next_state)]
        else:
            possible_next_states = [(1.0 - self._transition_noise, state)]

        if self._transition_noise > 0:
            other_actions = [0]  # No-OP as default case
            if action == 1 or action == 2:
                other_actions = [3, 4]
            elif action == 3 or action == 4:
                other_actions = [1, 2]

            for a in other_actions:
                pos_next_state = (
                    state[0] + self.action_pos_dict[a][0],
                    state[1] + self.action_pos_dict[a][1],
                )
                if self.can_move(state, a, pos_next_state):
                    possible_next_states.append(
                        (
                            self._transition_noise / len(other_actions),
                            pos_next_state,
                        )
                    )
                else:
                    possible_next_states.append(
                        (
                            self._transition_noise / len(other_actions),
                            state,
                        )
                    )

        if next_state is not None:
            # Return probability for specific next_state
            probability_mass_next_state = sum(
                prob for prob, state in possible_next_states 
                if state[0] == next_state[0] and state[1] == next_state[1]
            )
            return [(probability_mass_next_state, next_state)]

        return possible_next_states

    def _read_grid_map(self, grid_map_path: str) -> np.ndarray:
        """
        Read a grid map from a file.

        Args:
            grid_map_path: Path to the grid map file

        Returns:
            A numpy array representing the grid map

        Raises:
            FileNotFoundError: If the map file does not exist
        """
        if not os.path.exists(grid_map_path):
            raise FileNotFoundError(f"Map file not found: {grid_map_path}")

        grid_map = []
        with open(grid_map_path, "r") as f:
            for line in f:
                row = [int(x) for x in line.rstrip().split(" ")]
                grid_map.append(row)
        return np.array(grid_map)

    def _get_agent_start_state(self, start_grid_map: np.ndarray) -> State:
        """
        Find the agent's starting position in the grid map.

        Args:
            start_grid_map: The initial grid map array

        Returns:
            A tuple of (row, col) coordinates for the agent's start position

        Raises:
            ValueError: If no agent start state (value 2) is found in the map
        """
        start_state = None
        for i in range(start_grid_map.shape[0]):
            for j in range(start_grid_map.shape[1]):
                if start_grid_map[i, j] == 2:
                    start_state = (i, j)
                    break
            if start_state is not None:
                break
        if start_state is None:
            raise ValueError("No agent start state found in grid map")
        return start_state

    def _gridmap_to_observation(
        self, grid_map: np.ndarray, obs_shape: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Convert a grid map to an RGB observation array.

        Args:
            grid_map: The grid map to convert
            obs_shape: Optional shape for the output observation

        Returns:
            An RGB array representing the grid map
        """
        if obs_shape is None:
            obs_shape = self.render_shape

        observation = np.zeros(obs_shape, dtype=np.float32)
        gs0 = int(observation.shape[0] / grid_map.shape[0])
        gs1 = int(observation.shape[1] / grid_map.shape[1])

        for i in range(grid_map.shape[0]):
            for j in range(grid_map.shape[1]):
                observation[i * gs0:(i + 1) * gs0, j * gs1:(j + 1) * gs1] = np.array(
                    COLORS[grid_map[i, j]]
                )
        return observation

    def change_start_state(self, sp: State) -> bool:
        """
        Change the agent's start state to a new position.

        Args:
            sp: New start position as (row, col) tuple

        Returns:
            True if the change was successful, False otherwise
        """
        if self.start_grid_map[sp[0], sp[1]] == 0:
            return False

        st = self._get_agent_start_state(self.start_grid_map)
        self.start_grid_map[st[0], st[1]] = 1
        self.start_grid_map[sp[0], sp[1]] = 2

        self.current_grid_map = copy.deepcopy(self.start_grid_map)
        self.agent_start_state = copy.deepcopy(sp)
        self.agent_state = copy.deepcopy(sp)

        return True

    def get_agent_state(self) -> State:
        """
        Get the current state of the agent.

        Returns:
            The current position of the agent as (row, col) tuple
        """
        return copy.deepcopy(self.agent_state)

    def get_start_state(self) -> State:
        """
        Get the start state of the agent.

        Returns:
            The start position of the agent as (row, col) tuple
        """
        return copy.deepcopy(self.agent_start_state)
