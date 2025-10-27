# !/usr/bin/env python
"""
CSCI 5302 HW2: Tabular Solution Implementation

This module implements various reinforcement learning algorithms for solving gridworld
problems using tabular methods. It includes value iteration, deterministic policy iteration,
and stochastic policy iteration approaches.

The main components are:
- TabularPolicy: A class representing discrete state-action policies
- GridworldSolver: A class that handles policy computation and visualization
"""

import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from hw4_rl.envs import GridworldEnv

student_name = "Mandar Deshmukh"  # Set to your name
GRAD = True  # Set to True if graduate student


class TabularPolicy:
    """
    A tabular policy implementation for discrete state/action spaces.

    This class implements a tabular policy and value function for reinforcement learning
    in discrete state/action spaces. It maintains mappings from states to values and
    from states to action probability distributions.

    Attributes:
        num_states (int): Total number of discrete states
        num_actions (int): Total number of possible actions
        state_ranges (np.ndarray): Array of [min, max) ranges for each state dimension
        _value_function (np.ndarray): Array mapping states to their values
        _policy (np.ndarray): Array mapping states to action probability distributions
    """

    def __init__(self, n_states: int, state_ranges: np.ndarray, n_actions: int) -> None:
        """
        Initialize the tabular policy.

        Args:
            n_states: Number of discrete states
            state_ranges: Array of [min, max) ranges for each state dimension
            n_actions: Number of possible actions
        """
        self.num_states = n_states
        self.num_actions = n_actions
        self.state_ranges = state_ranges
        
        # Create data structure to store mapping from state to value
        self._value_function = np.zeros(shape=n_states)

        # Create data structure to store array with probability of each action for each state
        self._policy = np.random.uniform(0, 1, size=(n_states, self.num_actions))

    def get_action(self, state: Union[int, np.ndarray]) -> int:
        """
        Sample an action from the policy's action distribution for the given state.

        Args:
            state: The current state index or state coordinate vector

        Returns:
            The sampled action index
        """
        # Convert state to integer if it's a numpy array
        if isinstance(state, np.ndarray):
            state = self.get_state_index_from_coordinates(state)
        prob_dist = np.array(self._policy[state])
        assert prob_dist.ndim == 1

        # Sample from policy distribution for state
        idx = np.random.multinomial(1, prob_dist / np.sum(prob_dist))
        return np.argmax(idx)

    def set_state_value(self, state: int, value: float) -> None:
        """
        Set the value for a given state.

        Args:
            state: The state index
            value: The value to set for this state
        """
        self._value_function[state] = value

    def get_state_value(self, state: Union[int, np.ndarray]) -> float:
        """
        Get the value for a given state.

        Args:
            state: Either a state index or state coordinate vector

        Returns:
            The value for the given state
        """
        if isinstance(state, int):
            return self._value_function[state]
        else:
            # Map state vector to state index
            return self._value_function[self.get_state_index_from_coordinates(state)]

    def get_state_index_from_coordinates(self, state: np.ndarray) -> int:
        """
        Convert a state coordinate vector to its corresponding state index.

        Args:
            state: A numpy array containing the (x,y) coordinates of the state

        Returns:
            The integer index corresponding to the state coordinates
        """
        # Convert numpy array to tuple of integers
        if isinstance(state, np.ndarray):
            state = tuple(state.astype(int))
        return state[0] * (self.state_ranges[0][1] - self.state_ranges[0][0]) + state[1]

    def get_coordinates_from_state_index(self, state_idx: int) -> np.ndarray:
        """
        Convert a state index to its corresponding coordinate vector.

        Args:
            state_idx: The integer index of the state

        Returns:
            A numpy array containing the (x,y) coordinates corresponding to the state index
        """
        return np.array(
            [
                state_idx // (self.state_ranges[0][1] - self.state_ranges[0][0]),
                state_idx % (self.state_ranges[0][1] - self.state_ranges[0][0]),
            ]
        )

    def get_value_function(self) -> np.ndarray:
        """
        Get a deep copy of the current value function.

        Returns:
            A numpy array representing the value function table, where each entry
            maps a state index to its value
        """
        return copy.deepcopy(self._value_function)

    def set_value_function(self, v: np.ndarray) -> None:
        """
        Set the value function to a new array.

        Args:
            v: A numpy array containing the new value function table
        """
        self._value_function = copy.copy(v)

    def set_policy(self, state: int, action_prob_array: np.ndarray) -> None:
        """
        Set the action probability distribution for a given state.

        Args:
            state: The state index
            action_prob_array: A numpy array containing probabilities for each action
        """
        self._policy[state] = copy.copy(action_prob_array)

    def get_policy(self, state: Union[int, np.ndarray]) -> np.ndarray:
        """
        Get the action probability distribution for a given state.

        Args:
            state: Either a state index or state coordinate vector

        Returns:
            A numpy array containing probabilities for each action in the given state
        """
        if isinstance(state, int):
            return self._policy[state]
        else:
            # Map state vector to state index
            return self._policy[self.get_state_index_from_coordinates(state)]

    def get_policy_function(self) -> np.ndarray:
        """
        Get a deep copy of the current policy function.

        Returns:
            A numpy array representing the policy table, where each entry maps
            a state to a probability distribution over actions
        """
        return copy.deepcopy(self._policy)

    def set_policy_function(self, p: np.ndarray) -> None:
        """
        Set the policy function to a new array.

        Args:
            p: A numpy array containing the new policy table
        """
        self._policy = copy.copy(p)


class GridworldSolver:
    """
    A solver for gridworld reinforcement learning problems.

    This class implements various policy computation methods for gridworld environments,
    including deterministic value iteration, stochastic policy iteration, and
    deterministic policy iteration.

    Attributes:
        _policy_type (str): Type of policy computation method to use
        env (gym.Env): The gridworld environment
        env_name (str): Name of the environment
        temperature (float): Temperature parameter for stochastic policies
        eps (float): Small constant for numerical stability
        gamma (float): Discount factor for future rewards
        solver (TabularPolicy): The policy object that stores computed policies and values
        performance_history (List[float]): History of cumulative rewards from policy evaluations
    """

    def __init__(
        self,
        policy_type: str = "deterministic_vi",
        gridworld_map_number: int = 0,
        noisy_transitions: bool = False,
    ) -> None:
        """
        Initialize the GridworldSolver.

        Args:
            policy_type: The type of policy computation to use. Must be one of:
                ["deterministic_vi", "stochastic_pi", "deterministic_pi"]
            gridworld_map_number: Which gridworld map to use (0 or 1)
            noisy_transitions: Whether to use noisy state transitions
            max_ent_temperature: Temperature parameter for stochastic policies

        Raises:
            AssertionError: If policy_type is not one of the allowed values
        """
        self._policy_type = policy_type
        assert policy_type in ["deterministic_vi", "stochastic_pi", "deterministic_pi"]
        self.env: Optional[gym.Env] = None
        self.env_name = ""
        self.init_environment(gridworld_map_number, noisy_transitions)
        self.theta = 1e-3 #parameter to determine when to stop policy evaluation
        self.gamma = 0.99 #future return discount factor
        self.eps = 1e-8   #small constant for numerical stability

        # Get the unwrapped environment to access its attributes
        assert self.env is not None
        unwrapped_env = self.env.unwrapped
        self.solver = TabularPolicy(
            unwrapped_env.num_states,
            unwrapped_env.get_state_ranges(),
            unwrapped_env.num_actions,
        )
        self.performance_history: List[float] = []

    def init_environment(
        self, gridworld_map_number: int = 0, noisy_transitions: bool = False
    ) -> None:
        """
        Initialize the gridworld environment.

        Args:
            gridworld_map_number: Which gridworld map to use (0 or 1)
            noisy_transitions: Whether to use noisy state transitions

        Raises:
            AssertionError: If gridworld_map_number is not 0 or 1
        """
        assert gridworld_map_number in [0, 1]
        if noisy_transitions:
            self.env_name = f"gridworldnoisy-v{gridworld_map_number}"
        else:
            self.env_name = f"gridworld-v{gridworld_map_number}"

        self.env = gym.make(self.env_name)
        self.env.reset()

    def compute_policy(self) -> None:
        """
        Compute optimal policy using the specified algorithm.
        
        This method selects and runs the appropriate policy computation algorithm based on
        the policy_type specified during initialization.
        """
        if self._policy_type == "deterministic_vi":
            self._value_iteration()
        elif self._policy_type == "stochastic_pi":
            self._stochastic_policy_iteration()
        else:  # deterministic_pi
            self._deterministic_policy_iteration()

    def solve(
        self,
        start_state: Optional[np.ndarray] = None,
        visualize: bool = False,
        max_steps: float = float("inf"),
    ) -> Tuple[float, int]:
        """
        Execute the current policy in the environment.
        
        This method runs the current policy from a given start state (or default start state)
        and returns the cumulative reward and number of steps taken.

        Args:
            start_state: Optional starting state coordinates
            visualize: Whether to render the environment
            max_steps: Maximum number of steps to take

        Returns:
            Tuple of (cumulative_reward, num_steps)
        """
        assert self.env is not None
        state, _ = self.env.reset()
        if start_state is not None:
            self.env.unwrapped.change_start_state(start_state)
            state = start_state

        if visualize:
            self.env.render()

        episode_reward = 0
        num_steps = 0
        done = False

        while not done and num_steps < max_steps:
            # Get action using current policy
            action = self.solver.get_action(state)

            # Execute action
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            state = next_state
            done = terminated or truncated

            episode_reward += reward
            num_steps += 1

            if visualize:
                self.env.render()

        return episode_reward, num_steps

    def plot_policy_curve(
        self, reward_history: List[float], filename: Optional[str] = None
    ) -> None:
        """
        Plot the learning curve showing policy performance over iterations.

        Args:
            reward_history: List of rewards from each policy evaluation
            filename: Optional path to save the plot
        """
        plt.figure()
        plt.plot(range(len(reward_history)), reward_history)
        plt.xlabel("Iteration")
        plt.ylabel("Return")
        plt.title("Policy Iteration Performance")

        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, "..", "..", "hw4_rl", "figures")
            os.makedirs(figures_dir, exist_ok=True)
            filename = os.path.join(figures_dir, "gridworld_learning_curve.png")

        plt.savefig(filename)
        plt.close()

    def plot_value_function(
        self, value_function: np.ndarray, filename: Optional[str] = None
    ) -> Tuple[np.ndarray, Figure]:
        """
        Plot the value function as a heatmap.

        Args:
            value_function: Array of state values to plot
            filename: Optional path to save the plot

        Returns:
            Tuple of (image_array, matplotlib_figure)
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)

        # Normalize and reshape values
        V = (value_function - value_function.min()) / (
            value_function.max() - value_function.min() + self.eps
        )
        V = V.reshape(
            self.solver.state_ranges[0][1] - self.solver.state_ranges[0][0],
            self.solver.state_ranges[1][1] - self.solver.state_ranges[1][0],
        )

        V = np.flipud(V)

        # Create heatmap
        image = (plt.cm.coolwarm(V)[::-1, :, :-1] * 255.0).astype(np.uint8)
        ax.set_title(f"Env: {self.env_name}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.imshow(image)

        # Save plot
        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, "..", "figures")
            os.makedirs(figures_dir, exist_ok=True)
            filename = os.path.join(
                figures_dir, f"{self.env_name}_{self._policy_type}_value.png"
            )
        #plt.show()
        plt.savefig(filename)
        # plt.close()

        # Convert to image array
        canvas.draw()
        image = np.asarray(canvas.buffer_rgba()).reshape(
            int(fig.get_size_inches()[1] * fig.get_dpi()),
            int(fig.get_size_inches()[0] * fig.get_dpi()),
            4,
        )[:, :, :3]

        return image, fig
    
    def plot_policy(self, policy: Optional[np.ndarray] = None, filename: Optional[str] = None) -> None:
        """
        Plot the policy as arrows on the grid.
        
        Args:
            policy: Optional policy table to plot (defaults to solver policy)
            filename: Optional path to save the plot
        """
        if policy is None:
            policy = self.solver.get_policy_function()

        # Get dimensions of the grid
        rows = self.solver.state_ranges[0][1] - self.solver.state_ranges[0][0]
        cols = self.solver.state_ranges[1][1] - self.solver.state_ranges[1][0]

        fig, ax = plt.subplots(figsize=(cols, rows))
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_xticks(np.arange(0, cols + 1, 1))
        ax.set_yticks(np.arange(0, rows + 1, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)

        unwrapped_env = self.env.unwrapped

        # Arrow directions (assuming env.actions = [0:UP, 1:RIGHT, 2:DOWN, 3:LEFT])
        arrow_dict = {
            0: (0, 0),      # stay / no movement
            1: (0, 0.4),    # up
            2: (0, -0.4),   # down
            3: (-0.4, 0),   # left
            4: (0.4, 0),    # right          
        }

        for s in range(self.solver.num_states):
            coord = self.solver.get_coordinates_from_state_index(s)
            row, col = coord
            # Flip vertically for plotting (so row 0 is at top)
            y = rows - row - 1
            x = col

            # Choose best action(s) for deterministic policy
            best_actions = np.where(policy[s] == np.max(policy[s]))[0]
            for a in best_actions:
                if a == 0:
                    continue
                dx, dy = arrow_dict[a]
                ax.arrow(
                    x + 0.5, y + 0.5, dx, dy,
                    head_width=0.2, head_length=0.2, fc="k", ec="k"
                )

        ax.set_title(f"Policy for {self.env_name}")
        plt.tight_layout()

        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, "..", "..", "hw4_rl", "figures")
            os.makedirs(figures_dir, exist_ok=True)
            filename = os.path.join(
                figures_dir, f"{self.env_name}_{self._policy_type}_policy.png"
            )

        plt.savefig(filename)
        plt.show()
        plt.close()

    def _value_iteration(self) -> None:
        """
        Implement value iteration algorithm.
        
        This method iteratively updates state values based on the Bellman optimality
        equation until convergence. The student needs to implement:
        
        1. Value function update using the Bellman optimality equation:
           V(s) = max_a [ sum_s' T(s,a,s')[R(s,a,s') + gamma * V(s')] ]
           
        2. Policy update to be deterministic (probability 1 for best action):
           pi(s,a) = 1 if a = argmax_a Q(s,a), 0 otherwise
           where Q(s,a) = sum_s' T(s,a,s')[R(s,a,s') + gamma * V(s')]
           
        3. Check for convergence by comparing old and new value functions
        
        The transition probabilities T and rewards R are pre-computed and stored in
        the T and R matrices respectively.
        """
        horizon = 50 #run policy evaluation for a fixed number of iterations
        unwrapped_env = self.env.unwrapped  # Get unwrapped environment

        v_i = np.zeros(unwrapped_env.num_states)
        p_i = np.zeros((unwrapped_env.num_states, unwrapped_env.num_actions))

        # Pre-compute transition and reward matrices
        T = np.zeros(
            (
                unwrapped_env.num_states,
                unwrapped_env.num_actions,
                unwrapped_env.num_states,
            )
        )
        R = np.zeros(
            (
                unwrapped_env.num_states,
                unwrapped_env.num_actions,
                unwrapped_env.num_states,
            )
        )

        for s in range(unwrapped_env.num_states):
            s_coord = self.solver.get_coordinates_from_state_index(s)
            for a in unwrapped_env.actions:
                next_state = self.solver.get_state_index_from_coordinates(
                    unwrapped_env.T(s_coord, a)[0][1]
                )
                T[s, a, next_state] = unwrapped_env.T(
                    s_coord,
                    a,
                    self.solver.get_coordinates_from_state_index(next_state),
                )[0][0]
                R[s, a, next_state] = unwrapped_env.R(
                    s_coord,
                    a,
                    self.solver.get_coordinates_from_state_index(next_state),
                )

        for iteration in range(horizon):
            # Student code here     
            # Update value function
            v_old = v_i.copy()  # Store old values for convergence check
    
            # Compute Q-values for all state-action pairs
            # Q[s, a] = sum_s' T(s,a,s') * [R(s,a,s') + gamma * V(s')]
            q_values = T * (R + self.gamma * v_i)  # Broadcasting v_i
            q_values = np.sum(q_values, axis=2)  # Sum over next states
            
            # Update value function: V(s) = max_a Q(s,a)
            v_i = np.max(q_values, axis=1)
            
            # Update policy to be deterministic (greedy policy)
            p_i = np.zeros((unwrapped_env.num_states, unwrapped_env.num_actions))
            best_actions = np.argmax(q_values, axis=1)
            p_i[np.arange(unwrapped_env.num_states), best_actions] = 1.0
            
            # Check for convergence
            max_diff = np.max(np.abs(v_i - v_old))
            
            # Evaluate current policy every 10 iterations
            if iteration % 10 == 0:
                self.solver.set_policy_function(p_i)
                self.solver.set_value_function(v_i)
                reward, steps = self.solve(max_steps=100)
                self.performance_history.append(reward)
                print(f"VI Iteration {iteration}, max_diff: {max_diff:.6f}, reward: {reward:.2f}")
            
            # Stop if converged
            if max_diff < self.theta:
                print(f"Value Iteration converged after {iteration} iterations")
                break

                # Final evaluation if not done in loop
        if len(self.performance_history) == 0 or iteration % 10 != 0:
            self.solver.set_policy_function(p_i)
            self.solver.set_value_function(v_i)
            reward, steps = self.solve(max_steps=100)
            self.performance_history.append(reward)
        
        print(f"Value Iteration completed. Final reward: {reward:.2f}")

    def _deterministic_policy_iteration(self) -> None:
        """
        Implement deterministic policy iteration.
        
        This method alternates between policy evaluation and policy improvement steps,
        selecting the best action in each state deterministically. The student needs to implement:
        
        1. Policy Evaluation: Update value function using current policy:
           V(s) = sum_a pi(s,a) * sum_s' T(s,a,s')[R(s,a,s') + gamma * V(s')]
           Note: Since policy is deterministic, this simplifies to:
           V(s) = sum_s' T(s,a*,s')[R(s,a*,s') + gamma * V(s')]
           where a* is the action with probability 1 in state s
           
        2. Policy Improvement: Update policy to be deterministic for best action:
           pi(s,a) = 1 if a = argmax_a Q(s,a), 0 otherwise
           where Q(s,a) = sum_s' T(s,a,s')[R(s,a,s') + gamma * V(s')]
        
        The transition probabilities T and rewards R are pre-computed and stored in
        the T and R matrices respectively.
        """
        horizon = 50 #run policy evaluation for a fixed number of iterations
        unwrapped_env = self.env.unwrapped  # Get unwrapped environment

        p_i = self.solver.get_policy_function()

        # Pre-compute transition and reward matrices
        T = np.zeros(
            (
                unwrapped_env.num_states,
                unwrapped_env.num_actions,
                unwrapped_env.num_states,
            )
        )
        R = np.zeros(
            (
                unwrapped_env.num_states,
                unwrapped_env.num_actions,
                unwrapped_env.num_states,
            )
        )

        for s in range(unwrapped_env.num_states):
            s_coord = self.solver.get_coordinates_from_state_index(s)
            for a in unwrapped_env.actions:
                next_state = self.solver.get_state_index_from_coordinates(
                    unwrapped_env.T(s_coord, a)[0][1]
                )
                T[s, a, next_state] = unwrapped_env.T(
                    s_coord,
                    a,
                    self.solver.get_coordinates_from_state_index(next_state),
                )[0][0]
                R[s, a, next_state] = unwrapped_env.R(
                    s_coord,
                    a,
                    self.solver.get_coordinates_from_state_index(next_state),
                )

        for k in range(50):
            print("Policy Iteration %d" % k)

            # Policy Evaluation
            elapsed = time.time()
            v_i = np.zeros(unwrapped_env.num_states)

            # Computing policy-specific transition and reward matrices
            # policy_T[s, s'] = sum_a pi(s,a) * T(s,a,s')
            policy_T = np.sum(
                T * p_i[:, :, np.newaxis],  # Weight transitions by policy
                axis=1  # Sum over actions
            )

            # policy_R[s, s'] = sum_a pi(s,a) * R(s,a,s')
            policy_R = np.sum(
                R * p_i[:, :, np.newaxis],  # Weight rewards by policy
                axis=1  # Sum over actions
            )

            for _ in range(horizon):
                # Get expected value for current policy
                # Student code here
                # Following Bellman expectation equation for policy pi:
                # V^pi(s) = sum_s' T^pi(s,s') * [R^pi(s,s') + gamma * V^pi(s')]
                v_new = np.sum(
                    policy_T * (policy_R + self.gamma * v_i),
                    axis=1  # Sum over next states
                )
                
                # Check convergence of policy evaluation
                if np.max(np.abs(v_new - v_i)) < self.theta:
                    v_i = v_new
                    break
                v_i = v_new

            elapsed = time.time() - elapsed
            print(f".....Evaluate done in {elapsed:.3f}s")

            # Policy Improvement
            # Student code here
            # Saving old policy to check for convergence
            p_old = p_i.copy()

            # Computing Q-values: Q(s,a) = sum_s' T(s,a,s')[R(s,a,s') + gamma * V(s')]
            q_values = T * (R + self.gamma * v_i)  # Shape: (num_states, num_actions, num_states)
            q_values = np.sum(q_values, axis=2) 

            # Update policy to be deterministic
            p_i = np.zeros((unwrapped_env.num_states, unwrapped_env.num_actions))
            best_actions = np.argmax(q_values, axis=1)
            p_i[np.arange(unwrapped_env.num_states), best_actions] = 1.0

            elapsed = time.time() - elapsed
            print(f".....Improve done in {elapsed:.3f}s")

            #self.solver.set_policy_function(p_i)
            #self.performance_history.append(self.solve(max_steps=20)[0])

            # Update solver and evaluate performance
            self.solver.set_policy_function(p_i)
            self.solver.set_value_function(v_i)
            reward, steps = self.solve(max_steps=100)
            self.performance_history.append(reward)
            print(f"Performance: {reward:.2f} reward in {steps} steps")

            # Check if policy has converged (policy is stable)
            policy_stable = np.array_equal(p_i, p_old)
            if policy_stable:
                print(f"Policy converged after {k+1} iterations")
                break

            self.solver.set_value_function(v_i)

        print(f"Policy Iteration completed. Final reward: {self.solve(max_steps=100)[0]:.2f}")

if __name__ == "__main__":
    print("=" * 60)
    print("GRIDWORLD REINFORCEMENT LEARNING")
    print("=" * 60)

    ############ Q1.1 ############
    print("\n### Q1.1: Policy Iteration on Gridworld 0 ###")
    gw0_pi_solver = GridworldSolver(
        policy_type="deterministic_pi", 
        gridworld_map_number=0
    )

    start_time = time.time()
    gw0_pi_solver.compute_policy()
    elapsed_time = time.time() - start_time
    print(f"Computed Policy Iteration in {elapsed_time:.3f} seconds")
    
    # Plot results
    gw0_pi_solver.plot_value_function(gw0_pi_solver.solver.get_value_function())
    gw0_pi_solver.plot_policy()
    gw0_pi_solver.plot_policy_curve(gw0_pi_solver.performance_history)

    ############ Q1.2 ############
    
    print("\n### Q1.2: Value Iteration on Gridworld 0 ###")
    gw0_vi_solver = GridworldSolver(
        policy_type="deterministic_vi", 
        gridworld_map_number=0
    )
    
    start_time = time.time()
    gw0_vi_solver.compute_policy()
    elapsed_time = time.time() - start_time
    print(f"Computed Value Iteration in {elapsed_time:.3f} seconds")
    
    # Plot results
    gw0_vi_solver.plot_value_function(gw0_vi_solver.solver.get_value_function())
    gw0_vi_solver.plot_policy()
    gw0_vi_solver.plot_policy_curve(gw0_vi_solver.performance_history)

    # ========== TEST ON GRIDWORLD 1 ==========
    print("\n### Testing on Gridworld 1 ###")
    
    print("\n--- Policy Iteration on Gridworld 1 ---")
    gw1_pi_solver = GridworldSolver(
        policy_type="deterministic_pi", 
        gridworld_map_number=1
    )
    
    start_time = time.time()
    gw1_pi_solver.compute_policy()
    elapsed_time = time.time() - start_time
    print(f"Computed Policy Iteration in {elapsed_time:.3f} seconds")
    
    gw1_pi_solver.plot_value_function(gw1_pi_solver.solver.get_value_function())
    gw1_pi_solver.plot_policy()
    gw1_pi_solver.plot_policy_curve(gw1_pi_solver.performance_history)
    
    print("\n--- Value Iteration on Gridworld 1 ---")
    gw1_vi_solver = GridworldSolver(
        policy_type="deterministic_vi", 
        gridworld_map_number=1
    )
    
    start_time = time.time()
    gw1_vi_solver.compute_policy()
    elapsed_time = time.time() - start_time
    print(f"Computed Value Iteration in {elapsed_time:.3f} seconds")
    
    gw1_vi_solver.plot_value_function(gw1_vi_solver.solver.get_value_function())
    gw1_vi_solver.plot_policy()
    gw1_vi_solver.plot_policy_curve(gw1_vi_solver.performance_history)

    # ========== COMPARISON ==========
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print("\nGridworld 0:")
    print(f"  Policy Iteration - Final Reward: {gw0_pi_solver.performance_history[-1]:.2f}")
    print(f"  Value Iteration  - Final Reward: {gw0_vi_solver.performance_history[-1]:.2f}")
    print("\nGridworld 1:")
    print(f"  Policy Iteration - Final Reward: {gw1_pi_solver.performance_history[-1]:.2f}")
    print(f"  Value Iteration  - Final Reward: {gw1_vi_solver.performance_history[-1]:.2f}")
    print("\n" + "=" * 60)