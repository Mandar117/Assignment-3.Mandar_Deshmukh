#!/usr/bin/env python
"""
CSCI 5302 HW2: Mountain Car Continuous State Space Solution
Last updated 9/23, 5:00pm

This module implements various reinforcement learning algorithms for solving the Mountain Car
problem with continuous state spaces. It includes value iteration, deterministic policy iteration,
and stochastic policy iteration approaches, all using state space discretization.

The main components are:
- TabularPolicy: A class representing discrete state-action policies
- DiscretizedSolver: A class that handles continuous state space discretization and policy computation
- Various utility functions for policy evaluation and visualization
"""

version = "v2025.02.26.1400"

import copy
import itertools
import sys
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence, Callable

import gymnasium as gym
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.special import softmax

from hw4_rl.envs import MountainCarEnv

student_name = "Mandar Deshmukh"  # Set to your name
GRAD = True  # Set to True if graduate student


class TabularPolicy:
    """
    A tabular policy for discrete state and action spaces.
    
    This class implements a tabular policy that maintains state values, action probabilities,
    and transition/reward functions for a discretized state-action space.
    
    Attributes:
        num_states (int): Total number of discrete states
        num_actions (int): Total number of possible actions
        _transition_function (np.ndarray): State transition probabilities of shape (num_states, num_actions, num_states)
        _reward_function (np.ndarray): Reward values of shape (num_states, num_actions, num_states)
        _value_function (np.ndarray): State values of shape (num_states,)
        _policy (np.ndarray): Action probabilities of shape (num_states, num_actions)
    """

    def __init__(self, num_bins_per_dim: int, num_dims: int, num_actions: int) -> None:
        """
        Initialize the tabular policy.

        Args:
            num_bins_per_dim: Number of bins per state dimension
            num_dims: Number of state dimensions
            num_actions: Number of possible actions
        """
        self.num_states = num_bins_per_dim**num_dims
        self.num_actions = num_actions

        # Initialize transition and reward functions
        self._transition_function = np.zeros(
            (self.num_states, self.num_actions, self.num_states)
        )
        self._reward_function = np.zeros(
            (self.num_states, self.num_actions, self.num_states)
        )

        # Initialize value function and policy
        self._value_function = np.zeros(self.num_states)
        self._policy = np.random.uniform(0, 1, size=(self.num_states, self.num_actions))

    def get_action(self, state: int) -> int:
        """
        Sample an action from the policy's distribution at the given state.

        Args:
            state: The current state index

        Returns:
            The sampled action index
        """
        prob_dist = np.array(self._policy[state])
        assert prob_dist.ndim == 1
        idx = np.random.multinomial(1, prob_dist / np.sum(prob_dist))
        return np.argmax(idx)

    def set_state_value(self, state: int, value: float) -> None:
        """
        Set the value of a given state.

        Args:
            state: The state index
            value: The value to set
        """
        self._value_function[state] = value

    def get_state_value(self, state: int) -> float:
        """
        Get the value of a given state.

        Args:
            state: The state index

        Returns:
            The value of the state
        """
        return self._value_function[state]

    def get_value_function(self) -> np.ndarray:
        """
        Get a copy of the entire value function.

        Returns:
            A copy of the value function array
        """
        return copy.deepcopy(self._value_function)

    def set_value_function(self, values: np.ndarray) -> None:
        """
        Set the entire value function.

        Args:
            values: Array of values to set
        """
        self._value_function = copy.copy(values)

    def set_policy(self, state: int, action_probs: np.ndarray) -> None:
        """
        Set action probabilities for a given state.

        Args:
            state: The state index
            action_probs: Array of action probabilities
        """
        self._policy[state] = copy.copy(action_probs)

    def get_policy(self, state: int) -> np.ndarray:
        """
        Get action probabilities for a given state.

        Args:
            state: The state index

        Returns:
            Array of action probabilities
        """
        return self._policy[state]

    def get_policy_function(self) -> np.ndarray:
        """
        Get a copy of the entire policy function.

        Returns:
            A copy of the policy array
        """
        return copy.deepcopy(self._policy)

    def set_policy_function(self, policy: np.ndarray) -> None:
        """
        Set the entire policy function.

        Args:
            policy: Array of policy values to set
        """
        self._policy = copy.deepcopy(policy)

    def set_transition(self, state: int, action: int, next_state: int, prob: float) -> None:
        """
        Set transition probability for (s,a,s') tuple.

        Args:
            state: Current state index
            action: Action index
            next_state: Next state index
            prob: Transition probability
        """
        self._transition_function[state, action, next_state] = prob

    def get_transition(self, state: int, action: int, next_state: int) -> float:
        """
        Get transition probability for (s,a,s') tuple.

        Args:
            state: Current state index
            action: Action index
            next_state: Next state index

        Returns:
            The transition probability
        """
        return self._transition_function[state, action, next_state]

    def set_reward(self, state: int, action: int, next_state: int, reward: float) -> None:
        """
        Set reward for (s,a,s') tuple.

        Args:
            state: Current state index
            action: Action index
            next_state: Next state index
            reward: Reward value
        """
        self._reward_function[state, action, next_state] = reward

    def get_reward(self, state: int, action: int, next_state: int) -> float:
        """
        Get reward for (s,a,s') tuple.

        Args:
            state: Current state index
            action: Action index
            next_state: Next state index

        Returns:
            The reward value
        """
        return self._reward_function[state, action, next_state]


class DiscretizedSolver:
    """
    Solver for continuous state spaces using discretization.
    
    This class implements various reinforcement learning algorithms for solving continuous state
    space problems by discretizing the state space into bins. It supports different policy
    computation methods including value iteration, deterministic policy iteration, and
    stochastic policy iteration.
    
    Attributes:
        _mode (str): Discretization mode ('nn' for nearest neighbor or 'linear' for interpolation)
        _policy_type (str): Type of policy computation method
        temperature (float): Temperature parameter for softmax in stochastic policies
        eps (float): Small value for numerical stability
        _num_bins (int): Number of bins per dimension for discretization
        gamma (float): Discount factor for future rewards
        env (gym.Env): The Gymnasium environment instance
        env_name (str): Name of the environment
        goal_position (float): Target position for the mountain car
        state_lower_bound (np.ndarray): Lower bounds of the state space
        state_upper_bound (np.ndarray): Upper bounds of the state space
        bin_sizes (np.ndarray): Size of bins in each dimension
        num_dims (int): Number of state dimensions
        solver (TabularPolicy): The underlying tabular policy
        performance_history (List[float]): History of performance metrics during training
    """

    def __init__(
        self,
        mode: str,
        num_bins: int = 21,
        temperature: float = 1.0,
        policy_type: str = "deterministic_vi",
    ) -> None:
        """
        Initialize the discretized solver.

        Args:
            mode: Discretization mode ('nn' or 'linear')
            num_bins: Number of bins per dimension for discretization
            temperature: Temperature parameter for softmax in stochastic policies
            policy_type: Type of policy computation ('deterministic_vi', 'stochastic_pi', or 'deterministic_pi')
        
        Raises:
            AssertionError: If mode or policy_type are not valid values
        """
        # Validate inputs
        assert mode in ["nn", "linear"], "Mode must be 'nn' or 'linear'"
        assert policy_type in ["deterministic_vi", "stochastic_pi", "deterministic_pi"]

        # Store parameters
        self._mode = mode
        self._policy_type = policy_type
        self.temperature = temperature
        self.eps = 1e-6
        self._num_bins = num_bins
        self.gamma = 0.99  # Discount factor

        # Initialize environment
        self.env = gym.make("mountaincar5302-v0")
        self.env_name = "MountainCar"
        self.goal_position = 0.5

        # Get state space information
        self.state_lower_bound = self.env.observation_space.low
        self.state_upper_bound = self.env.observation_space.high
        self.bin_sizes = (
            self.state_upper_bound - self.state_lower_bound
        ) / self._num_bins
        self.num_dims = self.state_lower_bound.shape[0]

        # Initialize policy and tracking variables
        self.solver = TabularPolicy(
            self._num_bins, self.num_dims, self.env.action_space.n
        )
        self.performance_history = []

        # Build transition and reward functions
        self.populate_transition_and_reward_funcs()

    def populate_transition_and_reward_funcs(self) -> None:
        """
        Initialize transition and reward functions for all state-action pairs.
        
        This method iterates through all possible state-action pairs and computes
        their transition probabilities and expected rewards through sampling.
        """
        num_states = self._num_bins**self.num_dims
        for state in range(num_states):
            for action in range(self.env.action_space.n):
                self.add_transition(state, action)

    # Check if any states are in the goal region
        goal_states = []
        for state_idx in range(self.solver.num_states):
            coords = self.get_coordinates_from_state_index(state_idx)
            if coords[0] >= 0.5:  # Goal position
                goal_states.append(state_idx)

        print(f"\n[DIAGNOSTIC] Found {len(goal_states)} goal states out of {self.solver.num_states}")
        if len(goal_states) > 0:
            print(f"[DIAGNOSTIC] Example goal state {goal_states[0]}: position={self.get_coordinates_from_state_index(goal_states[0])[0]:.3f}")
        else:
            print(f"[DIAGNOSTIC] ❌ ERROR: No goal states in discretization!")
            print(f"[DIAGNOSTIC] Position range: [{self.state_lower_bound[0]:.3f}, {self.state_upper_bound[0]:.3f}]")
            print(f"[DIAGNOSTIC] Goal position: 0.5")

    def add_transition(self, state_idx: int, action_idx: int) -> None:
        """
        Compute and store transition and reward information for a state-action pair.

        This method samples multiple transitions from a given state-action pair to
        estimate transition probabilities and expected rewards.

        Args:
            state_idx: Index of the current state
            action_idx: Index of the action to take
        """
        state_vector = self.get_coordinates_from_state_index(state_idx)

        # Sample multiple transitions for better estimates
        n_samples = 10
        
        # Dictionary to accumulate: {next_state_idx: [list of (prob, reward) tuples]}
        transition_data = {}

        for _ in range(n_samples):
            # Reset environment and set state
            self.env.reset()
            self.env.unwrapped.state = np.array(state_vector, dtype=np.float32)

            # Take action and observe result
            next_state, reward, terminated, truncated, _ = self.env.step(action_idx)
            next_state = np.array(next_state, dtype=np.float32)
            
            # Store reward - don't zero out terminal rewards
            if terminated:
                actual_reward = 0.0  # Terminal state has no future value
            else:
                actual_reward = reward
            
            # Get discrete next state(s) with probabilities
            next_states_probs = self.get_discrete_state_probabilities(next_state)
            
            # Store this sample's data
            for next_state_idx, prob in next_states_probs:
                if next_state_idx not in transition_data:
                    transition_data[next_state_idx] = []
                transition_data[next_state_idx].append((prob, actual_reward))

        # Now compute and store the averaged transitions
        for next_state_idx, data_list in transition_data.items():
            # Average probability across all samples
            avg_prob = sum(prob for prob, _ in data_list) / n_samples
            
            # Average reward (weighted by probability within each sample)
            total_prob = sum(prob for prob, _ in data_list)
            if total_prob > 0:
                avg_reward = sum(prob * reward for prob, reward in data_list) / total_prob
            else:
                avg_reward = 0.0
            
            # Store the transition and reward
            self.solver.set_transition(state_idx, action_idx, next_state_idx, avg_prob)
            self.solver.set_reward(state_idx, action_idx, next_state_idx, avg_reward)

    def get_discrete_state_probabilities(self, continuous_state: np.ndarray) -> List[Tuple[int, float]]:
        """
        Convert continuous state to discrete state probabilities.
        
        For 'nn' mode, returns the nearest discrete state with probability 1.
        For 'linear' mode, returns interpolation weights for neighboring states.

        Students need to:
        For 'nn' mode:
        1. Calculate the nearest discrete state index
        2. Return a list with single tuple of (state_index, 1.0)

        For 'linear' mode:
        1. Find valid neighboring grid points within state bounds
        2. Convert valid neighbors to state indices
        3. Calculate interpolation weights based on distance to neighbors
        4. Return list of (state_index, weight) tuples

        Args:
            continuous_state: The continuous state vector

        Returns:
            List of tuples (state_index, probability) for the discrete states
        """
        if self._mode == "nn":
            # Student code here
            # Find the nearest neighbor
            # NEAREST NEIGHBOR MODE
            # Clip the continuous state to valid bounds
            continuous_state = np.clip(
                continuous_state, 
                self.state_lower_bound, 
                self.state_upper_bound - 1e-6
            )
            
            # Find the nearest discrete state index
            discrete_state_idx = self.get_state_index_from_coordinates(continuous_state)
            
            # Return single state with probability 1.0
            return [(discrete_state_idx, 1.0)]
            
        else:  # linear interpolation
            # Get neighboring grid points
            # Student code here
            # Find valid neighbors
            # BILINEAR INTERPOLATION MODE
            
            # Clip continuous state to valid bounds
            continuous_state = np.clip(
                continuous_state,
                self.state_lower_bound,
                self.state_upper_bound - 1e-6
            )
            
            # Calculate bin indices for each dimension
            bin_indices = ((continuous_state - self.state_lower_bound) / self.bin_sizes).astype(int)
            bin_indices = np.clip(bin_indices, 0, self._num_bins - 1)
            
            # Calculate the position within the bin (0.0 to 1.0)
            bin_positions = (continuous_state - self.state_lower_bound) / self.bin_sizes - bin_indices
            bin_positions = np.clip(bin_positions, 0.0, 1.0)
            
            # Student code here
            # Convert to state indices
            # Generate all 2^num_dims neighboring bins
            # For 2D: 4 neighbors (00, 01, 10, 11)
            neighbor_offsets = list(itertools.product([0, 1], repeat=self.num_dims))
            
            result = []
            
            # Student code here
            # Calculate interpolation weights
            for offset in neighbor_offsets:
                offset = np.array(offset)
                
                # Calculate neighbor bin indices
                neighbor_bins = bin_indices + offset
                
                # Check if neighbor is within valid bounds
                if np.all(neighbor_bins >= 0) and np.all(neighbor_bins < self._num_bins):
                    # Convert bin indices to state index
                    neighbor_state_idx = int(
                        neighbor_bins[0] * self._num_bins + neighbor_bins[1]
                    )
                    
                    # Calculate interpolation weight using bilinear formula
                    weight = 1.0
                    for dim in range(self.num_dims):
                        if offset[dim] == 0:
                            # Lower bin in this dimension
                            weight *= (1.0 - bin_positions[dim])
                        else:
                            # Upper bin in this dimension
                            weight *= bin_positions[dim]
                    
                    # Only add if weight is non-negligible
                    if weight > 1e-10:
                        result.append((neighbor_state_idx, weight))
            
            # Safety check: if result is empty (shouldn't happen but just in case)
            if len(result) == 0:
                # Fallback to nearest neighbor
                discrete_state_idx = self.get_state_index_from_coordinates(continuous_state)
                return [(discrete_state_idx, 1.0)]
            
            # Normalize weights to sum to 1.0 (for numerical stability)
            total_weight = sum(w for _, w in result)
            if total_weight > 1e-10:
                result = [(idx, w / total_weight) for idx, w in result]
            else:
                # Fallback: use nearest neighbor
                discrete_state_idx = self.get_state_index_from_coordinates(continuous_state)
                return [(discrete_state_idx, 1.0)]
            
            return result


    def compute_policy(self, max_iterations: int = 100, min_iter: int = 5, eval_sample_size: int = 15) -> None:
        """
        Compute optimal policy using the specified algorithm.
        
        This method selects and runs the appropriate policy computation algorithm based on
        the policy_type specified during initialization.

        Args:
            max_iterations: Maximum number of iterations to run
            min_iter: Minimum number of iterations before checking convergence
            eval_sample_size: Number of episodes to use for policy evaluation
        """
        if self._policy_type == "deterministic_vi":
            self._value_iteration(max_iterations, min_iter, eval_sample_size)
        elif self._policy_type == "stochastic_pi":
            self._stochastic_policy_iteration(
                max_iterations, min_iter, eval_sample_size
            )
        else:  # deterministic_pi
            self._deterministic_policy_iteration(
                max_iterations, min_iter, eval_sample_size
            )

    def _value_iteration(self, max_iterations: int, min_iter: int, eval_sample_size: int) -> None:
        """
        Implement value iteration algorithm.

        Students need to:
        1. Compute Q-values for all state-action pairs using:
           - Current value function
           - Transition probabilities
           - Reward function
           - Discount factor (self.gamma)
        2. Update value function with maximum Q-value for each state
        3. Update policy to be deterministic, choosing action with highest Q-value
        4. Check for convergence by comparing old and new value functions

        Args:
            max_iterations: Maximum number of iterations to run
            min_iter: Minimum number of iterations before checking convergence
            eval_sample_size: Number of episodes to use for policy evaluation
        """
        eps_value = 1e-5
        value_function = np.zeros(self.solver.num_states)
        policy = np.zeros((self.solver.num_states, self.solver.num_actions))

        for i in range(max_iterations):
            iter_start_time = time.time()

            old_values = value_function.copy()  # storing old values for convergence check
            # Compute Q-values
            # Student code here
            q_values = (
                self.solver._transition_function*
                (self.solver._reward_function + self.gamma * value_function)
            )

            # Sum over next states to get Q(s,a)
            # Shape: (num_states, num_actions)
            q_values = np.sum(q_values, axis=2)

            # Update value function and policy
            new_values = np.max(q_values, axis=1)

            # Update policy to be deterministic (greedy)
            policy = np.zeros((self.solver.num_states, self.solver.num_actions))
            best_actions = np.argmax(q_values, axis=1)
            policy[np.arange(self.solver.num_states), best_actions] = 1.0
            
            # Check convergence
            value_diff = np.max(np.abs(new_values - old_values))
            value_function = new_values

            # Evaluate current policy
            self.solver.set_policy_function(policy)
            self.solver.set_value_function(value_function)
            reward, steps = self.solve(max_steps=200, sample_size=eval_sample_size)
            self.performance_history.append(reward)

            print(
                f"VI Iteration {i}, diff {value_diff:.6f}, "
                f"elapsed {time.time() - iter_start_time:.3f}s, "
                f"performance {reward:.2f}"
            )
            
            # Check for convergence
            if value_diff < eps_value and i >= min_iter:
                print(f"Value Iteration converged after {i+1} iterations")
                break
            
            # Check if we've reached good performance
            if i >= min_iter and reward > self.expected_reward():
                print(f"Reached target performance after {i+1} iterations")
                break
        
        # Final evaluation
        final_reward, final_steps = self.solve(max_steps=200, sample_size=10)
        print(f"Final VI policy performance: {final_reward:.2f} reward in {final_steps:.1f} steps")
            
    def _deterministic_policy_iteration(self, max_iterations: int, min_iter: int, eval_sample_size: int) -> None:
        """
        Implement deterministic policy iteration.

        Students need to:
        1. Policy Evaluation:
           - Update value function using current policy's transition and reward functions
           - Iterate until convergence or horizon reached
        
        2. Policy Improvement:
           - Compute Q-values using current value function
           - Update policy to be deterministic, choosing action with highest Q-value
           - Check if policy is stable (unchanged from previous iteration)
           - Set policy_stable flag based on whether policy changed

        Args:
            max_iterations: Maximum number of iterations to run
            min_iter: Minimum number of iterations before checking convergence
            eval_sample_size: Number of episodes to use for policy evaluation
        """
         
        eps_value = 1e-5
        horizon = 100

        # Initialize random deterministic policy
        policy = np.zeros((self.solver.num_states, self.solver.num_actions))
        policy[
            np.arange(self.solver.num_states),
            np.random.randint(0, self.solver.num_actions, size=self.solver.num_states),
        ] = 1.0
        value_function = np.zeros(self.solver.num_states)

        for k in range(max_iterations):
            iter_start_time = time.time()

            # Policy evaluation
            policy_T = np.sum(
                self.solver._transition_function * policy[:, :, np.newaxis], axis=1
            )
            policy_R = np.sum(
                self.solver._reward_function * policy[:, :, np.newaxis], axis=1
            )

            for _ in range(horizon):
                # Student code here
                # Update value function
                # Bellman expectation equation:
                # V^pi(s) = sum_s' T^pi(s,s') * [R^pi(s,s') + gamma * V^pi(s')]
                v_new = np.sum(
                    policy_T * (policy_R + self.gamma * value_function),
                    axis=1
                )
                
                # Check convergence of policy evaluation
                if np.max(np.abs(v_new - value_function)) < eps_value:
                    value_function = v_new
                    break
                value_function = v_new

            # Policy improvement
            # Student code here
            # Save old policy to check for convergence
            old_policy = policy.copy()
            
            # Compute Q-values using current value function
            # Q(s,a) = sum_s' T(s,a,s') * [R(s,a,s') + gamma * V(s')]
            q_values = (
                self.solver._transition_function * 
                (self.solver._reward_function + self.gamma * value_function)
            )
            q_values = np.sum(q_values, axis=2)  # Sum over next states
            
            # Student code here
            # Update policy to be greedy (deterministic)
            policy = np.zeros((self.solver.num_states, self.solver.num_actions))
            best_actions = np.argmax(q_values, axis=1)
            policy[np.arange(self.solver.num_states), best_actions] = 1.0
            
            # Check convergence
            # Check if policy is stable (unchanged)
            policy_stable = np.array_equal(policy, old_policy)

            # Update and evaluate
            self.solver.set_policy_function(policy)
            self.solver.set_value_function(value_function)
            reward, steps = self.solve(max_steps=200, sample_size=eval_sample_size)
            self.performance_history.append(reward)

            print(
                f"Deterministic PI Iteration {k}, "
                f"elapsed {time.time() - iter_start_time:.3f}s, "
                f"performance {reward:.2f}"
            )

            if policy_stable and k >= min_iter:
                print(f"Policy converged after {k+1} iterations")
                break
            if k >= min_iter and reward > self.expected_reward():
                print(f"Reached target performance after {k+1} iterations")
                break

        # Final evaluation
        reward, steps = self.solve(max_steps=200, sample_size=10)
        print(f"Final PI policy performance: {reward:.2f} reward in {steps:.1f} steps")

    def _stochastic_policy_iteration(self, max_iterations: int, min_iter: int, eval_sample_size: int) -> None:
        """
        Implement stochastic policy iteration (soft policy iteration).
        
        This is similar to deterministic policy iteration, but uses soft policy updates
        with a temperature parameter to control exploration.
        """
        # Placeholder implementation for stochastic policy iteration
        # You can implement this if needed for bonus points
        print("Stochastic Policy Iteration not implemented")
        pass

    def solve(self, visualize: bool = False, max_steps: float = float("inf"), sample_size: int = 1) -> Tuple[float, float]:
        """
        Execute the current policy in the environment.
        """
        rewards = []
        steps = []

        for _ in range(sample_size):
            state, _ = self.env.reset()
            state = np.array(state, dtype=np.float32)
            if visualize:
                self.env.render()

            episode_reward = 0
            num_steps = 0
            done = False

            while not done and num_steps < max_steps:
                # Get action using current policy
                if self._mode == "nn":
                    discrete_state = self.get_discrete_state_probabilities(state)[0][0]
                    action = self.solver.get_action(discrete_state)
                else:
                    discrete_states = self.get_discrete_state_probabilities(state)
                    # Use the discrete state with highest probability
                    states, probs = zip(*discrete_states)
                    best_idx = np.argmax(probs)
                    action = self.solver.get_action(states[best_idx])

                # Execute action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                state = np.array(next_state, dtype=np.float32)
                done = terminated or truncated

                episode_reward += reward
                num_steps += 1

                if visualize:
                    self.env.render()

            rewards.append(episode_reward)
            steps.append(num_steps)

        return np.mean(rewards), np.mean(steps)

    def get_state_index_from_coordinates(self, continuous_state: np.ndarray) -> int:
        """
        Convert continuous state to discrete state index.
        
        This method maps a continuous state vector to its corresponding discrete state index
        based on the discretization scheme.

        Args:
            continuous_state: The continuous state vector

        Returns:
            Index of the corresponding discrete state
        """
        continuous_state = np.clip(
            continuous_state, self.state_lower_bound, self.state_upper_bound - 1e-6
        )

        bin_sizes = (self.state_upper_bound - self.state_lower_bound) / self._num_bins
        pos_bin = int((continuous_state[0] - self.state_lower_bound[0]) / bin_sizes[0])
        vel_bin = int((continuous_state[1] - self.state_lower_bound[1]) / bin_sizes[1])

        pos_bin = np.clip(pos_bin, 0, self._num_bins - 1)
        vel_bin = np.clip(vel_bin, 0, self._num_bins - 1)

        return pos_bin * self._num_bins + vel_bin

    def get_coordinates_from_state_index(self, state_idx: int) -> np.ndarray:
        """
        Convert discrete state index to continuous state coordinates.
        
        This method maps a discrete state index back to the center of its corresponding
        continuous state region.

        Args:
            state_idx: The discrete state index

        Returns:
            The continuous state vector at the center of the bin
        """
        bin_sizes = (self.state_upper_bound - self.state_lower_bound) / self._num_bins
        pos_idx = state_idx // self._num_bins
        vel_idx = state_idx % self._num_bins

        position = (pos_idx + 0.5) * bin_sizes[0] + self.state_lower_bound[0]
        velocity = (vel_idx + 0.5) * bin_sizes[1] + self.state_lower_bound[1]

        return np.array([position, velocity], dtype=np.float32)

    def expected_reward(self) -> float:
        """
        Compute expected reward threshold based on discretization.
        
        This method calculates a baseline expected reward threshold that depends on
        the granularity of the discretization.

        Returns:
            The expected reward threshold
        """
        return -110 - (30 * (((40**2) / self.solver.num_states)) ** 0.75)

    def plot_value_function(self, value_function: np.ndarray, filename: Optional[str] = None) -> Tuple[np.ndarray, Figure]:
        """
        Plot the value function as a heatmap.
        
        This method creates a visualization of the value function across the state space
        and optionally saves it to a file.

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
            value_function.max() - value_function.min() + 1e-6
        )
        V = V.reshape(self._num_bins, self._num_bins).T

        # Create heatmap
        image = (plt.cm.coolwarm(V)[::-1, :, :-1] * 255.0).astype(np.uint8)
        ax.set_title(f"Env: {self.env_name}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.imshow(image)

        # Save plot
        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, "..", "figures")
            os.makedirs(figures_dir, exist_ok=True)
            filename = os.path.join(
                figures_dir, f"{self.env_name}_{self._mode}_{self._num_bins}.png"
            )

        plt.savefig(filename)
        plt.close()

        # Convert to image array
        canvas.draw()
        image = np.asarray(canvas.buffer_rgba()).reshape(
            int(fig.get_size_inches()[1] * fig.get_dpi()),
            int(fig.get_size_inches()[0] * fig.get_dpi()),
            4,
        )[:, :, :3]

        return image, fig

    def plot_policy(self, filename: Optional[str] = None) -> Tuple[np.ndarray, Figure]:
        """
        Plot the policy as a heatmap.
        
        This method creates a visualization of the policy across the state space
        and optionally saves it to a file.

        Args:
            filename: Optional path to save the plot

        Returns:
            Tuple of (image_array, matplotlib_figure)
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)

        # Create policy grid
        policy_grid = np.zeros((self._num_bins, self._num_bins))
        for i in range(self._num_bins):
            for j in range(self._num_bins):
                state_idx = i * self._num_bins + j
                policy = self.solver.get_policy(state_idx)
                policy_grid[j, i] = np.argmax(policy)

        # Create custom colormap
        colors = ["red", "gray", "blue"]
        cmap = plt.cm.colors.ListedColormap(colors)
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

        # Plot policy
        im = ax.imshow(policy_grid[::-1], cmap=cmap, norm=norm)

        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor=c, label=l)
            for c, l in zip(colors, ["Left", "No Action", "Right"])
        ]
        ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))

        ax.set_title(f"Env: {self.env_name} Policy")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")

        # Save plot
        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            figures_dir = os.path.join(script_dir, "..", "..", "hw4_rl", "figures")
            os.makedirs(figures_dir, exist_ok=True)
            filename = os.path.join(
                figures_dir,
                f"{self.env_name}_{self._policy_type}_policy_{self._num_bins}.png",
            )

        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight")

        # Convert to image array
        canvas.draw()
        image = np.asarray(canvas.buffer_rgba()).reshape(
            int(fig.get_size_inches()[1] * fig.get_dpi()),
            int(fig.get_size_inches()[0] * fig.get_dpi()),
            4,
        )[:, :, :3]
        plt.close()

        return image, fig


def kl_divergence(policy_old: np.ndarray, policy_new: np.ndarray) -> float:
    """
    Compute KL divergence between two policies.
    
    This function calculates the Kullback-Leibler divergence between two policy
    distributions, which measures how much they differ.

    Args:
        policy_old: The original policy distribution
        policy_new: The new policy distribution

    Returns:
        The KL divergence value
    """
    eps = 1e-10
    policy_old = np.clip(policy_old, eps, 1.0)
    policy_new = np.clip(policy_new, eps, 1.0)
    return np.sum(policy_old * np.log(policy_old / policy_new))


def plot_policy_curves(
    reward_histories: List[List[float]], 
    labels: List[str], 
    filename: Optional[str] = None
) -> None:
    """
    Plot learning curves for different policies.
    
    This function creates a plot comparing the learning progress of different
    policy computation methods over iterations.

    Args:
        reward_histories: List of reward histories for each policy
        labels: List of labels for each policy
        filename: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    plt.clf()
    styles = ["-", "--", ":", "-."]
    colors = ["b", "r", "g", "m", "c", "y", "k"]

    for idx, (history, label) in enumerate(zip(reward_histories, labels)):
        style = styles[idx // len(colors)]
        color = colors[idx % len(colors)]
        plt.plot(range(len(history)), history, style, color=color, label=label)

    plt.xlabel("Iteration")
    plt.ylabel("Return")
    plt.title("Policy Iteration Performance")
    plt.legend()
    plt.grid(True)

    if filename is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        figures_dir = os.path.join(script_dir, "..", "..", "hw4_rl", "figures")
        os.makedirs(figures_dir, exist_ok=True)
        filename = os.path.join(figures_dir, "mountaincar_learning_curves.png")

    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    print("Testing Mountain Car with Value Iteration...")

    # Configuration
    bin_sizes = [21, 51, 101]  # Test all three resolutions like your friend
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, "..", "..", "hw4_rl", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    print(f"\n Figures will be saved to: {figures_dir}\n")

    for n_bins in bin_sizes:
        print(f"\n{'='*60}")
        print(f"=== Testing with {n_bins} bins ===")
        print(f"{'='*60}")

        solver = DiscretizedSolver(
            mode="linear",
            num_bins=n_bins,
            policy_type="deterministic_vi",
            temperature=1.0,
        )

        # Train
        start_time = time.time()
        solver.compute_policy()
        elapsed_time = time.time() - start_time
        print(f"\nComputed Value Iteration in {elapsed_time:.2f} seconds")

        # Save value function plot
        value_file = os.path.join(figures_dir, f"mountaincar_vi_value_{n_bins}.png")
        solver.plot_value_function(solver.solver.get_value_function(), value_file)
        print(f"Saved value function: {value_file}")
        
        # Save policy plot
        policy_file = os.path.join(figures_dir, f"mountaincar_vi_policy_{n_bins}.png")
        solver.plot_policy(policy_file)
        print(f"Saved policy: {policy_file}")

        # Test
        final_reward, final_steps = solver.solve(max_steps=200, sample_size=10)
        print(f"Performance: {final_reward:.2f} reward in {final_steps:.1f} steps\n")