"""
Mountain Car environment registration for CSCI 5302.

This module registers a custom version of the Mountain Car environment
with specific parameters for the CSCI 5302 course.
"""

from gymnasium.envs.registration import register

register(
    id="mountaincar5302-v0",
    entry_point="hw4_rl.envs.gym_mountaincar.envs:MountainCarEnv",
    max_episode_steps=200,
    reward_threshold=-110.0,
)
