"""
Gridworld environment registration for CSCI 5302.

This module registers different variants of the Gridworld environment:
- gridworld-v0: Basic gridworld with plan0.txt map
- gridworldnoisy-v0: Noisy version of gridworld-v0 with 0.2 transition noise
- gridworld-v1: Basic gridworld with plan1.txt map
- gridworldnoisy-v1: Noisy version of gridworld-v1 with 0.2 transition noise
"""

from gymnasium.envs.registration import register

register(
    id="gridworld-v0",
    entry_point="hw4_rl.envs.gym_gridworld.envs:GridworldEnv",
    kwargs={"map_file": "plan0.txt"},
)
register(
    id="gridworldnoisy-v0",
    entry_point="hw4_rl.envs.gym_gridworld.envs:GridworldEnv",
    kwargs={"map_file": "plan0.txt", "transition_noise": 0.2},
)
register(
    id="gridworld-v1",
    entry_point="hw4_rl.envs.gym_gridworld.envs:GridworldEnv",
    kwargs={"map_file": "plan1.txt"},
)
register(
    id="gridworldnoisy-v1",
    entry_point="hw4_rl.envs.gym_gridworld.envs:GridworldEnv",
    kwargs={"map_file": "plan1.txt", "transition_noise": 0.2},
)
