# HW4-RL: Reinforcement Learning Implementation

This package implements various reinforcement learning algorithms for solving gridworld and mountain car problems using tabular methods.

## Installation

You can install the package directly from the source:

```bash
pip install .
```

Or in development mode:

```bash
pip install -e .
```

## Usage

### Tabular Solution (Gridworld)

```python
from hw4_rl.solvers import GridworldSolver

# Create a solver for gridworld
solver = GridworldSolver(
    policy_type="deterministic_vi",  # or "stochastic_pi" or "deterministic_pi"
    gridworld_map_number=0,  # or 1
    noisy_transitions=False
)

# Compute the optimal policy
solver.compute_policy()

# Visualize the value function
solver.plot_value_function(solver.solver.get_value_function())
```

### Continuous Solution (Mountain Car)

```python
from hw4_rl.solvers import DiscretizedSolver

# Create a solver for mountain car
solver = DiscretizedSolver(
    num_bins_per_dim=51,  # Number of bins for discretization
    interpolation_mode="nn"  # or "linear"
)

# Compute the optimal policy
solver.compute_policy()

# Visualize the value function
solver.plot_value_function(solver.solver.get_value_function())
```

## Project Structure

```
hw4_rl/
├── solvers/
│   ├── __init__.py
│   ├── tabular_solution.py
│   └── continuous_solution.py
├── envs/
│   ├── __init__.py
│   ├── gym_gridworld/
│   └── gym_mountaincar/
└── __init__.py
```

## Requirements

- Python >= 3.6
- NumPy
- Matplotlib
- Gymnasium

## License

MIT License 