# OT2 Environment Setup

## Overview
The OT2 Environment is a custom Gym environment designed for a simulation involving pipette positioning in a virtual space. It integrates with the Gymnasium (formerly Gym) library to provide a reinforcement learning platform, particularly focusing on achieving precise pipette positioning based on specified goals.

## Dependencies
To run this environment, the following libraries are required:
- `gymnasium` (formerly known as `gym`): For creating custom gym environments.
- `numpy`: For numerical operations.
- `os`: For operating system interactions, typically for path and environment variable management.
- `sim_class`: A custom class (not included in standard libraries) for running the simulation.

Ensure these dependencies are installed in your Python environment. You can install them using pip:
```bash
# Note: 'sim_class' needs to be provided separately.
pip install gymnasium numpy
```

## Environment Description
`OT2Env` is a custom Gym environment that inherits from `gym.Env`. It simulates a scenario with pipette manipulation tasks, providing a reinforcement learning framework.

Key Components:
- `action_space`: Defined as a Box space with velocities in the x, y, z dimensions and a flag.
- `observation_space`: Includes pipette position and goal position.
- `reset`: Resets the environment to its initial state and sets a new goal position.
- `_calculate_reward`: Calculates the reward based on the distance to the goal and improvements.
- `step`: Runs one timestep of the environment's dynamics.

## Usage
To use this environment, follow these steps:
1. Initialize the environment:
   ```python
   env = OT2Env(render=True)
   ```
2. Load a pre-trained model (here, using PPO from Stable Baselines):
   ```python
   model = PPO.load("path_to_model.zip", env=env)
   ```
3. Run the simulation with specified goal points, predicting actions using the model and stepping through the environment:
   ```python
   # Example loop to move towards sorted goal points
   sorted_points = sorted(points[:5], key=lambda p: p[0])
   for point in sorted_points:
       # Reset environment and set new goal position
       # Execute actions and update the environment
   ```
4. Close the environment:
   ```python
   env.close()
   ```

## Custom Simulation Class (`sim_class`)
This environment relies on a custom `Simulation` class (referred to as `sim_class` in the import). Ensure you have this class implemented and accessible in your project.

## Notes
- The environment is designed for a specific simulation involving pipette manipulation.
- It assumes certain structure and data from the `sim_class`.
- Modify and extend as per your simulation requirements.