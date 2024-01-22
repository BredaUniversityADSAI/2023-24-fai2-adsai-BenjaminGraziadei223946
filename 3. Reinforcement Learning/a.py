from math import inf
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
import os


class OT2Env(gym.Env):
    def _init_(self, render=False, max_steps=1000):
        super(OT2Env, self)._init_()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1)

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # keep track of the number of steps
        self.steps = 1

    def reset(self, seed=None):
        # being able to set a seed is required for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Reset the state of the environment to an initial state
        # set a random goal position for the agent, consisting of x, y, and z coordinates within the working area (you determined these values in the previous datalab task)
        self.goal_position = np.array([np.random.uniform(-0.1872, 0.253),
                                   np.random.uniform(-0.1705, 0.2195),
                                   np.random.uniform(0.1693, 0.2895)], dtype=np.float32)

        # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32

        # Access the pipette position
        # Assuming that there's only one robot, get its ID from the keys of observation_data
        pipette_pos = self.sim.reset(1)

        # Convert to NumPy array and append goal position
        if 'pipette_position' in pipette_pos:
            pipette_pos_array = np.array(pipette_pos['pipette_position'], dtype=np.float32)
        else:
            # Handle the case where 'pipette_position' is not in the dictionary
            pipette_pos_array = np.zeros(3, dtype=np.float32)  # or any other default
 
        # Concatenate pipette position and goal position to form the observation
        observation = np.concatenate([pipette_pos_array, self.goal_position]).astype(np.float32)


        # Reset the number of steps
        self.steps = 1

        return observation

    def step(self, action):
        # Execute one time step within the environment
        # since we are only controlling the pipette position, we accept 3 values for the action and need to append 0 for the drop action
        action = np.append(action, 0)

        # Call the environment step function
        observation = self.sim.run([action]) # Why do we need to pass the action as a list? Think about the simulation class.

        # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32
        # Extract pipette position from the observation
        pipette_pos = observation[f'robotId_{self.sim.robotIds[0]}']['pipette_position']

        # Convert to NumPy array and append goal position
        observation = np.concatenate((pipette_pos, self.goal_position), axis=None).astype(np.float32)

        # Calculate the reward, this is something that you will need to experiment with to get the best results
        distance_to_goal = -np.linalg.norm(pipette_pos - self.goal_position)
        reward = -distance_to_goal

        # next we need to check if the if the task has been completed and if the episode should be terminated
        # To do this we need to calculate the distance between the pipette position and the goal position and if it is below a certain threshold, we will consider the task complete. 
        # What is a reasonable threshold? Think about the size of the pipette tip and the size of the plants.
        termination_threshold = 0.0005
        if distance_to_goal < termination_threshold:
            terminated = True
            # we can also give the agent a positive reward for completing the task
        else:
            terminated = False

        # next we need to check if the episode should be truncated, we can check if the current number of steps is greater than the maximum number of steps
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False

        info = {} # we don't need to return any additional information

        # increment the number of steps
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass
    
    def close(self):
        self.sim.close()