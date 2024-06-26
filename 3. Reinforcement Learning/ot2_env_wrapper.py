import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
from sim_class import Simulation

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        self.sim = Simulation(num_agents=1, render=self.render)

        # Define action space for x, y, z velocities
        self.action_space = gym.spaces.Box(low=np.array([-1,-1,-1, 0]), high=np.array([1,1,1, 0]), shape=(4,), dtype=np.float32)

        # Define observation space including pipette position and goal position
        self.observation_space = gym.spaces.Box(low=-np.inf,high=np.inf, shape=(6,), dtype=np.float32)

        self.goal_position = None
        self.steps = 0
        self.prev_pipette_pos = None
        self.consecutive_wrong_direction = 0
        self.consecutive_right_direction = 0

    def reset(self, seed=None, goal_position=None):
        if seed is not None:
            np.random.seed(seed)
        # Reset the environment to an initial state
        current_position = self.sim.reset(num_agents=1)
        if goal_position is not None:
            self.goal_position = goal_position
        else:
            self.goal_position = np.array([np.random.uniform(-0.1872, 0.253),
                                    np.random.uniform(-0.1705, 0.2195),
                                    np.random.uniform(0.1693, 0.2895)], dtype=np.float32)
        self.steps = 0

        if 'pipette_position' in current_position:
            pipette_pos_array = np.array(current_position['pipette_position'], dtype=np.float32)
        else:
            # Handle the case where 'pipette_position' is not in the dictionary
            pipette_pos_array = np.zeros(3, dtype=np.float32)  # or any other default
 
        # Concatenate pipette position and goal position to form the observation
        observation = np.concatenate([pipette_pos_array, self.goal_position]).astype(np.float32)
        info = {}
        return observation, info
    
    def _calculate_reward(self, pipette_pos):
        cur_distance_to_goal = np.linalg.norm(pipette_pos - self.goal_position)
        prev_distance_to_goal = np.linalg.norm(self.prev_pipette_pos - self.goal_position) if self.prev_pipette_pos is not None else cur_distance_to_goal

        # Calculate improvement in distance
        distance_improvement = prev_distance_to_goal - cur_distance_to_goal

        # Define constants for rewards, penalties, and scaling factors
        GOAL_REACHED_REWARD = 200.0
        SCALER = 10
        

        # Update penalty/reward based on direction of movement
        if distance_improvement > 0:  # Moving towards the goal
            self.consecutive_wrong_direction = 0
            self.consecutive_right_direction += 1
            distance_reward = distance_improvement * self.consecutive_right_direction * SCALER
        else:  # Moving away from the goal
            self.consecutive_right_direction = 0
            self.consecutive_wrong_direction += 1
            distance_reward = distance_improvement * self.consecutive_wrong_direction * SCALER

        # Check if the goal is reached
        termination_threshold = 0.001
        if cur_distance_to_goal < termination_threshold:
            terminated = True
            reward = GOAL_REACHED_REWARD + distance_reward
        else:
            terminated = False
            reward = distance_reward

        # Check for truncation
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False

        # Update the previous position for the next step
        self.prev_pipette_pos = pipette_pos.copy()

        return reward, terminated, truncated

    def step(self, action):
        observation = self.sim.run([action])
        self.steps += 1

        pipette_pos = np.array(observation[f'robotId_{self.sim.robotIds[0]}']['pipette_position'], dtype=np.float32)


        observation = np.concatenate([pipette_pos, self.goal_position]).astype(np.float32)

        reward, terminated, truncated = self._calculate_reward(pipette_pos)
        info = {}

        return observation, reward, terminated, truncated, info

    

    def close(self):
        # Clean up resources if needed
        self.sim.close()

