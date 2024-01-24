from stable_baselines3 import PPO
import numpy as np
from ot2_env_wrapper import OT2Env

def main(num_episodes=100):
    # Create the environment
    env = OT2Env(render=True)
    model = PPO.load(r"C:\Users\benjm\Downloads\model.zip", env=env)

    for episode in range(num_episodes):
        obs, info = env.reset()
        env.goal_position = np.array([0.1, 0.1, 0.16])
        steps = 0
        terminated = False
        truncated = False
        while not terminated and not truncated:
            obs = np.array(obs)
            action, _states = model.predict(obs,deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            # Add any additional logging or testing metrics here
        print(f"Episode {episode} took {steps} steps")

    # Close the environment after testing
    env.close()

if __name__ == "__main__":
    main()
