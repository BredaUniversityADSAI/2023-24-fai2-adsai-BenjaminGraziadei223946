import gymnasium as gym
from stable_baselines3 import PPO
from ot2_env_wrapper import OT2Env
from clearml import Task
import argparse

def main():
    # Create the environment
    env = OT2Env()

    # Instantiate the agent
    model = PPO("MlpPolicy", env, verbose=1,
                learning_rate=args.learning_rate, 
                batch_size=args.batch_size, 
                n_steps=args.n_steps, 
                n_epochs=args.n_epochs)
    #model = PPO.load("ot2_model", env=env)

    model.learn(total_timesteps=1000, progress_bar=True)
    model.save("ot2_model")
    test_model(env, model)

def test_model(env, model):
    observation = env.reset()
    for _ in range(1000):
        action = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        if truncated:
            observation = env.reset()
        if terminated:
            print("Goal Reached")
            break

    env.close()

if __name__ == "__main__":
    task = Task.init(project_name='Mentor Group D/Group 1', task_name='agent 007')
    task.set_base_docker('deanis/2023y2b-rl:latest')
    #setting the task to run remotely on the default queue
    task.execute_remotely(queue_name="default")

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--n_epochs", type=int, default=10)

    args = parser.parse_args()
    main()
