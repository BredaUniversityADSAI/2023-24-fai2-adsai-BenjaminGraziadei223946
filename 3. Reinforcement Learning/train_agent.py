import gymnasium as gym
from stable_baselines3 import PPO, ACER
from ot2_env_wrapper import OT2Env
from clearml import Task
import argparse
import os
import wandb
from wandb.integration.sb3 import WandbCallback

def main():
    # Create the environment
    env = OT2Env()
    
    os.environ['WANDB_API_KEY'] = '633d8379ec4fe0a5cd33330730b3d6a7f17b9c6f'
    run = wandb.init(project="Model-RL", sync_tensorboard=True)
    wandb_callback = WandbCallback(model_save_freq=1000,
                                model_save_path=f"models/{run.id}",
                                verbose=2)

    # Instantiate the agent
    model = ACER("MlpPolicy", env, verbose=1,
                    learning_rate=args.learning_rate,
                    batch_size=args.batch_size,
                    n_steps=args.n_steps,
                    n_epochs=args.n_epochs,
                tensorboard_log=f"runs/{run.id}")
    #model = PPO.load("ot2_model", env=env)

    model.learn(total_timesteps=5000000, callback=wandb_callback, progress_bar=True)

if __name__ == "__main__":
    task = Task.init(project_name='Mentor Group D/Group 1', task_name='agent 007')
    task.set_base_docker('deanis/2023y2b-rl:latest')
    #setting the task to run remotely on the default queue
    task.execute_remotely(queue_name="default")

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_steps", type=int, default=1024)
    parser.add_argument("--n_epochs", type=int, default=100)

    args = parser.parse_args()
    main()
