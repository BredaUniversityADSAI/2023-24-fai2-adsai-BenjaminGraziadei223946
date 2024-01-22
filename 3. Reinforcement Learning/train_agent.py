import gymnasium as gym
from stable_baselines3 import PPO
from ot2_env_wrapper import OT2Env
from clearml import Task

def main():
    # Create the environment
    env = OT2Env()

    # Instantiate the agent
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.005)
    #model = PPO.load("ot2_model", env=env)

    # Train the agent
    print("Starting training...")
    model.learn(total_timesteps=1000, progress_bar=True)
    print("Training completed!")

    # Save the model
    model.save("ot2_model")

    # Test the trained model
    print("Testing the trained model...")
    #test_model(env, model)

def test_model(env, model):
    observation = env.reset()
    for _ in range(1000):
        action = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Episode: {_ + 1}, Action: {action}, Reward: {reward}")
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
    main()
