import os
import cv2
from stable_baselines3 import DQN
import gymnasium as gym
from gymnasium.envs.registration import register
import random
from train_model import FixedObservationWrapper

if "tetris_gymnasium/Tetris" in gym.envs.registry.keys():
    if "tetris_gymnasium/Tetris" in gym.envs.registry:
        del gym.envs.registry["tetris_gymnasium/Tetris"]

register(
    id="tetris_gymnasium/Tetris",
    entry_point="tetris_gymnasium.envs.tetris:Tetris",
)


def evaluate_tetris_model(model_path, num_episodes=5):
    """
    Evaluate the Tetris model for a specified number of episodes or until the game is over.

    Args:
        model_path (str): Path to the trained model.
        num_episodes (int): Number of episodes to evaluate.
        time_limit (int): Maximum time (in seconds) to run each episode.
    """
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    env = FixedObservationWrapper(env, target_board_shape=(
        20, 10), target_holder_shape=(4, 4), target_queue_shape=(4, 16))

    model = DQN.load(model_path)
    print(f"Loaded model from {model_path}")

    for episode in range(num_episodes):
        print(f"Starting episode {episode + 1}")
        # observation, info = env.reset(seed=42)
        random_seed = random.randint(0, 10000)
        observation, info = env.reset(seed=random_seed)
        total_reward = 0

        while not env.unwrapped.game_over:
            env.render()

            action, _states = model.predict(observation, deterministic=True)
            observation, reward, _, truncated, info = env.step(action)

            total_reward += reward

            key = cv2.waitKey(10)
            if key == ord('q'):
                break

        # while not env.unwrapped.game_over:  # uncomment this if you only want to see the actions it took
        #     action, _states = model.predict(observation, deterministic=True)
        #     observation, reward, _, truncated, info = env.step(action)
        #     print(f"Action: {action}, Reward: {reward}, Info: {info}")
        print(
            f"Episode {episode + 1} finished with total reward: {total_reward}")
        print("Press any key to continue to the next episode...")
        cv2.waitKey(0)

    print("Evaluation complete!")
    env.close()


if __name__ == "__main__":
    trained_models_folder = "training_models"
    models = [f for f in os.listdir(
        trained_models_folder) if f.endswith(".zip")]
    if not models:
        print("No trained models found in the 'trained_models' folder.")
    else:
        models.sort(key=lambda x: os.path.getmtime(
            os.path.join(trained_models_folder, x)), reverse=True)
        latest_model_path = os.path.join(trained_models_folder, models[0])
        evaluate_tetris_model(latest_model_path, 5)
