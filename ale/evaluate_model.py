import os
from stable_baselines3 import DQN
from gymnasium_make import create_tetris_env

# Function to find the newest .zip file in the trained_models folder


def get_newest_model(folder):
    """Finds the newest .zip file in the specified folder."""
    zip_files = [f for f in os.listdir(folder) if f.endswith(".zip")]
    if not zip_files:
        raise FileNotFoundError(f"No .zip files found in the folder: {folder}")
    # Sort files by modification time (newest first)
    zip_files.sort(key=lambda f: os.path.getmtime(
        os.path.join(folder, f)), reverse=True)
    return os.path.join(folder, zip_files[0])


# Specify the folder where models are saved
trained_models_folder = "trained_models"

# Get the newest model file
newest_model_path = get_newest_model(trained_models_folder)
print(f"Loading the newest model: {newest_model_path}")

# Create the environment
# Use "human" to visualize the environment
env = create_tetris_env(render_mode="human")
# Load the trained model
model = DQN.load(newest_model_path, env=env)

# Evaluate the model
obs, info = env.reset()
action_counts = {i: 0 for i in range(env.action_space.n)}  # Track action usage
total_reward = 0
total_lines_completed = 0  # Track total lines completed
try:
    for _ in range(1000):  # Run for 1000 steps or until the game ends
        action, _states = model.predict(obs, deterministic=False)
        action_counts[int(action)] += 1
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Extract lines completed from the reward logic
        lines_completed = reward  # Reward is proportional to lines completed
        total_lines_completed += lines_completed

        if done or truncated:
            obs, info = env.reset()
finally:
    print("Action counts during evaluation:", action_counts)
    print("Total reward during evaluation:", total_reward)
    print("Total lines completed during evaluation:", total_lines_completed)
    env.close()
