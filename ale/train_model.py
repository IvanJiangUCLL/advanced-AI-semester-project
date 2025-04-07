from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium_make import create_tetris_env
from stable_baselines3.common.callbacks import BaseCallback
import datetime
import os

# Create the environment
# Grayscale frames are compatible with MlpPolicy
env = DummyVecEnv([lambda: create_tetris_env(render_mode="human")])


def get_newest_model(folder):
    """Finds the newest .zip file in the specified folder."""
    zip_files = [f for f in os.listdir(folder) if f.endswith(".zip")]
    if not zip_files:
        raise FileNotFoundError(f"No .zip files found in the folder: {folder}")
    # Sort files by modification time (newest first)
    zip_files.sort(key=lambda f: os.path.getmtime(
        os.path.join(folder, f)), reverse=True)
    return os.path.join(folder, zip_files[0])


# Path to the previously trained model
trained_models_folder = "trained_models"
try:
    newest_model_path = get_newest_model(trained_models_folder)
    print(f"Loading previous model from {newest_model_path}")
    # Load the model and attach the environment
    model = DQN.load(newest_model_path, env=env)
except FileNotFoundError:
    print("No previous model found. Initializing a new model.")
    model = DQN(
        "MlpPolicy",  # Use MLP policy for grayscale observations
        env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=100000,
        batch_size=64,
        target_update_interval=1000,
        exploration_fraction=0.5,  # Fraction of training where exploration is used
        exploration_final_eps=0.1,  # Final value of epsilon
    )

# Train the model


class ActionLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.action_counts = {i: 0 for i in range(env.action_space.n)}

    def _on_step(self) -> bool:
        # Convert action to an integer if it's a numpy array
        action = int(self.locals["actions"].item())
        self.action_counts[action] += 1
        return True

    def _on_training_end(self) -> None:
        print("Action counts during training:", self.action_counts)


class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Log rewards
        reward = self.locals["rewards"]
        self.episode_rewards.append(reward)
        return True

    def _on_training_end(self) -> None:
        print("Episode rewards during training:", self.episode_rewards)


# Add the callbacks to the model's training
callback = ActionLoggerCallback()
reward_logger_callback = RewardLoggerCallback()
model.learn(total_timesteps=100000, callback=[
            callback, reward_logger_callback])

# Generate a unique filename using the current timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = "trained_models"  # Specify the folder where models will be saved
# Create the folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
model_filename = os.path.join(
    output_folder, f"tetris_dqn_model_{timestamp}.zip"
)

# Save the trained model
model.save(model_filename)
print(f"Model saved as {model_filename}")

# Close the environment
env.close()
