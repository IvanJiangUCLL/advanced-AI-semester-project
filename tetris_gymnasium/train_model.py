import os
import datetime
import torch
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from tetris_gymnasium.components.tetromino_queue import TetrominoQueue
from tetris_gymnasium.components.tetromino_randomizer import BagRandomizer

# Check CUDA availability and setup
print(f"Is CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch using device: {torch.cuda.current_device()}")
    # Enable CUDA optimization
    torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


class FixedObservationWrapper(gym.ObservationWrapper):
    """
    A wrapper to ensure the observation space has a fixed size.
    Pads or crops observations to match the target dimensions.
    """

    def __init__(self, env, target_board_shape=(20, 10), target_holder_shape=(4, 4), target_queue_shape=(4, 16)):
        super().__init__(env)
        self.target_board_shape = target_board_shape
        self.target_holder_shape = target_holder_shape
        self.target_queue_shape = target_queue_shape

        # Update the observation space to reflect the fixed dimensions
        self.observation_space = gym.spaces.Dict({
            "board": gym.spaces.Box(
                low=0,
                # Adjust high
                high=np.full(self.target_board_shape,
                             self.env.observation_space["board"].high.max()),
                shape=self.target_board_shape,
                dtype=np.uint8,
            ),
            "active_tetromino_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=self.target_board_shape,
                dtype=np.uint8,
            ),
            "holder": gym.spaces.Box(
                low=0,
                # Adjust high
                high=np.full(self.target_holder_shape,
                             self.env.observation_space["holder"].high.max()),
                shape=self.target_holder_shape,
                dtype=np.uint8,
            ),
            "queue": gym.spaces.Box(
                low=0,
                # Adjust high
                high=np.full(self.target_queue_shape,
                             self.env.observation_space["queue"].high.max()),
                shape=self.target_queue_shape,
                dtype=np.uint8,
            ),
        })

    def observation(self, obs):
        """
        Modify the observation to match the fixed dimensions.
        Pads or crops each component as needed.
        """
        obs["board"] = self._resize(obs["board"], self.target_board_shape)
        obs["active_tetromino_mask"] = self._resize(
            obs["active_tetromino_mask"], self.target_board_shape)
        obs["holder"] = self._resize(obs["holder"], self.target_holder_shape)
        obs["queue"] = self._resize(obs["queue"], self.target_queue_shape)
        return obs

    def _resize(self, array, target_shape):
        """
        Resize an array to the target shape by padding or cropping.
        """
        current_shape = array.shape
        pad_height = max(0, target_shape[0] - current_shape[0])
        pad_width = max(0, target_shape[1] - current_shape[1])

        # Pad the array if it's smaller than the target shape
        if pad_height > 0 or pad_width > 0:
            array = np.pad(
                array,
                ((0, pad_height), (0, pad_width)),
                mode="constant",
                constant_values=0,
            )

        # Crop the array if it's larger than the target shape
        if current_shape[0] > target_shape[0] or current_shape[1] > target_shape[1]:
            array = array[:target_shape[0], :target_shape[1]]

        return array


# Register the Tetris environment
if "tetris_gymnasium/Tetris" in gym.envs.registry.keys():
    if "tetris_gymnasium/Tetris" in gym.envs.registry:
        del gym.envs.registry["tetris_gymnasium/Tetris"]

register(
    id="tetris_gymnasium/Tetris",
    entry_point="tetris_gymnasium.envs.tetris:Tetris",
)

# Create output folder
output_folder = "training_models"
os.makedirs(output_folder, exist_ok=True)


def get_latest_model(folder):
    models = [f for f in os.listdir(folder) if f.endswith(".zip")]
    if not models:
        return None
    models.sort(key=lambda x: os.path.getmtime(
        os.path.join(folder, x)), reverse=True)
    return os.path.join(folder, models[0])


def create_training_environment(width=10, height=20, tetromino_types=None):
    # Create the base environment
    env = gym.make(
        "tetris_gymnasium/Tetris",
        render_mode="rgb_array",
        width=width,
        height=height,
    )

    # Modify tetromino types if specified
    if tetromino_types is not None:
        env.unwrapped.tetrominoes = [
            t for t in env.unwrapped.tetrominoes if t.id in tetromino_types
        ]
        if not env.unwrapped.tetrominoes:
            raise ValueError(
                f"No valid tetrominoes found for types: {tetromino_types}.")
        env.unwrapped.queue = TetrominoQueue(
            BagRandomizer(len(env.unwrapped.tetrominoes)))

    # Wrap the environment to ensure fixed observation space
    env = FixedObservationWrapper(env, target_board_shape=(
        20, 10), target_holder_shape=(4, 4), target_queue_shape=(4, 16))

    return env


def make_env(env_params):
    def _init():
        return create_training_environment(**env_params)
    return _init


# Define training stages for curriculum learning
training_stages = [
    {
        "name": "stage1_basic",
        "env_params": {"width": 10, "height": 20},
        "timesteps": 10000000,
        "learning_rate": 1e-4
    },
    {
        "name": "stage2_intermediate",
        "env_params": {"width": 10, "height": 20},
        "timesteps": 20000000,
        "learning_rate": 5e-5
    },
    {
        "name": "stage3_advanced",
        "env_params": {"width": 10, "height": 20},
        "timesteps": 20000000,
        "learning_rate": 1e-5
    }
]

# Main training function


def train_tetris_curriculum():
    model = None
    models_base_folder = "training_models"  # Base folder for all models

    for i, stage in enumerate(training_stages):
        print(f"\n=== Starting training stage: {stage['name']} ===")

        # Create vectorized environment for parallel processing
        n_envs = 2  # Number of parallel environments
        env = DummyVecEnv([make_env(stage["env_params"])
                          for _ in range(n_envs)])

        # If first stage, check if we have any previous models to load
        if i == 0:
            model_path = get_latest_model(models_base_folder)
            if model_path:
                print(f"Loading most recent model: {model_path}")
                try:
                    model = DQN.load(
                        model_path,
                        env=env,
                        verbose=0,
                        learning_rate=stage["learning_rate"],
                        batch_size=512,           # Larger batch size for GPU
                        buffer_size=100000,      # Larger replay buffer
                        device=device,            # Use GPU
                    )
                    print("Successfully loaded previous model")
                except Exception as e:
                    print(f"Error loading model: {e}")
                    print("Creating new model instead")
                    model = None
            else:
                print("No previous model found, creating new model")

        # If model is None or loading failed, create new model
        if model is None:
            print(f"Initializing new model for {stage['name']}")
            model = DQN(
                "MultiInputPolicy",
                env,
                verbose=0,
                learning_rate=stage["learning_rate"],
                buffer_size=200000,              # 500000 experiences in buffer
                batch_size=512,                   # Larger batches for GPU processing
                train_freq=(1, "step"),           # Update more frequently
                gradient_steps=16,                 # Multiple gradient steps per update
                learning_starts=10000,            # Collect more experiences before learning
                target_update_interval=1000,
                exploration_fraction=0.3,
                exploration_final_eps=0.05,
                device=device,                    # Use GPU
            )
        else:
            # If we're continuing to next stage, update environment and parameters
            if i > 0:
                print(f"Updating model for {stage['name']}")
                model.set_env(env)
                model.learning_rate = stage["learning_rate"]
                model.exploration_final_eps = 0.05

        # Train the model
        print(f"Training model for {stage['timesteps']} timesteps")
        model.learn(
            total_timesteps=stage["timesteps"],
            tb_log_name=stage["name"],
            progress_bar=True
        )

        # Save the final model for this stage
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(
            output_folder, f"tetris_dqn_{stage['name']}_{timestamp}.zip")
        model.save(model_path)
        print(f"Model saved to {model_path}")

    env.close()
    print("Curriculum training complete!")


if __name__ == "__main__":
    train_tetris_curriculum()
