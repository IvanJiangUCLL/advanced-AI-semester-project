import os
import datetime
from stable_baselines3 import DQN
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit

if "tetris_gymnasium/Tetris" in gym.envs.registry.keys():
    if "tetris_gymnasium/Tetris" in gym.envs.registry:
        del gym.envs.registry["tetris_gymnasium/Tetris"]

register(
    id="tetris_gymnasium/Tetris",
    entry_point="tetris_gymnasium.envs.tetris:Tetris",
)

env = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array")

output_folder = "training_models"
os.makedirs(output_folder, exist_ok=True)


def get_latest_model(folder):
    models = [f for f in os.listdir(folder) if f.endswith(".zip")]
    if not models:
        return None
    models.sort(key=lambda x: os.path.getmtime(
        os.path.join(folder, x)), reverse=True)
    return os.path.join(folder, models[0])


model_path = get_latest_model(output_folder)
if model_path:
    print(f"Loading model from {model_path}")
    model = DQN.load(model_path, env=env)
    model.learning_rate = lambda x: 1e-3 * (1 - x)
    # model.train_freq = (4, "step") # doesn't work, don't know why
    model.exploration_rate = 0.4
    model.exploration_fraction = 0.1
else:
    print("No previous model found. Initializing a new model.")
    model = DQN(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=100000,
        batch_size=64,
        train_freq=(4, "step"),
        target_update_interval=10000,
        exploration_fraction=0.4,
        exploration_final_eps=0.1,
    )

model.learn(total_timesteps=1000000)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = os.path.join(output_folder, f"tetris_dqn_{timestamp}.zip")
model.save(model_filename)
print(f"Model saved to {model_filename}")

env.close()
