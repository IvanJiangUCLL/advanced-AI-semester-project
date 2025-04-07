import cv2
import gymnasium as gym
from gymnasium.envs.registration import register
import inspect

# Clear the environment registry if the environment is already registered
if "tetris_gymnasium/Tetris" in gym.envs.registry.keys():
    if "tetris_gymnasium/Tetris" in gym.envs.registry:
        del gym.envs.registry["tetris_gymnasium/Tetris"]

# Re-register the Tetris environment
register(
    id="tetris_gymnasium/Tetris",
    entry_point="tetris_gymnasium.envs.tetris:Tetris",
)

if __name__ == "__main__":
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    env.reset(seed=42)

    # Print the module and file path of the environment
    print(f"Module: {env.unwrapped.__class__.__module__}")
    print(f"File: {inspect.getfile(env.unwrapped.__class__)}")

    terminated = False
    while not terminated:
        env.render()
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        key = cv2.waitKey(100)  # timeout to see the movement
    print("Game Over!")
