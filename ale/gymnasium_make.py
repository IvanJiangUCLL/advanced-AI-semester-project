import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
import ale_py
import numpy as np
from gymnasium.spaces import Discrete


class TetrisActionWrapper(gym.ActionWrapper):
    """Custom wrapper to restrict the action space to Tetris-relevant actions."""

    def __init__(self, env):
        super().__init__(env)
        # Define the new action space (only 5 actions relevant to Tetris)
        self.action_space = Discrete(5)
        # Map the new action space to the original action space
        self._action_map = [0, 1, 2, 3, 4]  # NOOP, FIRE, RIGHT, LEFT, DOWN

    def action(self, action):
        """Map the restricted action space to the original action space."""
        return self._action_map[action]

# Define a custom reward wrapper to modify the reward function


class CustomRewardWrapper(gym.RewardWrapper):
    """Custom wrapper to modify the reward function based on lines completed."""

    def __init__(self, env):
        super().__init__(env)
        self.previous_score = 0  # Track the score from the previous step

    def reward(self, reward):
        """Modify the reward function to count lines completed."""
        # Get the current score from the info dictionary
        # Replace with actual score logic
        current_score = self.env.unwrapped.ale.getEpisodeFrameNumber()

        # Calculate the difference in score (lines completed)
        lines_completed = current_score - self.previous_score

        # Update the previous score
        self.previous_score = current_score

        # Reward is proportional to the number of lines completed
        return lines_completed

# Function to create and wrap the Tetris environment


def create_tetris_env(render_mode="human"):
    """Creates and wraps the Tetris environment with grayscale observations."""
    gym.register_envs(ale_py)  # Register ALE environments
    env = gym.make(
        "ALE/Tetris-v5",  # Use the Tetris environment with image-based observations
        obs_type="grayscale",  # Set observation type to grayscale
        render_mode=render_mode,
        frameskip=1,  # Disable frame-skipping in the base environment
        full_action_space=True,  # Use the full action space
    )
    # Apply Atari preprocessing (grayscale, resizing, etc.)
    env = AtariPreprocessing(env)
    # Apply the custom action wrapper
    env = TetrisActionWrapper(env)
    # Apply the custom reward wrapper
    env = CustomRewardWrapper(env)
    return env


# Main logic
if __name__ == "__main__":
    # Create the environment
    env = create_tetris_env(render_mode="human")

    # Initialize the environment
    obs, info = env.reset()

    # Example random agent loop
    try:
        for _ in range(500000):  # Run for 100000 steps or until the game ends
            action = env.action_space.sample()  # Take a random action
            obs, reward, done, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward}")
            if done or truncated:
                obs, info = env.reset()

            # Check if the game is over
            if done or truncated:
                print("Game over! Resetting environment.")
                obs, info = env.reset()
    finally:
        env.close()  # Ensure the environment is closed properly
