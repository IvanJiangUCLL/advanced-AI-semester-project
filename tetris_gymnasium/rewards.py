"""This module contains the improved mapping for the rewards that the agent can receive."""
from dataclasses import dataclass


@dataclass
class RewardsMapping:
    """Mapping for the rewards that the agent can receive.

    The mapping is designed to encourage efficient play, line clearing, and avoiding game over.
    """
    # Basic movements - small negative rewards to encourage efficiency
    move_left: float = -0.01
    move_right: float = -0.01
    hard_drop: float = 0  # Neutral as this is just committing a placement
    move_down: float = 0.01  # Small positive to encourage playing faster
    rotate_clockwise: float = -0.01
    rotate_counterclockwise: float = -0.01
    swap: float = -0.1  # Slightly higher cost as this should be a strategic choice
    no_op: float = -0.1  # Discourage doing nothing
    
    # Staying alive has a small positive reward
    alife: float = 0.1
    
    # Line clearing rewards are proportional but much more reasonable
    clear_line: float = 100  # Base reward for clearing a line
    
    # Major penalties for bad outcomes
    game_over: float = -50  # Significant penalty for ending the game
    invalid_action: float = -1  # Penalty for invalid actions
    
    # New rewards to shape behavior
    clear_multiple_lines_bonus: float = 10  # Additional bonus per line when clearing multiple
    height_penalty_factor: float = -0.1  # Penalty per unit of height
    hole_penalty: float = -1  # Penalty for each hole created
    smoothness_reward: float = 1  # Reward for keeping the surface smooth
