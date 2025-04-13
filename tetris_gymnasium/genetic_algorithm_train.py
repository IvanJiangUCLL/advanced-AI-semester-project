import os
import time
import random
import gymnasium as gym
from gymnasium.envs.registration import register
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy
from datetime import datetime
import torch
import torch.multiprocessing as mp
from torch.jit import script

# Uncomment this section if you need to re-register the environment
if "tetris_gymnasium/Tetris" in gym.envs.registry.keys():
    if "tetris_gymnasium/Tetris" in gym.envs.registry:
        del gym.envs.registry["tetris_gymnasium/Tetris"]

register(
    id="tetris_gymnasium/Tetris",
    entry_point="tetris_gymnasium.envs.tetris:Tetris",
)

# JIT-compiled feature extraction functions for better performance


@script
def _count_holes_jit(board):
    """Count cells that are empty but have a block above them"""
    holes = 0
    for col in range(board.shape[1]):
        block_found = False
        for row in range(board.shape[0]):
            if board[row, col] > 0:
                block_found = True
            elif block_found and board[row, col] == 0:
                holes += 1
    return holes


@script
def _calculate_bumpiness_jit(board):
    """Calculate the sum of differences in heights between adjacent columns"""
    heights = torch.zeros(board.shape[1], device=board.device)

    for col in range(board.shape[1]):
        for row in range(board.shape[0]):
            if board[row, col] > 0:
                heights[col] = board.shape[0] - row
                break

    bumpiness = torch.sum(torch.abs(heights[:-1] - heights[1:]))
    return bumpiness


@script
def _calculate_height_jit(board):
    """Calculate the total height of all columns"""
    total_height = 0
    for col in range(board.shape[1]):
        for row in range(board.shape[0]):
            if board[row, col] > 0:
                total_height += board.shape[0] - row
                break
    return total_height


@script
def _count_complete_lines_jit(board):
    """Count the number of complete lines (rows filled with blocks)"""
    complete_lines = 0
    for row in range(board.shape[0]):
        if torch.all(board[row] > 0):
            complete_lines += 1
    return complete_lines


@script
def _get_landing_height_jit(board, active_mask):
    """Estimate the landing height of the current piece"""
    if torch.sum(active_mask) == 0:
        return torch.tensor(0.0, device=board.device)

    active_positions = torch.nonzero(active_mask > 0)
    if active_positions.size(0) == 0:
        return torch.tensor(0.0, device=board.device)

    return torch.tensor(board.shape[0] - torch.min(active_positions[:, 0]).item(), device=board.device)


class TetrisFeatureExtractor:
    """Extract features from Tetris board state for decision making"""

    @staticmethod
    def get_features(observation):
        """
        Extract relevant features from the Tetris observation.

        Features:
        1. Number of holes
        2. Bumpiness (sum of differences between adjacent columns)
        3. Total height
        4. Lines cleared (from info)
        5. Number of complete lines
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        board = torch.tensor(
            observation["board"], device=device, dtype=torch.float32)
        active_mask = torch.tensor(
            observation["active_tetromino_mask"], device=device, dtype=torch.float32)

        features = {
            "holes": _count_holes_jit(board),
            "bumpiness": _calculate_bumpiness_jit(board),
            "height": _calculate_height_jit(board),
            "complete_lines": _count_complete_lines_jit(board),
            "landing_height": _get_landing_height_jit(board, active_mask)
        }

        return features

    @staticmethod
    def get_features_batch(observations):
        """Extract features for a batch of observations"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = len(observations)

        # Initialize feature tensors
        features = {
            "holes": torch.zeros(batch_size, device=device),
            "bumpiness": torch.zeros(batch_size, device=device),
            "height": torch.zeros(batch_size, device=device),
            "complete_lines": torch.zeros(batch_size, device=device),
            "landing_height": torch.zeros(batch_size, device=device)
        }

        # Process each observation individually but in a batch context
        for i, obs in enumerate(observations):
            board = torch.tensor(
                obs["board"], device=device, dtype=torch.float32)
            active_mask = torch.tensor(
                obs["active_tetromino_mask"], device=device, dtype=torch.float32)

            features["holes"][i] = _count_holes_jit(board)
            features["bumpiness"][i] = _calculate_bumpiness_jit(board)
            features["height"][i] = _calculate_height_jit(board)
            features["complete_lines"][i] = _count_complete_lines_jit(board)
            features["landing_height"][i] = _get_landing_height_jit(
                board, active_mask)

        return features


class TetrisGenome:
    """Individual genome for the genetic algorithm"""

    def __init__(self, weights=None):
        """
        Initialize a genome with weights for different features

        Weights (if None, random weights are generated):
        - holes_weight: Penalty for holes
        - bumpiness_weight: Penalty for bumpiness
        - height_weight: Penalty for height
        - lines_weight: Reward for lines cleared
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if weights is None:
            self.weights = {
                "holes": torch.tensor(random.uniform(-1.0, 0.0), device=device),
                "bumpiness": torch.tensor(random.uniform(-1.0, 0.0), device=device),
                "height": torch.tensor(random.uniform(-1.0, 0.0), device=device),
                "complete_lines": torch.tensor(random.uniform(0.0, 1.0), device=device),
                "landing_height": torch.tensor(random.uniform(-1.0, 0.0), device=device)
            }
        else:
            self.weights = {k: torch.tensor(v, device=device)
                            for k, v in weights.items()}

        self.fitness = 0
        self.lines_cleared = 0
        self.games_played = 0

    def evaluate_position(self, features):
        """Calculate the value of a position based on features and weights"""
        value = torch.tensor(0.0, device=next(
            iter(self.weights.values())).device)
        for feature, weight in self.weights.items():
            value += features[feature] * weight
        return value.item()

    def evaluate_positions_batch(self, features_batch):
        """Calculate values for a batch of positions"""
        device = next(iter(self.weights.values())).device
        batch_size = features_batch["holes"].size(0)
        values = torch.zeros(batch_size, device=device)

        for feature, weight in self.weights.items():
            values += features_batch[feature] * weight

        return values

    def mutate(self, mutation_rate=0.1, mutation_scale=0.2):
        """Mutate weights randomly"""
        new_weights = self.weights.copy()

        for key in new_weights:
            if random.random() < mutation_rate:
                mutation = random.gauss(0, mutation_scale)
                new_weights[key] += mutation

                if key in ["holes", "bumpiness", "height", "landing_height"]:
                    new_weights[key] = max(-1.0, min(0.0, new_weights[key]))
                elif key in ["complete_lines"]:
                    new_weights[key] = max(0.0, min(1.0, new_weights[key]))

        return TetrisGenome(new_weights)

    @staticmethod
    def crossover(parent1, parent2):
        """Create a child by combining weights from two parents"""
        child_weights = {}

        for key in parent1.weights:
            choice = random.randint(0, 2)
            if choice == 0:
                child_weights[key] = parent1.weights[key]
            elif choice == 1:
                child_weights[key] = parent2.weights[key]
            else:
                child_weights[key] = (
                    parent1.weights[key] + parent2.weights[key]) / 2.0

        return TetrisGenome(child_weights)


# Function to be used with multiprocessing
def evaluate_genome_worker(args):
    genome, games_per_genome, max_steps_per_game = args
    env = gym.make(
        "tetris_gymnasium/Tetris",
        render_mode=None,
        gravity=True
    )

    total_reward = 0
    total_lines = 0

    for game in range(games_per_genome):
        observation, _ = env.reset()
        game_reward = 0
        game_over = False
        steps = 0

        while not game_over and steps < max_steps_per_game:
            # Select action using vectorized approach
            best_action = select_action_vectorized(observation, env, genome)

            observation, reward, game_over, _, info = env.step(best_action)
            game_reward += reward

            if "lines_cleared" in info:
                total_lines += info["lines_cleared"]

            steps += 1

        total_reward += game_reward

    env.close()

    # Return results
    return {
        "fitness": total_reward / games_per_genome,
        "lines_cleared": total_lines,
        "games_played": games_per_genome
    }


def select_action_vectorized(observation, env, genome):
    """Vectorized approach to select the best action"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_value = float('-inf')
    best_action = 0

    original_state = env.unwrapped.get_state()

    # Try all possible actions in a more GPU-friendly way
    valid_actions = []
    observations_after_actions = []

    for action in range(env.action_space.n):
        if action == env.unwrapped.actions.no_op:
            continue

        env.unwrapped.set_state(original_state)
        next_obs, reward, done, _, info = env.step(action)

        if done:
            continue

        valid_actions.append(action)
        observations_after_actions.append(next_obs)

    # Reset environment to original state
    env.unwrapped.set_state(original_state)

    if not valid_actions:
        return 0  # Default action if no valid actions

    # Extract features for all valid actions
    features_batch = TetrisFeatureExtractor.get_features_batch(
        observations_after_actions)

    # Evaluate all positions in one go
    values = genome.evaluate_positions_batch(features_batch)

    # Find best action
    best_idx = torch.argmax(values).item()
    best_action = valid_actions[best_idx]

    return best_action


class GeneticTetrisAI:
    """Genetic algorithm for evolving Tetris-playing AIs"""

    def __init__(self, population_size=50, elite_size=5, mutation_rate=0.1,
                 games_per_genome=3, max_steps_per_game=5000):
        """
        Initialize the genetic algorithm

        Args:
            population_size: Number of individuals in the population
            elite_size: Number of top individuals to keep unchanged
            mutation_rate: Chance of mutation for each weight
            games_per_genome: Number of games to play for fitness evaluation
            max_steps_per_game: Maximum number of steps allowed per game
        """
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.games_per_genome = games_per_genome
        self.max_steps_per_game = max_steps_per_game

        # Set PyTorch to use the maximum number of threads
        torch.set_num_threads(os.cpu_count())

        self.population = [TetrisGenome() for _ in range(population_size)]
        self.best_genome = None
        self.generations = 0
        self.history = {
            "best_fitness": [],
            "avg_fitness": [],
            "best_lines": []
        }

        # Configure CUDA for better performance
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            print(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    def select_action(self, observation, env, genome):
        """
        Select the best action using vectorized approach for better GPU utilization
        """
        return select_action_vectorized(observation, env, genome)

    def evaluate_genome(self, genome, render=False):
        """Evaluate a genome by playing several games and averaging performance"""
        env = gym.make(
            "tetris_gymnasium/Tetris",
            render_mode="rgb_array" if render else None,
            gravity=True
        )

        total_reward = 0
        total_lines = 0

        for game in range(self.games_per_genome):
            observation, _ = env.reset()
            game_reward = 0
            game_over = False
            steps = 0

            while not game_over and steps < self.max_steps_per_game:
                action = self.select_action(observation, env, genome)

                observation, reward, game_over, _, info = env.step(action)
                game_reward += reward

                if "lines_cleared" in info:
                    total_lines += info["lines_cleared"]

                steps += 1

                if render:
                    env.render()
                    time.sleep(0.01)

            total_reward += game_reward

        env.close()

        genome.fitness = total_reward / self.games_per_genome
        genome.lines_cleared = total_lines
        genome.games_played = self.games_per_genome

        return genome.fitness

    def evaluate_population_parallel(self):
        """Evaluate all genomes in the population in parallel"""
        # Prepare arguments for multiprocessing
        args_list = [(genome, self.games_per_genome, self.max_steps_per_game)
                     for genome in self.population]

        # Get number of CPU cores for parallel processing
        num_processes = max(1, os.cpu_count() // 2)

        # Use multiprocessing to evaluate genomes in parallel
        with mp.Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(evaluate_genome_worker, args_list),
                total=len(self.population),
                desc="Evaluating population"
            ))

        # Update population with results
        for i, result in enumerate(results):
            self.population[i].fitness = result["fitness"]
            self.population[i].lines_cleared = result["lines_cleared"]
            self.population[i].games_played = result["games_played"]

    def select_parents(self):
        """Select parents using tournament selection"""
        parents = []

        sorted_population = sorted(
            self.population, key=lambda x: x.fitness, reverse=True)

        elites = sorted_population[:self.elite_size]
        parents.extend(elites)

        while len(parents) < self.population_size:
            tournament_size = 3
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)

        return parents, elites

    def create_next_generation(self, parents, elites):
        """Create the next generation through crossover and mutation"""
        next_generation = []

        next_generation.extend(elites)

        while len(next_generation) < self.population_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

            child = TetrisGenome.crossover(parent1, parent2)

            child = child.mutate(self.mutation_rate)

            next_generation.append(child)

        return next_generation

    def evolve(self, generations=50):
        """Run the genetic algorithm for a number of generations"""
        output_folder = "genetic_tetris_results"
        os.makedirs(output_folder, exist_ok=True)

        # Set up the process start method for multiprocessing
        mp.set_start_method('spawn', force=True)

        for gen in range(generations):
            self.generations += 1
            print(f"\n=== Generation {self.generations} ===")

            # Use parallel evaluation for better performance
            self.evaluate_population_parallel()

            self.population.sort(key=lambda x: x.fitness, reverse=True)

            current_best = self.population[0]
            if self.best_genome is None or current_best.fitness > self.best_genome.fitness:
                self.best_genome = deepcopy(current_best)

            best_fitness = self.population[0].fitness
            avg_fitness = sum(
                g.fitness for g in self.population) / len(self.population)
            best_lines = self.population[0].lines_cleared

            print(f"Best fitness: {best_fitness:.2f}")
            print(f"Average fitness: {avg_fitness:.2f}")
            print(f"Best lines cleared: {best_lines}")
            print(f"Best weights: {self.population[0].weights}")

            self.history["best_fitness"].append(best_fitness)
            self.history["avg_fitness"].append(avg_fitness)
            self.history["best_lines"].append(best_lines)

            if gen % 5 == 0 or gen == generations - 1:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(
                    output_folder, f"best_genome_gen{self.generations}_{timestamp}.pkl")
                with open(filename, 'wb') as f:
                    pickle.dump(self.best_genome, f)
                print(f"Saved best genome to {filename}")

                self.plot_progress(output_folder)

            if gen < generations - 1:
                parents, elites = self.select_parents()
                self.population = self.create_next_generation(parents, elites)

    def plot_progress(self, output_folder):
        """Plot the progress of the genetic algorithm"""
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(self.history["best_fitness"], label="Best Fitness")
        plt.plot(self.history["avg_fitness"], label="Average Fitness")
        plt.title("Fitness over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(self.history["best_lines"], label="Best Lines Cleared")
        plt.title("Lines Cleared over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Lines Cleared")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            output_folder, f"progress_gen{self.generations}_{timestamp}.png")
        plt.savefig(filename)
        plt.close()

    def render_best_genome(self, games=1, steps_per_game=1000):
        """Render games played by the best genome"""
        if self.best_genome is None:
            print("No best genome to render.")
            return

        print(f"Rendering {games} games with the best genome...")
        print(f"Best genome weights: {self.best_genome.weights}")

        env = gym.make(
            "tetris_gymnasium/Tetris",
            render_mode="human",
            gravity=True
        )

        for game in range(games):
            print(f"Game {game+1}")
            observation, _ = env.reset()
            done = False
            total_reward = 0
            total_lines = 0
            steps = 0

            while not done and steps < steps_per_game:
                action = self.select_action(observation, env, self.best_genome)
                observation, reward, done, _, info = env.step(action)

                total_reward += reward
                steps += 1

                if "lines_cleared" in info:
                    total_lines += info["lines_cleared"]

                env.render()
                time.sleep(0.05)

            print(
                f"Game {game+1} - Steps: {steps}, Reward: {total_reward:.2f}, Lines: {total_lines}")

        env.close()


def load_best_genome(filename):
    """Load a saved genome from a file"""
    with open(filename, 'rb') as f:
        genome = pickle.load(f)
    return genome


if __name__ == "__main__":
    # Print system information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CPU count: {os.cpu_count()}")

    mp.set_start_method("spawn", force=True)

    # Increase algorithm parameters for better GPU utilization
    POPULATION_SIZE = 50  # Increased from 100
    ELITE_SIZE = 5  # Increased from 10
    MUTATION_RATE = 0.1
    GENERATIONS = 20
    GAMES_PER_GENOME = 3  # Increased from 3

    # Initialize genetic algorithm with optimized parameters
    genetic_ai = GeneticTetrisAI(
        population_size=POPULATION_SIZE,
        elite_size=ELITE_SIZE,
        mutation_rate=MUTATION_RATE,
        games_per_genome=GAMES_PER_GENOME
    )

    # Start evolution
    genetic_ai.evolve(generations=GENERATIONS)

    # Render best genome
    genetic_ai.render_best_genome(games=3)

    print("Training complete!")
