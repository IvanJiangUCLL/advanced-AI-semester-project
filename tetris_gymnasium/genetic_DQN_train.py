import os
import datetime
import torch
import numpy as np
import random
import copy
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from tetris_gymnasium.components.tetromino_queue import TetrominoQueue
from tetris_gymnasium.components.tetromino_randomizer import BagRandomizer

# Keep your existing environment setup and wrapper
# [Your existing FixedObservationWrapper and environment setup code]


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
                high=np.full(self.target_holder_shape,
                             self.env.observation_space["holder"].high.max()),
                shape=self.target_holder_shape,
                dtype=np.uint8,
            ),
            "queue": gym.spaces.Box(
                low=0,
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

# Check CUDA availability and setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create output folder
output_folder = "training_models"
os.makedirs(output_folder, exist_ok=True)


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


class GeneticDQN:
    def __init__(self, population_size=10, generations=20, base_timesteps=200000):
        self.population_size = population_size
        self.generations = generations
        self.base_timesteps = base_timesteps
        self.population = []
        self.best_model = None
        self.best_fitness = float('-inf')
        self.env_params = {"width": 10, "height": 20}
        self.models_base_folder = "training_models/genetic_dqn"
        os.makedirs(self.models_base_folder, exist_ok=True)

    def initialize_population(self):
        """Create an initial population of DQN models with random hyperparameters"""
        print("Initializing population...")
        self.population = []

        for i in range(self.population_size):
            # Create a set of hyperparameters with random variations
            hyperparams = {
                # Random learning rate between 1e-6 and 1e-3
                "learning_rate": 10**random.uniform(-6, -3),
                "buffer_size": random.choice([50000, 100000, 200000]),
                "batch_size": random.choice([128, 256, 512]),
                "gradient_steps": random.choice([1, 4, 8, 16]),
                "exploration_fraction": random.uniform(0.1, 0.3),
                "exploration_final_eps": random.uniform(0.01, 0.1),
                "target_update_interval": random.choice([500, 1000, 2000])
            }

            # Create the environment
            env = DummyVecEnv([make_env(self.env_params) for _ in range(1)])

            # Create the model with these hyperparameters
            model = DQN(
                "MultiInputPolicy",
                env,
                verbose=0,
                learning_rate=hyperparams["learning_rate"],
                buffer_size=hyperparams["buffer_size"],
                batch_size=hyperparams["batch_size"],
                gradient_steps=hyperparams["gradient_steps"],
                learning_starts=min(10000, self.base_timesteps // 10),
                exploration_fraction=hyperparams["exploration_fraction"],
                exploration_final_eps=hyperparams["exploration_final_eps"],
                target_update_interval=hyperparams["target_update_interval"],
                device=device,
            )

            self.population.append({
                "model": model,
                "hyperparams": hyperparams,
                "fitness": 0,
                "env": env,
                "id": i
            })

        print(f"Population initialized with {self.population_size} models")

    def evaluate_fitness(self, individual, n_episodes=5):
        """Evaluate the fitness of an individual by playing n episodes"""
        model = individual["model"]
        env = individual["env"]

        total_reward = 0
        for _ in range(n_episodes):
            # Fix the reset call to handle different return formats
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                if len(reset_result) == 2:
                    obs, info = reset_result
                else:
                    obs = reset_result[0]  # Take just the observation
            else:
                obs = reset_result  # For older versions that just return obs

            done = False
            episode_reward = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                step_result = env.step(action)

                # Handle both Gym and Gymnasium return formats
                if len(step_result) == 5:  # Gymnasium format
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:  # Gym format
                    obs, reward, done, info = step_result

                episode_reward += reward[0] if isinstance(
                    reward, np.ndarray) else reward

            total_reward += episode_reward

        # Average reward over episodes
        fitness = total_reward / n_episodes
        return fitness

    def select_parents(self, tournament_size=3):
        """Select parents using tournament selection"""
        parents = []
        for _ in range(self.population_size):
            # Select tournament_size individuals randomly
            tournament = random.sample(self.population, tournament_size)
            # Choose the best one
            winner = max(tournament, key=lambda x: x["fitness"])
            parents.append(winner)
        return parents

    def crossover(self, parent1, parent2):
        """Create a new individual by mixing hyperparameters from two parents"""
        child_hyperparams = {}

        # Randomly select hyperparameters from either parent
        for key in parent1["hyperparams"]:
            if random.random() < 0.5:
                child_hyperparams[key] = parent1["hyperparams"][key]
            else:
                child_hyperparams[key] = parent2["hyperparams"][key]

        # Create a new environment for the child
        env = DummyVecEnv([make_env(self.env_params) for _ in range(1)])

        # Create a new model with the mixed hyperparameters
        model = DQN(
            "MultiInputPolicy",
            env,
            verbose=0,
            learning_rate=child_hyperparams["learning_rate"],
            buffer_size=child_hyperparams["buffer_size"],
            batch_size=child_hyperparams["batch_size"],
            gradient_steps=child_hyperparams["gradient_steps"],
            learning_starts=min(10000, self.base_timesteps // 10),
            exploration_fraction=child_hyperparams["exploration_fraction"],
            exploration_final_eps=child_hyperparams["exploration_final_eps"],
            target_update_interval=child_hyperparams["target_update_interval"],
            device=device,
        )

        # If one of the parents is the best model found so far, transfer some weights
        if parent1["fitness"] == self.best_fitness or parent2["fitness"] == self.best_fitness:
            best_parent = parent1 if parent1["fitness"] == self.best_fitness else parent2
            # Transfer policy network weights (partial transfer to maintain diversity)
            for target_param, source_param in zip(model.policy.parameters(), best_parent["model"].policy.parameters()):
                if random.random() < 0.7:  # 70% chance to inherit weights from best parent
                    target_param.data.copy_(source_param.data)

        return {
            "model": model,
            "hyperparams": child_hyperparams,
            "fitness": 0,
            "env": env,
            "id": random.randint(1000, 9999)  # New random ID
        }

    def mutate(self, individual, mutation_rate=0.2):
        """Mutate hyperparameters with some probability"""
        mutated_hyperparams = copy.deepcopy(individual["hyperparams"])

        for key in mutated_hyperparams:
            # Mutate each hyperparameter with probability mutation_rate
            if random.random() < mutation_rate:
                if key == "learning_rate":
                    mutated_hyperparams[key] = 10**random.uniform(-6, -3)
                elif key == "buffer_size":
                    mutated_hyperparams[key] = random.choice(
                        [50000, 100000, 200000])
                elif key == "batch_size":
                    mutated_hyperparams[key] = random.choice([128, 256, 512])
                elif key == "gradient_steps":
                    mutated_hyperparams[key] = random.choice([1, 4, 8, 16])
                elif key == "exploration_fraction":
                    mutated_hyperparams[key] = random.uniform(0.1, 0.3)
                elif key == "exploration_final_eps":
                    mutated_hyperparams[key] = random.uniform(0.01, 0.1)
                elif key == "target_update_interval":
                    mutated_hyperparams[key] = random.choice([500, 1000, 2000])

        # Create a new model with the mutated hyperparameters
        # We're keeping the weights from the original model but updating the hyperparams
        original_model = individual["model"]
        env = individual["env"]

        mutated_model = DQN(
            "MultiInputPolicy",
            env,
            verbose=0,
            learning_rate=mutated_hyperparams["learning_rate"],
            buffer_size=mutated_hyperparams["buffer_size"],
            batch_size=mutated_hyperparams["batch_size"],
            gradient_steps=mutated_hyperparams["gradient_steps"],
            learning_starts=min(10000, self.base_timesteps // 10),
            exploration_fraction=mutated_hyperparams["exploration_fraction"],
            exploration_final_eps=mutated_hyperparams["exploration_final_eps"],
            target_update_interval=mutated_hyperparams["target_update_interval"],
            device=device,
        )

        # Transfer policy network weights
        for target_param, source_param in zip(mutated_model.policy.parameters(), original_model.policy.parameters()):
            target_param.data.copy_(source_param.data)

        return {
            "model": mutated_model,
            "hyperparams": mutated_hyperparams,
            "fitness": 0,
            "env": env,
            "id": individual["id"]
        }

    def train_generation(self, gen_number):
        """Train all models in the current generation"""
        print(f"\n=== Training Generation {gen_number} ===")

        # Adjust timesteps based on generation (more training for later generations)
        timesteps = int(self.base_timesteps *
                        (1 + gen_number / self.generations))

        for i, individual in enumerate(self.population):
            print(
                f"Training individual {i+1}/{self.population_size} with hyperparams: {individual['hyperparams']}")

            # Train the model
            individual["model"].learn(
                total_timesteps=timesteps, progress_bar=True)

            # Evaluate the fitness
            fitness = self.evaluate_fitness(individual)
            individual["fitness"] = fitness

            print(f"Individual {i+1} fitness: {fitness}")

            # Update best model if this is better
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_model = individual["model"]

                # Save the best model so far
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join(
                    self.models_base_folder,
                    f"best_model_gen{gen_number}_fitness{int(fitness)}_{timestamp}.zip"
                )
                self.best_model.save(model_path)
                print(
                    f"New best model saved to {model_path} with fitness {fitness}")

    def create_next_generation(self):
        """Create the next generation through selection, crossover, and mutation"""
        print("\n=== Creating Next Generation ===")

        # Sort population by fitness
        self.population.sort(key=lambda x: x["fitness"], reverse=True)

        # Keep the best individual (elitism)
        new_population = [self.population[0]]

        # Select parents for producing offspring
        parents = self.select_parents()

        # Create offspring until we fill the population
        while len(new_population) < self.population_size:
            # Select two parents
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

            # Create a child through crossover
            child = self.crossover(parent1, parent2)

            # Possibly mutate the child
            if random.random() < 0.3:  # 30% chance to mutate
                child = self.mutate(child)

            new_population.append(child)

        # Close the environments of the old population to free resources
        for individual in self.population:
            individual["env"].close()

        # Update the population
        # Ensure we keep exactly population_size individuals
        self.population = new_population[:self.population_size]

    def run(self):
        """Run the genetic algorithm"""
        print("Starting Genetic DQN optimization")

        # Initialize population
        self.initialize_population()

        # Run for specified number of generations
        for gen in range(self.generations):
            print(f"\n=== Generation {gen+1}/{self.generations} ===")

            # Train and evaluate the current generation
            self.train_generation(gen)

            # Print generation stats
            fitnesses = [ind["fitness"] for ind in self.population]
            avg_fitness = sum(fitnesses) / len(fitnesses)
            max_fitness = max(fitnesses)
            min_fitness = min(fitnesses)

            print(f"Generation {gen+1} stats:")
            print(f"  Avg fitness: {avg_fitness:.2f}")
            print(f"  Max fitness: {max_fitness:.2f}")
            print(f"  Min fitness: {min_fitness:.2f}")
            print(f"  Best fitness so far: {self.best_fitness:.2f}")

            # Create the next generation (except for the last one)
            if gen < self.generations - 1:
                self.create_next_generation()

        print("\n=== Genetic DQN optimization completed ===")
        print(f"Best fitness achieved: {self.best_fitness}")

        # Save the final best model
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = os.path.join(
            self.models_base_folder,
            f"final_best_model_fitness{int(self.best_fitness)}_{timestamp}.zip"
        )
        self.best_model.save(final_model_path)
        print(f"Final best model saved to {final_model_path}")

        return self.best_model, self.best_fitness


if __name__ == "__main__":
    # Set smaller values for testing, increase for actual training
    genetic_dqn = GeneticDQN(
        population_size=6, generations=10, base_timesteps=100000)
    best_model, best_fitness = genetic_dqn.run()
    print(f"Training complete! Best fitness: {best_fitness}")
