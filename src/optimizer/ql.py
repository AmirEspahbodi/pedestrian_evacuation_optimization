import collections
import random
import math
import numpy as np
from typing import List, Tuple, Any, Optional
from dataclasses import dataclass
import heapq
from src.simulator.domain import Domain
from src.config import SimulationConfig, OptimizerStrategy, IEAConfig
from .common import FitnessEvaluator


# --- Helper Functions for Geometric Constraints ---


def _are_exits_overlapping(pos1: int, pos2: int, omega: int) -> bool:
    """Checks if two exits, defined by their start positions and a fixed width, overlap."""
    # Interval for exit 1 is [pos1, pos1 + omega - 1]
    # Interval for exit 2 is [pos2, pos2 + omega - 1]
    return pos1 < pos2 + omega and pos2 < pos1 + omega


def _is_configuration_valid(config: List[int], omega: int) -> bool:
    """Checks if a given k-exit configuration is valid (i.e., no exits overlap)."""
    for i in range(len(config)):
        for j in range(i + 1, len(config)):
            if _are_exits_overlapping(config[i], config[j], omega):
                return False
    return True


def _generate_initial_state(k: int, max_pos: int, omega: int) -> Optional[List[int]]:
    """
    Generates a random, valid initial configuration of k exits.
    Returns None if it's impossible to place k non-overlapping exits.
    """
    if k * omega > max_pos + 1:
        # Not enough space on the perimeter for k non-overlapping exits
        return None

    config = []
    # Create a list of all possible start positions and shuffle it for randomness
    possible_positions = list(range(max_pos + 1))
    random.shuffle(possible_positions)

    for pos in possible_positions:
        if all(
            not _are_exits_overlapping(pos, existing_pos, omega)
            for existing_pos in config
        ):
            config.append(pos)
            if len(config) == k:
                return sorted(config)  # Start with a sorted config

    return None  # Should not be reached if initial check is correct


# --- Main Q-Learning Algorithm ---


def q_learning_exit_optimizer(
    domain: Domain,
    num_episodes: int = 20,
    learning_rate_alpha: float = 0.1,
    discount_factor_gamma: float = 0.9,
    exploration_rate_epsilon: float = 1.0,
    exploration_decay_rate: float = 0.999,
    min_exploration_rate: float = 0.01,
) -> Tuple[Optional[List[int]], float]:
    omega = SimulationConfig.omega
    perimeter_length = 2 * (domain.width + domain.height)
    k_exits = SimulationConfig.num_emergency_exits
    psi_evaluator = FitnessEvaluator(domain, OptimizerStrategy.IEA)
    max_evals: int = IEAConfig.islands[0].maxevals

    # Define the valid range for an exit's starting position
    max_val_for_element = perimeter_length - omega

    # Q-table: Q[state_tuple][action_tuple] -> q_value
    # A state is a tuple of k exit positions. An action is (exit_index, new_position).
    q_table = collections.defaultdict(lambda: collections.defaultdict(float))

    best_solution: Optional[List[int]] = None
    best_fitness = float("inf")
    evaluations_count = 0

    for episode in range(num_episodes):
        print(f"episode {episode}")
        if evaluations_count >= max_evals:
            print(f"INFO: Maximum evaluations ({max_evals}) reached. Stopping.")
            break

        # 1. Initialize State
        current_solution = _generate_initial_state(k_exits, max_val_for_element, omega)
        if current_solution is None:
            raise ValueError(
                f"Cannot place {k_exits} non-overlapping exits of width {omega} "
                f"on a perimeter of length {perimeter_length}."
            )

        current_fitness = psi_evaluator.evaluate(current_solution)
        evaluations_count += 1

        # Iteratively improve the solution within the episode (a single trajectory)
        # Using a fixed number of steps per episode, e.g., k_exits * some_factor
        max_steps_per_episode = k_exits * 20

        for step in range(max_steps_per_episode):
            if evaluations_count >= max_evals:
                break

            # State is the tuple representation of the solution vector
            state_key = tuple(current_solution)

            # 2. Select Action (Epsilon-Greedy)
            action_to_perform = None
            if random.uniform(0, 1) < exploration_rate_epsilon:
                # Explore: Choose a random valid action
                # This is more efficient than generating all valid actions first
                found_valid_action = False
                for _ in range(100):  # Try 100 times to find a random valid action
                    idx_to_change = random.randint(0, k_exits - 1)
                    new_pos = random.randint(0, max_val_for_element)

                    temp_solution = list(current_solution)
                    temp_solution[idx_to_change] = new_pos

                    if _is_configuration_valid(temp_solution, omega):
                        action_to_perform = (idx_to_change, new_pos)
                        found_valid_action = True
                        break
                if not found_valid_action:
                    continue  # Skip this step if we fail to find a valid random move
            else:
                # Exploit: Choose the best known action
                # We must find the action with the max Q-value among all *valid* actions.
                best_q_for_state = -float("inf")
                # Iterate through all possible single-exit modifications
                for idx_to_change in range(k_exits):
                    for new_pos in range(max_val_for_element + 1):
                        potential_action = (idx_to_change, new_pos)

                        # Check if modifying the exit creates a valid configuration
                        other_exits = (
                            current_solution[:idx_to_change]
                            + current_solution[idx_to_change + 1 :]
                        )
                        if all(
                            not _are_exits_overlapping(new_pos, p, omega)
                            for p in other_exits
                        ):
                            q_val = q_table[state_key][potential_action]
                            if q_val > best_q_for_state:
                                best_q_for_state = q_val
                                action_to_perform = potential_action

            if action_to_perform is None:
                continue  # No valid action found, proceed to next step

            # 3. Take Action & Observe Reward and Next State
            idx, new_pos = action_to_perform
            next_solution = list(current_solution)
            next_solution[idx] = new_pos

            next_fitness = psi_evaluator.evaluate(next_solution)

            evaluations_count += 1

            # Reward is the improvement in fitness score
            reward = current_fitness - next_fitness

            next_state_key = tuple(next_solution)

            # 4. Update Q-Table using the Bellman equation
            old_q_value = q_table[state_key][action_to_perform]

            # Find max Q-value for the next state over all its valid actions
            max_q_next_state = 0.0
            if q_table[next_state_key]:
                max_q_next_state = max(q_table[next_state_key].values())

            # Q-update formula
            new_q_value = old_q_value + learning_rate_alpha * (
                reward + discount_factor_gamma * max_q_next_state - old_q_value
            )
            q_table[state_key][action_to_perform] = new_q_value

            # 5. Transition to the next state
            current_solution = next_solution
            current_fitness = next_fitness

            # Track the best solution found so far
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_solution = list(current_solution)
                print(
                    f"Episode {episode + 1}, Step {step + 1}: New best fitness: {best_fitness:.4f} -> {best_solution}"
                )

        # Decay epsilon after each episode to shift from exploration to exploitation
        exploration_rate_epsilon = max(
            min_exploration_rate, exploration_rate_epsilon * exploration_decay_rate
        )

    return best_solution, best_fitness
