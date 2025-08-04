import collections
import random
import math
import time
import numpy as np
from typing import List, Tuple, Any, Optional, Dict
from dataclasses import dataclass
import heapq
from src.simulator.domain import Domain
from src.config import SimulationConfig, OptimizerStrategy, IEAConfig
from .common import FitnessEvaluator


# --- Helper Functions for Geometric Constraints ---


def _are_exits_overlapping(pos1: int, pos2: int, omega: int) -> bool:
    """Checks if two exits, defined by their start positions and a fixed width, overlap."""
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
        return None

    config = []
    possible_positions = list(range(max_pos + 1))
    random.shuffle(possible_positions)

    for pos in possible_positions:
        if all(not _are_exits_overlapping(pos, p, omega) for p in config):
            config.append(pos)
            if len(config) == k:
                return sorted(config)

    return None


# --- Main Q-Learning Algorithm ---


def q_learning_exit_optimizer(
    domain: Domain,
    num_episodes: int = 20,
    learning_rate_alpha: float = 0.1,
    discount_factor_gamma: float = 0.9,
    exploration_rate_epsilon: float = 1.0,
    exploration_decay_rate: float = 0.999,
    min_exploration_rate: float = 0.01,
) -> Tuple[Optional[List[int]], float, Dict[str, List[float]], float]:
    omega = SimulationConfig.omega
    perimeter_length = 2 * (domain.width + domain.height)
    k_exits = SimulationConfig.num_emergency_exits
    psi_evaluator = FitnessEvaluator(domain, OptimizerStrategy.IEA)
    max_evals: int = IEAConfig.islands[0].maxevals
    max_val_for_element = perimeter_length - omega

    # Q-table and best‐so‐far
    q_table = collections.defaultdict(lambda: collections.defaultdict(float))
    best_solution: Optional[List[int]] = None
    best_fitness = float("inf")
    evaluations_count = 0

    start_time = time.perf_counter()
    time_to_best: Optional[float] = None
    history: Dict[str, List[float]] = {
        f"episode-{i + 1}": [] for i in range(num_episodes)
    }

    for episode in range(num_episodes):
        print(f"start episode: {episode+1}")
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

        # Possibly record first-best at initialization
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_solution = list(current_solution)
            time_to_best = time.perf_counter() - start_time

        max_steps_per_episode = k_exits * 30

        # 2. Inner Loop: steps within one episode
        for step in range(max_steps_per_episode):
            if evaluations_count >= max_evals:
                break

            state_key = tuple(current_solution)
            action_to_perform = None

            # 2a. ε‐Greedy: Explore
            if random.uniform(0, 1) < exploration_rate_epsilon:
                for _ in range(100):
                    idx = random.randint(0, k_exits - 1)
                    new_pos = random.randint(0, max_val_for_element)
                    temp = list(current_solution)
                    temp[idx] = new_pos
                    if _is_configuration_valid(temp, omega):
                        action_to_perform = (idx, new_pos)
                        break
            else:
                # 2b. Exploit: choose best Q
                best_q = -float("inf")
                for idx in range(k_exits):
                    for new_pos in range(max_val_for_element + 1):
                        other = current_solution[:idx] + current_solution[idx + 1 :]
                        if all(
                            not _are_exits_overlapping(new_pos, p, omega) for p in other
                        ):
                            qv = q_table[state_key][(idx, new_pos)]
                            if qv > best_q:
                                best_q = qv
                                action_to_perform = (idx, new_pos)

            if action_to_perform is None:
                continue

            # 3. Take Action
            idx, new_pos = action_to_perform
            next_solution = list(current_solution)
            next_solution[idx] = new_pos
            next_fitness = psi_evaluator.evaluate(next_solution)

            if next_fitness < best_fitness:
                best_fitness = next_fitness
                best_solution = list(next_solution)
                time_to_best = time.perf_counter() - start_time

            evaluations_count += 1

            reward = current_fitness - next_fitness
            next_key = tuple(next_solution)

            # 4. Q‐Update
            old_q = q_table[state_key][action_to_perform]
            max_q_next = max(q_table[next_key].values()) if q_table[next_key] else 0.0
            new_q = old_q + learning_rate_alpha * (
                reward + discount_factor_gamma * max_q_next - old_q
            )
            q_table[state_key][action_to_perform] = new_q

            # 5. Transition
            current_solution = next_solution
            current_fitness = next_fitness

            # Track new global best
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_solution = list(current_solution)
                time_to_best = time.perf_counter() - start_time
                print(
                    f"Episode {episode + 1}, Step {step + 1}: "
                    f"New best fitness: {best_fitness:.4f} -> {best_solution}"
                )

            # Record the final fitness of this episode
            history[f"episode-{episode + 1}"].append(float(current_fitness))

        # Decay exploration rate
        exploration_rate_epsilon = max(
            min_exploration_rate, exploration_rate_epsilon * exploration_decay_rate
        )

    # Ensure time_to_best is not None
    if time_to_best is None:
        time_to_best = time.perf_counter() - start_time

    return best_solution, best_fitness, history, time_to_best
