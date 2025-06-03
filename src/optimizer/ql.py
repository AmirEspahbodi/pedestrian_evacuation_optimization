import collections
import random
import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass
import heapq
from src.simulator.domain import Domain
from src.config import SimulationConfig, OptimizerStrategy, IEAConfig
from .common import FitnessEvaluator


@dataclass
class QLearningConfig:
    """Configuration parameters for Q-learning algorithm"""

    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    q_init_value: float = 0.0
    use_optimistic_init: bool = True
    use_relative_reward: bool = True
    early_stopping_patience: int = 100
    convergence_threshold: float = 1e-6


class AdvancedQLearningExitOptimizer:
    """
    Advanced Q-Learning optimizer for emergency exit placement using iterative improvement.

    Features:
    - Efficient omega constraint handling
    - Adaptive exploration strategies
    - Experience-based learning with memory optimization
    - Early stopping and convergence detection
    - Multiple initialization strategies
    """

    def __init__(self, config: QLearningConfig = None):
        self.config = config or QLearningConfig()
        self.q_table: Dict[Tuple, Dict[Tuple, float]] = collections.defaultdict(
            lambda: collections.defaultdict(lambda: self._get_init_q_value())
        )
        self.best_fitness = float("inf")
        self.best_solution: Optional[List[int]] = None
        self.episode_rewards: List[float] = []
        self.fitness_history: List[float] = []
        self.eval_count = 0

    def _get_init_q_value(self) -> float:
        """Get initial Q-value based on configuration"""
        if self.config.use_optimistic_init:
            # Optimistic initialization to encourage exploration
            return 10.0 / (1 - self.config.discount_factor)
        return self.config.q_init_value

    def _generate_valid_k_vector(
        self, k_exits: int, perimeter_length: int, omega: int
    ) -> List[int]:
        """
        Generate a valid K-exit vector ensuring no overlaps given omega constraint.
        Uses greedy placement with random ordering for diversity.
        """
        if k_exits * omega > perimeter_length:
            raise ValueError(
                f"Cannot place {k_exits} exits of width {omega} on perimeter of length {perimeter_length}"
            )

        # Create available segments
        available_positions = list(range(perimeter_length - omega + 1))
        random.shuffle(available_positions)

        exits = []
        used_segments: Set[int] = set()

        for pos in available_positions:
            if len(exits) >= k_exits:
                break

            # Check if this position conflicts with existing exits
            segment = set(range(pos, pos + omega))
            if not segment.intersection(used_segments):
                exits.append(pos)
                used_segments.update(segment)

        if len(exits) < k_exits:
            # Fallback: try systematic placement
            exits = []
            used_segments = set()
            step = max(omega, perimeter_length // (k_exits + 1))

            for i in range(k_exits):
                pos = (i * step) % (perimeter_length - omega + 1)
                segment = set(range(pos, pos + omega))

                # Find next available position if current conflicts
                attempts = 0
                while (
                    segment.intersection(used_segments) and attempts < perimeter_length
                ):
                    pos = (pos + 1) % (perimeter_length - omega + 1)
                    segment = set(range(pos, pos + omega))
                    attempts += 1

                if attempts < perimeter_length:
                    exits.append(pos)
                    used_segments.update(segment)

        return (
            exits[:k_exits]
            if len(exits) >= k_exits
            else self._fallback_placement(k_exits, perimeter_length, omega)
        )

    def _fallback_placement(
        self, k_exits: int, perimeter_length: int, omega: int
    ) -> List[int]:
        """Fallback placement strategy for difficult configurations"""
        spacing = perimeter_length // k_exits
        exits = []
        for i in range(k_exits):
            pos = min(
                (i * spacing) % (perimeter_length - omega + 1), perimeter_length - omega
            )
            exits.append(pos)
        return exits

    def _get_valid_actions(
        self, current_vector: List[int], perimeter_length: int, omega: int
    ) -> List[Tuple[int, int]]:
        """
        Generate all valid actions (exit_index, new_position) that don't violate omega constraint.
        Optimized for efficiency with early pruning.
        """
        valid_actions = []
        k_exits = len(current_vector)

        for exit_idx in range(k_exits):
            current_segments = set()
            # Build used segments excluding the exit we're modifying
            for i, pos in enumerate(current_vector):
                if i != exit_idx:
                    current_segments.update(range(pos, pos + omega))

            # Check each possible new position
            for new_pos in range(perimeter_length - omega + 1):
                new_segment = set(range(new_pos, new_pos + omega))
                if not new_segment.intersection(current_segments):
                    valid_actions.append((exit_idx, new_pos))

        return valid_actions

    def _calculate_reward(self, prev_fitness: float, curr_fitness: float) -> float:
        """Calculate reward based on fitness improvement"""
        if self.config.use_relative_reward:
            # Relative improvement with scaling
            improvement = prev_fitness - curr_fitness
            # Add small bonus for any improvement to encourage exploration
            if improvement > 0:
                return improvement + 0.1
            else:
                return improvement - 0.05  # Small penalty for worsening
        else:
            return -curr_fitness

    def _select_action(
        self, state_tuple: Tuple, valid_actions: List[Tuple[int, int]], epsilon: float
    ) -> Tuple[int, int]:
        """
        Select action using epsilon-greedy with enhanced exploration strategies.
        """
        if not valid_actions:
            raise ValueError("No valid actions available")

        if random.random() < epsilon:
            # Exploration: choose random valid action
            return random.choice(valid_actions)
        else:
            # Exploitation: choose best known action
            if state_tuple not in self.q_table or not self.q_table[state_tuple]:
                # If state is new, choose random action
                return random.choice(valid_actions)

            best_q = float("-inf")
            best_actions = []

            for action in valid_actions:
                q_val = self.q_table[state_tuple][action]
                if q_val > best_q:
                    best_q = q_val
                    best_actions = [action]
                elif q_val == best_q:
                    best_actions.append(action)

            return random.choice(best_actions)

    def _update_epsilon(self, episode: int) -> float:
        """Calculate current epsilon with adaptive decay"""
        return max(
            self.config.epsilon_min,
            self.config.epsilon_start * (self.config.epsilon_decay**episode),
        )

    def _get_canonical_state(self, vector: List[int]) -> Tuple:
        """
        Convert vector to canonical state representation.
        Uses sorted tuple for state space reduction if exits are indistinguishable.
        """
        return tuple(sorted(vector))

    def _check_convergence(self, window_size: int = 50) -> bool:
        """Check if algorithm has converged based on fitness history"""
        if len(self.fitness_history) < window_size:
            return False

        recent_fitness = self.fitness_history[-window_size:]
        fitness_variance = np.var(recent_fitness)
        return fitness_variance < self.config.convergence_threshold

    def _should_early_stop(self, no_improvement_count: int) -> bool:
        """Determine if early stopping criteria are met"""
        return no_improvement_count >= self.config.early_stopping_patience

    def optimize(
        self,
        domain: Any,
        fitness_function,
        k_exits: int,
        perimeter_length: int,
        omega: int,
        num_episodes: int,
        max_evals: int,
    ) -> Tuple[List[int], float]:
        """
        Main optimization loop using Q-learning with advanced features.

        Args:
            domain: Domain object passed to fitness function
            fitness_function: Function that evaluates exit configurations
            k_exits: Number of emergency exits
            perimeter_length: Length of environment perimeter
            omega: Width of each emergency exit
            num_episodes: Maximum number of episodes
            max_evals: Maximum number of fitness evaluations

        Returns:
            Tuple of (best_exit_configuration, best_fitness_score)
        """
        self.eval_count = 0
        no_improvement_count = 0
        last_improvement_fitness = float("inf")

        print(
            f"Starting Q-Learning optimization with {k_exits} exits, perimeter {perimeter_length}, omega {omega}"
        )

        for episode in range(num_episodes):
            if self.eval_count >= max_evals:
                print(f"Reached maximum evaluations: {max_evals}")
                break

            # Initialize episode
            current_vector = self._generate_valid_k_vector(
                k_exits, perimeter_length, omega
            )
            current_fitness = fitness_function(current_vector)
            self.eval_count += 1

            # Track best solution
            if current_fitness < self.best_fitness:
                self.best_fitness = current_fitness
                self.best_solution = current_vector.copy()
                no_improvement_count = 0
                last_improvement_fitness = current_fitness
            else:
                no_improvement_count += 1

            self.fitness_history.append(current_fitness)

            # Calculate current exploration rate
            epsilon = self._update_epsilon(episode)
            episode_reward = 0

            # Episode steps
            max_steps = min(100, max_evals - self.eval_count)  # Adaptive step limit

            for step in range(max_steps):
                if self.eval_count >= max_evals:
                    break

                # Get current state
                state_tuple = self._get_canonical_state(current_vector)

                # Get valid actions
                valid_actions = self._get_valid_actions(
                    current_vector, perimeter_length, omega
                )
                if not valid_actions:
                    break  # No valid moves available

                # Select action
                action = self._select_action(state_tuple, valid_actions, epsilon)
                exit_idx, new_pos = action

                # Apply action
                next_vector = current_vector.copy()
                next_vector[exit_idx] = new_pos

                # Evaluate new state
                next_fitness = fitness_function(next_vector)
                self.eval_count += 1
                next_state_tuple = self._get_canonical_state(next_vector)

                # Calculate reward
                reward = self._calculate_reward(current_fitness, next_fitness)
                episode_reward += reward

                # Q-learning update
                old_q = self.q_table[state_tuple][action]

                # Find max Q-value for next state
                next_valid_actions = self._get_valid_actions(
                    next_vector, perimeter_length, omega
                )
                max_next_q = 0.0
                if next_valid_actions and next_state_tuple in self.q_table:
                    max_next_q = max(
                        self.q_table[next_state_tuple].get(
                            next_action, self._get_init_q_value()
                        )
                        for next_action in next_valid_actions
                    )

                # Bellman update
                new_q = old_q + self.config.learning_rate * (
                    reward + self.config.discount_factor * max_next_q - old_q
                )
                self.q_table[state_tuple][action] = new_q

                # Update current state
                current_vector = next_vector
                current_fitness = next_fitness

                # Track global best
                if current_fitness < self.best_fitness:
                    self.best_fitness = current_fitness
                    self.best_solution = current_vector.copy()
                    no_improvement_count = 0

            self.episode_rewards.append(episode_reward)

            # Progress reporting
            if episode % 100 == 0 or episode < 10:
                print(
                    f"Episode {episode}: Best fitness = {self.best_fitness:.6f}, "
                    f"Current epsilon = {epsilon:.4f}, Evaluations = {self.eval_count}"
                )

            # Early stopping check
            if self._should_early_stop(no_improvement_count):
                print(
                    f"Early stopping at episode {episode} (no improvement for {no_improvement_count} episodes)"
                )
                break

            # Convergence check
            if episode > 200 and self._check_convergence():
                print(f"Convergence detected at episode {episode}")
                break

        print(
            f"Optimization completed. Best fitness: {self.best_fitness:.6f} "
            f"after {self.eval_count} evaluations"
        )

        return self.best_solution, self.best_fitness


def q_learning_exit_optimizer(
    domain: Any, num_episodes: int = 15, **kwargs
) -> Tuple[List[int], float]:
    """
    Main function interface for Q-learning emergency exit optimization.

    Args:
        domain: Domain object containing environment parameters
        k_exits: Number of emergency exits to place
        perimeter_length: Total perimeter length (2 * (width + height))
        omega: Width of each emergency exit
        psi_evaluator: Fitness evaluator object with evaluate method
        num_episodes: Maximum number of training episodes
        max_evals: Maximum number of fitness function evaluations
        **kwargs: Additional configuration parameters

    Returns:
        Tuple of (best_exit_positions, best_fitness_score)
    """

    perimeter_length = 2 * (domain.width + domain.height)
    k_exits = SimulationConfig.num_emergency_exits
    psi_evaluator = FitnessEvaluator(domain, OptimizerStrategy.IEA)
    omega = SimulationConfig.omega

    max_evals: int = IEAConfig.islands[0].maxevals

    # Create configuration with any provided overrides
    config = QLearningConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Create and run optimizer
    optimizer = AdvancedQLearningExitOptimizer(config)

    try:
        best_solution, best_fitness = optimizer.optimize(
            domain=domain,
            fitness_function=psi_evaluator.evaluate,
            k_exits=k_exits,
            perimeter_length=perimeter_length,
            omega=omega,
            num_episodes=num_episodes,
            max_evals=max_evals,
        )

        return best_solution, best_fitness, optimizer.fitness_history

    except Exception as e:
        print(f"Error during optimization: {e}")
        # Return a valid fallback solution
        fallback_solution = optimizer._generate_valid_k_vector(
            k_exits, perimeter_length, omega
        )
        fallback_fitness = psi_evaluator.evaluate(fallback_solution)
        return fallback_solution, fallback_fitness
