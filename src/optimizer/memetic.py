import time
import numpy as np
import random
from .common import FitnessEvaluator
from src.config import SimulationConfig, OptimizerStrategy, IEAConfig
from src.simulator.domain import Domain


class MemeticAlgorithm:
    def __init__(
        self,
        domain: Domain,
        population_size=100,
        crossover_rate=0.8,
        mutation_rate=0.2,
        local_search_rate=0.1,
        local_search_iterations=10,
    ):
        self.domain = domain
        self.psi_evaluator = FitnessEvaluator(
            domain=domain, optimizer_strategy=OptimizerStrategy.IEA
        )
        self.k_exits = SimulationConfig.num_emergency_exits
        self.omega = SimulationConfig.omega
        self.perimeter_length = 2 * (domain.width + domain.height)
        self.max_val_for_element = self.perimeter_length - self.omega
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.local_search_rate = local_search_rate
        self.local_search_iterations = local_search_iterations
        self.population = []

    def _initialize_population(self):
        """Initializes the population with random valid individuals."""
        self.population = []
        for _ in range(self.population_size):
            individual = sorted(
                np.random.choice(
                    self.max_val_for_element + 1, self.k_exits, replace=False
                )
            )
            self.population.append(list(individual))

    def _evaluate_population(self):
        """Evaluates the fitness of the entire population."""
        fitness_scores = []
        for individual in self.population:
            fitness = self.psi_evaluator.evaluate(individual)
            fitness_scores.append(fitness)
        return fitness_scores

    def _selection(self, fitness_scores):
        """Selects parents for the next generation using tournament selection."""
        parents = []
        for _ in range(self.population_size):
            tournament = random.sample(list(enumerate(self.population)), k=3)
            winner = min(tournament, key=lambda x: fitness_scores[x[0]])
            parents.append(winner[1])
        return parents

    def _crossover(self, parent1, parent2):
        """Performs a two-point crossover to create two new offspring."""
        if random.random() < self.crossover_rate and self.k_exits > 1:
            point1, point2 = sorted(random.sample(range(self.k_exits), 2))
            offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
            offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
            return sorted(offspring1), sorted(offspring2)
        return parent1, parent2

    def _mutate(self, individual):
        """Applies a creep mutation to an individual."""
        for i in range(self.k_exits):
            if random.random() < self.mutation_rate:
                change = random.randint(
                    -int(self.max_val_for_element * 0.05),
                    int(self.max_val_for_element * 0.05),
                )
                individual[i] = max(
                    0, min(self.max_val_for_element, individual[i] + change)
                )
        return sorted(individual)

    def _local_search(self, individual):
        """
        Performs a simple hill-climbing local search to refine an individual.
        """
        best_individual = list(individual)
        best_fitness = self.psi_evaluator.evaluate(best_individual)

        for _ in range(self.local_search_iterations):
            if self.psi_evaluator.get_evaluation_count() >= self.max_evals:
                break

            neighbor = list(best_individual)
            gene_to_mutate = random.randint(0, self.k_exits - 1)
            change = random.randint(-5, 5)
            neighbor[gene_to_mutate] = max(
                0, min(self.max_val_for_element, neighbor[gene_to_mutate] + change)
            )
            neighbor = sorted(neighbor)

            neighbor_fitness = self.psi_evaluator.evaluate(neighbor)

            if neighbor_fitness < best_fitness:
                best_individual = neighbor
                best_fitness = neighbor_fitness

        return best_individual

    def run(self, num_episodes=20):
        """
        Runs the Memetic Algorithm for a given number of episodes or evaluations.

        Args:
            num_episodes (int): The maximum number of generations.

        Returns:
            A tuple containing the best individual, its fitness score, the history of fitness per episode,
            and the time to reach the best fitness.
        """
        # Setup
        self.max_evals = IEAConfig.islands[0].maxevals
        start_time = time.perf_counter()
        history = {f"episode-{i + 1}": [] for i in range(num_episodes)}

        # Initialize
        self._initialize_population()
        best_overall_individual = None
        best_overall_fitness = float("inf")
        time_to_best = None

        # Main loop
        for episode in range(num_episodes):
            print(f"episode {episode}")
            if self.psi_evaluator.get_evaluation_count() >= self.max_evals:
                break

            # Evaluate current population
            fitness_scores = self._evaluate_population()

            # Update global best and time-to-best
            current_best_idx = np.argmin(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]
            if current_best_fitness < best_overall_fitness:
                best_overall_fitness = current_best_fitness
                best_overall_individual = self.population[current_best_idx]
                time_to_best = time.perf_counter() - start_time

            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Best Fitness: {best_overall_fitness:.4f} | "
                f"Evaluations: {self.psi_evaluator.get_evaluation_count()}/{self.max_evals}"
            )

            # Selection
            parents = self._selection(fitness_scores)
            next_generation = []

            # Crossover & Mutation
            for i in range(0, self.population_size, 2):
                p1 = parents[i]
                p2 = parents[i + 1] if i + 1 < self.population_size else parents[0]
                off1, off2 = self._crossover(p1, p2)
                next_generation.append(self._mutate(off1))
                next_generation.append(self._mutate(off2))

            self.population = next_generation[: self.population_size]

            # Local search on top individuals
            sorted_pop = sorted(
                zip(self.population, fitness_scores), key=lambda x: x[1]
            )
            num_ls = int(self.population_size * self.local_search_rate)
            for i in range(num_ls):
                if self.psi_evaluator.get_evaluation_count() >= self.max_evals:
                    break
                indiv = sorted_pop[i][0]
                refined = self._local_search(indiv)
                for j, ind in enumerate(self.population):
                    if ind == indiv:
                        self.population[j] = refined
                        break

            # Record history after local search (final generation)
            final_fitness = self._evaluate_population()
            history[f"episode-{episode + 1}"] = final_fitness

        return best_overall_individual, best_overall_fitness, history, time_to_best
