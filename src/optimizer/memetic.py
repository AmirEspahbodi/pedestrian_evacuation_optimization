import numpy as np
import random
from .common import FitnessEvaluator
from src.config import SimulationConfig, OptimizerStrategy, IEAConfig
from src.simulator.domain import Domain


class MemeticAlgorithm:
    def __init__(self, domain: Domain,
                 population_size=100, crossover_rate=0.8, mutation_rate=0.2,
                 local_search_rate=0.1, local_search_iterations=10):
        self.domain = domain
        self.psi_evaluator = FitnessEvaluator(domain=domain, optimizer_strategy=OptimizerStrategy.IEA)
        self.k_exits = SimulationConfig.num_emergency_exits
        self.omega = SimulationConfig.omega
        self.perimeter_length = 2*(domain.width+domain.height)
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
            individual = sorted(np.random.choice(self.max_val_for_element + 1,
                                                 self.k_exits, replace=False))
            self.population.append(list(individual))

    def _evaluate_population(self, history:dict[str, list[int]], episode):
        """Evaluates the fitness of the entire population."""
        fitness_scores = []
        for individual in self.population:
            fitness = self.psi_evaluator.evaluate(individual)
            if f'episode{episode+1}' in history.keys():
                history[f'episode{episode+1}'].append(fitness)
            else:
                history[f'episode{episode+1}'] = [fitness]
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
        if random.random() < self.crossover_rate:
            if self.k_exits > 1:
                point1, point2 = sorted(random.sample(range(self.k_exits), 2))
                offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
                offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
                return sorted(offspring1), sorted(offspring2)
        return parent1, parent2

    def _mutate(self, individual):
        """Applies a creep mutation to an individual."""
        for i in range(self.k_exits):
            if random.random() < self.mutation_rate:
                # Creep mutation: small perturbation to the gene value
                change = random.randint(-int(self.max_val_for_element * 0.05),
                                        int(self.max_val_for_element * 0.05))
                individual[i] = max(0, min(self.max_val_for_element, individual[i] + change))
        return sorted(individual)

    def _local_search(self, individual, history:dict[str, list[int]], episode):
        """
        Performs a simple hill-climbing local search to refine an individual.
        """
        best_individual = list(individual)
        best_fitness = self.psi_evaluator.evaluate(best_individual)

        for _ in range(self.local_search_iterations):
            if self.psi_evaluator.get_evaluation_count() >= self.max_evals:
                break
            
            neighbor = list(best_individual)
            # Mutate one gene to create a neighbor
            gene_to_mutate = random.randint(0, self.k_exits - 1)
            change = random.randint(-5, 5) # Small neighborhood search
            neighbor[gene_to_mutate] = max(0, min(self.max_val_for_element,
                                               neighbor[gene_to_mutate] + change))
            neighbor = sorted(neighbor)
            
            neighbor_fitness = self.psi_evaluator.evaluate(neighbor)
            
            if f'episode{episode+1}' in history.keys():
                history[f'episode{episode+1}'].append(neighbor_fitness)
            else:
                history[f'episode{episode+1}'] = [neighbor_fitness]

            if neighbor_fitness < best_fitness:
                best_individual = neighbor
                best_fitness = neighbor_fitness
                
        return best_individual

    def run(self, num_episodes=100):
        """
        Runs the Memetic Algorithm for a given number of episodes or evaluations.

        Args:
            num_episodes (int): The maximum number of generations.
            max_evals (int): The maximum number of fitness evaluations.

        Returns:
            A tuple containing the best individual and its fitness score.
        """
        self.max_evals = IEAConfig.islands[0].maxevals
        self._initialize_population()
        best_overall_individual = None
        best_overall_fitness = float('inf')
        history: dict[str, list[int]] = {}

        for episode in range(num_episodes):
            print(f"episode {episode}")
            if self.psi_evaluator.get_evaluation_count() >= self.max_evals:
                break

            fitness_scores = self._evaluate_population(history, episode)
            
            current_best_idx = np.argmin(fitness_scores)
            if fitness_scores[current_best_idx] < best_overall_fitness:
                best_overall_fitness = fitness_scores[current_best_idx]
                best_overall_individual = self.population[current_best_idx]
            
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Best Fitness: {best_overall_fitness:.4f} | "
                  f"Evaluations: {self.psi_evaluator.get_evaluation_count()}/{self.max_evals}")

            parents = self._selection(fitness_scores)
            next_generation = []

            for i in range(0, self.population_size, 2):
                parent1 = parents[i]
                parent2 = parents[i + 1] if i + 1 < self.population_size else parents[0]
                offspring1, offspring2 = self._crossover(parent1, parent2)
                next_generation.append(self._mutate(offspring1))
                next_generation.append(self._mutate(offspring2))

            self.population = next_generation[:self.population_size]
            
            # Apply local search to a portion of the best individuals
            sorted_population = sorted(zip(self.population, fitness_scores), key=lambda x: x[1])
            num_local_search = int(self.population_size * self.local_search_rate)
            
            for i in range(num_local_search):
                if self.psi_evaluator.get_evaluation_count() >= self.max_evals:
                    break
                individual_to_refine = sorted_population[i][0]
                refined_individual = self._local_search(individual_to_refine, history, episode)
                
                # Replace the original with the refined individual
                for j, ind in enumerate(self.population):
                    if ind == individual_to_refine:
                        self.population[j] = refined_individual
                        break

        return best_overall_individual, best_overall_fitness
