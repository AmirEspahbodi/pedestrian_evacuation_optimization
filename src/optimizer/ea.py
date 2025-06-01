import random
import numpy as np
from typing import List, Tuple, Set
from src.simulator.domain import Domain
from src.config import SimulationConfig, OptimizerStrategy, EAConfig
from .common import FitnessEvaluator, Individual


def _gaussian_perturbation(
    value: float, gamma: float, perimeter_length: float
) -> float:
    perturbed_value_float = value * (1 + gamma * np.random.normal(0, 1))
    perturbed_value_discrete = np.round(perturbed_value_float).astype(int)
    if int(perimeter_length) == 0:
        return 0
    return int(perturbed_value_discrete % int(perimeter_length))


def _set_based_recombination(
    parent1_genes: List[float], parent2_genes: List[float], k_exits: int
) -> List[float]:
    """
    Implements set-based recombination (Algorithm 2)
    """
    combined_genes: Set[float] = set(parent1_genes) | set(parent2_genes)

    available_genes: List[float] = list(combined_genes)

    offspring_genes: List[float] = []

    if len(available_genes) < k_exits:
        offspring_genes = random.choices(available_genes, k=k_exits)
    else:
        offspring_genes = random.sample(available_genes, k_exits)

    return offspring_genes


def _mutate_individual_ea(
    individual_genes: List[float],
    mutation_gamma: float,
    perimeter_length: float,
    k_exits: int,
):
    exit_to_mutate_idx = random.randrange(k_exits)
    individual_genes[exit_to_mutate_idx] = _gaussian_perturbation(
        individual_genes[exit_to_mutate_idx], mutation_gamma, perimeter_length
    )


def _binary_tournament_selection(population: List[Individual]) -> Individual:
    idx1 = random.randrange(len(population))
    idx2 = random.randrange(len(population))

    competitor1 = population[idx1]
    competitor2 = population[idx2]

    return competitor1 if competitor1.fitness < competitor2.fitness else competitor2


def evolutionary_algorithm(
    domain: Domain,
) -> Tuple[List[float], float, int]:
    perimeter_length = 2 * (domain.width + domain.height)
    k_exits = SimulationConfig.num_emergency_exits
    psi_evaluator = FitnessEvaluator(domain, OptimizerStrategy.EA)
    population_size: int = EAConfig.islands[0].popsize
    recombination_prob: float = EAConfig.islands[0].recombination_prob
    mutation_gamma: float = EAConfig.islands[0].mutation_gamma
    max_evals: int = EAConfig.islands[0].maxevals
    history: dict[int,list[float]] = {}
    
    generation = 0
    print(f"generation = {generation}")
    population: List[Individual] = []
    for _ in range(population_size):
        genes = [random.randint(0, perimeter_length) for _ in range(k_exits)]
        individual = Individual(genes)
        population.append(individual)
    
    history[generation] = []
    for ind in population:
        if psi_evaluator.get_evaluation_count() >= max_evals:
            break
        ind.fitness = psi_evaluator.evaluate(ind.genes)
        history[generation].append(ind.fitness)
    

    population.sort()
    best_overall_individual = population[0]

    while psi_evaluator.get_evaluation_count() < max_evals:
        generation += 1
        print(f"generation = {generation}")
        history[generation] = []

        next_population: List[Individual] = []

        for _ in range(population_size // 2):
            if psi_evaluator.get_evaluation_count() >= max_evals:
                break

            parent1 = _binary_tournament_selection(population)
            parent2 = _binary_tournament_selection(population)

            offspring1_genes: List[float]
            offspring2_genes: List[float]

            if random.random() < recombination_prob:
                offspring1_genes = _set_based_recombination(
                    parent1.genes, parent2.genes, k_exits
                )
                offspring2_genes = _set_based_recombination(
                    parent2.genes, parent1.genes, k_exits
                )
            else:
                offspring1_genes = parent1.genes[:]
                offspring2_genes = parent2.genes[:]

            _mutate_individual_ea(
                offspring1_genes, mutation_gamma, perimeter_length, k_exits
            )
            _mutate_individual_ea(
                offspring2_genes, mutation_gamma, perimeter_length, k_exits
            )

            offspring1 = Individual(offspring1_genes)
            if psi_evaluator.get_evaluation_count() < max_evals:
                offspring1.fitness = psi_evaluator.evaluate(offspring1.genes)
                history[generation].append(offspring1.fitness)
            else:
                offspring1.fitness = float("inf")
            next_population.append(offspring1)

            if len(next_population) < population_size:
                offspring2 = Individual(offspring2_genes)
                if psi_evaluator.get_evaluation_count() < max_evals:
                    offspring2.fitness = psi_evaluator.evaluate(offspring2.genes)
                    history[generation].append(offspring2.fitness)
                else:
                    offspring2.fitness = float("inf")
                next_population.append(offspring2)

        if not next_population:
            break

        population.extend(next_population)
        population.sort()
        population = population[:population_size]

        if population[0].fitness < best_overall_individual.fitness:
            best_overall_individual = population[0]

    return (
        best_overall_individual.genes,
        best_overall_individual.fitness,
        psi_evaluator.get_evaluation_count(),
        history
    )
