import random
import math
import time
import numpy as np
from typing import List, Tuple, Dict
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
    combined_genes = set(parent1_genes) | set(parent2_genes)
    available = list(combined_genes)
    if len(available) < k_exits:
        return random.choices(available, k=k_exits)
    return random.sample(available, k=k_exits)


def _mutate_individual_ea(
    individual_genes: List[float],
    mutation_gamma: float,
    perimeter_length: float,
    k_exits: int,
):
    idx = random.randrange(k_exits)
    individual_genes[idx] = _gaussian_perturbation(
        individual_genes[idx], mutation_gamma, perimeter_length
    )


def _binary_tournament_selection(population: List[Individual]) -> Individual:
    i1, i2 = random.randrange(len(population)), random.randrange(len(population))
    c1, c2 = population[i1], population[i2]
    return c1 if c1.fitness < c2.fitness else c2


def evolutionary_algorithm(
    domain: Domain,
) -> Tuple[List[float], float, Dict[int, List[float]], float]:
    start_time = time.time()

    perimeter_length = 2 * (domain.width + domain.height)
    k_exits = SimulationConfig.num_emergency_exits
    psi = FitnessEvaluator(domain, OptimizerStrategy.EA)
    popsize = EAConfig.islands[0].popsize
    pr = EAConfig.islands[0].recombination_prob
    gamma = EAConfig.islands[0].mutation_gamma
    maxevals = EAConfig.islands[0].maxevals
    history: Dict[int, List[float]] = {}

    generation = 0
    print(f"generation = {generation}")

    # Initialize population
    population: List[Individual] = []
    for _ in range(popsize):
        genes = [random.randint(0, perimeter_length) for _ in range(k_exits)]
        population.append(Individual(genes))

    # Evaluate initial population
    history[generation] = []
    for ind in population:
        if psi.get_evaluation_count() >= maxevals:
            break
        ind.fitness = psi.evaluate(ind.genes)
        history[generation].append(ind.fitness)

    population.sort()
    best_overall = population[0]
    time_to_best = time.time() - start_time

    # Evolutionary loop
    while psi.get_evaluation_count() < maxevals:
        generation += 1
        print(f"generation = {generation}")
        history[generation] = []
        next_pop: List[Individual] = []

        for _ in range(math.ceil(popsize / 2)):
            if psi.get_evaluation_count() >= maxevals:
                break

            # Selection
            p1 = _binary_tournament_selection(population)
            p2 = _binary_tournament_selection(population)

            # Recombination
            if random.random() < pr:
                o1_genes = _set_based_recombination(p1.genes, p2.genes, k_exits)
                o2_genes = _set_based_recombination(p2.genes, p1.genes, k_exits)
            else:
                o1_genes, o2_genes = p1.genes[:], p2.genes[:]

            # Mutation
            _mutate_individual_ea(o1_genes, gamma, perimeter_length, k_exits)
            _mutate_individual_ea(o2_genes, gamma, perimeter_length, k_exits)

            # Create & evaluate offspring
            off1 = Individual(o1_genes)
            if psi.get_evaluation_count() < maxevals:
                off1.fitness = psi.evaluate(off1.genes)
                history[generation].append(off1.fitness)
            else:
                off1.fitness = float('inf')
            next_pop.append(off1)

            if len(next_pop) < popsize:
                off2 = Individual(o2_genes)
                if psi.get_evaluation_count() < maxevals:
                    off2.fitness = psi.evaluate(off2.genes)
                    history[generation].append(off2.fitness)
                else:
                    off2.fitness = float('inf')
                next_pop.append(off2)

        if not next_pop:
            break

        # Survivor selection
        population.extend(next_pop)
        population.sort()
        population = population[:popsize]

        # Check for new best
        if population[0].fitness < best_overall.fitness:
            best_overall = population[0]
            time_to_best = time.time() - start_time

    return best_overall.genes, best_overall.fitness, history, time_to_best
