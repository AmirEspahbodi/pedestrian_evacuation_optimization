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
) -> int:
    perturbed = value * (1 + gamma * np.random.normal(0, 1))
    discrete = int(np.round(perturbed))
    if int(perimeter_length) == 0:
        return 0
    return discrete % int(perimeter_length)

def _set_based_recombination(
    p1: List[float], p2: List[float], k_exits: int
) -> List[float]:
    combined = set(p1) | set(p2)
    genes = list(combined)
    if len(genes) < k_exits:
        return random.choices(genes, k=k_exits)
    return random.sample(genes, k=k_exits)

def _mutate_individual_ea(
    genes: List[float],
    gamma: float,
    perimeter_length: float,
    k_exits: int,
) -> None:
    idx = random.randrange(k_exits)
    genes[idx] = _gaussian_perturbation(genes[idx], gamma, perimeter_length)

def _binary_tournament_selection(pop: List[Individual]) -> Individual:
    i, j = random.randrange(len(pop)), random.randrange(len(pop))
    return pop[i] if pop[i].fitness < pop[j].fitness else pop[j]

def evolutionary_algorithm(
    domain: Domain,
) -> Tuple[List[float], float, float, Dict[int, List[float]]]:
    """
    Returns:
      best_genes:     List of exit positions for best solution
      best_fitness:   Its fitness value
      time_to_best:   Seconds elapsed when that best was first discovered
      history:        Per-generation list of all fitnesses evaluated
    """
    # --- Setup ---
    perimeter = 2 * (domain.width + domain.height)
    k_exits = SimulationConfig.num_emergency_exits
    psi = FitnessEvaluator(domain, OptimizerStrategy.EA)

    cfg = EAConfig.islands[0]
    popsize, pr, gamma, maxevals = (
        cfg.popsize, cfg.recombination_prob, cfg.mutation_gamma, cfg.maxevals
    )

    history: Dict[int, List[float]] = {}

    # --- Start timer ---
    start_time = time.perf_counter()
    time_to_best = 0.0  # will store elapsed seconds when best first set

    # --- Initial population ---
    generation = 0
    population: List[Individual] = []
    for _ in range(popsize):
        genes = [random.randint(0, perimeter) for _ in range(k_exits)]
        population.append(Individual(genes))

    # Evaluate gen 0
    history[0] = []
    for ind in population:
        if psi.get_evaluation_count() >= maxevals:
            break
        ind.fitness = psi.evaluate(ind.genes)
        history[0].append(ind.fitness)

    # Sort & record initial best
    population.sort()
    best_overall = population[0]
    # record the time when this first best was obtained
    time_to_best = time.perf_counter() - start_time

    # --- Generations ---
    while psi.get_evaluation_count() < maxevals:
        generation += 1
        history[generation] = []
        next_pop: List[Individual] = []

        # Generate offspring in pairs
        for _ in range(math.ceil(popsize / 2)):
            if psi.get_evaluation_count() >= maxevals:
                break

            # Selection
            p1 = _binary_tournament_selection(population)
            p2 = _binary_tournament_selection(population)

            # Recombination or cloning
            if random.random() < pr:
                o1_genes = _set_based_recombination(p1.genes, p2.genes, k_exits)
                o2_genes = _set_based_recombination(p2.genes, p1.genes, k_exits)
            else:
                o1_genes, o2_genes = p1.genes[:], p2.genes[:]

            # Mutation
            _mutate_individual_ea(o1_genes, gamma, perimeter, k_exits)
            _mutate_individual_ea(o2_genes, gamma, perimeter, k_exits)

            # Evaluate offspring1
            o1 = Individual(o1_genes)
            if psi.get_evaluation_count() < maxevals:
                o1.fitness = psi.evaluate(o1.genes)
                history[generation].append(o1.fitness)
            else:
                o1.fitness = float('inf')
            next_pop.append(o1)

            # Evaluate offspring2 (if room)
            if len(next_pop) < popsize:
                o2 = Individual(o2_genes)
                if psi.get_evaluation_count() < maxevals:
                    o2.fitness = psi.evaluate(o2.genes)
                    history[generation].append(o2.fitness)
                else:
                    o2.fitness = float('inf')
                next_pop.append(o2)

        if not next_pop:
            break

        # Merge, sort, truncate
        population.extend(next_pop)
        population.sort()
        population = population[:popsize]

        # Check for new global best
        if population[0].fitness < best_overall.fitness:
            best_overall = population[0]
            time_to_best = time.perf_counter() - start_time

    return best_overall.genes, best_overall.fitness, time_to_best, history
