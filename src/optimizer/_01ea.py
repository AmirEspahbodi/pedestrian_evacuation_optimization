import math
import random
import time
from typing import Dict, List, Tuple

import numpy as np

# Assuming these exist in your project structure
from .common import FitnessEvaluator, Individual


def _gaussian_perturbation(value: int, gamma: float, perimeter_length: float) -> int:
    """
    OPTIMIZED: Uses additive mutation scaled by the perimeter length.

    Args:
        value: Current location of the exit (integer index on perimeter).
        gamma: Mutation strength (0.03 = standard deviation is 3% of perimeter).
        perimeter_length: Total length of the building perimeter.

    Returns:
        New discrete integer location.
    """
    if perimeter_length == 0:
        return 0

    # Calculate step size based on perimeter size (Additive noise)
    # This ensures exits at index 10 and index 1000 have equal mobility.
    sigma = perimeter_length * gamma
    perturbation = np.random.normal(0, sigma)

    new_value = value + perturbation

    # Round to nearest integer and wrap around (Periodic Boundary)
    discrete = int(np.round(new_value))
    return discrete % int(perimeter_length)


def _set_based_recombination(
    p1: List[int], p2: List[int], k_exits: int, perimeter: float
) -> List[int]:
    """
    OPTIMIZED: Set-based recombination that guarantees unique exits.
    If parents don't provide enough unique genes, we inject fresh random exits
    instead of duplicating existing ones.
    """
    combined = set(p1) | set(p2)
    genes = list(combined)

    # If we have enough unique genes, pick k_exits randomly
    if len(genes) >= k_exits:
        return random.sample(genes, k=k_exits)

    # If parents are too similar (low diversity), fill the rest with random spots
    # This prevents the "duplicate exit" bug and injects necessary diversity.
    needed = k_exits - len(genes)
    new_random_genes = [random.randint(0, int(perimeter) - 1) for _ in range(needed)]
    return genes + new_random_genes


def _mutate_individual_ea(
    genes: List[int],
    gamma: float,
    perimeter_length: float,
    k_exits: int,
) -> None:
    # Mutate exactly one exit per individual to fine-tune the solution
    idx = random.randrange(k_exits)
    genes[idx] = _gaussian_perturbation(genes[idx], gamma, perimeter_length)


def _binary_tournament_selection(pop: List[Individual]) -> Individual:
    i, j = random.randrange(len(pop)), random.randrange(len(pop))
    # Minimization problem: return the one with lower fitness
    return pop[i] if pop[i].fitness < pop[j].fitness else pop[j]


def ea_algorithm(
    pedestrian_confs, gird, simulator_config, ea_config
) -> Tuple[List[int], float, float, Dict[str, List[float]]]:
    # 1. Setup Environment
    # Calculate perimeter correctly based on grid dimensions
    perimeter = 2 * (len(gird) + len(gird[0]))
    k_exits = simulator_config.numEmergencyExits
    psi = FitnessEvaluator(gird, pedestrian_confs, simulator_config)

    # 2. Load Optimized Hyperparameters
    popsize = 50
    pr = 0.9
    gamma = 0.05
    # popsize = ea_config.popsize
    # pr = ea_config.recombination_prob
    # gamma = ea_config.mutation_gamma
    maxevals = 1050

    history: Dict[str, List[float]] = {}
    start_time = time.perf_counter()
    time_to_best = 0.0

    # 3. Initialization
    generation = 0
    population: List[Individual] = []

    # Create initial random population
    for _ in range(popsize):
        # Ensure integers 0 to perimeter
        genes = [random.randint(0, int(perimeter) - 1) for _ in range(k_exits)]
        ind = Individual(genes)
        population.append(ind)

    # Evaluate Generation 0
    history["0"] = []
    for ind in population:
        if psi.get_evaluation_count() >= maxevals:
            ind.fitness = float("inf")  # Should not happen typically on init
        else:
            ind.fitness = psi.evaluate(ind.genes)
            history["0"].append(ind.fitness)

    # Sort: best (lowest fitness) at index 0
    population.sort()
    best_overall = population[0]
    time_to_best = time.perf_counter() - start_time

    # 4. Main Evolutionary Loop
    while psi.get_evaluation_count() < maxevals:
        generation += 1
        history[f"{generation}"] = []
        offspring_pop: List[Individual] = []

        # Generate Offspring (size = popsize) to compete with parents
        while len(offspring_pop) < popsize:
            if psi.get_evaluation_count() >= maxevals:
                break

            # Selection
            p1 = _binary_tournament_selection(population)
            p2 = _binary_tournament_selection(population)

            # Recombination
            if random.random() < pr:
                # Use optimized recomb with perimeter for random injection
                o1_genes = _set_based_recombination(
                    p1.genes, p2.genes, k_exits, perimeter
                )
                o2_genes = _set_based_recombination(
                    p2.genes, p1.genes, k_exits, perimeter
                )
            else:
                o1_genes, o2_genes = p1.genes[:], p2.genes[:]

            # Mutation
            _mutate_individual_ea(o1_genes, gamma, perimeter, k_exits)
            _mutate_individual_ea(o2_genes, gamma, perimeter, k_exits)

            # Create Individuals
            o1 = Individual(o1_genes)
            o2 = Individual(o2_genes)

            # Evaluate Offspring 1
            if psi.get_evaluation_count() < maxevals:
                o1.fitness = psi.evaluate(o1.genes)
                offspring_pop.append(o1)
                history[f"{generation}"].append(o1.fitness)

            # Evaluate Offspring 2
            if len(offspring_pop) < popsize and psi.get_evaluation_count() < maxevals:
                o2.fitness = psi.evaluate(o2.genes)
                offspring_pop.append(o2)
                history[f"{generation}"].append(o2.fitness)

        if not offspring_pop:
            break

        # 5. Survival Selection (Mu + Lambda)
        # We combine parents + offspring and keep the top 'popsize' best individuals.
        # This guarantees monotonicity (fitness never gets worse).
        combined_pool = population + offspring_pop
        combined_pool.sort()  # Assumes Individual.__lt__ compares fitness
        population = combined_pool[:popsize]

        # Check for new global best
        if population[0].fitness < best_overall.fitness:
            best_overall = population[0]
            time_to_best = time.perf_counter() - start_time

    return best_overall.genes, best_overall.fitness, time_to_best, history
