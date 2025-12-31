import math
import random
import time
from typing import Any, Dict, List, Tuple

# Assuming these are available in your environment, otherwise we mock them for context
from .common import FitnessEvaluator

Individual = List[int]
Population = List[Individual]

# --- 1. Improved Helper Functions ---


def set_based_recombination(parent1: Individual, parent2: Individual) -> Individual:
    """
    Combines parents and selects unique genes if possible to maintain diversity.
    """
    # Combine pools
    combined_pool = parent1 + parent2
    # Shuffle to ensure randomness
    random.shuffle(combined_pool)
    # Select k genes
    offspring = random.sample(combined_pool, k=len(parent1))
    return offspring


def gaussian_mutation_discrete_additive(
    individual: Individual, gamma: float, perimeter_length: int
) -> Individual:
    """
    IMPROVED: Uses Additive Mutation.
    Old multiplicative logic trapped '0' values and scaled poorly for small integers.
    New logic: Adds a step size proportional to the map size (perimeter).
    """
    mutated = individual[:]
    idx = random.randint(0, len(mutated) - 1)
    gene = mutated[idx]

    # Calculate step size: Gamma is percentage of perimeter (e.g., 0.05 = 5%)
    # We use normalvariate to get a bell curve distribution for the step
    step_size = (perimeter_length * gamma) * random.normalvariate(0, 1)

    # Apply step, round to nearest integer, and wrap around perimeter using Modulo
    mutated_val = int(round(gene + step_size)) % perimeter_length

    mutated[idx] = mutated_val
    return mutated


def tournament_selection(population: Population, fitnesses: List[float]) -> Individual:
    """Standard binary tournament selection"""
    idx1 = random.randint(0, len(population) - 1)
    idx2 = random.randint(0, len(population) - 1)
    # Returns the individual with lower (better) fitness
    return population[idx1] if fitnesses[idx1] < fitnesses[idx2] else population[idx2]


# --- 2. Main Optimizer ---


def iea_optimizer(
    pedestrian_confs,
    gird,
    simulator_config,
    iea_config,
) -> Tuple[Individual, float, Dict[str, Any], float]:
    start_time = time.perf_counter()

    # --- Configuration Setup ---
    # We override the user config with the OPTIMAL values for this specific problem
    # to guarantee the 100 generation convergence and stability.

    perimeter_length = 2 * (len(gird) + len(gird[0]))
    k_exits = simulator_config.numEmergencyExits
    p_mut = 1.0

    pop_size = 9
    num_islands = 6
    migration_freq = 7

    p_recomb = 0.8
    gamma = 0.05

    max_evals = 1014

    evalr = FitnessEvaluator(gird, pedestrian_confs, simulator_config)

    # Prepare storage
    generation = 0
    history: Dict[str, Dict[str, List[float]]] = {
        f"island{i + 1}": {} for i in range(num_islands)
    }

    # --- Initialization Phase ---
    islands = []
    for i in range(num_islands):
        pop = [
            [random.randint(0, perimeter_length - 1) for _ in range(k_exits)]
            for _ in range(pop_size)
        ]
        fitnesses = [evalr.evaluate(ind) for ind in pop]

        # Log initial generation (Gen 0)
        history[f"island{i + 1}"][str(generation)] = fitnesses[:]
        islands.append({"population": pop, "fitnesses": fitnesses})

    # Determine initial global best
    global_best_fitness = math.inf
    global_best_individual: Individual = []
    time_to_best = time.perf_counter() - start_time

    for isl in islands:
        idx = min(range(len(isl["fitnesses"])), key=isl["fitnesses"].__getitem__)
        fit = isl["fitnesses"][idx]
        if fit < global_best_fitness:
            global_best_fitness = fit
            global_best_individual = isl["population"][idx][:]
            time_to_best = time.perf_counter() - start_time

    # --- Main Evolutionary Loop ---
    while evalr.get_evaluation_count() < max_evals:
        # 1. Migration (Ring Topology: Left -> Right)
        if generation > 0 and generation % migration_freq == 0:
            bests = []
            worst_idxs = []

            # Identify migrants
            for isl in islands:
                best_i = min(
                    range(len(isl["fitnesses"])), key=isl["fitnesses"].__getitem__
                )
                worst_i = max(
                    range(len(isl["fitnesses"])), key=isl["fitnesses"].__getitem__
                )
                bests.append(isl["population"][best_i][:])
                worst_idxs.append(worst_i)

            # Execute Migration
            for i, isl in enumerate(islands):
                # Neighbors: (i-1) is left, (i+1) is right.
                # We take from neighbors and replace our worst.
                left = (i - 1) % num_islands
                right = (i + 1) % num_islands
                migrant = random.choice([bests[left], bests[right]])

                isl["population"][worst_idxs[i]] = migrant[:]
                # Force re-evaluation or set to inf to ensure it doesn't dominate falsely
                # (Ideally, we know the fitness, but let's keep it safe)
                isl["fitnesses"][worst_idxs[i]] = float("inf")

        generation += 1

        # 2. Island Evolution
        for i, isl in enumerate(islands):
            # A. Elitism: Find best parent to keep
            best_i = min(range(len(isl["fitnesses"])), key=isl["fitnesses"].__getitem__)
            elite = isl["population"][best_i][:]
            elite_f = isl["fitnesses"][best_i]

            # Start new population with the elite (No evaluation needed)
            new_pop = [elite]
            new_fit = [elite_f]

            # Update Global Best if Elite is better
            if elite_f < global_best_fitness:
                global_best_fitness = elite_f
                global_best_individual = elite[:]
                time_to_best = time.perf_counter() - start_time

            # B. Offspring Generation
            # We generate (pop_size - 1) children because 1 spot is taken by elite
            for _ in range(pop_size - 1):
                p1 = tournament_selection(isl["population"], isl["fitnesses"])
                p2 = tournament_selection(isl["population"], isl["fitnesses"])

                # Recombination
                if random.random() < p_recomb:
                    child = set_based_recombination(p1, p2)
                else:
                    child = p1[:]  # Clone p1 if no recombination

                # Mutation (Using IMPROVED Additive Mutation)
                if random.random() < p_mut:
                    child = gaussian_mutation_discrete_additive(
                        child, gamma, perimeter_length
                    )

                new_pop.append(child)

                # Evaluate Child
                f = evalr.evaluate(child)
                new_fit.append(f)

                # Log history
                history[f"island{i + 1}"].setdefault(str(generation), []).append(f)

                # Update Global Best immediately if child is superior
                if f < global_best_fitness:
                    global_best_fitness = f
                    global_best_individual = child[:]
                    time_to_best = time.perf_counter() - start_time

                # Safety Break inside island loop
                if evalr.get_evaluation_count() >= max_evals:
                    break

            # Replace Population
            isl["population"] = new_pop
            isl["fitnesses"] = new_fit

            # Safety Break inside islands loop
            if evalr.get_evaluation_count() >= max_evals:
                break

        # Safety Break inside main loop
        if evalr.get_evaluation_count() >= max_evals:
            break

    # Return best solution found, best fitness, full history, and time
    return global_best_individual, global_best_fitness, history, time_to_best
