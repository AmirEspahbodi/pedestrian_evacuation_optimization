import math
import random
import time
from typing import Any, Dict, List, Tuple

from .common import FitnessEvaluator

Individual = List[int]  # Changed to int for discrete environment
Population = List[Individual]

# --- Core Algorithm Components ---


def set_based_recombination(parent1: Individual, parent2: Individual) -> Individual:
    combined_pool = parent1 + parent2
    random.shuffle(combined_pool)
    offspring = random.sample(combined_pool, k=len(parent1))
    return offspring


def gaussian_mutation_discrete(
    individual: Individual, gamma: float, perimeter_length: int
) -> Individual:
    mutated = individual[:]
    idx = random.randint(0, len(mutated) - 1)
    gene = mutated[idx]
    perturbation = gamma * random.normalvariate(0, 1)
    mutated_val = int(round(gene * (1 + perturbation))) % perimeter_length
    mutated[idx] = mutated_val
    return mutated


def tournament_selection(population: Population, fitnesses: List[float]) -> Individual:
    idx1 = random.randint(0, len(population) - 1)
    idx2 = random.randint(0, len(population) - 1)
    return population[idx1] if fitnesses[idx1] < fitnesses[idx2] else population[idx2]


# --- Main iEA Optimizer with Timing ---


def iea_optimizer(
    pedestrian_confs,
    gird,
    simulator_config,
    iea_config,
) -> Tuple[Individual, float, Dict[str, Any], float]:
    """
    Runs the island-based EA, returns:
      - global_best_individual
      - global_best_fitness
      - history of fitnesses per island and generation
      - timestamp when global best was first found
    """
    start_time = time.perf_counter()

    # Algorithm parameters
    perimeter_length = 2 * (len(gird) + len(gird[0]))
    k_exits = simulator_config.numEmergencyExits
    p_mut = 1.0

    evalr = FitnessEvaluator(gird, pedestrian_confs, simulator_config)

    gamma, p_recomb, pop_size, num_islands, max_evals, migration_freq = (
        iea_config.mutation_gamma,
        iea_config.recombination_prob,
        iea_config.popsize,
        iea_config.numislands,
        iea_config.max_evals,
        iea_config.migration_frequency_generations,
    )

    # Prepare evaluator and storage
    generation = 0
    history: Dict[str, Dict[str, List[float]]] = {
        f"island{i + 1}": {} for i in range(num_islands)
    }

    # Initialization
    islands = []
    for i in range(num_islands):
        pop = [
            [random.randint(0, perimeter_length - 1) for _ in range(k_exits)]
            for _ in range(pop_size)
        ]
        fitnesses = [evalr.evaluate(ind) for ind in pop]
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

    # Main loop
    while evalr.get_evaluation_count() < max_evals:
        # Migration
        if generation > 0 and generation % migration_freq == 0:
            bests = []
            worst_idxs = []
            for isl in islands:
                best_i = min(
                    range(len(isl["fitnesses"])), key=isl["fitnesses"].__getitem__
                )
                worst_i = max(
                    range(len(isl["fitnesses"])), key=isl["fitnesses"].__getitem__
                )
                bests.append(isl["population"][best_i][:])
                worst_idxs.append(worst_i)
            for i, isl in enumerate(islands):
                left = (i - 1) % num_islands
                right = (i + 1) % num_islands
                migrant = random.choice([bests[left], bests[right]])
                isl["population"][worst_idxs[i]] = migrant[:]
                isl["fitnesses"][worst_idxs[i]] = float("inf")

        generation += 1
        for i, isl in enumerate(islands):
            # Elitism
            best_i = min(range(len(isl["fitnesses"])), key=isl["fitnesses"].__getitem__)
            elite = isl["population"][best_i][:]
            elite_f = isl["fitnesses"][best_i]

            new_pop = [elite]
            new_fit = [elite_f]

            # Check new elite against global best
            if elite_f < global_best_fitness:
                global_best_fitness = elite_f
                global_best_individual = elite[:]
                time_to_best = time.perf_counter() - start_time

            # Offspring
            for _ in range(pop_size - 1):
                p1 = tournament_selection(isl["population"], isl["fitnesses"])
                p2 = tournament_selection(isl["population"], isl["fitnesses"])
                child = (
                    set_based_recombination(p1, p2)
                    if random.random() < p_recomb
                    else p1[:]
                )
                if random.random() < p_mut:
                    child = gaussian_mutation_discrete(child, gamma, perimeter_length)

                new_pop.append(child)
                f = evalr.evaluate(child)
                new_fit.append(f)
                history[f"island{i + 1}"].setdefault(str(generation), []).append(f)

                # Check child against global best
                if f < global_best_fitness:
                    global_best_fitness = f
                    global_best_individual = child[:]
                    time_to_best = time.perf_counter() - start_time

                if evalr.get_evaluation_count() >= max_evals:
                    break
            isl["population"] = new_pop
            isl["fitnesses"] = new_fit
            if evalr.get_evaluation_count() >= max_evals:
                break
        if evalr.get_evaluation_count() >= max_evals:
            break

    # Return best solution and time found
    return global_best_individual, global_best_fitness, history, time_to_best
