import math
import random
import time
from typing import Dict, List, Tuple

import numpy as np

from src.config import EAConfig, SimulationConfig

from .common import FitnessEvaluator, Individual


def _gaussian_perturbation(value: float, gamma: float, perimeter_length: float) -> int:
    perturbed = value * (1 + gamma * np.random.normal(0, 1))
    discrete = int(np.round(perturbed))
    if int(perimeter_length) == 0:
        return 0
    return discrete % int(perimeter_length)


def _set_based_recombination(p1: List[int], p2: List[int], k_exits: int) -> List[int]:
    combined = set(p1) | set(p2)
    genes = list(combined)
    if len(genes) < k_exits:
        return random.choices(genes, k=k_exits)
    return random.sample(genes, k=k_exits)


def _mutate_individual_ea(
    genes: List[int],
    gamma: float,
    perimeter_length: float,
    k_exits: int,
) -> None:
    idx = random.randrange(k_exits)
    genes[idx] = _gaussian_perturbation(genes[idx], gamma, perimeter_length)


def _binary_tournament_selection(pop: List[Individual]) -> Individual:
    i, j = random.randrange(len(pop)), random.randrange(len(pop))
    return pop[i] if pop[i].fitness < pop[j].fitness else pop[j]


def ea_algorithm(
    pedestrian_confs, gird, simulator_config, ea_config
) -> tuple[List[int], float, float, Dict[str, List[float]]]:
    perimeter = 2 * (len(gird) + len(gird[0]))
    k_exits = simulator_config.numEmergencyExits
    psi = FitnessEvaluator(gird, pedestrian_confs, simulator_config)

    popsize, pr, gamma, _, maxevals = (
        ea_config.popsize,
        ea_config.recombination_prob,
        ea_config.mutation_gamma,
        ea_config.offspring,
        ea_config.max_evals,
    )

    history: Dict[str, List[float]] = {}

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
    history["0"] = []
    for ind in population:
        if psi.get_evaluation_count() >= maxevals:
            break
        ind.fitness = psi.evaluate(ind.genes)
        history["0"].append(ind.fitness)

    # Sort & record initial best
    population.sort()
    best_overall = population[0]
    # record the time when this first best was obtained
    time_to_best = time.perf_counter() - start_time

    # --- Generations ---
    while psi.get_evaluation_count() < maxevals:
        generation += 1
        history[f"{generation}"] = []
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
                history[f"{generation}"].append(o1.fitness)
            else:
                o1.fitness = float("inf")
            next_pop.append(o1)

            # Evaluate offspring2 (if room)
            if len(next_pop) < popsize:
                o2 = Individual(o2_genes)
                if psi.get_evaluation_count() < maxevals:
                    o2.fitness = psi.evaluate(o2.genes)
                    history[f"{generation}"].append(o2.fitness)
                else:
                    o2.fitness = float("inf")
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
