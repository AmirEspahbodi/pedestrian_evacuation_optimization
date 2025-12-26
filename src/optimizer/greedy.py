import math
import random
import time
from typing import List, Tuple

from src.config import SimulationConfig

from .common import FitnessEvaluator


def greedy_algorithm(
    pedestrian_confs,
    gird,
    simulator_config: SimulationConfig,
) -> Tuple[List[float], float, float]:
    psi_evaluator = FitnessEvaluator(gird, pedestrian_confs, simulator_config)
    omega_exit_width = simulator_config.omega
    perimeter_length = 2 * (len(gird) + len(gird[0]))
    k_exits = simulator_config.numEmergencyExits

    # Precompute number of scan points
    eta = math.ceil(perimeter_length / omega_exit_width) if omega_exit_width > 0 else 0

    # Initialize solutions and fitness
    E_solutions: List[float] = []
    current_fitness = float("inf")

    # Timer
    start_time = time.perf_counter()
    best_overall_fitness = float("inf")
    time_of_best = 0.0

    # Edge case: no exits to place
    if k_exits <= 0 or eta == 0:
        return E_solutions, current_fitness, 0.0

    print(f"k_exits = {k_exits}")
    print(f"eta = {eta}")

    # Outer loop: place each exit
    for i in range(k_exits):
        print(f"looking for {i + 1}th emergency exit place")
        best_psi_for_this_exit = float("inf")
        chosen_exit_for_this_iteration = -1.0

        # Random start point on the perimeter
        p_start_scan = random.randint(0, perimeter_length)
        candidate_location = p_start_scan

        # Inner loop: scan eta candidate positions
        for j in range(eta):
            temp_accesses = E_solutions + [candidate_location]
            current_eval_psi = psi_evaluator.evaluate(temp_accesses)

            # Track best for this exit
            if current_eval_psi < best_psi_for_this_exit:
                best_psi_for_this_exit = current_eval_psi
                chosen_exit_for_this_iteration = candidate_location

                # If it's also the best overall, record the time
                if current_eval_psi < best_overall_fitness:
                    best_overall_fitness = current_eval_psi
                    time_of_best = time.perf_counter() - start_time

            # Advance candidate, with wrap-around
            candidate_location += omega_exit_width
            if candidate_location >= perimeter_length:
                candidate_location -= perimeter_length

        # Commit this exit and update global fitness
        E_solutions.append(chosen_exit_for_this_iteration)
        current_fitness = best_psi_for_this_exit

    return E_solutions, best_overall_fitness, time_of_best
