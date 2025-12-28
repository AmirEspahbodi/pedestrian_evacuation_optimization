import time
from typing import Any, Dict, List, Tuple

import numpy as np
from cmaes import CMAwM

from .common import FitnessEvaluator


def run_cma_es_optimization(
    grid,
    pedestrian_confs,
    simulator_config,
    iea_config,
    seed: int = 42,
) -> Tuple[List[int], float, float, Dict[str, List[float]]]:
    """
    Runs CMA-ES with Margin and tracks full history.

    Returns:
        best_global_solution_gens (List[int]): The best integer coordinates found.
        best_global_solution_fitness_value (float): The minimal evacuation time.
        time_to_find_best_global_solution (float): Seconds elapsed when the best was found.
        history (Dict[str, List[float]]): Mapping of generation (str) -> list of fitness values.
    """

    # 1. Setup
    start_time = time.time()
    evaluator = FitnessEvaluator(grid, pedestrian_confs, simulator_config)
    k_exits = simulator_config.numEmergencyExits
    max_evaluations = iea_config.max_evals

    def fitness_function(x: np.ndarray) -> float:
        # Cast to pure python integers for the simulator
        int_vector = [int(val) for val in x]
        return evaluator.evaluate(int_vector)

    # 2. Bounds (0-400) and Steps (1 for integers)
    bounds = np.column_stack((np.zeros(k_exits), np.full(k_exits, 400.0)))
    steps = np.ones(k_exits)

    optimizer = CMAwM(
        mean=np.full(k_exits, 200.0),
        sigma=100.0,
        bounds=bounds,
        steps=steps,
        seed=seed,
        n_max_resampling=10 * k_exits,
    )

    # 3. Tracking Variables
    best_solution = None
    best_value = float("inf")
    time_to_find_best = 0.0

    history = {}  # Key: "generation_index", Value: [f1, f2, f3...]

    eval_count = 0
    generation_idx = 0

    print(f"--- Starting Optimization (k={k_exits}) ---")

    # 4. Optimization Loop
    while True:
        population_solutions = []  # Stores (x_tell, fitness) for the optimizer
        generation_fitness_values = []  # Stores just fitness for history

        # ASK: Generate population
        for _ in range(optimizer.population_size):
            x_for_eval, x_for_tell = optimizer.ask()

            # EVALUATE
            try:
                value = fitness_function(x_for_eval)
            except Exception:
                value = float("inf")  # Handle simulator crashes

            eval_count += 1

            # Store data
            population_solutions.append((x_for_tell, value))
            generation_fitness_values.append(value)

            # Check for new Global Best
            if value < best_value:
                best_value = value
                best_solution = [int(x) for x in x_for_eval]  # Store as pure ints
                time_to_find_best = time.time() - start_time

            if eval_count >= max_evaluations:
                break

        # RECORD HISTORY
        # Using string keys as requested: "0", "1", "2"...
        history[str(generation_idx)] = generation_fitness_values

        # TELL: Update optimizer
        optimizer.tell(population_solutions)

        # LOGGING (Optional)
        if generation_idx % 5 == 0:
            avg_gen = sum(generation_fitness_values) / len(generation_fitness_values)
            print(f"Gen {generation_idx}: Best={best_value:.4f} | Avg={avg_gen:.4f}")

        generation_idx += 1

        # STOP CONDITIONS
        if optimizer.should_stop() or eval_count >= max_evaluations:
            break

    # 5. Final Outputs
    print(f"--- Finished ---")
    print(f"Best Found at: {time_to_find_best:.2f}s | Value: {best_value}")

    return (best_solution, best_value, time_to_find_best, history)
