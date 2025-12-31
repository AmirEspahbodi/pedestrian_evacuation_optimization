import time

import numpy as np
from cmaes import CatCMAwM

from .common import FitnessEvaluator


def get_robust_optimizer(num_exits_k, perimeter_max=400, seed=42):
    valid_integers = list(range(perimeter_max + 1))
    z_space = [valid_integers] * num_exits_k

    x_space = np.empty((0, 2))
    c_space = []

    initial_mean = np.full(num_exits_k, perimeter_max / 2.0)

    initial_sigma = perimeter_max / 5.0

    algo_pop_size = 16 + num_exits_k

    optimizer = CatCMAwM(
        x_space=x_space,
        z_space=z_space,
        c_space=c_space,
        mean=initial_mean,
        sigma=initial_sigma,
        population_size=algo_pop_size,
        seed=seed,
    )

    return optimizer, algo_pop_size


def _05cat_ma_awm(
    grid,
    pedestrian_confs,
    simulator_config,
    iea_config,
    num_exits_k=2,
):
    perimeter_max = 2 * (len(grid) + len(grid[0]))

    # --- SETTINGS ---
    # Increased to allow the covariance matrix to adapt to the geometry
    max_generations = 60

    # --- INITIALIZE EVALUATOR ---
    evaluator = FitnessEvaluator(grid, pedestrian_confs, simulator_config)

    # --- GET ROBUST OPTIMIZER ---
    optimizer, pop_size = get_robust_optimizer(
        num_exits_k, perimeter_max=perimeter_max, seed=42
    )

    print(
        f"--- Starting Elite Optimization (k={num_exits_k}, Pop={pop_size}, Sigma={optimizer._sigma}) ---"
    )

    # --- TRACKING ---
    history = {}
    best_global_solution_vector = None
    best_global_solution_fitness_value = float("inf")
    best_global_solution_gen = 0
    start_time = time.time()
    time_to_find_best_global_solution = 0.0

    # --- OPTIMIZATION LOOP ---
    for generation in range(max_generations):
        solutions = []
        current_gen_fitnesses = []

        for _ in range(optimizer.population_size):
            sol = optimizer.ask()

            exit_positions = sol.z

            try:
                fitness_value = evaluator.evaluate(exit_positions)
            except Exception as e:
                print(f"Sim crash on {exit_positions}: {e}")
                fitness_value = 1e9

            # Update best global
            if fitness_value < best_global_solution_fitness_value:
                best_global_solution_fitness_value = fitness_value
                best_global_solution_vector = exit_positions
                best_global_solution_gen = generation
                time_to_find_best_global_solution = time.time() - start_time

            solutions.append((sol, fitness_value))
            current_gen_fitnesses.append(fitness_value)

        # Tell the optimizer the results
        optimizer.tell(solutions)

        # Logging
        history[str(generation)] = current_gen_fitnesses
        min_gen = min(current_gen_fitnesses)
        print(
            f"Gen {generation:02d}: Min={min_gen:.2f} | Global Best={best_global_solution_fitness_value:.2f}"
        )

    total_time = time.time() - start_time
    print(f"--- Finished in {total_time:.2f}s ---")

    if isinstance(best_global_solution_vector, np.ndarray):
        best_global_solution_vector = best_global_solution_vector.tolist()

    history = {k: [float(v) for v in vals] for k, vals in history.items()}

    return (
        best_global_solution_vector,
        best_global_solution_gen,
        float(best_global_solution_fitness_value),
        float(time_to_find_best_global_solution),
        history,
    )
