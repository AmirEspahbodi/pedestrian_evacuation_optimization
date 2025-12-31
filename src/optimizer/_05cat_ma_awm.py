import time

import numpy as np
from cmaes import CatCMAwM

from .common import FitnessEvaluator


def _05cat_ma_awm(
    grid,
    pedestrian_confs,
    simulator_config,
    iea_config,
    num_exits_k=2,
):
    """
    Optimizes exit locations and tracks full history and timing metrics.
    """
    perimeter_max = 2 * (len(grid) + len(grid[0]))

    max_generations = 20
    pop_size = 12

    # --- 2. INITIALIZE EVALUATOR ---
    evaluator = FitnessEvaluator(grid, pedestrian_confs, simulator_config)

    # --- 3. DEFINE SEARCH SPACE ---
    valid_integers = list(range(perimeter_max + 1))
    z_space = [valid_integers] * num_exits_k

    # x_space: Must be shape (0, 2) to indicate ZERO continuous variables without crashing
    x_space = np.empty((0, 2))

    optimizer = CatCMAwM(
        x_space=x_space,
        z_space=z_space,
        population_size=pop_size,
        c_space=[],
        sigma=20.0,
        seed=42,
    )

    # --- 4. TRACKING VARIABLES ---
    history = {}
    best_global_solution_vector = None
    best_global_solution_fitness_value = float("inf")
    best_global_solution_gen = 0
    time_to_find_best_global_solution = 0.0

    start_time = time.time()

    print(f"--- Starting Optimization (k={num_exits_k}, Gen={max_generations}) ---")

    # --- 5. OPTIMIZATION LOOP ---
    for generation in range(max_generations):
        solutions = []
        current_gen_fitnesses = []

        for _ in range(optimizer.population_size):
            sol = optimizer.ask()

            try:
                exit_positions = sol.z

                # --- EVALUATION ---
                fitness_value = evaluator.evaluate(exit_positions)

                solutions.append((sol, fitness_value))
                current_gen_fitnesses.append(fitness_value)

                # Track Global Best
                if fitness_value < best_global_solution_fitness_value:
                    best_global_solution_fitness_value = fitness_value
                    best_global_solution_vector = exit_positions
                    best_global_solution_gen = generation
                    time_to_find_best_global_solution = time.time() - start_time

            except Exception as e:
                print(f"Error on inputs {sol.z}: {e}")
                penalty = float("inf")
                solutions.append((sol, penalty))
                current_gen_fitnesses.append(penalty)

        optimizer.tell(solutions)

        # Log History
        history[str(generation)] = current_gen_fitnesses

        # Print Status
        best_in_gen = (
            min(current_gen_fitnesses) if current_gen_fitnesses else float("inf")
        )
        print(
            f"Gen {generation}: Min={best_in_gen:.2f} | Global Best={best_global_solution_fitness_value:.2f}"
        )

    total_time = time.time() - start_time
    print(f"--- Finished in {total_time:.2f}s ---")

    # --- CONVERT TO JSONABLE TYPES ---
    # Convert numpy array to python list
    if isinstance(best_global_solution_vector, np.ndarray):
        best_global_solution_vector = best_global_solution_vector.tolist()

    # Ensure fitness values are native python floats (handles numpy.float64)
    best_global_solution_fitness_value = float(best_global_solution_fitness_value)
    time_to_find_best_global_solution = float(time_to_find_best_global_solution)

    # Sanitize history dictionary to ensure all values are native python floats
    history = {k: [float(v) for v in vals] for k, vals in history.items()}

    return (
        best_global_solution_vector,
        best_global_solution_gen,
        best_global_solution_fitness_value,
        time_to_find_best_global_solution,
        history,
    )
