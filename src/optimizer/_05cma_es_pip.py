from typing import List, Tuple

import numpy as np
from cmaes import CMAwM

from .common import FitnessEvaluator


def run_cma_es_optimization(
    grid,
    pedestrian_confs,
    simulator_config,
    iea_config,
    seed: int = 42,
):
    """
    Runs CMA-ES with Margin to optimize emergency exit placement.

    Args:
        grid: Floor plan object.
        pedestrian_confs: Configuration for pedestrians.
        simulator_config: Simulator settings.
        k_exits: Number of emergency exits to place (dimensionality).
        max_evaluations: Budget for fitness function calls.
        seed: Random seed for reproducibility.
    """

    # 1. Setup the Fitness Function
    # We initialize the evaluator once.
    evaluator = FitnessEvaluator(grid, pedestrian_confs, simulator_config)
    k_exits = simulator_config.numEmergencyExits
    max_evaluations = iea_config.max_evals

    def fitness_function(x: np.ndarray) -> float:
        # CRITICAL: Explicitly cast to Python integers.
        # Even if numpy has 200.0, your simulator might want pure int(200).
        int_vector = [int(val) for val in x]
        return evaluator.evaluate(int_vector)

    # 2. Define Problem Bounds and Steps
    # Domain: [0, 400] for all dimensions.
    lower_bounds = np.zeros(k_exits)
    upper_bounds = np.full(k_exits, 400.0)

    # Shape needs to be (k, 2) for CMAwM bounds
    bounds = np.column_stack((lower_bounds, upper_bounds))

    # Steps: Define the minimal modification step.
    # Since all vars are integers, step is 1.0 for all dimensions.
    # If variables were continuous, step would be 0.
    steps = np.ones(k_exits)

    # 3. Initialize CMAwM
    # Mean: Start in the center of the perimeter range (200).
    # Sigma: High initial variance (100) to explore the whole [0, 400] space initially.
    optimizer = CMAwM(
        mean=np.full(k_exits, 200.0),
        sigma=100.0,
        bounds=bounds,
        steps=steps,
        seed=seed,
        n_max_resampling=10 * k_exits,  # Safety for bound handling
    )

    print(f"--- Starting CMAwM Optimization (k={k_exits}) ---")
    print(f"{'Eval #':<10} | {'Best Fitness':<15} | {'Current Best Solution'}")
    print("-" * 60)

    best_solution = None
    best_value = float("inf")
    eval_count = 0

    # 4. Optimization Loop
    while True:
        solutions = []

        # Ask for a population of candidates
        for _ in range(optimizer.population_size):
            # x_for_eval: Discrete values (integers) -> Send to Simulator
            # x_for_tell: Continuous values -> Send back to Optimizer
            x_for_eval, x_for_tell = optimizer.ask()

            try:
                value = fitness_function(x_for_eval)
                eval_count += 1
            except Exception as e:
                # Fallback in case simulator crashes unexpectedly
                print(f"Simulator crashed on input {x_for_eval}: {e}")
                value = float("inf")

            solutions.append((x_for_tell, value))

            # Track global best
            if value < best_value:
                best_value = value
                best_solution = x_for_eval

            # Check Budget
            if eval_count >= max_evaluations:
                break

        # Tell the optimizer the results
        optimizer.tell(solutions)

        # Logging
        if eval_count % optimizer.population_size == 0:
            # Just showing the first 3 dims for brevity in logs
            sol_repr = str(best_solution[: min(k_exits, 3)]) + (
                "..." if k_exits > 3 else ""
            )
            print(f"{eval_count:<10} | {best_value:<15.4f} | {sol_repr}")

        # Stop conditions
        if optimizer.should_stop() or eval_count >= max_evaluations:
            break

    print("-" * 60)
    print(f"Optimization Finished.")
    print(f"Best Evacuation Time: {best_value}")
    print(f"Best Exit Locations: {[int(x) for x in best_solution]}")

    return [int(x) for x in best_solution], best_value
