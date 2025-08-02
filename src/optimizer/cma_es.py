import numpy as np
import time
from src.config import SimulationConfig, OptimizerStrategy, EAConfig
from src.optimizer.common import FitnessEvaluator
from src.simulator.domain import Domain

def adapted_cma_es(
    domain: Domain,
):
    fitness_function = FitnessEvaluator(
        domain=domain, optimizer_strategy=OptimizerStrategy.IEA
    )
    k_exits = SimulationConfig.num_emergency_exits
    omega = SimulationConfig.omega
    perimeter_length = 2 * (domain.width + domain.height)
    max_evaluations = EAConfig.islands[0].maxevals
    population_size = EAConfig.islands[0].popsize

    # Epsilon for numerical stability
    epsilon = 1e-8

    # --- 1. Initialization ---
    # Set population size (lambda) if not provided
    if population_size is None:
        population_size = 4 + int(3 * np.log(k_exits))

    # Number of parents (mu)
    num_parents = population_size // 2

    # Recombination weights
    weights = np.log(num_parents + 0.5) - np.log(np.arange(1, num_parents + 1))
    weights /= np.sum(weights)

    # Variance-effective size of the selection
    mueff = np.sum(weights) ** 2 / np.sum(weights**2)

    # --- 2. Strategy Parameters Initialization ---
    # Learning rates for covariance matrix and step-size adaptation
    cc = (4 + mueff / k_exits) / (k_exits + 4 + 2 * mueff / k_exits)
    cs = (mueff + 2) / (k_exits + mueff + 5)
    c1 = 2 / ((k_exits + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((k_exits + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (k_exits + 1)) - 1) + cs

    # Evolution paths for C and sigma
    pc = np.zeros(k_exits)
    ps = np.zeros(k_exits)

    # Covariance matrix and its Cholesky decomposition
    B = np.eye(k_exits)
    D = np.ones(k_exits)
    C = B @ np.diag(D**2) @ B.T
    invsqrtC = B @ np.diag(D**-1) @ B.T

    # Initial mean and step-size
    mean = np.random.uniform(0, perimeter_length - omega, size=k_exits)
    mean.sort()
    sigma = 0.3 * (perimeter_length - omega)

    # --- History & Timing Tracking ---
    start_time = time.perf_counter()
    time_to_best = None
    fitness_pop_hist = []

    # --- 3. Evolutionary Loop ---
    evaluation_count = 0
    generation = 0

    while evaluation_count < max_evaluations:
        generation += 1

        # Store history before generation

        # Generate new population
        population = []
        for _ in range(population_size):
            z = np.random.randn(k_exits)
            y = B @ (D * z)
            x = mean + sigma * y
            population.append(x)

        # Project and evaluate fitness
        projected_population = []
        fitness_values = []
        for individual in population:
            sorted_individual = np.sort(individual)
            projected = np.clip(sorted_individual, 0, perimeter_length - omega)
            for i in range(1, k_exits):
                if projected[i] < projected[i - 1] + omega:
                    projected[i] = projected[i - 1] + omega
            projected = np.clip(projected, 0, perimeter_length - omega)
            projected = np.round(projected).astype(int)
            projected_population.append(projected)

            fitness = fitness_function.evaluate(projected)
            fitness_values.append(fitness)
            evaluation_count += 1
            if evaluation_count >= max_evaluations:
                break

        # Record fitnesses of this population
        fitness_pop_hist.append(fitness_values.copy())

        # Sort by fitness
        sorted_indices = np.argsort(fitness_values)

        # Update best solution found so far
        current_best = fitness_values[sorted_indices[0]]
        if generation == 1 or current_best < best_fitness:
            best_fitness = current_best
            best_solution = projected_population[sorted_indices[0]]
            time_to_best = time.perf_counter() - start_time

        # --- 4. Update Strategy Parameters ---
        old_mean = mean.copy()
        selected_individuals = [population[i] for i in sorted_indices[:num_parents]]
        mean = weights @ selected_individuals

        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (
            mean - old_mean
        ) / sigma

        h_sig = np.linalg.norm(ps) / np.sqrt(
            1 - (1 - cs) ** (2 * evaluation_count / population_size)
        ) / k_exits < (1.4 + 2 / (k_exits + 1))

        pc = (1 - cc) * pc + h_sig * np.sqrt(cc * (2 - cc) * mueff) * (
            mean - old_mean
        ) / sigma

        artmp = (1 / sigma) * (np.array(selected_individuals) - old_mean)
        C = (
            (1 - c1 - cmu) * C
            + c1
            * (pc[:, np.newaxis] @ pc[np.newaxis, :] + (1 - h_sig) * cc * (2 - cc) * C)
            + cmu * (artmp.T @ np.diag(weights) @ artmp)
        )

        sigma *= np.exp(
            (cs / damps)
            * (np.linalg.norm(ps) / np.linalg.norm(np.random.randn(k_exits)) - 1)
        )

        # --- 5. Covariance Matrix Decomposition ---
        if (
            evaluation_count - getattr(C, "eigen_eval", 0)
            > population_size / (c1 + cmu) / k_exits / 10
        ):
            C = np.triu(C) + np.triu(C, 1).T
            D_squared, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D_squared, epsilon))
            invsqrtC = B @ np.diag(D**-1) @ B.T
            C.eigen_eval = evaluation_count

    # Return best solution, its fitness, history, and timing
    return best_solution, best_fitness, fitness_pop_hist, time_to_best