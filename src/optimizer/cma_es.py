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

    # Ensure a minimal population size
    population_size = max(population_size, 2)

    # Epsilon for numerical stability
    epsilon = 1e-8

    # --- 1. Initialization ---
    # Number of parents (mu)
    num_parents = max(1, population_size // 2)

    # Recombination weights
    weights = np.log(num_parents + 0.5) - np.log(np.arange(1, num_parents + 1))
    weights /= np.sum(weights)

    # Variance-effective size of the selection
    mueff = np.sum(weights) ** 2 / np.sum(weights**2)

    # --- 2. Strategy Parameters Initialization ---
    cc = (4 + mueff / k_exits) / (k_exits + 4 + 2 * mueff / k_exits)
    cs = (mueff + 2) / (k_exits + mueff + 5)
    c1 = 2 / ((k_exits + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((k_exits + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (k_exits + 1)) - 1) + cs

    pc = np.zeros(k_exits)
    ps = np.zeros(k_exits)

    B = np.eye(k_exits)
    D = np.ones(k_exits)
    C = B @ np.diag(D**2) @ B.T
    invsqrtC = B @ np.diag(D**-1) @ B.T

    mean = np.random.uniform(0, perimeter_length - omega, size=k_exits)
    mean.sort()
    sigma = 0.3 * (perimeter_length - omega)

    # Initialize tracking variables
    best_fitness = float('inf')
    best_solution = None
    last_eigen_eval = 0
    fitness_pop_hist = {}
    start_time = time.perf_counter()
    time_to_best = None

    evaluation_count = 0
    generation = 0

    # --- 3. Evolutionary Loop ---
    while evaluation_count < max_evaluations:
        generation += 1

        population = []
        for _ in range(population_size):
            z = np.random.randn(k_exits)
            y = B @ (D * z)
            x = mean + sigma * y
            population.append(x)

        projected_population = []
        fitness_values = []
        for individual in population:
            sorted_individual = np.sort(individual)
            projected = np.clip(sorted_individual, 0, perimeter_length - omega)
            for i in range(1, k_exits):
                projected[i] = max(projected[i], projected[i - 1] + omega)
            projected = np.clip(projected, 0, perimeter_length - omega)
            projected = np.round(projected).astype(int)
            projected_population.append(projected)

            fitness = fitness_function.evaluate(projected)
            fitness_values.append(fitness)
            evaluation_count += 1
            if evaluation_count >= max_evaluations:
                break

        fitness_pop_hist[generation] = fitness_values.copy()

        sorted_indices = np.argsort(fitness_values)
        current_best = fitness_values[sorted_indices[0]]
        if generation == 1 or current_best < best_fitness:
            best_fitness = current_best
            best_solution = projected_population[sorted_indices[0]]
            time_to_best = time.perf_counter() - start_time

        # --- 4. Update Strategy Parameters ---
        old_mean = mean.copy()
        selected = [population[i] for i in sorted_indices[:num_parents]]
        mean = weights @ selected

        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (
            (mean - old_mean) / sigma
        )

        # Heaviside for sigma update
        h_sig = (np.linalg.norm(ps) / np.sqrt(
            1 - (1 - cs) ** (2 * evaluation_count / population_size)
        ) / k_exits) < (1.4 + 2 / (k_exits + 1))

        pc = (1 - cc) * pc + h_sig * np.sqrt(cc * (2 - cc) * mueff) * (
            (mean - old_mean) / sigma
        )

        artmp = (1 / sigma) * (np.array(selected) - old_mean)
        C = (
            (1 - c1 - cmu) * C
            + c1 * (np.outer(pc, pc) + (1 - h_sig) * cc * (2 - cc) * C)
            + cmu * (artmp.T @ np.diag(weights) @ artmp)
        )

        sigma *= np.exp(
            (cs / damps)
            * (np.linalg.norm(ps) / np.linalg.norm(np.random.randn(k_exits)) - 1)
        )

        # --- 5. Covariance Matrix Decomposition ---
        if (evaluation_count - last_eigen_eval) > (population_size / (c1 + cmu) / k_exits / 10):
            C = np.triu(C) + np.triu(C, 1).T
            D_squared, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D_squared, epsilon))
            invsqrtC = B @ np.diag(D**-1) @ B.T
            last_eigen_eval = evaluation_count

    return best_solution, best_fitness, fitness_pop_hist, time_to_best
