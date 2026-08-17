import numpy as np
from typing import List, Tuple, Optional, Any, Dict
from .common import FitnessEvaluator


def pa_dgwo(
    k: int,
    P: int,
    Q: int,
    B_max: int,
    pedestrian_confs,
    gird,
    simulator_config,
    gamma: float = 2.0,
    lambda_init: float = 0.5,
    lambda_final: float = 0.3,
    q_neighbors: int = 5,
    p_local: float = 0.3,
    seed: Optional[int] = None,
) -> Tuple[List[int], float, List[List[List[Any]]], int, int]:
    """
    Perimeter-Aware Discrete Grey Wolf Optimizer (PA-DGWO).

    Parameters
    ----------
    k : int
        Number of emergency exits (decision vector length), k in {2,3,4,5}.
    P : int
        Perimeter length (total number of cells on the perimeter).
    Q : int
        Width of each emergency exit (number of cells it occupies).
    B_max : int
        Maximum evaluation budget.
    gamma : float
        Shape parameter for nonlinear decay of exploration parameter 'a'.
    lambda_init : float
        Initial screening threshold for surrogate.
    lambda_final : float
        Final screening threshold for surrogate.
    q_neighbors : int
        Number of nearest neighbors for surrogate prediction.
    p_local : float
        Probability of local perturbation in Phase 3.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    best_solution : np.ndarray of shape (k,), dtype int
        Best solution found (vector of k integers in [0, P)).
    best_fitness : float
        Fitness value of the best solution.
    history : List[List[Tuple[np.ndarray, float]]]
        Evaluated individuals and their fitness values grouped by episode (index 0 is init).
    best_fitness_eval_count : int
        Evaluation count when the best fitness value was discovered.
    best_fitness_episode : int
        Episode index in which the best fitness value was discovered (0 for init).
    """

    evalr = FitnessEvaluator(gird, pedestrian_confs, simulator_config)

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Tracking variables
    history: List[List[Tuple[np.ndarray, float]]] = []
    best_fitness = float("inf")
    best_solution: Optional[np.ndarray] = None
    best_fitness_eval_count = 0
    best_fitness_episode = 0

    # ------------------------------------------------------------------
    # HELPER FUNCTIONS
    # ------------------------------------------------------------------

    def circular_distance(a: int, b: int) -> int:
        """Circular distance between two positions on perimeter of length P."""
        diff = abs(a - b)
        return min(diff, P - diff)

    def signed_circular_displacement(a: int, b: int) -> int:
        """
        Signed displacement from position a to position b along the
        shortest arc on the perimeter. Positive = clockwise.
        Result in range [-P//2, P//2].
        """
        delta = (b - a) % P
        if delta <= P // 2:
            return delta
        else:
            return delta - P

    def circular_mean(positions: List[int]) -> int:
        """
        Circular mean of a list of integer positions on perimeter Z_P.
        Uses angle embedding and atan2.
        """
        angles = [2.0 * np.pi * x / P for x in positions]
        sin_sum = sum(np.sin(th) for th in angles)
        cos_sum = sum(np.cos(th) for th in angles)
        mean_angle = np.arctan2(sin_sum, cos_sum)
        mean_pos = int(np.round(P * mean_angle / (2.0 * np.pi))) % P
        return mean_pos

    def check_overlap(x_i: int, x_j: int) -> bool:
        """Check if two exits starting at x_i and x_j overlap."""
        return circular_distance(x_i, x_j) < Q

    def repair_overlaps(x: np.ndarray) -> np.ndarray:
        """
        Repair overlapping exits by shifting them apart.
        Alternates shift direction based on a parity counter.
        Ensures all exits are non-overlapping on the circular perimeter.
        """
        x = x.copy()
        if k * Q >= P:
            return x

        max_repairs = 100
        for _ in range(max_repairs):
            overlap_found = False
            order = np.argsort(x)
            sorted_x = x[order].copy()

            for i in range(k):
                for j in range(i + 1, k):
                    if circular_distance(sorted_x[i], sorted_x[j]) < Q:
                        overlap_found = True
                        deficit = Q - circular_distance(sorted_x[i], sorted_x[j])
                        sorted_x[j] = (sorted_x[j] + deficit) % P

            x[order] = sorted_x

            if not overlap_found:
                break

        for _ in range(50):
            any_overlap = False
            for i in range(k):
                for j in range(i + 1, k):
                    if check_overlap(x[i], x[j]):
                        any_overlap = True
                        x[j] = (x[i] + Q) % P
            if not any_overlap:
                break

        return x

    def aggregate_circular_distance(x: np.ndarray, y: np.ndarray) -> float:
        """Sum of circular distances across all dimensions."""
        return sum(circular_distance(int(x[j]), int(y[j])) for j in range(k))

    def surrogate_predict(
        candidate: np.ndarray,
        archive_X: np.ndarray,
        archive_f: np.ndarray,
        q: int
    ) -> float:
        """
        k-NN surrogate prediction for a candidate solution.
        Returns predicted fitness value.
        """
        n_archive = len(archive_f)
        if n_archive == 0:
            return float('inf')

        distances = np.array([
            aggregate_circular_distance(candidate, archive_X[i])
            for i in range(n_archive)
        ])

        q_actual = min(q, n_archive)
        nn_indices = np.argpartition(distances, q_actual - 1)[:q_actual]

        eps = 1e-6
        weights = np.array([
            1.0 / (distances[idx] + eps) for idx in nn_indices
        ])
        predicted = np.sum(weights * archive_f[nn_indices]) / np.sum(weights)
        return predicted

    def perimeter_coverage_init(n_wolves: int) -> np.ndarray:
        """
        Perimeter-Coverage Initialization (PCI).
        Generates n_wolves solutions with exits spread around the perimeter.
        """
        solutions = np.zeros((n_wolves, k), dtype=int)
        delta_0 = max(1, P // (4 * k))

        for i in range(n_wolves):
            for j in range(k):
                base = (j * P) // k
                offset = (i * P) // (n_wolves * k)
                jitter = int(rng.integers(-delta_0, delta_0 + 1))
                solutions[i, j] = (base + offset + jitter) % P

            solutions[i] = repair_overlaps(solutions[i])

        return solutions

    def generate_candidate(
        wolf: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        delta: np.ndarray,
        a_val: float
    ) -> np.ndarray:
        """
        Generate a new candidate position for an omega wolf using
        the Circular Discrete Encirclement Operator and
        Circular Three-Leader Hunting.
        """
        new_pos = np.zeros(k, dtype=int)

        for j in range(k):
            targets = []

            for leader in [alpha, beta, delta]:
                r1 = rng.random()
                r2 = rng.random()

                A_j = 2.0 * a_val * r1 - a_val
                C_j = 2.0 * r2

                delta_j = signed_circular_displacement(int(wolf[j]), int(leader[j]))
                D_j = abs(C_j * delta_j)

                sign_delta = 1 if delta_j >= 0 else -1
                if delta_j == 0:
                    sign_delta = 0

                displacement = int(np.round(A_j * D_j * sign_delta))
                target_j = (int(leader[j]) - displacement) % P
                targets.append(target_j)

            max_spread = max(
                circular_distance(targets[0], targets[1]),
                circular_distance(targets[1], targets[2]),
                circular_distance(targets[0], targets[2])
            )

            if max_spread > P // 3:
                dists_to_wolf = [circular_distance(t, int(wolf[j])) for t in targets]
                new_pos[j] = targets[int(np.argmin(dists_to_wolf))]
            else:
                new_pos[j] = circular_mean(targets)

        return new_pos

    # ------------------------------------------------------------------
    # PHASE 0: INITIALIZATION (Episode 0)
    # ------------------------------------------------------------------

    N0 = 2 * k + 6
    T = max(1, B_max // N0)

    wolves = perimeter_coverage_init(N0)

    archive_X = []
    archive_f = []
    init_episode_history: List[Tuple[np.ndarray, float]] = []

    for i in range(N0):
        if evalr.get_evaluation_count() >= B_max:
            break
        fitness_val = evalr.evaluate(wolves[i].tolist())
        current_eval_count = evalr.get_evaluation_count()

        archive_X.append(wolves[i].copy())
        archive_f.append(fitness_val)
        init_episode_history.append((wolves[i].copy(), fitness_val))

        if fitness_val < best_fitness:
            best_fitness = fitness_val
            best_solution = wolves[i].copy()
            best_fitness_eval_count = current_eval_count
            best_fitness_episode = 0

    history.append(init_episode_history)

    if len(archive_X) == 0:
        raise ValueError("Budget too small to evaluate even one solution.")

    archive_X_arr = np.array(archive_X)
    archive_f_arr = np.array(archive_f)

    sorted_indices = np.argsort(archive_f_arr)
    alpha_idx = sorted_indices[0]
    beta_idx = sorted_indices[1] if len(sorted_indices) > 1 else sorted_indices[0]
    delta_idx = sorted_indices[2] if len(sorted_indices) > 2 else sorted_indices[0]

    x_alpha = archive_X_arr[alpha_idx].copy()
    x_beta = archive_X_arr[beta_idx].copy()
    x_delta = archive_X_arr[delta_idx].copy()
    f_alpha = archive_f_arr[alpha_idx]
    f_beta = archive_f_arr[beta_idx]
    f_delta = archive_f_arr[delta_idx]

    current_wolves = wolves.copy()
    current_fitness = archive_f_arr[:N0].copy()

    t = 0  # iteration counter

    # ------------------------------------------------------------------
    # PHASE 1-3: MAIN LOOP (Episodes 1 to T)
    # ------------------------------------------------------------------

    while evalr.get_evaluation_count() < B_max and t < T:
        t += 1

        progress = t / T
        a_val = 2.0 * (1.0 - progress ** gamma)
        N_t = max(2 * k + 2, int(np.floor(N0 * (1.0 - progress) ** 0.5)))
        lambda_t = lambda_init + (lambda_final - lambda_init) * progress

        # Generate Candidates
        candidates = []
        n_omegas = max(1, N_t - 3)

        for i in range(n_omegas):
            wolf_idx = rng.integers(0, len(current_wolves))
            wolf = current_wolves[wolf_idx].copy()

            candidate = generate_candidate(wolf, x_alpha, x_beta, x_delta, a_val)
            candidate = repair_overlaps(candidate)
            candidates.append(candidate)

        # Local Perturbation
        if progress > 0.7 and rng.random() < p_local and len(candidates) > 0:
            pert_idx = rng.integers(0, len(candidates))
            pert_candidate = x_alpha.copy()
            j_rand = rng.integers(0, k)
            shift = int(rng.integers(-max(1, Q // 2), max(1, Q // 2) + 1))
            pert_candidate[j_rand] = (int(pert_candidate[j_rand]) + shift) % P
            pert_candidate = repair_overlaps(pert_candidate)
            candidates[pert_idx] = pert_candidate

        # Surrogate Screening
        surviving_candidates = []
        n_archive = len(archive_f_arr)

        if n_archive >= q_neighbors + 2:
            f_worst = np.max(archive_f_arr)
            for cand in candidates:
                predicted_f = surrogate_predict(cand, archive_X_arr, archive_f_arr, q_neighbors)
                if predicted_f < f_alpha + lambda_t * (f_worst - f_alpha):
                    surviving_candidates.append(cand)
        else:
            surviving_candidates = candidates

        # Evaluate Surviving Candidates
        new_evaluations_X = []
        new_evaluations_f = []
        current_episode_history: List[Tuple[np.ndarray, float]] = []

        for cand in surviving_candidates:
            if evalr.get_evaluation_count() >= B_max:
                break
            fitness_val = evalr.evaluate(cand.tolist())
            current_eval_count = evalr.get_evaluation_count()

            new_evaluations_X.append(cand.copy())
            new_evaluations_f.append(fitness_val)
            current_episode_history.append((cand.copy(), fitness_val))

            if fitness_val < best_fitness:
                best_fitness = fitness_val
                best_solution = cand.copy()
                best_fitness_eval_count = current_eval_count
                best_fitness_episode = t

        history.append(current_episode_history)

        # Update Archive
        if len(new_evaluations_X) > 0:
            new_X_arr = np.array(new_evaluations_X)
            new_f_arr = np.array(new_evaluations_f)
            archive_X_arr = np.vstack([archive_X_arr, new_X_arr])
            archive_f_arr = np.concatenate([archive_f_arr, new_f_arr])

        # Update Hierarchy
        sorted_all = np.argsort(archive_f_arr)
        alpha_idx = sorted_all[0]
        beta_idx = sorted_all[1] if len(sorted_all) > 1 else sorted_all[0]
        delta_idx = sorted_all[2] if len(sorted_all) > 2 else sorted_all[0]

        x_alpha = archive_X_arr[alpha_idx].copy()
        x_beta = archive_X_arr[beta_idx].copy()
        x_delta = archive_X_arr[delta_idx].copy()
        f_alpha = archive_f_arr[alpha_idx]

        # Update Working Pack
        top_indices = sorted_all[:N_t]
        current_wolves = archive_X_arr[top_indices].copy()
        current_fitness = archive_f_arr[top_indices].copy()

        if evalr.get_evaluation_count() >= B_max:
            break

    # ------------------------------------------------------------------
    # PHASE 4: TERMINATION
    # ------------------------------------------------------------------

    if best_solution is None:
        best_solution = x_alpha.copy()
        best_fitness = f_alpha

    # ------------------------------------------------------------------
    # PHASE 4: TERMINATION & JSON SERIALIZATION PATCH
    # ------------------------------------------------------------------

    if best_solution is None:
        best_solution = x_alpha.copy()
        best_fitness = f_alpha

    # Convert all outputs into native Python types for JSON compatibility
    best_solution_json = [int(x) for x in best_solution]
    best_fitness_json = float(best_fitness)
    best_fitness_eval_count_json = int(best_fitness_eval_count)
    best_fitness_episode_json = int(best_fitness_episode)

    history_json = [
        [
            [[int(x) for x in ind], float(fit)]
            for ind, fit in episode
        ]
        for episode in history
    ]

    return (
        best_solution_json,
        best_fitness_json,
        history_json,
        best_fitness_eval_count_json,
        best_fitness_episode_json,
    )
