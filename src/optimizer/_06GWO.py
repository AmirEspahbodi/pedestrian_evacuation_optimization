import numpy as np
from typing import List, Tuple, Optional


def pa_dgwo(
    k: int,
    P: int,
    Q: int,
    B_max: int,
    gamma: float = 2.0,
    lambda_init: float = 0.5,
    lambda_final: float = 0.3,
    q_neighbors: int = 5,
    p_local: float = 0.3,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, float]:
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
    """

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

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
        # Check if problem is feasible
        if k * Q >= P:
            # Infeasible: exits cannot fit. Return as-is (should not happen).
            return x

        max_repairs = 100
        for _ in range(max_repairs):
            overlap_found = False
            # Sort positions circularly starting from the smallest
            order = np.argsort(x)
            sorted_x = x[order].copy()

            for i in range(k):
                for j in range(i + 1, k):
                    if circular_distance(sorted_x[i], sorted_x[j]) < Q:
                        overlap_found = True
                        # Shift the second exit clockwise by the deficit
                        deficit = Q - circular_distance(sorted_x[i], sorted_x[j])
                        sorted_x[j] = (sorted_x[j] + deficit) % P

            x[order] = sorted_x

            if not overlap_found:
                break

        # Final verification and brute-force fix if needed
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

        # Select q nearest neighbors
        q_actual = min(q, n_archive)
        nn_indices = np.argpartition(distances, q_actual - 1)[:q_actual]

        # Inverse distance weighting
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
        Returns array of shape (n_wolves, k) with integer positions.
        """
        solutions = np.zeros((n_wolves, k), dtype=int)
        delta_0 = max(1, P // (4 * k))

        for i in range(n_wolves):
            for j in range(k):
                base = (j * P) // k
                offset = (i * P) // (n_wolves * k)
                jitter = int(rng.integers(-delta_0, delta_0 + 1))
                solutions[i, j] = (base + offset + jitter) % P

            # Repair overlaps
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
            # Generate random coefficients for each leader
            targets = []

            for leader in [alpha, beta, delta]:
                r1 = rng.random()
                r2 = rng.random()

                A_j = 2.0 * a_val * r1 - a_val
                C_j = 2.0 * r2

                # Signed circular displacement from wolf to leader
                delta_j = signed_circular_displacement(int(wolf[j]), int(leader[j]))

                # Encirclement distance
                D_j = abs(C_j * delta_j)

                # Circular position update
                sign_delta = 1 if delta_j >= 0 else -1
                if delta_j == 0:
                    sign_delta = 0

                displacement = int(np.round(A_j * D_j * sign_delta))
                target_j = (int(leader[j]) - displacement) % P
                targets.append(target_j)

            # Circular averaging of three targets
            # Check if targets are too spread (> P/3 apart)
            max_spread = max(
                circular_distance(targets[0], targets[1]),
                circular_distance(targets[1], targets[2]),
                circular_distance(targets[0], targets[2])
            )

            if max_spread > P // 3:
                # Fallback: select target closest to current wolf position
                dists_to_wolf = [circular_distance(t, int(wolf[j])) for t in targets]
                new_pos[j] = targets[int(np.argmin(dists_to_wolf))]
            else:
                new_pos[j] = circular_mean(targets)

        return new_pos

    # ------------------------------------------------------------------
    # PHASE 0: INITIALIZATION
    # ------------------------------------------------------------------

    # Initial pack size
    N0 = 2 * k + 6

    # Estimated max iterations
    T = max(1, B_max // N0)

    # Initialize wolves using Perimeter-Coverage Initialization
    wolves = perimeter_coverage_init(N0)

    # Evaluate all initial wolves
    archive_X = []
    archive_f = []
    B_used = 0

    for i in range(N0):
        if B_used >= B_max:
            break
        fitness_val = psi_evaluate(wolves[i].tolist())
        archive_X.append(wolves[i].copy())
        archive_f.append(fitness_val)
        B_used += 1

    # Convert archive to arrays
    if len(archive_X) == 0:
        raise ValueError("Budget too small to evaluate even one solution.")

    archive_X_arr = np.array(archive_X)
    archive_f_arr = np.array(archive_f)

    # Sort by fitness to assign hierarchy
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

    # Current pack (working population)
    current_wolves = wolves.copy()
    current_fitness = archive_f_arr[:N0].copy()

    t = 0  # iteration counter

    # ------------------------------------------------------------------
    # PHASE 1-3: MAIN LOOP
    # ------------------------------------------------------------------

    while B_used < B_max and t < T:
        t += 1

        # --- Parameter Update (Budget-Aware Nonlinear Schedule) ---
        progress = t / T  # in (0, 1]
        a_val = 2.0 * (1.0 - progress ** gamma)

        # Adaptive pack size
        N_t = max(2 * k + 2, int(np.floor(N0 * (1.0 - progress) ** 0.5)))

        # Screening threshold (linearly interpolated)
        lambda_t = lambda_init + (lambda_final - lambda_init) * progress

        # --- Generate Candidates ---
        candidates = []
        n_omegas = max(1, N_t - 3)  # wolves excluding alpha, beta, delta roles

        for i in range(n_omegas):
            # Select a random wolf from current pack as the "omega" to update
            wolf_idx = rng.integers(0, len(current_wolves))
            wolf = current_wolves[wolf_idx].copy()

            # Generate new candidate via Circular Three-Leader Hunting
            candidate = generate_candidate(wolf, x_alpha, x_beta, x_delta, a_val)

            # Apply overlap repair
            candidate = repair_overlaps(candidate)

            candidates.append(candidate)

        # --- Local Perturbation (Phase 3: last 30% of iterations) ---
        if progress > 0.7 and rng.random() < p_local and len(candidates) > 0:
            # Replace one candidate with a local perturbation of alpha
            pert_idx = rng.integers(0, len(candidates))
            pert_candidate = x_alpha.copy()
            j_rand = rng.integers(0, k)
            shift = int(rng.integers(-max(1, Q // 2), max(1, Q // 2) + 1))
            pert_candidate[j_rand] = (int(pert_candidate[j_rand]) + shift) % P
            pert_candidate = repair_overlaps(pert_candidate)
            candidates[pert_idx] = pert_candidate

        # --- Surrogate Screening ---
        surviving_candidates = []
        n_archive = len(archive_f_arr)

        if n_archive >= q_neighbors + 2:
            f_worst = np.max(archive_f_arr)
            for cand in candidates:
                predicted_f = surrogate_predict(cand, archive_X_arr, archive_f_arr, q_neighbors)
                # Accept if predicted fitness is not too bad
                if predicted_f < f_alpha + lambda_t * (f_worst - f_alpha):
                    surviving_candidates.append(cand)
                # If rejected, skip evaluation (saves budget)
        else:
            # Not enough data for surrogate; evaluate all
            surviving_candidates = candidates

        # --- Evaluate Surviving Candidates ---
        new_evaluations_X = []
        new_evaluations_f = []

        for cand in surviving_candidates:
            if B_used >= B_max:
                break
            fitness_val = psi_evaluate(cand.tolist())
            new_evaluations_X.append(cand.copy())
            new_evaluations_f.append(fitness_val)
            B_used += 1

        # --- Update Archive ---
        if len(new_evaluations_X) > 0:
            new_X_arr = np.array(new_evaluations_X)
            new_f_arr = np.array(new_evaluations_f)
            archive_X_arr = np.vstack([archive_X_arr, new_X_arr])
            archive_f_arr = np.concatenate([archive_f_arr, new_f_arr])

        # --- Update Hierarchy ---
        # Find top 3 from entire archive
        sorted_all = np.argsort(archive_f_arr)
        alpha_idx = sorted_all[0]
        beta_idx = sorted_all[1] if len(sorted_all) > 1 else sorted_all[0]
        delta_idx = sorted_all[2] if len(sorted_all) > 2 else sorted_all[0]

        x_alpha = archive_X_arr[alpha_idx].copy()
        x_beta = archive_X_arr[beta_idx].copy()
        x_delta = archive_X_arr[delta_idx].copy()
        f_alpha = archive_f_arr[alpha_idx]

        # --- Update Working Pack ---
        # Keep the best N_t solutions from archive as current wolves
        top_indices = sorted_all[:N_t]
        current_wolves = archive_X_arr[top_indices].copy()
        current_fitness = archive_f_arr[top_indices].copy()

        # --- Budget Check ---
        if B_used >= B_max:
            break

    # ------------------------------------------------------------------
    # PHASE 4: TERMINATION
    # ------------------------------------------------------------------

    best_solution = x_alpha.copy()
    best_fitness = f_alpha

    return best_solution, best_fitness


# ======================================================================
# USAGE EXAMPLE (fitness function stub for testing)
# ======================================================================

def psi_evaluate(solution: List[int]) -> float:
    """
    Placeholder fitness function.
    Replace this with the actual evacuation simulator call.

    Parameters
    ----------
    solution : list of int
        Vector of k integers, each in [0, perimeter_length).
        Represents start positions of emergency exits.

    Returns
    -------
    float
        Fitness value (lower is better).
    """
    # --- REPLACE THIS WITH YOUR ACTUAL SIMULATOR CALL ---
    # Example: return your_simulator.evaluate(solution)
    raise NotImplementedError(
        "Replace this stub with the actual psi_evaluate fitness function."
    )


# ======================================================================
# MAIN ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    # Problem parameters
    k = 3              # number of emergency exits
    P = 400            # perimeter length (e.g., 100x100 grid)
    Q = 5              # exit width in cells
    B_max = 600        # evaluation budget
    gamma = 2.0        # nonlinear decay shape
    seed = 42          # reproducibility

    # Run PA-DGWO
    best_sol, best_fit = pa_dgwo(
        k=k,
        P=P,
        Q=Q,
        B_max=B_max,
        gamma=gamma,
        seed=seed
    )

    print(f"Best solution (exit positions): {best_sol.tolist()}")
    print(f"Best fitness: {best_fit:.6f}")
