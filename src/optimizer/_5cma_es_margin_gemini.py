import math

import numpy as np
from scipy.stats import norm

from .common import FitnessEvaluator


class CMAES_Margin_Integer:
    """
    CMA-ES with Margin for Mixed-Integer Optimization.
    Reference: Hamano et al., "CMA-ES with margin: lower-bounding marginal probability
    for mixed-integer black-box optimization", GECCO 2022.
    """

    def __init__(self, dim, bounds, popsize=20, sigma_init=None, alpha=None):
        """
        Args:
            dim (int): Dimensionality of the problem (k).
            bounds (tuple): (min_val, max_val) e.g., (0, 400).
            population_size (int): Lambda. If None, calculated via default heuristic.
            sigma_init (float): Initial step size. Defaults to 1/4 of bound range.
        """
        self.N = dim
        self.bounds = bounds
        self.min_bound, self.max_bound = bounds

        # 1. Strategy Parameters (Table 1 defaults)
        self.lam = popsize if popsize else 4 + int(3 * np.log(self.N))
        self.mu = self.lam // 2

        # Weights for recombination
        weights_prime = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights_prime / np.sum(weights_prime)
        self.mueff = 1 / np.sum(self.weights**2)

        # Adaptation constants
        self.cc = (4 + self.mueff / self.N) / (self.N + 4 + 2 * self.mueff / self.N)
        self.cs = (self.mueff + 2) / (self.N + self.mueff + 5)
        self.c1 = 2 / ((self.N + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1 - self.c1,
            2 * (self.mueff - 2 + 1 / self.mueff) / ((self.N + 2) ** 2 + self.mueff),
        )
        self.ds = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.N + 1)) - 1) + self.cs

        # Chi-squared expectation for N dimensions
        self.chiN = np.sqrt(self.N) * (1 - 1 / (4 * self.N) + 1 / (21 * self.N**2))

        # 2. Dynamic State Initialization
        # Initialize mean at center of bounds
        self.m = np.full(
            self.N, (self.max_bound - self.min_bound) / 2.0 + self.min_bound
        )

        # Initialize sigma
        if sigma_init:
            self.sigma = sigma_init
        else:
            # Heuristic: 1/4 of the range
            self.sigma = (self.max_bound - self.min_bound) / 4.0

        self.C = np.eye(self.N)  # Covariance Matrix
        self.pc = np.zeros(self.N)  # Evolution path for C
        self.ps = np.zeros(self.N)  # Evolution path for sigma
        self.B = np.eye(self.N)  # Eigenvectors
        self.D = np.ones(self.N)  # Eigenvalues (sqrt)

        self.gen = 0

        # 3. Margin Parameters (Section 4 & 5.1)
        # alpha determines the minimum probability mass kept on adjacent integers
        # Default recommendation from Sec 5.1: 1 / (N * lambda)
        self.alpha = 1.0 / (self.N * self.lam) if alpha is None else alpha

        # Affine transformation matrix A (Initialized to Identity)
        # This scales the variance to enforce the margin
        self.A = np.eye(self.N)

        # Storage for current generation
        self.y_raw = None  # The raw samples from N(0, C)
        self.v_affine = None  # The affine transformed samples

    def ask(self):
        """
        Generate candidate solutions.
        Returns:
            candidates (np.array): (lambda, N) array of integer vectors in [0, 400].
        """
        # Step 1: Sample from standard CMA-ES distribution N(0, C)
        z = np.random.randn(self.lam, self.N)
        # y = B * D * z
        self.y_raw = (self.B @ (self.D * z).T).T  # Shape (lam, N)

        # Step 2: Apply Margin Affine Transformation (Section 4.2 Step 2)
        # v = m + sigma * A * y
        # A is diagonal, so we apply element-wise multiplication
        diag_A = np.diag(self.A)
        self.v_affine = self.m + self.sigma * (self.y_raw * diag_A)

        # Step 3: Discretize and Bound
        # Rounding simulates the thresholds at k.5
        candidates_float = np.round(self.v_affine)

        # Clip to bounds [0, 400]
        # While the margin logic tries to keep distribution wide, we must
        # strictly respect the floor plan limits for the simulator.
        candidates = np.clip(candidates_float, self.min_bound, self.max_bound).astype(
            int
        )

        return candidates

    def tell(self, fitness_values):
        """
        Update strategy parameters based on fitness values.
        Args:
            fitness_values (list or array): Length lambda list of fitness scores (lower is better).
        """
        # Sort by fitness (lowest is best)
        idx_sorted = np.argsort(fitness_values)
        y_sorted = self.y_raw[idx_sorted]

        # Selection: weighted sum of the mu best raw samples
        y_w = np.sum(self.weights[:, None] * y_sorted[: self.mu], axis=0)

        # --- Standard CMA-ES Updates (Section 2.1) ---

        # 1. Update Mean (Note: This is the "raw" mean update before margin correction)
        # m(t+1) = m(t) + cm * sigma * y_w
        # In standard CMA formulation where y is drawn relative to m(t), cm=1.
        self.m = self.m + self.sigma * y_w

        # 2. Update Evolution Paths
        # C^(-1/2) = B * D^(-1) * B'
        inv_sqrt_C = self.B @ np.diag(1 / self.D) @ self.B.T

        # p_sigma
        self.ps = (1 - self.cs) * self.ps + np.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * (inv_sqrt_C @ y_w)

        # h_sigma check (stall update if ps is growing too fast)
        hsig_check = (
            np.linalg.norm(self.ps)
            / np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1)))
            / self.chiN
        )
        hsig = 1 if hsig_check < 1.4 + 2 / (self.N + 1) else 0

        # p_c
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) * y_w

        # 3. Update Covariance Matrix C
        # Rank-1 update
        pc_tensor = np.outer(self.pc, self.pc)

        # Rank-mu update
        rank_mu = np.zeros((self.N, self.N))
        for i in range(self.mu):
            rank_mu += self.weights[i] * np.outer(y_sorted[i], y_sorted[i])

        dh = (1 - hsig) * self.cc * (2 - self.cc)
        c1_a = self.c1 * dh

        self.C = (
            (1 - self.c1 - self.cmu * np.sum(self.weights) + c1_a) * self.C
            + self.c1 * pc_tensor
            + self.cmu * rank_mu
        )

        # 4. Update Step-Size Sigma
        self.sigma = self.sigma * np.exp(
            (self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chiN - 1)
        )

        # 5. Eigendecomposition (update B and D)
        # Enforce symmetry
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        vals, vecs = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(vals, 1e-12))  # prevent numerical errors
        self.B = vecs

        # --- MARGIN CORRECTION (Section 4.4) ---
        # Modify mean m and affine matrix A to prevent integer stagnation
        self._apply_margin_correction_integer()

        self.gen += 1

    def _apply_margin_correction_integer(self):
        """
        Implementation of Section 4.4: Margin for Integer Variables.
        Updates self.m and self.A.
        """
        # Calculate current coordinate-wise standard deviations of the search distribution
        # The actual search distribution is N(m, sigma^2 * A * C * A^T)
        # Since A is diagonal, the std for dimension j is: sigma * A_jj * sqrt(C_jj)
        diag_C = np.diag(self.C)
        diag_A = np.diag(self.A)
        current_stds = self.sigma * diag_A * np.sqrt(diag_C)

        new_m = self.m.copy()
        new_A_diag = diag_A.copy()

        for j in range(self.N):
            mj = self.m[j]
            std_j = current_stds[j]

            # Determine thresholds for the current mean
            # In integer rounding, thresholds are at k.5
            # l_low: the highest threshold strictly less than mj
            # l_up: the lowest threshold greater than or equal to mj
            # E.g., if m=200.2, l_low=199.5, l_up=200.5
            l_low = np.floor(mj - 0.5) + 0.5
            l_up = np.floor(mj - 0.5) + 1.5

            # 1. Calculate current marginal probabilities
            # Pr(v <= l_low)
            p_low = norm.cdf((l_low - mj) / std_j)
            # Pr(v > l_up) = 1 - Pr(v <= l_up)
            p_up = 1.0 - norm.cdf((l_up - mj) / std_j)
            p_mid = 1.0 - p_low - p_up

            # 2. Enforce Margin Constraints (Eq 20-21)
            p_low_prime = max(self.alpha / 2.0, p_low)
            p_up_prime = max(self.alpha / 2.0, p_up)

            # 3. Smoothing (Eq 22-23)
            # This ensures probabilities sum to 1 while respecting bounds
            denom = p_low_prime + p_up_prime + p_mid - 3 * (self.alpha / 2.0)

            # Safety check for denominator close to zero
            if denom < 1e-10:
                p_low_pp = p_low_prime
                p_up_pp = p_up_prime
            else:
                factor = (1.0 - p_low_prime - p_up_prime - p_mid) / denom
                p_low_pp = p_low_prime + factor * (p_low_prime - self.alpha / 2.0)
                p_up_pp = p_up_prime + factor * (p_up_prime - self.alpha / 2.0)

            # 4. Solve for new Mean and Std (Eq 24 logic)
            # We want P(v <= l_low) = p_low_pp
            # We want P(v > l_up) = p_up_pp

            z_low = norm.ppf(p_low_pp)  # Z-score for lower tail
            z_high = norm.ppf(1.0 - p_up_pp)  # Z-score for upper tail (note: 1 - prob)

            # System of equations:
            # l_low - m_new = z_low * std_new
            # l_up - m_new = z_high * std_new

            # Subtracting: l_up - l_low = std_new * (z_high - z_low)
            delta_l = l_up - l_low  # Should be 1.0 for standard integers
            z_diff = z_high - z_low

            if z_diff <= 1e-10:
                # Should not happen given alpha < 0.5
                continue

            std_new = delta_l / z_diff
            m_new = l_low - z_low * std_new

            # Update mean
            new_m[j] = m_new

            # Update A (Scale factor)
            # std_new = sigma * A_new * sqrt(C_jj)
            # -> A_new = std_new / (sigma * sqrt(C_jj))
            new_A_diag[j] = std_new / (self.sigma * np.sqrt(diag_C[j]))

        self.m = new_m
        self.A = np.diag(new_A_diag)


def run_cma_es_optimization(gird, pedestrian_confs, simulator_config, iea_config):
    psi = FitnessEvaluator(gird, pedestrian_confs, simulator_config)
    print("Fitness Evaluator Initialized.")

    # 2. Setup Optimizer
    # k = vector length (3 to 5), bounds = 0 to 400
    # You might need to derive k from your grid/config, or hardcode it.
    VECTOR_LENGTH = simulator_config.numEmergencyExits
    BOUNDS = (0, 2 * (len(gird) + len(gird[0])))

    popsize = 7
    MAX_GENERATIONS = 200
    alpha = 1.0 / (VECTOR_LENGTH * popsize)

    # Initialize Algorithm
    optimizer = CMAES_Margin_Integer(
        dim=VECTOR_LENGTH, bounds=BOUNDS, popsize=popsize, alpha=alpha
    )

    print(f"Optimization started: Dim={VECTOR_LENGTH}, PopSize={optimizer.lam}")

    # --- Optimization Loop ---
    best_fitness_global = float("inf")
    best_solution_global = None

    for gen in range(MAX_GENERATIONS):
        # 1. Ask for candidates (Integer vectors in range 0-400)
        candidates = optimizer.ask()

        fitness_values = []

        # 2. Evaluate candidates
        for i in range(len(candidates)):
            vector_candidate = candidates[i]

            # --- YOUR SIMULATOR CALL ---
            # Using your provided snippet style:
            fitness_val = psi.evaluate(vector_candidate)

            # Placeholder for testing:
            # Simple Sphere function targetting [200, 200, 200...]

            fitness_values.append(fitness_val)

            # Track best
            if fitness_val < best_fitness_global:
                best_fitness_global = fitness_val
                best_solution_global = vector_candidate.copy()

        # 3. Tell algorithm the results
        optimizer.tell(fitness_values)

        # Logging
        current_best = min(fitness_values)
        print(
            f"Gen {gen}: Best Fitness = {current_best:.4f}, Sigma = {optimizer.sigma:.2f}"
        )
        print(f"       Current Mean: {np.round(optimizer.m, 1)}")

    print("\nOptimization Finished.")
    print(f"Best Solution: {best_solution_global}")
    print(f"Best Fitness: {best_fitness_global}")
