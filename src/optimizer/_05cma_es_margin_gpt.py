import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .common import FitnessEvaluator


def norm_cdf(x: float) -> float:
    """Standard normal CDF using error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0,1) for norm_ppf")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


# chi2_ppf for df=1 via relation with normal: chi2_ppf(q) = norm_ppf((1+q)/2)^2
def chi2_ppf_df1(q: float) -> float:
    return norm_ppf((1.0 + q) / 2.0) ** 2


# ---------- Encoding / thresholds helpers ----------
def build_thresholds(discrete_vals: Sequence[int]) -> np.ndarray:
    """Return thresholds ℓ_k|k+1 as midpoints between consecutive discrete values."""
    z = np.asarray(sorted(discrete_vals), dtype=float)
    if len(z) <= 1:
        return np.array([], dtype=float)
    return 0.5 * (z[:-1] + z[1:])


def encode_vector_continuous_to_discrete(
    v: np.ndarray, discrete_values: List[Sequence[int]]
) -> np.ndarray:
    """
    Given a real vector v, encode each component into the nearest discrete bin
    defined by midpoints between allowed values. Returns integer vector.
    """
    N = len(v)
    out = np.empty(N, dtype=int)
    for j in range(N):
        z = np.asarray(sorted(discrete_values[j]), dtype=float)
        if z.size == 0:
            raise ValueError(f"discrete_values[{j}] empty")
        if z.size == 1:
            out[j] = int(z[0])
            continue
        # thresholds
        th = build_thresholds(z)
        # find index k such that v_j <= threshold[k] ... use np.searchsorted on thresholds
        # thresholds partition real line: (-inf, th0], (th0, th1], ..., (th_{K-2}, +inf)
        k = np.searchsorted(th, v[j], side="right")
        # k in [0, len(z)-1] corresponds to z[k]
        out[j] = int(z[k])
    return out


# helpers: find l_low and l_up thresholds around m_j for integer margin correction
def find_neighbor_thresholds(
    m_j: float, z_j: Sequence[int]
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    For sorted z_j (allowed discrete values), return:
      - l_closest := threshold closest to m_j (for binary correction)
      - l_low := maximum threshold < m_j  (or None)
      - l_up  := minimum threshold >= m_j (or None)
    Thresholds are midpoints between successive z's.
    """
    z = np.asarray(sorted(z_j), dtype=float)
    if z.size == 1:
        return (None, None, None)
    th = build_thresholds(z)  # length K-1
    # l_low: max threshold < m_j => th[idx-1] if exist
    idx = np.searchsorted(th, m_j, side="right")  # number of thresholds <= m_j
    l_low = th[idx - 1] if idx - 1 >= 0 else None
    l_up = th[idx] if idx < th.size else None
    # For closest threshold to m_j: the nearest threshold among th (if any). for binary case paper uses threshold directly
    if th.size == 0:
        l_closest = None
    else:
        # if m_j outside, choose nearest boundary threshold
        diffs = np.abs(th - m_j)
        l_closest = float(th[np.argmin(diffs)])
    return (l_closest, l_low, l_up)


# ---------- Main CMA-ES with Margin class ----------
class CMAESWithMargin:
    def __init__(
        self,
        N: int,
        fitness: Callable[[Sequence[int]], float],
        discrete_values: Optional[List[Sequence[int]]] = None,
        sigma0: float = 50.0,
        x0: Optional[np.ndarray] = None,
        max_evals: int = 4000,
        seed: Optional[int] = None,
        alpha: Optional[float] = None,
    ):
        """
        N: problem dimension (number of integer variables here)
        fitness: function(list_of_ints) -> float (minimize)
        discrete_values: list (len N) of allowed integer values for each dim.
                         If None, default to 0..400 for every dimension.
        sigma0: initial step-size
        x0: optional initial mean (real vector of length N); if None, use midpoints of discrete sets
        max_evals: termination evaluations limit
        alpha: margin parameter; default uses (N*lambda)^-1 as recommended
        """
        self.N = N
        self.fitness = fitness
        self.max_evals = int(max_evals)
        self.rng = np.random.RandomState(seed)

        if discrete_values is None:
            # default full range 0..400
            self.discrete_values = [list(range(401)) for _ in range(N)]
        else:
            if len(discrete_values) != N:
                raise ValueError("discrete_values length must equal N")
            self.discrete_values = [
                sorted(list(map(int, dv))) for dv in discrete_values
            ]

        # initial mean
        if x0 is None:
            # set to center of each discrete set (float)
            m0 = np.array(
                [0.5 * (min(dv) + max(dv)) for dv in self.discrete_values], dtype=float
            )
        else:
            m0 = np.asarray(x0, dtype=float)
            if m0.shape != (N,):
                raise ValueError("x0 shape mismatch")
        self.m = m0.copy()

        # initialize CMA-ES parameters (standard defaults)
        self.lam = 4 + int(3 * math.log(self.N))
        self.mu = self.lam // 2
        # recombination weights
        w_prime = np.array(
            [math.log((self.lam + 1) / 2.0) - math.log(i + 1) for i in range(self.mu)]
        )
        w_prime = np.maximum(0.0, w_prime)
        self.w = w_prime / np.sum(w_prime)
        # mu_eff: effective selection mass
        self.mu_eff = (np.sum(self.w) ** 2) / np.sum(
            self.w**2
        )  # since sum(w)=1 this equals 1/sum(w^2)
        # learning rates and other hyperparams (as in standard CMA-ES default)
        self.c_m = 1.0
        self.c_sigma = (self.mu_eff + 2.0) / (self.N + self.mu_eff + 5.0)
        self.d_sigma = (
            1.0
            + self.c_sigma
            + 2.0 * max(0.0, math.sqrt((self.mu_eff - 1.0) / (self.N + 1.0)) - 1.0)
        )
        self.c_c = (4.0 + self.mu_eff / self.N) / (
            self.N + 4.0 + 2.0 * self.mu_eff / self.N
        )
        self.c1 = 2.0 / (((self.N + 1.3) ** 2) + self.mu_eff)
        self.c_mu = min(
            1.0 - self.c1,
            2.0
            * (self.mu_eff - 2.0 + 1.0 / self.mu_eff)
            / (((self.N + 2.0) ** 2) + self.mu_eff),
        )
        # initialize sigma, C, pc, ps
        self.sigma = float(sigma0)
        self.C = np.eye(self.N)
        self.p_c = np.zeros(self.N)
        self.p_sigma = np.zeros(self.N)
        # diagonal A (affine transform) initial identity
        self.A_diag = np.ones(
            self.N, dtype=float
        )  # A is diagonal matrix; store diagonal
        # prepare eigen decomposition holder
        self.B = np.eye(self.N)
        self.D = np.ones(self.N)  # sqrt eigenvalues (D)
        self.C_sqrt = np.eye(self.N)
        # termination and counters
        self.eval_count = 0
        self.generation = 0
        # margin alpha
        if alpha is None:
            self.alpha = 1.0 / (self.N * self.lam)
        else:
            self.alpha = float(alpha)

        # bookkeeping best
        self.best_int = None
        self.best_val = float("inf")

    def _update_eigendecomp(self):
        # Ensure C is symmetric positive definite (numerical clipping)
        C_sym = 0.5 * (self.C + self.C.T)
        eigvals, eigvecs = np.linalg.eigh(C_sym)
        # numeric stability: clip tiny negative eigenvalues
        eps = 1e-20
        eigvals = np.maximum(eigvals, eps)
        self.D = np.sqrt(eigvals)  # sqrt of eigenvalues
        self.B = eigvecs
        self.C_sqrt = self.B @ np.diag(self.D)

    def _sample_population(self):
        # sample y_i ~ N(0, C) using C^{1/2} @ z; z ~ N(0,I)
        self._update_eigendecomp()
        Z = self.rng.randn(self.N, self.lam)  # each column is z
        Y = self.C_sqrt @ Z  # shape N x lam
        X = self.m.reshape(-1, 1) + self.sigma * Y  # continuous samples used for update
        # build transformed v = m + sigma * A * y (A is diagonal)
        A_mat = np.diag(self.A_diag)
        V = self.m.reshape(-1, 1) + self.sigma * (A_mat @ Y)
        return Y, X, V  # Y: N x lam, X: N x lam, V: N x lam (real)

    def _evaluate_population(self, V):
        # encode each column of V into discrete vector and evaluate
        lam = V.shape[1]
        vals = np.empty(lam, dtype=float)
        ints = np.empty((self.N, lam), dtype=int)
        for i in range(lam):
            v_real = V[:, i]
            v_int = encode_vector_continuous_to_discrete(v_real, self.discrete_values)
            ints[:, i] = v_int
            vals[i] = self.fitness(list(v_int))
            self.eval_count += 1
            # update best
            if vals[i] < self.best_val:
                self.best_val = float(vals[i])
                self.best_int = v_int.copy()
        return vals, ints

    def _update_cma(self, X, Y, vals):
        # X, Y are N x lam arrays; vals is length lam
        idx = np.argsort(vals)  # ascending (minimization)
        X_sorted = X[:, idx]
        Y_sorted = Y[:, idx]
        # recombination for mean
        old_m = self.m.copy()
        weighted = np.zeros_like(self.m)
        for i in range(self.mu):
            weighted += self.w[i] * (X_sorted[:, i] - old_m)
        self.m = old_m + self.c_m * weighted
        # evolution paths
        # ps
        # compute expected norm: E||N(0,I)|| = sqrt(N)*(1 - 1/(4N) + ...)
        expected_norm = math.sqrt(self.N) * (
            1.0 - 1.0 / (4.0 * self.N) + 1.0 / (21.0 * (self.N**2))
        )
        y_w = np.zeros(self.N)
        for i in range(self.mu):
            y_w += self.w[i] * Y_sorted[:, i]
        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + math.sqrt(
            self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff
        ) * (self.B @ (np.linalg.solve(np.diag(self.D), (self.B.T @ y_w))))
        # h_sigma
        norm_p_sigma = np.linalg.norm(self.p_sigma)
        threshold = (1.4 + 2.0 / (self.N + 1.0)) * expected_norm
        h_sigma = 1.0 if norm_p_sigma < threshold else 0.0
        # p_c
        self.p_c = (1.0 - self.c_c) * self.p_c + h_sigma * math.sqrt(
            self.c_c * (2.0 - self.c_c) * self.mu_eff
        ) * y_w
        # covariance update
        # w◦i adaptation: if w_i >= 0 -> use w_i; else scale by N / ||C^{-1/2} y||^2
        rank_mu = np.zeros_like(self.C)
        for i in range(self.mu):
            yi = Y_sorted[:, i].reshape(-1, 1)
            w_i = self.w[i]
            rank_mu += w_i * (yi @ yi.T)
        rank_one = np.outer(self.p_c, self.p_c)
        # compute scaling for negative weights (rare here since weights positive after truncation)
        self.C = (
            (1.0 - self.c1 - self.c_mu) * self.C
            + self.c1 * rank_one
            + self.c_mu * rank_mu
        )
        # step-size
        self.sigma *= math.exp(
            (self.c_sigma / self.d_sigma)
            * ((np.linalg.norm(self.p_sigma) / expected_norm) - 1.0)
        )

    # ---------- margin corrections ----------
    def _apply_margin_corrections(self):
        """
        Apply margin corrections dimension-wise to self.m and self.A_diag.
        Follows Section 4.3 and 4.4 of the paper. :contentReference[oaicite:5]{index=5}
        """
        sigma = self.sigma
        alpha = self.alpha
        # for convenience, get diagonal C_jj
        C_diag = np.diag(self.C).copy()
        for j in range(self.N):
            z_j = np.asarray(self.discrete_values[j], dtype=float)
            if z_j.size == 0:
                continue
            if z_j.size == 1:
                # trivial
                continue
            m_j = self.m[j]
            a_j = self.A_diag[j]
            # thresholds
            th = build_thresholds(z_j)
            # If binary-like variable (only 2 values), apply binary margin (eq 13)
            if z_j.size == 2:
                # binary threshold is midpoint th[0]
                l_closest = th[0]
                # compute CI_j(1-2alpha)
                chi2_q = chi2_ppf_df1(1.0 - 2.0 * alpha)  # chi2 ppf for df=1
                var_j = (a_j**2) * C_diag[j] * (sigma**2)
                CI = math.sqrt(max(0.0, chi2_q * var_j))
                # eq (13)
                diff = m_j - l_closest
                new_mj = l_closest + math.copysign(min(abs(diff), CI), diff)
                self.m[j] = new_mj
                # eq (14): A unchanged
                continue
            # For integer variables with >=3 discrete values:
            # Check edge cases: mean outside first/last threshold => use binary-like correction toward nearest edge
            l_closest, l_low, l_up = find_neighbor_thresholds(m_j, z_j)
            first_thresh = th[0]
            last_thresh = th[-1]
            if (l_low is None and m_j <= first_thresh) or (
                l_up is None and m_j > last_thresh
            ):
                # use binary-like correction toward nearest threshold (eq 13)
                # find encoding threshold closest to m_j (choose first or last as appropriate)
                if m_j <= first_thresh:
                    nearest = first_thresh
                else:
                    nearest = last_thresh
                chi2_q = chi2_ppf_df1(1.0 - 2.0 * alpha)
                var_j = (a_j**2) * C_diag[j] * (sigma**2)
                CI = math.sqrt(max(0.0, chi2_q * var_j))
                diff = m_j - nearest
                new_mj = nearest + math.copysign(min(abs(diff), CI), diff)
                self.m[j] = new_mj
                # A unchanged
                continue
            # Otherwise we are in interior threshold region; apply integer margin (eqs 17-24).
            # compute probabilities p_low = Pr( v_j <= l_low ), p_up = Pr( v_j > l_up ), p_mid = the rest.
            assert l_low is not None and l_up is not None
            var_marginal = (a_j**2) * C_diag[j] * (sigma**2)  # var of v_j
            std = math.sqrt(max(var_marginal, 1e-20))
            p_low = norm_cdf((l_low - m_j) / std)
            p_up = 1.0 - norm_cdf((l_up - m_j) / std)
            p_mid = 1.0 - p_low - p_up
            # restrict p_low, p_up
            p_low_p = max(alpha / 2.0, p_low)
            p_up_p = max(alpha / 2.0, p_up)
            denom = (p_low_p + p_up_p + p_mid) - 1.5 * alpha
            if abs(denom) < 1e-20:
                denom = 1e-20
            ppp_low_num = (1.0 - p_low_p - p_up_p - p_mid) * (p_low_p - alpha / 2.0)
            ppp_up_num = (1.0 - p_low_p - p_up_p - p_mid) * (p_up_p - alpha / 2.0)
            ppp_low = p_low_p + ppp_low_num / denom
            ppp_up = p_up_p + ppp_up_num / denom
            # ensure >= alpha/2
            ppp_low = max(ppp_low, alpha / 2.0)
            ppp_up = max(ppp_up, alpha / 2.0)
            # ppp_mid = 1 - ppp_low - ppp_up
            # Now we need to set m_j and a_j so that:
            # m_j - l_low = CI_j(1 - 2 p''_low)
            # l_up - m_j = CI_j(1 - 2 p''_up)
            # where CI_j(q) = sqrt(chi2_ppf(q) * sigma^2 * (A C A)_{jj})
            # but (A C A)_{jj} = a_j^2 * C_jj  (A diag), so CI_j = a_j * sigma * sqrt(chi2_ppf * C_jj)
            # Let s_low = sqrt(chi2_ppf(1 - 2 ppp_low)), s_up = sqrt(chi2_ppf(1 - 2 ppp_up))
            q_low = 1.0 - 2.0 * ppp_low
            q_up = 1.0 - 2.0 * ppp_up
            # numerical safety
            q_low = min(max(q_low, 1e-12), 1.0 - 1e-12)
            q_up = min(max(q_up, 1e-12), 1.0 - 1e-12)
            s_low = math.sqrt(max(0.0, chi2_ppf_df1(q_low)))
            s_up = math.sqrt(max(0.0, chi2_ppf_df1(q_up)))
            # Solve for a_j and m_j analytically:
            # (l_up - l_low) = a_j * sigma * sqrt(C_jj) * (s_up + s_low)
            denom2 = sigma * math.sqrt(max(C_diag[j], 1e-20)) * (s_up + s_low)
            if denom2 <= 1e-20:
                # cannot do anything robustly; skip correction
                continue
            a_j_new = (l_up - l_low) / denom2
            # enforce positive A
            a_j_new = max(a_j_new, 1e-12)
            # compute new m_j using first eq: m_j = l_low + a_j * sigma * sqrt(C_jj) * s_low
            m_j_new = l_low + a_j_new * sigma * math.sqrt(max(C_diag[j], 1e-20)) * s_low
            # apply updates
            self.A_diag[j] = a_j_new
            self.m[j] = m_j_new

    # ---------- main run method ----------
    def run(
        self, max_gens: Optional[int] = None, tol: float = 1e-8, verbose: bool = False
    ):
        if max_gens is None:
            max_gens = 1000000
        while self.eval_count < self.max_evals and self.generation < max_gens:
            Y, X, V = self._sample_population()
            vals, ints = self._evaluate_population(V)
            # update using X and Y but ranks determined by evals of V (per paper)
            self._update_cma(X, Y, vals)
            # apply margin correction to m and A
            self._apply_margin_corrections()
            self.generation += 1
            # optional verbose
            if verbose and (self.generation % 10 == 0):
                print(
                    f"gen {self.generation:4d} evals {self.eval_count:6d} best {self.best_val:.6g}"
                )
            # simple stopping: if sigma gets extremely tiny or best is zero
            if self.sigma < 1e-12:
                break
            if abs(self.best_val) < tol:
                break
        return {
            "best_int": self.best_int,
            "best_val": self.best_val,
            "evals": self.eval_count,
            "generations": self.generation,
        }


def run_cma_es_optimization(gird, pedestrian_confs, simulator_config, iea_config):
    psi = FitnessEvaluator(gird, pedestrian_confs, simulator_config)

    # Dummy example: minimize a toy objective: sum((pos-100)**2) but some positions invalid (simulate furniture)
    N = simulator_config.numEmergencyExits
    # define allowed discrete positions for each dim (simulate invalid points)
    discrete_values = []
    for j in range(N):
        allowed = list(range(0, 401))
        # mark some forbidden positions: e.g., every 37th index forbidden
        allowed = [x for x in allowed]
        discrete_values.append(allowed)

    cma = CMAESWithMargin(
        N=N, fitness=psi.evaluate, discrete_values=discrete_values, sigma0=40.0, seed=1
    )
    res = cma.run(max_gens=200, verbose=True)
    print("RESULT:", res)
