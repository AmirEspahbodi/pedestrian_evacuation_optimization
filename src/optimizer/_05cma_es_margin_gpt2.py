from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .common import FitnessEvaluator

# ----------------------------
# Math helpers (no SciPy)
# ----------------------------


def norm_cdf(x: float) -> float:
    """Standard normal CDF Φ(x)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """
    Standard normal inverse CDF Φ^{-1}(p).
    Acklam's rational approximation. Accurate to ~1e-9 in double precision.
    """
    if not (0.0 < p < 1.0):
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError(f"p must be in (0,1); got {p}")

    # Coefficients in rational approximations
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

    # Define break-points
    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        # Rational approximation for lower region
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )

    if p > phigh:
        # Rational approximation for upper region
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    # Rational approximation for central region
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def expected_norm_chi(n: int) -> float:
    """E||N(0,I_n)|| ≈ sqrt(n) * (1 - 1/(4n) + 1/(21n^2))."""
    n = int(n)
    if n <= 0:
        return 0.0
    return math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))


# ----------------------------
# Discrete axis encoding (Encoding_f + thresholds)
# ----------------------------


@dataclass(frozen=True)
class DiscreteAxis:
    """
    Represents a 1D discrete set {z_{k}} and thresholds ℓ_{k|k+1} (midpoints).
    Encoding_f follows the paper's piecewise definition using thresholds.
    """

    values: np.ndarray  # sorted shape (K,)
    thresholds: np.ndarray  # shape (K-1,), thresholds[k] = (values[k]+values[k+1])/2

    @staticmethod
    def from_values(values: Sequence[int]) -> "DiscreteAxis":
        v = np.array(sorted(set(values)), dtype=float)
        if v.ndim != 1 or v.size < 2:
            raise ValueError("Each axis needs at least 2 discrete values.")
        thr = 0.5 * (v[:-1] + v[1:])
        return DiscreteAxis(values=v, thresholds=thr)

    def encode_scalar(self, x: float) -> float:
        """
        Encoding_f(x) for a scalar (paper's definition).
        Returns a value from self.values.
        """
        # idx = number of thresholds < x (strictly) if using right-closed intervals.
        # We want:
        #  z1 if x <= ℓ1|2
        #  zk if ℓk-1|k < x <= ℓk|k+1
        #  zK if ℓK-1|K < x
        idx = int(np.searchsorted(self.thresholds, x, side="right"))
        return float(self.values[idx])

    def low_up_thresholds_for_mean(
        self, m: float
    ) -> Tuple[Optional[float], Optional[float], str]:
        """
        For integer-margin logic:
        - edge-low:   m <= ℓ1|2
        - edge-high:  ℓK-1|K < m
        - interior:   ℓlow(m) = max{ℓ in ℓ_j : ℓ < m}, ℓup(m) = min{ℓ in ℓ_j : m <= ℓ}
        Returns (ℓlow, ℓup, case) where case in {"edge_low","edge_high","interior"}.
        """
        l = self.thresholds
        if m <= float(l[0]):
            return None, float(l[0]), "edge_low"
        if float(l[-1]) < m:
            return float(l[-1]), None, "edge_high"
        # interior
        i = int(np.searchsorted(l, m, side="left"))  # first index with l[i] >= m
        # because we're not in edge_low, i >= 1; because not in edge_high, i <= len(l)-1
        return float(l[i - 1]), float(l[i]), "interior"

    def closest_threshold(self, m: float) -> float:
        """Closest encoding threshold to mean (used in Eq. 13 style correction)."""
        l = self.thresholds
        i = int(np.argmin(np.abs(l - m)))
        return float(l[i])


# ----------------------------
# CMA-ES with Margin (integer-only)
# ----------------------------


@dataclass
class CMAESMarginConfig:
    # Domain
    k: int
    discrete_values_per_dim: Optional[List[Sequence[int]]] = (
        None  # if None -> range(bounds)
    )
    bounds: Tuple[int, int] = (0, 400)

    # Budget / termination
    max_evals: int = 5000
    stop_min_eig_sigma2C: float = 1e-30
    stop_max_cond_C: float = 1e14

    # Initialization
    mean0: Optional[Sequence[float]] = None
    sigma0: Optional[float] = None
    seed: Optional[int] = None

    # Margin hyperparameter α
    alpha: Optional[float] = None  # if None -> 1/(N*lambda) (recommended in paper)

    # Practical robustness
    penalty_value: float = 1e30  # used if user fitness crashes
    use_cache: bool = True  # deterministic fitness -> caching helps
    jitter_diag: float = 1e-14  # numerical safety for C
    min_std: float = 1e-30  # avoid division by 0 in margin math


@dataclass
class CMAESMarginResult:
    best_x: np.ndarray
    best_f: float
    evals: int
    iters: int
    history_best_f: List[float]


class CMAESWithMarginInteger:
    """
    Integer-only CMA-ES with Margin:
      - Sample y ~ N(0, C), x = m + σ y
      - Evaluate v = m + σ A y (A diagonal) -> discretize via Encoding_f -> fitness
      - Standard CMA-ES parameter update uses x,y (not v) (paper's Step 4)
      - Margin correction updates m and A using Eqs. 13–14 (edge/binary) and 15–24 (integer interior)
    """

    def __init__(self, fitness: Callable[[np.ndarray], float], cfg: CMAESMarginConfig):
        self.fitness = fitness
        self.cfg = cfg

        N = int(cfg.k)
        if N < 1:
            raise ValueError("k must be >= 1")
        self.N = N

        # Build discrete axes
        lo, hi = cfg.bounds
        if cfg.discrete_values_per_dim is None:
            values = [list(range(lo, hi + 1)) for _ in range(N)]
        else:
            if len(cfg.discrete_values_per_dim) != N:
                raise ValueError("discrete_values_per_dim must have length k")
            values = [list(v) for v in cfg.discrete_values_per_dim]

        self.axes: List[DiscreteAxis] = [DiscreteAxis.from_values(v) for v in values]

        # RNG
        self.rng = np.random.default_rng(cfg.seed)

        # Default CMA-ES hyperparameters (Table 1)
        self.lam = 4 + int(math.floor(3.0 * math.log(N)))  # λ
        self.mu = self.lam // 2  # μ

        w_prime = np.array(
            [
                math.log((self.lam + 1.0) / 2.0) - math.log(i)
                for i in range(1, self.lam + 1)
            ],
            dtype=float,
        )

        # Positive weights (i <= μ): normalize by sum of w'_i
        w_pos = w_prime[: self.mu].copy()
        w_pos /= np.sum(w_pos)

        # μ_w = 1 / sum_{i=1..μ} w_i^2  (Table 1)
        self.mu_w = 1.0 / float(np.sum(w_pos**2))

        # CMA learning rates (Table 1)
        self.c_m = 1.0
        self.c_sigma = (self.mu_w + 2.0) / (N + self.mu_w + 5.0)
        self.c_c = (4.0 + self.mu_w / N) / (N + 4.0 + 2.0 * self.mu_w / N)
        self.c1 = 2.0 / (((N + 1.3) ** 2) + self.mu_w)
        self.c_mu = min(
            1.0 - self.c1,
            2.0 * (self.mu_w - 2.0 + 1.0 / self.mu_w) / (((N + 2.0) ** 2) + self.mu_w),
        )
        self.d_sigma = (
            1.0
            + self.c_sigma
            + 2.0 * max(0.0, math.sqrt((self.mu_w - 1.0) / (N + 1.0)) - 1.0)
        )

        # Negative weights (i > μ): Table 1 scaling via min(...)
        w_neg_raw = w_prime[self.mu :].copy()  # negative values
        denom = float(np.sum(np.abs(w_neg_raw)))
        w_neg = (w_neg_raw / denom) if denom > 0 else np.zeros_like(w_neg_raw)

        # μ_w^- = (sum w'_i)^2 / sum (w'_i)^2  for i>μ (Table 1)
        num = float(np.sum(w_neg_raw))
        den = float(np.sum(w_neg_raw**2))
        self.mu_w_minus = (num * num / den) if den > 0 else 0.0

        # α^- = min(1 + c1/cμ, 1 + 2 μ_w^-/(μ_w+2), (1-c1-cμ)/(N cμ))  (Table 1)
        if self.c_mu > 0:
            alpha_minus = min(
                1.0 + self.c1 / self.c_mu,
                1.0 + 2.0 * self.mu_w_minus / (self.mu_w + 2.0),
                (1.0 - self.c1 - self.c_mu) / (N * self.c_mu),
            )
        else:
            alpha_minus = 1.0

        w_neg *= alpha_minus

        # Full weights for covariance update (positive + negative)
        self.w = np.concatenate([w_pos, w_neg], axis=0)
        self.w_pos = w_pos

        # Default margin α recommended ~ 1/(N*λ)
        self.alpha = cfg.alpha if cfg.alpha is not None else 1.0 / (self.N * self.lam)

        # State: m, C, σ, evolution paths, A (diagonal)
        if cfg.mean0 is None:
            # Middle of (lo,hi) in each dim
            mid = 0.5 * (lo + hi)
            self.m = np.full((N,), float(mid), dtype=float)
        else:
            if len(cfg.mean0) != N:
                raise ValueError("mean0 must have length k")
            self.m = np.array(cfg.mean0, dtype=float)

        if cfg.sigma0 is None:
            # A reasonable default scale for [0,400]
            self.sigma = 0.3 * (hi - lo)  # 120 for [0,400]
        else:
            self.sigma = float(cfg.sigma0)

        self.C = np.eye(N, dtype=float)
        self.p_sigma = np.zeros((N,), dtype=float)
        self.p_c = np.zeros((N,), dtype=float)
        self.A_diag = np.ones((N,), dtype=float)  # A(0)=I

        # Bookkeeping
        self.evals = 0
        self.iters = 0
        self.best_x: Optional[np.ndarray] = None
        self.best_f: float = math.inf
        self.history_best_f: List[float] = []

        self.cache: Dict[Tuple[int, ...], float] = {}

    def _encode_vector(self, v: np.ndarray) -> np.ndarray:
        out = np.empty((self.N,), dtype=int)
        for j in range(self.N):
            out[j] = int(round(self.axes[j].encode_scalar(float(v[j]))))
        return out

    def _safe_fitness(self, x_int: np.ndarray) -> float:
        # Hard guarantee: within user bounds (avoid crashes)
        lo, hi = self.cfg.bounds
        if np.any(x_int < lo) or np.any(x_int > hi):
            return self.cfg.penalty_value

        key = tuple(int(t) for t in x_int)
        if self.cfg.use_cache and key in self.cache:
            return self.cache[key]

        try:
            val = float(self.fitness(x_int))
        except Exception:
            val = float(self.cfg.penalty_value)

        if self.cfg.use_cache:
            self.cache[key] = val
        return val

    def _eigendecomp_C(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (B, D, sqrtC, invsqrtC) where:
          C = B diag(D^2) B^T, D = sqrt(eigvals)
        """
        # Symmetrize to reduce numerical drift
        C = 0.5 * (self.C + self.C.T)
        # Add tiny diagonal jitter to help SPD
        C = C + self.cfg.jitter_diag * np.eye(self.N)
        eigvals, B = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, self.cfg.min_std)
        D = np.sqrt(eigvals)
        sqrtC = B @ (D[:, None] * B.T)
        invsqrtC = B @ ((1.0 / D)[:, None] * B.T)
        self.C = C
        return B, D, sqrtC, invsqrtC

    def _margin_correction_integer_only(self) -> None:
        """
        Applies the paper's margin correction after the CMA-ES update:
          - edge cases use Eq. (13)-(14) with margin α (or α for 'inner neighbor')
          - interior integer case uses Eqs. (15)-(24), with margin α/2 outside plateau
        Updates self.m and self.A_diag in-place.
        """
        N = self.N
        alpha = self.alpha
        lo, hi = self.cfg.bounds

        for j in range(N):
            axis = self.axes[j]
            Kj = axis.values.size

            # Marginal std for v_j under N(m, σ^2 A C A^T): std = σ * a_j * sqrt(C_jj)
            Cjj = float(self.C[j, j])
            base = self.sigma * self.A_diag[j] * math.sqrt(max(Cjj, self.cfg.min_std))
            base = max(base, self.cfg.min_std)

            if Kj == 2:
                # Binary variable: enforce min(Pr(v<ℓ), Pr(v>=ℓ)) >= α via Eq. (13)-(14)
                ell = float(axis.thresholds[0])
                CI = norm_ppf(1.0 - alpha) * base  # CI_j(1-2α) = z_{1-α} * std
                delta = float(self.m[j]) - ell
                if abs(delta) > CI:
                    self.m[j] = ell + math.copysign(CI, delta)
                # Eq (14): A unchanged
                continue

            # Integer variable with Kj >= 3:
            ell_low, ell_up, case = axis.low_up_thresholds_for_mean(float(self.m[j]))

            if case in ("edge_low", "edge_high"):
                # Paper: when m <= ℓ_{1|2} or ℓ_{K-1|K} < m,
                # correct probability of generating one inner integer above α using Eq. (13)-(14).
                ell = float(axis.closest_threshold(float(self.m[j])))
                CI = norm_ppf(1.0 - alpha) * base
                delta = float(self.m[j]) - ell
                if abs(delta) > CI:
                    self.m[j] = ell + math.copysign(CI, delta)
                # Eq (14): A unchanged
                continue

            # Interior case: use Eqs. (15)-(24)
            assert ell_low is not None and ell_up is not None
            m_j = float(self.m[j])
            std = base

            # (17)(18)(19): p_low, p_up, p_mid
            p_low = norm_cdf((ell_low - m_j) / std)
            p_up = 1.0 - norm_cdf((ell_up - m_j) / std)
            p_mid = 1.0 - p_low - p_up

            # Numerical clamp
            p_low = min(max(p_low, 0.0), 1.0)
            p_up = min(max(p_up, 0.0), 1.0)
            p_mid = min(max(p_mid, 0.0), 1.0)

            # (20)(21): p'_low, p'_up with margin α/2
            p_low_p = max(alpha / 2.0, p_low)
            p_up_p = max(alpha / 2.0, p_up)

            # (22)(23): adjust to keep sum 1 while preserving lower bounds
            denom = p_low_p + p_up_p + p_mid - 3.0 * alpha / 2.0
            if denom <= 0:
                # Degenerate (should not happen for reasonable α); fall back to simple clamping.
                p_low_pp, p_up_pp = p_low_p, p_up_p
            else:
                factor = (1.0 - p_low_p - p_up_p - p_mid) / denom
                p_low_pp = p_low_p + factor * (p_low_p - alpha / 2.0)
                p_up_pp = p_up_p + factor * (p_up_p - alpha / 2.0)

            # Guardrails
            p_low_pp = max(alpha / 2.0, min(0.5, p_low_pp))
            p_up_pp = max(alpha / 2.0, min(0.5, p_up_pp))

            # (24): solve simultaneous equations for updated m_j and A_j (diagonal)
            # m - ℓlow = CI(1-2 p''low) = z_{1-p''low} * σ * a * sqrt(Cjj)
            # ℓup - m = CI(1-2 p''up)  = z_{1-p''up}  * σ * a * sqrt(Cjj)
            z_low = norm_ppf(1.0 - p_low_pp)
            z_up = norm_ppf(1.0 - p_up_pp)
            width = float(ell_up - ell_low)

            # s = σ * a * sqrt(Cjj)
            s = width / max(z_low + z_up, self.cfg.min_std)
            m_new = ell_low + z_low * s
            denom_a = self.sigma * math.sqrt(max(Cjj, self.cfg.min_std))
            a_new = s / max(denom_a, self.cfg.min_std)

            self.m[j] = m_new
            self.A_diag[j] = max(a_new, self.cfg.min_std)

        # (Optional sanity) Keep mean from drifting to crazy magnitudes.
        # Encoding_f will still keep evaluations feasible, but this can stabilize numerics.
        # We keep it very loose to not fight the intended behavior.
        self.m = np.clip(self.m, lo - 2.0 * (hi - lo), hi + 2.0 * (hi - lo))

    def step(self) -> None:
        """Run one CMA-ES-with-margin iteration."""
        B, D, sqrtC, invsqrtC = self._eigendecomp_C()
        chiN = expected_norm_chi(self.N)

        # Step 1: sample y ~ N(0,C), x = m + σ y
        Z = self.rng.standard_normal(size=(self.N, self.lam))
        Y = (sqrtC @ Z).T  # shape (λ, N)
        X = self.m[None, :] + self.sigma * Y

        # Step 2: affine transform for evaluation only: v = m + σ A y (A diagonal)
        V = self.m[None, :] + self.sigma * (Y * self.A_diag[None, :])

        # Step 3: discretize v -> vbar and evaluate + rank by f(vbar)
        Vbar = np.empty((self.lam, self.N), dtype=int)
        fvals = np.empty((self.lam,), dtype=float)
        for i in range(self.lam):
            vb = self._encode_vector(V[i])
            Vbar[i] = vb
            fvals[i] = self._safe_fitness(vb)
        self.evals += self.lam

        order = np.argsort(fvals)  # ascending (minimization)
        Y_sorted = Y[order]
        X_sorted = X[order]
        Vbar_sorted = Vbar[order]
        f_sorted = fvals[order]

        # Track best
        if f_sorted[0] < self.best_f:
            self.best_f = float(f_sorted[0])
            self.best_x = Vbar_sorted[0].copy()
        self.history_best_f.append(self.best_f)

        # Step 4: standard CMA-ES updates based on x,y (paper says unchanged)
        # Mean update (Eq. 3): m <- m + c_m * sum_{i=1..μ} w_i (x_i - m)
        y_w = np.sum(self.w_pos[:, None] * Y_sorted[: self.mu], axis=0)
        self.m = self.m + self.c_m * self.sigma * y_w

        # Evolution path p_sigma (Eq. 4)
        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + math.sqrt(
            self.c_sigma * (2.0 - self.c_sigma) * self.mu_w
        ) * (invsqrtC @ y_w)

        # h_sigma (indicator in Eq. 5)
        norm_ps = float(np.linalg.norm(self.p_sigma))
        lhs = norm_ps
        rhs = (
            math.sqrt(1.0 - (1.0 - self.c_sigma) ** (2.0 * (self.iters + 1)))
            * (1.4 + 2.0 / (self.N + 1.0))
            * chiN
        )
        h_sigma = 1.0 if lhs < rhs else 0.0

        # Evolution path p_c (Eq. 5)
        self.p_c = (1.0 - self.c_c) * self.p_c + h_sigma * math.sqrt(
            self.c_c * (2.0 - self.c_c) * self.mu_w
        ) * y_w

        # Step-size update (Eq. 7)
        self.sigma = self.sigma * math.exp(
            (self.c_sigma / self.d_sigma) * (norm_ps / chiN - 1.0)
        )

        # Covariance update (Eq. 6) with active weights w° (negative weight correction)
        invsqrtC_Y = (invsqrtC @ Y_sorted.T).T  # shape (λ, N)
        norms_sq = np.sum(invsqrtC_Y**2, axis=1)
        w_circ = np.where(
            self.w >= 0.0,
            self.w,
            self.w * (self.N / np.maximum(norms_sq, self.cfg.min_std)),
        )

        rank_mu = np.zeros_like(self.C)
        for i in range(self.lam):
            rank_mu += w_circ[i] * np.outer(Y_sorted[i], Y_sorted[i])

        c_factor = (
            1.0
            - self.c1
            - self.c_mu * float(np.sum(self.w))
            + (1.0 - h_sigma) * self.c1 * self.c_c * (2.0 - self.c_c)
        )
        self.C = (
            c_factor * self.C
            + self.c1 * np.outer(self.p_c, self.p_c)
            + self.c_mu * rank_mu
        )
        self.C = 0.5 * (self.C + self.C.T)  # symmetrize

        # Step 5: margin correction updates m and A (integer-only here)
        self._margin_correction_integer_only()

        self.iters += 1

    def should_stop(self) -> bool:
        # Termination criteria used in the paper experiments
        # (min eigenvalue of σ^2 C too small OR condition number too large)
        C = 0.5 * (self.C + self.C.T)
        eigvals = np.linalg.eigvalsh(C)
        eigvals = np.maximum(eigvals, self.cfg.min_std)
        min_eig_sigma2C = (self.sigma**2) * float(np.min(eigvals))
        cond_C = float(np.max(eigvals) / np.min(eigvals))
        return (
            (self.evals >= self.cfg.max_evals)
            or (min_eig_sigma2C < self.cfg.stop_min_eig_sigma2C)
            or (cond_C > self.cfg.stop_max_cond_C)
        )

    def run(self) -> CMAESMarginResult:
        while not self.should_stop():
            self.step()

        if self.best_x is None:
            # Should never happen because we evaluate from iteration 0.
            self.best_x = self._encode_vector(self.m)

        return CMAESMarginResult(
            best_x=self.best_x,
            best_f=self.best_f,
            evals=self.evals,
            iters=self.iters,
            history_best_f=self.history_best_f,
        )


# ----------------------------
# Convenience wrapper
# ----------------------------


def run_cma_es_optimization(
    gird,
    pedestrian_confs,
    simulator_config,
    iea_config,
    *,
    discrete_values_per_dim: Optional[List[Sequence[int]]] = None,
    max_evals: int = 5000,
    mean0: Optional[Sequence[float]] = None,
    sigma0: Optional[float] = None,
    alpha: Optional[float] = None,
    seed: Optional[int] = None,
    penalty_value: float = 1e30,
    use_cache: bool = True,
) -> CMAESMarginResult:
    fitness = FitnessEvaluator(gird, pedestrian_confs, simulator_config).evaluate
    k = simulator_config.numEmergencyExits
    bounds = (0, 2 * (len(gird) + len(gird[0])))

    cfg = CMAESMarginConfig(
        k=k,
        bounds=bounds,
        discrete_values_per_dim=discrete_values_per_dim,
        max_evals=max_evals,
        mean0=mean0,
        sigma0=sigma0,
        alpha=alpha,
        seed=seed,
        penalty_value=penalty_value,
        use_cache=use_cache,
    )
    opt = CMAESWithMarginInteger(fitness, cfg)
    return opt.run()
