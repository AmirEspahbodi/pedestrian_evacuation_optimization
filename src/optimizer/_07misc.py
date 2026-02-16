from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import FitnessEvaluator

# ============================================================
# Utilities: Latin Hypercube (symmetric) + distance helpers
# ============================================================


def _symmetric_lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """
    Symmetric Latin Hypercube in [0,1]^d with n points (n must be even).
    Mirrors the first n/2 points: x -> 1-x (per dimension), then shuffles.
    This matches the paper's symmetric LHD idea used for initialization. :contentReference[oaicite:1]{index=1}
    """
    if n % 2 != 0:
        raise ValueError("Symmetric LHS requires an even number of points n.")
    half = n // 2

    # Standard LHS for half points
    X = np.empty((half, d), dtype=float)
    for j in range(d):
        perm = rng.permutation(half)
        u = rng.random(half)
        X[:, j] = (perm + u) / half  # in (0,1)

    X_mirror = 1.0 - X
    X_full = np.vstack([X, X_mirror])
    rng.shuffle(X_full, axis=0)
    return X_full


def _min_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    For each row in A, compute min Euclidean distance to rows in B.
    A: (m,d), B: (n,d)
    returns: (m,)
    """
    if B.shape[0] == 0:
        return np.full((A.shape[0],), np.inf, dtype=float)
    # squared distances
    diff = A[:, None, :] - B[None, :, :]
    dsq = np.sum(diff * diff, axis=2)
    return np.sqrt(np.min(dsq, axis=1))


def _scale_to_unit_interval(v: np.ndarray) -> np.ndarray:
    """
    Scale array v to [0,1]. If constant, return zeros.
    """
    vmin = np.min(v)
    vmax = np.max(v)
    if np.isclose(vmax, vmin):
        return np.zeros_like(v, dtype=float)
    return (v - vmin) / (vmax - vmin)


# ============================================================
# Cubic RBF surrogate with linear tail (Eq. 4-6) + bumpiness ln(z)
# ============================================================


class CubicRBF:
    """
    Cubic RBF interpolant with linear tail:
        s(z) = sum_i k_i * ||z - z_i||^3 + (a + b^T z)
    Fit by solving Eq. (5)-(6). :contentReference[oaicite:2]{index=2}

    Also provides bumpiness ln(z) used by the target-value strategy
    (Eq. 12-16) via a fast Schur-complement identity derived from Eq. (13)-(14).
    :contentReference[oaicite:3]{index=3}
    """

    def __init__(self, nugget: float = 1e-12):
        self.nugget = float(nugget)
        self.X: Optional[np.ndarray] = None  # (n,d)
        self.y: Optional[np.ndarray] = None  # (n,)
        self.k: Optional[np.ndarray] = None  # (n,)
        self.b: Optional[np.ndarray] = None  # (d,)
        self.a: Optional[float] = None

        # For bumpiness ln(z): store A and lazily-computed A_inv where
        # A = [[U,P],[P^T,0]] for current design (old points only).
        self._A: Optional[np.ndarray] = None
        self._A_inv: Optional[np.ndarray] = None

    @staticmethod
    def _phi(r: np.ndarray) -> np.ndarray:
        return r**3

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("X must be (n,d) and y must be (n,) with matching n.")
        n, d = X.shape
        if n < d + 1:
            raise ValueError(
                f"Need at least d+1 points to fit linear tail (have n={n}, d={d})."
            )

        # Build U and P (Eq. 5-6) :contentReference[oaicite:4]{index=4}
        D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        U = self._phi(D)
        if self.nugget > 0:
            U = U + np.eye(n) * self.nugget

        P = np.hstack([X, np.ones((n, 1), dtype=float)])  # (n, d+1)

        A_top = np.hstack([U, P])
        A_bot = np.hstack([P.T, np.zeros((d + 1, d + 1), dtype=float)])
        A = np.vstack([A_top, A_bot])

        rhs = np.concatenate([y, np.zeros(d + 1, dtype=float)])

        # Solve for [k; c] where c = [b; a]  :contentReference[oaicite:5]{index=5}
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            # robust fallback: tiny ridge on bottom-right can help degeneracy
            ridge = 1e-10
            A2 = A.copy()
            A2[n:, n:] += np.eye(d + 1) * ridge
            sol = np.linalg.lstsq(A2, rhs, rcond=None)[0]

        k = sol[:n]
        c = sol[n:]
        b = c[:d]
        a = float(c[d])

        self.X, self.y = X, y
        self.k, self.b, self.a = k, b, a

        # cache for ln(z)
        self._A = A
        self._A_inv = None  # compute lazily when needed

    def predict(self, Xq: np.ndarray) -> np.ndarray:
        if self.X is None or self.k is None or self.b is None or self.a is None:
            raise RuntimeError("Model is not fit.")
        Xq = np.asarray(Xq, dtype=float)
        if Xq.ndim == 1:
            Xq = Xq[None, :]
        # phi(||x - x_i||)
        D = np.linalg.norm(Xq[:, None, :] - self.X[None, :, :], axis=2)  # (m,n)
        Phi = self._phi(D)  # (m,n)
        return Phi @ self.k + (Xq @ self.b + self.a)

    def _ensure_A_inv(self) -> np.ndarray:
        if self._A is None:
            raise RuntimeError("Model is not fit.")
        if self._A_inv is None:
            # Invert once; then ln(z) evaluations are O(m^2) via BLAS
            try:
                self._A_inv = np.linalg.inv(self._A)
            except np.linalg.LinAlgError:
                # fallback: pseudo-inverse if near-singular
                self._A_inv = np.linalg.pinv(self._A, rcond=1e-12)
        return self._A_inv

    def ln_bumpiness(self, Xcand: np.ndarray) -> np.ndarray:
        """
        Compute ln(z) for candidate points z using the Schur-complement identity
        equivalent to solving Eq. (13)-(14) and extracting the (n+1)th component. :contentReference[oaicite:6]{index=6}

        Fast formula:
            ln(z) = -1 / (b(z)^T A^{-1} b(z))
        where
            b(z) = [phi_z ; p(z)] with phi_z_i = ||z - z_i||^3 and p(z) = [z; 1].
        """
        if self.X is None:
            raise RuntimeError("Model is not fit.")
        Xcand = np.asarray(Xcand, dtype=float)
        if Xcand.ndim == 1:
            Xcand = Xcand[None, :]

        Ainv = self._ensure_A_inv()
        X = self.X
        n, d = X.shape

        D = np.linalg.norm(Xcand[:, None, :] - X[None, :, :], axis=2)  # (m,n)
        phi_z = self._phi(D)  # (m,n)
        pz = np.hstack([Xcand, np.ones((Xcand.shape[0], 1), dtype=float)])  # (m,d+1)
        bvec = np.hstack([phi_z, pz])  # (m, n+d+1)

        # denom_i = b_i^T Ainv b_i
        # vectorized: denom = sum(b * (Ainv @ b^T)^T, axis=1)
        tmp = (Ainv @ bvec.T).T  # (m, n+d+1)
        denom = np.sum(bvec * tmp, axis=1)

        # Numerical safety
        eps = 1e-18
        denom = np.where(np.abs(denom) < eps, np.sign(denom) * eps + eps, denom)
        return -1.0 / denom


# ============================================================
# Mixed-Integer GA subsolver (used for t-step auxiliary problems)
# ============================================================


@dataclass
class GAConfig:
    pop_size: int = 60
    generations: int = 60
    tournament_k: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.25
    mutation_sigma: float = 0.15  # as fraction of variable range
    elite: int = 2


def _tournament_select(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    idx = rng.integers(0, len(fitness), size=k)
    return idx[np.argmin(fitness[idx])]


def _uniform_crossover(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    mask = rng.random(a.shape[0]) < 0.5
    c1 = a.copy()
    c2 = b.copy()
    c1[mask] = b[mask]
    c2[mask] = a[mask]
    return c1, c2


# ============================================================
# MISO-CPTV(+local) for PURE INTEGER VECTORS
# ============================================================


@dataclass
class MISOConfig:
    nmax: int = 300  # total expensive eval budget
    n0: Optional[int] = 7

    # c-step parameters (paper defaults) :contentReference[oaicite:8]{index=8}
    Tc_s: int = 1
    Tc_f: Optional[int] = None  # default max(5,d)
    Tc_r: int = 5
    W: Tuple[float, ...] = (0.3, 0.5, 0.8, 0.95)
    N_candidates: Optional[int] = None  # default min(500*d, 5000)
    r0_frac: float = 0.2
    r_min_frac: float = 2**-6  # rl = 2^-6 r0 :contentReference[oaicite:9]{index=9}

    # t-step parameters (paper defaults) :contentReference[oaicite:10]{index=10}
    P: int = 10  # pattern G = {0,1,...,P,P+1}
    delta: float = (
        0.5  # "points closer than delta considered equal" (integer => 0.5 is safe)
    )

    # local search (memetic) - adapted for pure integer
    enable_local_search: bool = True
    local_max_evals: int = 40

    # numerical
    penalty_value: float = 1e12  # assigned to invalid/crashed evaluations


class MISOIntegerOptimizer:
    """
    Implements MISO-CPTV plus a memetic local search phase, adapted to PURE INTEGER problems.
    Core logic follows Müller (2016), Algorithms 3-5, with:
      - c-step: coordinate perturbation + weighted surrogate/distance scores (Alg. 4) :contentReference[oaicite:11]{index=11}
      - t-step: target value strategy minimizing ln(z)*(s(z)-t)^2 etc. (Alg. 5, Eq. 12-16) :contentReference[oaicite:12]{index=12}
    """

    def __init__(
        self,
        gird,
        pedestrian_confs,
        simulator_config,
        iea_config,
        config: Optional[MISOConfig] = None,
        ga_config: Optional[GAConfig] = None,
        seed: Optional[int] = None,
        is_valid_fn: Optional[Callable[[np.ndarray], bool]] = None,
        repair_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.fitness_fn = FitnessEvaluator(
            gird, pedestrian_confs, simulator_config
        ).evaluate

        self.k = int(simulator_config.numEmergencyExits)
        bounds = (0, 2 * (len(gird) + len(gird[0])))
        lower = [bounds[0]] * self.k
        upper = [bounds[1]] * self.k
        self.lower = np.asarray(lower, dtype=int)
        self.upper = np.asarray(upper, dtype=int)
        if self.lower.shape != (self.k,) or self.upper.shape != (self.k,):
            raise ValueError("lower/upper must be length-k.")
        if np.any(self.lower > self.upper):
            raise ValueError("lower must be <= upper.")
        if np.any(self.lower < 0) or np.any(self.upper > 400):
            # You said your simulator crashes out of 0..400; enforce that.
            raise ValueError(
                "Bounds must be within [0,400] to avoid simulator crashes."
            )

        self.cfg = config or MISOConfig()
        self.ga_cfg = ga_config or GAConfig()
        self.rng = np.random.default_rng(seed)
        self.is_valid_fn = is_valid_fn
        self.repair_fn = repair_fn

        # storage
        self.X: List[np.ndarray] = []
        self.y: List[float] = []
        self.valid: List[bool] = []
        self._cache: Dict[Tuple[int, ...], Tuple[float, bool]] = {}

        self.model = CubicRBF(nugget=1e-12)

    # -----------------------------
    # Safe evaluation with caching
    # -----------------------------
    def _project_int_bounds(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=int).copy()
        x = np.clip(x, self.lower, self.upper)
        return x

    def _maybe_repair(self, x: np.ndarray) -> np.ndarray:
        if self.repair_fn is None:
            return x
        xr = self.repair_fn(x.copy())
        xr = np.asarray(xr, dtype=int)
        return self._project_int_bounds(xr)

    def _evaluate(self, x: np.ndarray) -> Tuple[float, bool]:
        """
        Returns (objective, is_valid).
        Invalid includes: is_valid_fn returns False OR fitness_fn crashes.
        """
        x = self._project_int_bounds(x)
        x = self._maybe_repair(x)
        key = tuple(int(v) for v in x)

        if key in self._cache:
            return self._cache[key]

        if self.is_valid_fn is not None:
            try:
                ok = bool(self.is_valid_fn(x))
            except Exception:
                ok = False
            if not ok:
                val = float(self.cfg.penalty_value)
                self._cache[key] = (val, False)
                return val, False

        try:
            val = float(self.fitness_fn(x))
            self._cache[key] = (val, True)
            return val, True
        except Exception:
            val = float(self.cfg.penalty_value)
            self._cache[key] = (val, False)
            return val, False

    # -----------------------------
    # Initialization (Alg. 3, step 4) :contentReference[oaicite:13]{index=13}
    # -----------------------------
    def _initial_design(self, n0: int) -> None:
        d = self.k
        # symmetric LHS in [0,1]^d then scale to integer bounds
        U = _symmetric_lhs(n0, d, self.rng)
        X0 = self.lower + np.round(U * (self.upper - self.lower)).astype(int)
        # ensure uniqueness as much as possible
        seen = set()
        for i in range(n0):
            x = X0[i]
            x = self._project_int_bounds(x)
            x = self._maybe_repair(x)
            key = tuple(int(v) for v in x)
            if key in seen:
                # resample random point
                x = self.rng.integers(self.lower, self.upper + 1, size=d, dtype=int)
                x = self._maybe_repair(x)
                key = tuple(int(v) for v in x)
            seen.add(key)
            val, ok = self._evaluate(x)
            self.X.append(x)
            self.y.append(val)
            self.valid.append(ok)

    # -----------------------------
    # Surrogate fit (RBF Eq. 4-6) :contentReference[oaicite:14]{index=14}
    # -----------------------------
    def _fit_surrogate(self) -> None:
        X = np.vstack(self.X).astype(float)
        y = np.asarray(self.y, dtype=float)

        # Prefer fitting to valid points (keeps surrogate meaningful when crashes happen)
        valid_mask = np.asarray(self.valid, dtype=bool)
        if np.sum(valid_mask) >= (self.k + 1):
            Xf = X[valid_mask]
            yf = y[valid_mask]
        else:
            # if too few valid points, fall back to all (penalized) points
            Xf, yf = X, y

        self.model.fit(Xf, yf)

    # -----------------------------
    # q(n) perturb probability (Eq. 9) :contentReference[oaicite:15]{index=15}
    # -----------------------------
    def _q_of_n(self, n: int, n0: int, nmax: int, d: int) -> float:
        if nmax <= n0 + 1:
            return 1.0
        base = min(20.0 / d, 1.0)
        frac = 1.0 - (math.log(n - n0 + 1.0) / math.log(nmax - n0))
        return float(np.clip(base * frac, 0.0, 1.0))

    def _weight_cycle(self, i: int) -> float:
        W = self.cfg.W
        # Eq. (10) cycles weights; simplest: W[i mod |W|] :contentReference[oaicite:16]{index=16}
        return W[(i - 1) % len(W)]  # i starts at 1 in paper counters

    def _is_far_enough(self, x: np.ndarray, X_seen: np.ndarray, delta: float) -> bool:
        if X_seen.shape[0] == 0:
            return True
        d = np.min(np.linalg.norm(X_seen - x[None, :], axis=1))
        return bool(d > delta)

    def _random_point_far(self, X_seen: np.ndarray, delta: float) -> np.ndarray:
        d = self.k
        for _ in range(5000):
            x = self.rng.integers(self.lower, self.upper + 1, size=d, dtype=int)
            x = self._maybe_repair(x)
            if self._is_far_enough(x.astype(float), X_seen.astype(float), delta):
                return x
        # fallback (accept even if close)
        return self._maybe_repair(
            self.rng.integers(self.lower, self.upper + 1, size=d, dtype=int)
        )

    # ============================================================
    # c-step (Algorithm 4) :contentReference[oaicite:17]{index=17}
    # ============================================================
    def _c_step_propose(
        self,
        zbest: np.ndarray,
        n: int,
        n0: int,
        nmax: int,
        r: float,
        c_iter: int,
    ) -> np.ndarray:
        d = self.k
        qn = self._q_of_n(n, n0, nmax, d)

        N = self.cfg.N_candidates or min(
            500 * d, 5000
        )  # paper default :contentReference[oaicite:18]{index=18}
        Z = np.vstack(self.X).astype(float)

        candidates = np.tile(zbest[None, :], (N, 1)).astype(int)

        # For each candidate and dimension, decide whether to perturb
        perturb_mask = self.rng.random((N, d)) < qn
        # ensure at least one variable perturbed per candidate
        none = np.where(~perturb_mask.any(axis=1))[0]
        if none.size > 0:
            j = self.rng.integers(0, d, size=none.size)
            perturb_mask[none, j] = True

        # Apply integer perturbations (Eq. 7-8 integer case) :contentReference[oaicite:19]{index=19}
        q = self.rng.normal(0.0, 1.0, size=(N, d))
        steps = np.sign(q) * np.maximum(1.0, np.round(np.abs(r * q)))
        steps = steps.astype(int)
        candidates = candidates + (perturb_mask * steps)

        # clip + optional repair
        candidates = np.clip(candidates, self.lower, self.upper)
        for i in range(N):
            candidates[i] = self._maybe_repair(candidates[i])

        # score candidates: surrogate score Ss + distance score Sd (Alg.4 steps 7-9) :contentReference[oaicite:20]{index=20}
        s_pred = self.model.predict(candidates.astype(float))
        Ss = _scale_to_unit_interval(s_pred)  # lower better => closer to 0

        dist = _min_distances(candidates.astype(float), Z)
        # paper: far => Sd close to 0, near => Sd close to 1
        Sd = 1.0 - _scale_to_unit_interval(dist)

        wd = self._weight_cycle(c_iter)
        ws = 1.0 - wd
        St = ws * Ss + wd * Sd

        # avoid re-evaluating exact duplicates if possible
        seen = set(tuple(int(v) for v in x) for x in self.X)
        order = np.argsort(St)
        for idx in order:
            key = tuple(int(v) for v in candidates[idx])
            if key not in seen:
                return candidates[idx]

        # fallback: random far point
        return self._random_point_far(np.vstack(self.X), self.cfg.delta)

    # ============================================================
    # t-step (Algorithm 5) + GA subsolver for (12), (15), (16) :contentReference[oaicite:21]{index=21}
    # ============================================================
    def _ga_solve(
        self,
        obj: Callable[[np.ndarray], np.ndarray],
        seed_points: List[np.ndarray],
    ) -> np.ndarray:
        """
        Solve min obj(x) over integer box using a simple MI-GA.
        obj must accept (m,d) and return (m,).
        """
        d = self.k
        ga = self.ga_cfg

        pop = self.rng.integers(
            self.lower, self.upper + 1, size=(ga.pop_size, d), dtype=int
        )
        for i in range(pop.shape[0]):
            pop[i] = self._maybe_repair(pop[i])

        # inject seeds
        si = 0
        for sp in seed_points:
            if si >= ga.pop_size:
                break
            pop[si] = self._maybe_repair(self._project_int_bounds(sp))
            si += 1

        best_x = pop[0].copy()
        best_f = float("inf")

        for _ in range(ga.generations):
            fvals = obj(pop.astype(float))
            # update best
            j = int(np.argmin(fvals))
            if float(fvals[j]) < best_f:
                best_f = float(fvals[j])
                best_x = pop[j].copy()

            # elitism
            elite_idx = np.argsort(fvals)[: ga.elite]
            new_pop = [pop[i].copy() for i in elite_idx]

            # breeding
            while len(new_pop) < ga.pop_size:
                p1 = pop[_tournament_select(fvals, ga.tournament_k, self.rng)]
                p2 = pop[_tournament_select(fvals, ga.tournament_k, self.rng)]
                c1, c2 = p1.copy(), p2.copy()
                if self.rng.random() < ga.crossover_rate:
                    c1, c2 = _uniform_crossover(c1, c2, self.rng)

                for child in (c1, c2):
                    if self.rng.random() < ga.mutation_rate:
                        # gaussian-like integer step scaled by range
                        span = (self.upper - self.lower).astype(float)
                        sig = ga.mutation_sigma * span
                        step = np.round(self.rng.normal(0.0, sig)).astype(int)
                        # mutate a random subset of coordinates
                        mask = self.rng.random(d) < 0.5
                        if not mask.any():
                            mask[self.rng.integers(0, d)] = True
                        child[mask] += step[mask]
                    child[:] = np.clip(child, self.lower, self.upper)
                    child[:] = self._maybe_repair(child)
                    new_pop.append(child)
                    if len(new_pop) >= ga.pop_size:
                        break

            pop = np.vstack(new_pop[: ga.pop_size])

        return best_x

    def _t_step_propose(
        self,
        fbest: float,
        zbest: np.ndarray,
        Ct_i: int,
    ) -> np.ndarray:
        """
        Implements Algorithm 5 logic (g in G={0..P,P+1}) and solves the auxiliary problems
        via MI-GA (paper's MI-GA subsolver). :contentReference[oaicite:22]{index=22}
        """
        P = self.cfg.P
        G = list(range(0, P + 1)) + [P + 1]
        g = G[
            (Ct_i - 1) % len(G)
        ]  # Eq. (11) cycling :contentReference[oaicite:23]{index=23}

        X_seen = np.vstack(self.X).astype(float)
        fmax = float(np.max(np.asarray(self.y, dtype=float)))

        # Seed points for GA
        seeds = [zbest]

        if g == 0:
            # Inf-step: min ln(z)  (Eq. 12) :contentReference[oaicite:24]{index=24}
            def obj(Xcand: np.ndarray) -> np.ndarray:
                ln = self.model.ln_bumpiness(Xcand)
                # discourage duplicates/too-close
                dmin = _min_distances(Xcand, X_seen)
                penalty = np.where(dmin <= self.cfg.delta, 1e6, 0.0)
                return ln + penalty

            znew = self._ga_solve(obj, seeds)

        elif 1 <= g <= P:
            # Cycle step (global search): (15) then (16) :contentReference[oaicite:25]{index=25}
            def obj_s(Xcand: np.ndarray) -> np.ndarray:
                s = self.model.predict(Xcand)
                dmin = _min_distances(Xcand, X_seen)
                penalty = np.where(dmin <= self.cfg.delta, 1e6, 0.0)
                return s + penalty

            zs = self._ga_solve(obj_s, seeds)
            ss = float(self.model.predict(zs.astype(float))[0])

            wg = (1.0 - g / len(G)) ** 2
            t = ss - wg * (fmax - ss)  # step 10 :contentReference[oaicite:26]{index=26}

            def obj_tv(Xcand: np.ndarray) -> np.ndarray:
                s = self.model.predict(Xcand)
                ln = self.model.ln_bumpiness(Xcand)
                dmin = _min_distances(Xcand, X_seen)
                penalty = np.where(dmin <= self.cfg.delta, 1e6, 0.0)
                return ln * (s - t) ** 2 + penalty

            znew = self._ga_solve(obj_tv, seeds + [zs])

        else:
            # Cycle step (local search branch): (15) then either zs or solve (16) :contentReference[oaicite:27]{index=27}
            def obj_s(Xcand: np.ndarray) -> np.ndarray:
                s = self.model.predict(Xcand)
                dmin = _min_distances(Xcand, X_seen)
                penalty = np.where(dmin <= self.cfg.delta, 1e6, 0.0)
                return s + penalty

            zs = self._ga_solve(obj_s, seeds)
            ss = float(self.model.predict(zs.astype(float))[0])

            if ss < fbest - 1e-6 * abs(fbest):
                znew = zs
            else:
                t = fbest - 1e-2 * abs(fbest)

                def obj_tv(Xcand: np.ndarray) -> np.ndarray:
                    s = self.model.predict(Xcand)
                    ln = self.model.ln_bumpiness(Xcand)
                    dmin = _min_distances(Xcand, X_seen)
                    penalty = np.where(dmin <= self.cfg.delta, 1e6, 0.0)
                    return ln * (s - t) ** 2 + penalty

                znew = self._ga_solve(obj_tv, seeds + [zs])

        # Enforce paper's "if too close, resample random until far" (steps 22-25) :contentReference[oaicite:28]{index=28}
        if not self._is_far_enough(znew.astype(float), X_seen, self.cfg.delta):
            znew = self._random_point_far(X_seen, self.cfg.delta)

        return znew

    # ============================================================
    # Discrete local search (memetic adaptation for pure integer)
    # ============================================================
    def _local_search(
        self, zstart: np.ndarray, remaining_budget: int
    ) -> Tuple[np.ndarray, float, int]:
        """
        A small-budget discrete pattern/coordinate search on TRUE objective.
        This is the "memetic" improvement phase analogous in spirit to Algorithm 6,
        adapted because we have no continuous variables. :contentReference[oaicite:29]{index=29}
        """
        max_evals = min(self.cfg.local_max_evals, remaining_budget)
        used = 0

        zbest = zstart.copy()
        fbest, ok = self._evaluate(zbest)
        if not ok:
            return zbest, fbest, used

        # start with step ~ 5% of span, then halve
        span = self.upper - self.lower
        step = int(max(1, round(0.05 * float(np.min(span)))))
        step = max(step, 1)

        seen = set(tuple(int(c) for c in x) for x in self.X)

        while used < max_evals and step >= 1:
            improved = False
            # explore coordinate neighbors
            for j in range(self.k):
                for sgn in (-1, +1):
                    if used >= max_evals:
                        break
                    cand = zbest.copy()
                    cand[j] = int(cand[j] + sgn * step)
                    cand = self._project_int_bounds(cand)
                    cand = self._maybe_repair(cand)
                    key = tuple(int(v) for v in cand)
                    if key in seen:
                        continue
                    val, ok = self._evaluate(cand)
                    used += 1
                    seen.add(key)
                    self.X.append(cand)
                    self.y.append(val)
                    self.valid.append(ok)

                    if ok and val < fbest:
                        zbest, fbest = cand, val
                        improved = True
                        break
                if improved or used >= max_evals:
                    break
            if not improved:
                step = step // 2

        return zbest, fbest, used

    # ============================================================
    # Main optimize loop (Algorithm 3) :contentReference[oaicite:30]{index=30}
    # ============================================================
    def optimize(self) -> Dict[str, object]:
        start_time = time.perf_counter()
        d = self.k
        nmax = int(self.cfg.nmax)

        n0 = (
            self.cfg.n0 if self.cfg.n0 is not None else 2 * (d + 1)
        )  # paper default :contentReference[oaicite:31]{index=31}
        if n0 % 2 != 0:
            n0 += 1

        Tc_f = self.cfg.Tc_f if self.cfg.Tc_f is not None else max(5, d)
        Tt_f = self.cfg.P + 2  # |G| = P+2 :contentReference[oaicite:32]{index=32}

        # radii per paper: r0=0.2*l(D), rl=2^-6*r0 :contentReference[oaicite:33]{index=33}
        lD = float(np.min(self.upper - self.lower))
        r0 = self.cfg.r0_frac * lD
        rl = max(1.0, self.cfg.r_min_frac * r0)
        r = r0

        # init counters (Algorithm 3, step 3) :contentReference[oaicite:34]{index=34}
        c_step = True
        t_step = False

        Cc_f = 0
        Cc_s = 0
        Cc_r = 0
        Cc_i = 0

        Ct_f = 0
        Ct_s = 0
        Ct_i = 0

        # init design + surrogate fit (Algorithm 3, steps 4-6) :contentReference[oaicite:35]{index=35}
        self._initial_design(n0)
        self._fit_surrogate()

        history: List[Dict[str, object]] = []

        def current_best() -> Tuple[np.ndarray, float]:
            yarr = np.asarray(self.y, dtype=float)
            # prefer valid best
            vmask = np.asarray(self.valid, dtype=bool)
            if np.any(vmask):
                idx = int(np.argmin(yarr[vmask]))
                # map back
                real_idx = int(np.where(vmask)[0][idx])
            else:
                real_idx = int(np.argmin(yarr))
            return self.X[real_idx].copy(), float(self.y[real_idx])

        zbest, fbest = current_best()
        time_to_best = time.perf_counter() - start_time
        n = len(self.X)

        # Track whether (c+t) phase improved best; used to trigger local search.
        fbest_at_phase_start = fbest

        while n < nmax:
            if c_step:
                # Algorithm 3: c-step branch, increment counter, use Alg.4 :contentReference[oaicite:36]{index=36}
                Cc_i += 1
                znew = self._c_step_propose(zbest, n, n0, nmax, r, Cc_i)

                fnew, ok = self._evaluate(znew)
                self.X.append(znew)
                self.y.append(fnew)
                self.valid.append(ok)
                n += 1

                history.append({"phase": "c", "x": znew.copy(), "f": fnew, "ok": ok})

                # Update best from true evaluations
                if ok and fnew < fbest:
                    fbest = fnew
                    zbest = znew.copy()
                    time_to_best = time.perf_counter() - start_time
                    Cc_s += 1
                    Cc_f = 0
                    if Cc_s > self.cfg.Tc_s:
                        r = min(
                            r0, 2.0 * r
                        )  # step 24 :contentReference[oaicite:37]{index=37}
                        Cc_s = 0
                else:
                    Cc_f += 1
                    Cc_s = 0
                    if Cc_f > Tc_f:
                        if Cc_r > self.cfg.Tc_r:
                            # switch to t-step (step 16) :contentReference[oaicite:38]{index=38}
                            c_step = False
                            t_step = True
                            Cc_r = 0
                            Cc_f = 0
                        else:
                            # decrease radius and reset fail counter (step 19) :contentReference[oaicite:39]{index=39}
                            Cc_r += 1
                            r = max(rl, r / 2.0)
                            Cc_f = 0

            elif t_step:
                Ct_i += 1
                znew = self._t_step_propose(fbest=fbest, zbest=zbest, Ct_i=Ct_i)

                fnew, ok = self._evaluate(znew)
                self.X.append(znew)
                self.y.append(fnew)
                self.valid.append(ok)
                n += 1

                history.append({"phase": "t", "x": znew.copy(), "f": fnew, "ok": ok})

                if ok and fnew < fbest:
                    fbest = fnew
                    zbest = znew.copy()
                    time_to_best = time.perf_counter() - start_time
                    Ct_s += 1
                    Ct_f = 0
                else:
                    Ct_f += 1
                    Ct_s = 0
                    if Ct_f > Tt_f:
                        # t-step ends -> switch back to c-step (Alg.5 step 31) :contentReference[oaicite:40]{index=40}
                        t_step = False
                        c_step = True
                        Ct_f = 0

                        # memetic local search trigger: if (c+t) didn't improve best
                        if self.cfg.enable_local_search and (
                            fbest >= fbest_at_phase_start - 1e-12
                        ):
                            remaining = nmax - n
                            if remaining > 0:
                                zl, fl, used = self._local_search(
                                    zbest, remaining_budget=remaining
                                )
                                n = len(self.X)  # local search appended points
                                if fl < fbest:
                                    fbest = fl
                                    zbest = zl.copy()
                                    time_to_best = time.perf_counter() - start_time
                                history.append(
                                    {
                                        "phase": "local",
                                        "x": zbest.copy(),
                                        "f": fbest,
                                        "ok": True,
                                        "used": used,
                                    }
                                )

                        fbest_at_phase_start = fbest

            else:
                # Shouldn't happen in CPTV; keep safe
                c_step = True

            # Update surrogate after each expensive evaluation (Algorithm 3, step 24) :contentReference[oaicite:41]{index=41}
            self._fit_surrogate()
            zbest, fbest = current_best()

        return {
            "best_x": zbest.copy(),
            "best_f": float(fbest),
            "time_to_best": time_to_best,  # <--- Add this
            "total_time": time.perf_counter()
            - start_time,  # <--- Optional: Adds total runtime too
            "evaluations": len(self.X),
            "X": np.vstack(self.X).astype(int),
            "y": np.asarray(self.y, dtype=float),
            "valid": np.asarray(self.valid, dtype=bool),
            "history": history,
        }
