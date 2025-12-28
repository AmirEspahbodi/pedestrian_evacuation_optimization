from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .common import FitnessEvaluator

ArrayLikeInt = Union[np.ndarray, Sequence[int]]
Bounds = Union[Tuple[int, int], Sequence[Tuple[int, int]]]


@dataclass
class GWOResult:
    best_x: np.ndarray
    best_f: float
    n_evals: int
    history_best: List[float]
    history_alpha: List[float]


def integer_enhanced_gwo(
    gird,
    pedestrian_confs,
    simulator_config,
    iea_config,
    *,
    n_wolves: int = 16,
    max_iters: int = 80,
    seed: Optional[int] = None,
    # Feasibility handling (recommended for “not all integers are valid”):
    valid_fn: Optional[Callable[[np.ndarray], bool]] = None,
    repair_fn: Optional[Callable[[np.ndarray, np.random.Generator], np.ndarray]] = None,
    penalty_value: float = 1e30,
    # Optional per-dimension allowed integer sets (best way to encode obstacles):
    choices: Optional[Sequence[Sequence[int]]] = None,
    # GWO control:
    a_schedule: str = "linear",  # "linear" (canonical), "cosine" (slower early decay)
    # Enhancements:
    opposition_init: bool = True,
    epd_rate: float = 0.20,  # fraction of worst wolves replaced (0 disables)
    epd_period: int = 5,  # every epd_period iters
    restart_on_stall: bool = True,
    stall_patience: int = 15,
    local_search_steps: int = 6,  # 0 disables (low-dim integer problems benefit)
    verbose: bool = False,
) -> GWOResult:
    """
    Integer / constraint-aware Enhanced Grey Wolf Optimizer (IE-GWO).

    Core update is canonical GWO:
      A = 2*a*r1 - a
      C = 2*r2
      D = |C * X_leader - X|
      X1 = X_alpha - A1 * D_alpha
      X2 = X_beta  - A2 * D_beta
      X3 = X_delta - A3 * D_delta
      X_new = (X1 + X2 + X3) / 3

    with 'a' decreasing 2 -> 0 over iterations (canonical schedule).
    The equations match the chapter’s encircling/hunting model and pseudocode. (See Eq. 9.1–9.6 and the loop
    structure.)  :contentReference[oaicite:2]{index=2}

    Enhancements for your setting:
      - Always returns integer vectors inside bounds.
      - Optional `valid_fn` and `repair_fn` to handle invalid integers (obstacles).
      - Fitness caching (deterministic objective).
      - Elitism, EPD weak-wolf replacement, optional opposition-based initialization,
        optional local search around alpha.

    Parameters
    ----------
    fitness_fn:
        Callable that takes np.ndarray shape (dim,) integer vector and returns fitness (to minimize).
    dim:
        Dimensionality (k).
    bounds:
        Either (lb, ub) applied to all dims, or per-dimension [(lb0, ub0), ...].
    choices:
        Optional per-dimension allowed integer values.
        If provided, each coordinate is snapped to the nearest allowed value in that dimension.
        This is often the best way to encode “cannot place exit on desk/column”.
    valid_fn:
        Optional feasibility predicate. If provided and returns False, candidate is penalized (or repaired if repair_fn).
    repair_fn:
        Optional repair operator. Called as repair_fn(x_int, rng) and must return a valid integer vector.
    """
    fitness_fn = FitnessEvaluator(gird, pedestrian_confs, simulator_config).evaluate
    dim = simulator_config.numEmergencyExits
    bounds = (0, 2 * (len(gird) + len(gird[0])))
    max_evals = iea_config.max_evals

    if dim <= 0:
        raise ValueError("dim must be positive.")
    if n_wolves < 4:
        raise ValueError("n_wolves should be >= 4 (need alpha/beta/delta + others).")

    rng = np.random.default_rng(seed)

    # Normalize bounds to per-dimension arrays
    if (
        isinstance(bounds, tuple)
        and len(bounds) == 2
        and all(isinstance(v, (int, np.integer)) for v in bounds)
    ):
        lb = np.full(dim, int(bounds[0]), dtype=int)
        ub = np.full(dim, int(bounds[1]), dtype=int)
    else:
        b = list(bounds)  # type: ignore[arg-type]
        if len(b) != dim:
            raise ValueError("If bounds is per-dimension, it must have length == dim.")
        lb = np.array([int(x[0]) for x in b], dtype=int)
        ub = np.array([int(x[1]) for x in b], dtype=int)

    if np.any(lb > ub):
        raise ValueError("Lower bound > upper bound in bounds.")

    # Prepare choices arrays (sorted, unique)
    choices_arr: Optional[List[np.ndarray]] = None
    if choices is not None:
        if len(choices) != dim:
            raise ValueError("choices must have length == dim.")
        choices_arr = []
        for j in range(dim):
            arr = np.array(list(choices[j]), dtype=int)
            arr = np.unique(arr)
            arr.sort()
            # Keep only values inside bounds
            arr = arr[(arr >= lb[j]) & (arr <= ub[j])]
            if arr.size == 0:
                raise ValueError(f"choices[{j}] has no values within bounds.")
            choices_arr.append(arr)

    def _snap_to_choices(x_int: np.ndarray) -> np.ndarray:
        """Snap each coordinate to nearest allowed choice value (if choices_arr is provided)."""
        if choices_arr is None:
            return x_int
        y = x_int.copy()
        for j in range(dim):
            arr = choices_arr[j]
            v = int(y[j])
            # nearest in sorted arr
            idx = int(np.searchsorted(arr, v))
            if idx <= 0:
                y[j] = arr[0]
            elif idx >= arr.size:
                y[j] = arr[-1]
            else:
                lo = arr[idx - 1]
                hi = arr[idx]
                y[j] = lo if (v - lo) <= (hi - v) else hi
        return y

    def _project_integer(x_cont: np.ndarray) -> np.ndarray:
        """Round -> clip to bounds -> optional snap to allowed sets."""
        x_int = np.rint(x_cont).astype(int)
        x_int = np.clip(x_int, lb, ub)
        x_int = _snap_to_choices(x_int)
        return x_int

    def _is_valid(x_int: np.ndarray) -> bool:
        if np.any(x_int < lb) or np.any(x_int > ub):
            return False
        if valid_fn is None:
            return True
        try:
            return bool(valid_fn(x_int))
        except Exception:
            # If user's valid_fn throws, treat as invalid
            return False

    def _attempt_repair(x_int: np.ndarray) -> np.ndarray:
        """Try repair_fn if provided; otherwise randomized attempts near x; otherwise random feasible."""
        x0 = x_int.copy()

        if repair_fn is not None:
            try:
                xr = np.asarray(repair_fn(x0.copy(), rng), dtype=int)
                xr = np.clip(xr, lb, ub)
                xr = _snap_to_choices(xr)
                if _is_valid(xr):
                    return xr
            except Exception:
                pass  # fall through to randomized repair

        # Randomized repair: try perturbations around the candidate
        # (useful when invalidity is “blocked cells”)
        span = (ub - lb).astype(int)
        max_step = np.maximum(2, (0.10 * span).astype(int))  # 10% of range
        for _ in range(40):
            step = rng.integers(-max_step, max_step + 1, size=dim)
            xr = np.clip(x0 + step, lb, ub)
            xr = _snap_to_choices(xr)
            if _is_valid(xr):
                return xr

        # Last resort: random sampling
        for _ in range(200):
            if choices_arr is None:
                xr = rng.integers(lb, ub + 1, size=dim, dtype=int)
            else:
                xr = np.array(
                    [rng.choice(choices_arr[j]) for j in range(dim)], dtype=int
                )
            if _is_valid(xr):
                return xr

        # If nothing found, return x0 (will be penalized safely later).
        return x0

    # Deterministic cache
    cache: Dict[Tuple[int, ...], float] = {}
    n_evals = 0

    def _safe_eval(x_int: np.ndarray) -> float:
        """Never crash: bounds guaranteed; invalid gets penalty; exceptions get penalty. Cached."""
        nonlocal n_evals
        key = tuple(int(v) for v in x_int.tolist())
        if key in cache:
            return cache[key]

        if not _is_valid(x_int):
            cache[key] = float(penalty_value)
            return cache[key]

        try:
            val = float(fitness_fn(x_int.copy()))
        except Exception:
            val = float(penalty_value)

        cache[key] = val
        n_evals += 1
        return val

    def _init_population() -> np.ndarray:
        if choices_arr is None:
            X = rng.integers(lb, ub + 1, size=(n_wolves, dim), dtype=int)
        else:
            X = np.vstack(
                [
                    np.array(
                        [rng.choice(choices_arr[j]) for j in range(dim)], dtype=int
                    )
                    for _ in range(n_wolves)
                ]
            )
        # Repair any invalids
        for i in range(n_wolves):
            if not _is_valid(X[i]):
                X[i] = _attempt_repair(X[i])
        return X

    def _opposition(X: np.ndarray) -> np.ndarray:
        # Opposite in continuous: x_opp = lb + ub - x; then project to integer/choices
        Xopp = (lb + ub) - X
        Xopp = np.clip(Xopp, lb, ub)
        if choices_arr is not None:
            for i in range(Xopp.shape[0]):
                Xopp[i] = _snap_to_choices(Xopp[i])
        return Xopp.astype(int)

    def _rank_population(
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        F = np.array([_safe_eval(X[i]) for i in range(X.shape[0])], dtype=float)
        order = np.argsort(F)  # ascending (minimization)
        return X[order], F[order], order, F

    def _a_value(t: int) -> float:
        # canonical: linear decrease 2 -> 0
        if max_iters <= 1:
            return 0.0
        frac = t / (max_iters - 1)
        if a_schedule == "linear":
            return 2.0 * (1.0 - frac)
        if a_schedule == "cosine":
            # smoother: starts exploring longer, then exploits
            return 2.0 * (0.5 * (1.0 + np.cos(np.pi * frac)))
        raise ValueError("a_schedule must be 'linear' or 'cosine'.")

    def _local_search(best_x: np.ndarray, best_f: float) -> Tuple[np.ndarray, float]:
        """Small random-neighborhood search around alpha (good for low-dim integers)."""
        if local_search_steps <= 0:
            return best_x, best_f

        span = (ub - lb).astype(int)
        # Small steps: 1..max(2, 3% range)
        max_step = np.maximum(2, (0.03 * span).astype(int))
        cur_x, cur_f = best_x.copy(), float(best_f)

        for _ in range(local_search_steps):
            # Randomly perturb a subset of dims
            mask = rng.random(dim) < 0.7
            if not np.any(mask):
                mask[rng.integers(0, dim)] = True
            step = np.zeros(dim, dtype=int)
            step[mask] = rng.integers(-max_step[mask], max_step[mask] + 1)
            cand = np.clip(cur_x + step, lb, ub)
            cand = _snap_to_choices(cand)
            if not _is_valid(cand):
                cand = _attempt_repair(cand)
            f = _safe_eval(cand)
            if f < cur_f:
                cur_x, cur_f = cand, f
        return cur_x, cur_f

    # -------------------------
    # Initialization
    # -------------------------
    X = _init_population()

    if opposition_init:
        Xopp = _opposition(X)
        # Pairwise choose best between X and Xopp
        for i in range(n_wolves):
            fi = _safe_eval(X[i])
            fo = _safe_eval(Xopp[i])
            if fo < fi:
                X[i] = Xopp[i]

    X_sorted, F_sorted, _, _ = _rank_population(X)
    alpha_x, alpha_f = X_sorted[0].copy(), float(F_sorted[0])
    beta_x, beta_f = X_sorted[1].copy(), float(F_sorted[1])
    delta_x, delta_f = X_sorted[2].copy(), float(F_sorted[2])

    best_x, best_f = alpha_x.copy(), alpha_f
    history_best: List[float] = [best_f]
    history_alpha: List[float] = [alpha_f]

    stall = 0

    # -------------------------
    # Main loop
    # -------------------------
    for t in range(max_iters):
        if max_evals is not None and n_evals >= max_evals:
            if verbose:
                print(f"[stop] eval budget reached: {n_evals}/{max_evals}")
            break

        a = _a_value(t)

        # Vectorized leadership update (for each wolf i, compute X1, X2, X3 then average)
        X_new_cont = np.empty((n_wolves, dim), dtype=float)

        # Leaders as float
        Xa = alpha_x.astype(float)
        Xb = beta_x.astype(float)
        Xd = delta_x.astype(float)

        for i in range(n_wolves):
            Xi = X[i].astype(float)

            # For each leader, sample independent r1, r2 (as in canonical GWO).
            r1a, r2a = rng.random(dim), rng.random(dim)
            r1b, r2b = rng.random(dim), rng.random(dim)
            r1d, r2d = rng.random(dim), rng.random(dim)

            Aa = 2.0 * a * r1a - a
            Ab = 2.0 * a * r1b - a
            Ad = 2.0 * a * r1d - a

            Ca = 2.0 * r2a
            Cb = 2.0 * r2b
            Cd = 2.0 * r2d

            Da = np.abs(Ca * Xa - Xi)
            Db = np.abs(Cb * Xb - Xi)
            Dd = np.abs(Cd * Xd - Xi)

            X1 = Xa - Aa * Da
            X2 = Xb - Ab * Db
            X3 = Xd - Ad * Dd

            X_new_cont[i] = (X1 + X2 + X3) / 3.0

        # Project to integer + bounds (+ choices snapping)
        X_new = np.vstack([_project_integer(X_new_cont[i]) for i in range(n_wolves)])

        # Repair invalids (optional)
        for i in range(n_wolves):
            if not _is_valid(X_new[i]):
                X_new[i] = _attempt_repair(X_new[i])

        X = X_new

        # EPD: replace weakest wolves periodically
        if epd_rate > 0.0 and epd_period > 0 and ((t + 1) % epd_period == 0):
            X_sorted, F_sorted, order_sorted, _Fall = _rank_population(X)
            m = int(np.ceil(epd_rate * n_wolves))
            if m > 0:
                worst_idx = order_sorted[-m:]  # indices in original X
                # Replacement: half near alpha (exploitation), half random (exploration)
                span = (ub - lb).astype(int)
                step_scale = np.maximum(
                    2, (0.15 * (1.0 - t / max(1, max_iters - 1)) * span).astype(int)
                )
                for j, idx in enumerate(worst_idx):
                    if j < m // 2:
                        # around alpha
                        step = rng.integers(-step_scale, step_scale + 1, size=dim)
                        cand = np.clip(alpha_x + step, lb, ub)
                        cand = _snap_to_choices(cand)
                    else:
                        # random
                        if choices_arr is None:
                            cand = rng.integers(lb, ub + 1, size=dim, dtype=int)
                        else:
                            cand = np.array(
                                [rng.choice(choices_arr[d]) for d in range(dim)],
                                dtype=int,
                            )

                    if not _is_valid(cand):
                        cand = _attempt_repair(cand)
                    X[idx] = cand

        # Rank and update leaders
        X_sorted, F_sorted, _, _ = _rank_population(X)
        alpha_x, alpha_f = X_sorted[0].copy(), float(F_sorted[0])
        beta_x, beta_f = X_sorted[1].copy(), float(F_sorted[1])
        delta_x, delta_f = X_sorted[2].copy(), float(F_sorted[2])

        # Local search around alpha (optional)
        alpha_x2, alpha_f2 = _local_search(alpha_x, alpha_f)
        if alpha_f2 < alpha_f:
            alpha_x, alpha_f = alpha_x2, alpha_f2

        # Global best / elitism
        if alpha_f < best_f:
            best_x, best_f = alpha_x.copy(), float(alpha_f)
            stall = 0
        else:
            stall += 1

        # Enforce elitism: keep best_x inside population by replacing the current worst
        # (helps prevent losing best due to stochastic update)
        worst_i = int(np.argmax([_safe_eval(X[i]) for i in range(n_wolves)]))
        X[worst_i] = best_x.copy()

        # Optional restart on stall
        if restart_on_stall and stall >= stall_patience:
            stall = 0
            # Reinitialize a fraction of population (keep best)
            frac = 0.4
            k_re = max(1, int(frac * n_wolves))
            idxs = rng.choice(np.arange(n_wolves), size=k_re, replace=False)
            for idx in idxs:
                if choices_arr is None:
                    cand = rng.integers(lb, ub + 1, size=dim, dtype=int)
                else:
                    cand = np.array(
                        [rng.choice(choices_arr[d]) for d in range(dim)], dtype=int
                    )
                if not _is_valid(cand):
                    cand = _attempt_repair(cand)
                X[idx] = cand
            # Put best back explicitly
            X[rng.integers(0, n_wolves)] = best_x.copy()

        history_best.append(best_f)
        history_alpha.append(alpha_f)

        if verbose and (t % 10 == 0 or t == max_iters - 1):
            print(
                f"iter={t:4d}  a={a:.3f}  alpha={alpha_f:.6g}  best={best_f:.6g}  evals={n_evals}"
            )

        if max_evals is not None and n_evals >= max_evals:
            if verbose:
                print(f"[stop] eval budget reached: {n_evals}/{max_evals}")
            break

    return GWOResult(
        best_x=best_x.astype(int),
        best_f=float(best_f),
        n_evals=int(n_evals),
        history_best=history_best,
        history_alpha=history_alpha,
    )
