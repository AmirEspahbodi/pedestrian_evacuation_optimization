from __future__ import annotations

import time  # <--- Added for tracking discovery time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .common import FitnessEvaluator

ArrayLikeInt = Union[np.ndarray, Sequence[int]]
Bounds = Union[Tuple[int, int], Sequence[Tuple[int, int]]]


def integer_enhanced_gwo(
    gird,
    pedestrian_confs,
    simulator_config,
    iea_config,
    *,
    n_wolves: int = 16,
    max_iters: int = 80,
    seed: Optional[int] = None,
    # Feasibility handling:
    valid_fn: Optional[Callable[[np.ndarray], bool]] = None,
    repair_fn: Optional[Callable[[np.ndarray, np.random.Generator], np.ndarray]] = None,
    penalty_value: float = 1e30,
    # Optional per-dimension allowed integer sets:
    choices: Optional[Sequence[Sequence[int]]] = None,
    # GWO control:
    a_schedule: str = "linear",
    # Enhancements:
    opposition_init: bool = True,
    epd_rate: float = 0.20,
    epd_period: int = 5,
    restart_on_stall: bool = True,
    stall_patience: int = 15,
    local_search_steps: int = 15,
    verbose: bool = False,
):
    """
    Integer / constraint-aware Enhanced Grey Wolf Optimizer (IE-GWO).
    """

    # --- 1. Start Timer ---
    t0 = time.time()

    fitness_fn = FitnessEvaluator(gird, pedestrian_confs, simulator_config).evaluate
    dim = simulator_config.numEmergencyExits
    bounds = (0, 2 * (len(gird) + len(gird[0])))
    max_evals = 1310

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
            arr = arr[(arr >= lb[j]) & (arr <= ub[j])]
            if arr.size == 0:
                raise ValueError(f"choices[{j}] has no values within bounds.")
            choices_arr.append(arr)

    def _snap_to_choices(x_int: np.ndarray) -> np.ndarray:
        if choices_arr is None:
            return x_int
        y = x_int.copy()
        for j in range(dim):
            arr = choices_arr[j]
            v = int(y[j])
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
            return False

    def _attempt_repair(x_int: np.ndarray) -> np.ndarray:
        x0 = x_int.copy()
        if repair_fn is not None:
            try:
                xr = np.asarray(repair_fn(x0.copy(), rng), dtype=int)
                xr = np.clip(xr, lb, ub)
                xr = _snap_to_choices(xr)
                if _is_valid(xr):
                    return xr
            except Exception:
                pass

        span = (ub - lb).astype(int)
        max_step = np.maximum(2, (0.10 * span).astype(int))
        for _ in range(40):
            step = rng.integers(-max_step, max_step + 1, size=dim)
            xr = np.clip(x0 + step, lb, ub)
            xr = _snap_to_choices(xr)
            if _is_valid(xr):
                return xr

        for _ in range(200):
            if choices_arr is None:
                xr = rng.integers(lb, ub + 1, size=dim, dtype=int)
            else:
                xr = np.array(
                    [rng.choice(choices_arr[j]) for j in range(dim)], dtype=int
                )
            if _is_valid(xr):
                return xr
        return x0

    cache: Dict[Tuple[int, ...], float] = {}
    n_evals = 0

    def _safe_eval(x_int: np.ndarray) -> float:
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
        for i in range(n_wolves):
            if not _is_valid(X[i]):
                X[i] = _attempt_repair(X[i])
        return X

    def _opposition(X: np.ndarray) -> np.ndarray:
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
        order = np.argsort(F)
        return X[order], F[order], order, F

    def _a_value(t: int) -> float:
        if max_iters <= 1:
            return 0.0
        frac = t / (max_iters - 1)
        if a_schedule == "linear":
            return 2.0 * (1.0 - frac)
        if a_schedule == "cosine":
            return 2.0 * (0.5 * (1.0 + np.cos(np.pi * frac)))
        raise ValueError("a_schedule must be 'linear' or 'cosine'.")

    def _local_search(best_x: np.ndarray, best_f: float) -> Tuple[np.ndarray, float]:
        if local_search_steps <= 0:
            return best_x, best_f
        span = (ub - lb).astype(int)
        max_step = np.maximum(2, (0.03 * span).astype(int))
        cur_x, cur_f = best_x.copy(), float(best_f)
        for _ in range(local_search_steps):
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

    # --- 2. Track initial best time and history containers ---
    best_found_time = time.time() - t0
    history_pop: Dict[str, List[float]] = {}

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

        X_new_cont = np.empty((n_wolves, dim), dtype=float)
        Xa = alpha_x.astype(float)
        Xb = beta_x.astype(float)
        Xd = delta_x.astype(float)

        for i in range(n_wolves):
            Xi = X[i].astype(float)
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

        X_new = np.vstack([_project_integer(X_new_cont[i]) for i in range(n_wolves)])

        for i in range(n_wolves):
            if not _is_valid(X_new[i]):
                X_new[i] = _attempt_repair(X_new[i])

        X = X_new

        if epd_rate > 0.0 and epd_period > 0 and ((t + 1) % epd_period == 0):
            X_sorted, F_sorted, order_sorted, _ = _rank_population(X)
            m = int(np.ceil(epd_rate * n_wolves))
            if m > 0:
                worst_idx = order_sorted[-m:]
                span = (ub - lb).astype(int)
                step_scale = np.maximum(
                    2, (0.15 * (1.0 - t / max(1, max_iters - 1)) * span).astype(int)
                )
                for j, idx in enumerate(worst_idx):
                    if j < m // 2:
                        step = rng.integers(-step_scale, step_scale + 1, size=dim)
                        cand = np.clip(alpha_x + step, lb, ub)
                        cand = _snap_to_choices(cand)
                    else:
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

        # --- 3. Capture Population History ---
        # Changed unpacking to capture F_all (4th return value)
        X_sorted, F_sorted, _, F_all = _rank_population(X)
        history_pop[str(t)] = F_all.tolist()

        alpha_x, alpha_f = X_sorted[0].copy(), float(F_sorted[0])
        beta_x, beta_f = X_sorted[1].copy(), float(F_sorted[1])
        delta_x, delta_f = X_sorted[2].copy(), float(F_sorted[2])

        alpha_x2, alpha_f2 = _local_search(alpha_x, alpha_f)
        if alpha_f2 < alpha_f:
            alpha_x, alpha_f = alpha_x2, alpha_f2

        # Global best / elitism
        if alpha_f < best_f:
            best_x, best_f = alpha_x.copy(), float(alpha_f)
            # --- 4. Capture Discovery Time ---
            best_found_time = time.time() - t0
            stall = 0
        else:
            stall += 1

        worst_i = int(np.argmax([_safe_eval(X[i]) for i in range(n_wolves)]))
        X[worst_i] = best_x.copy()

        if restart_on_stall and stall >= stall_patience:
            stall = 0
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

    return {
        "best_x": best_x.astype(int).tolist(),
        "best_f": best_f,
        "history_best": history_best,
        "history_alpha": history_alpha,
        "history_pop": history_pop,
        "discovery_time": best_found_time,
    }
