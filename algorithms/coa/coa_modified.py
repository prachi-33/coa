"""
MCOA — Vectorized (NumPy batch population updates, no inner Python for-loop)
Modifications: exponential C2 decay, dynamic C3, reflective boundaries.
"""

import numpy as np
import math


def MCOA(func, lb, ub, dim, N=30, MaxFES=60000):
    lb = np.array([lb] * dim) if np.isscalar(lb) else np.array(lb, dtype=float)
    ub = np.array([ub] * dim) if np.isscalar(ub) else np.array(ub, dtype=float)

    MaxIter = MaxFES // N
    FES     = 0
    C1 = 2.0; sigma = 0.1; mu = 25.0; k = 3.0

    X       = lb + (ub - lb) * np.random.rand(N, dim)
    fitness = np.array([func(X[i]) for i in range(N)]); FES += N

    best_idx     = np.argmin(fitness)
    X_G          = X[best_idx].copy()
    best_fitness = fitness[best_idx]
    X_L          = X_G.copy()

    convergence_curve = np.zeros(MaxIter)

    for t in range(1, MaxIter + 1):
        # MOD 1: exponential C2
        C2 = 2.0 * math.exp(-k * t / MaxIter)
        # MOD 2: dynamic C3
        C3 = max(3.0 * (1.0 - t / MaxIter), 1e-6)

        temp = np.random.rand() * 15.0 + 20.0
        p    = C1 / (sigma * math.sqrt(2 * math.pi)) * \
               math.exp(-((temp - mu) ** 2) / (2 * sigma ** 2))
        X_shade = (X_G + X_L) / 2.0

        if temp > 30.0:
            R         = np.random.rand(N, dim)
            X_resort  = X + C2 * R * (X_shade - X)

            perm      = np.random.permutation(N)
            self_mask = perm == np.arange(N)
            perm[self_mask] = (perm[self_mask] + 1) % N
            X_comp    = X - X[perm] + X_shade

            coin      = np.random.rand(N) < 0.5
            X_new     = np.where(coin[:, None], X_resort, X_comp)
        else:
            denom   = abs(fitness[best_idx]) if fitness[best_idx] != 0.0 else 1e-10
            Q       = C3 * np.random.rand(N) * (fitness / denom)
            R       = np.random.rand(N, dim)
            X_shred = X - X_G * p * R
            X_eat   = (1.0 - p) * X + p * X_G
            X_new   = np.where((Q > (C3 + 1.0) / 2.0)[:, None], X_shred, X_eat)

        # MOD 3: reflective bounds (vectorized)
        X_new = _reflect_bounds_batch(X_new, lb, ub)

        new_fitness = np.array([func(X_new[i]) for i in range(N)])
        FES += N

        imp = new_fitness < fitness
        X[imp]       = X_new[imp]
        fitness[imp] = new_fitness[imp]

        local_idx = np.argmin(fitness)
        X_L       = X[local_idx].copy()
        best_idx  = local_idx
        if fitness[local_idx] < best_fitness:
            best_fitness = fitness[local_idx]
            X_G = X[local_idx].copy()

        convergence_curve[t - 1] = best_fitness
        if FES >= MaxFES:
            convergence_curve[t:] = best_fitness
            break

    return best_fitness, X_G, convergence_curve


def _reflect_bounds_batch(X, lb, ub):
    """Vectorised reflective boundary — handles the whole population at once."""
    for _ in range(10):   # bounded while-loop; avoids infinite loop
        lo = X < lb
        hi = X > ub
        if not (lo.any() or hi.any()):
            break
        X = np.where(lo, 2 * lb - X, X)
        X = np.where(hi, 2 * ub - X, X)
    return np.clip(X, lb, ub)   # safety clamp for edge cases


if __name__ == "__main__":
    def sphere(x): return float(np.sum(x ** 2))
    bf, _, cv = MCOA(sphere, -100, 100, 30)
    print(f"MCOA | Best: {bf:.6e}")
