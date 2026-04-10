"""
COA Base — Vectorized (NumPy batch population updates, no inner Python for-loop)
"""

import numpy as np
import math


def COA(func, lb, ub, dim, N=30, MaxFES=60000):
    lb = np.array([lb] * dim) if np.isscalar(lb) else np.array(lb, dtype=float)
    ub = np.array([ub] * dim) if np.isscalar(ub) else np.array(ub, dtype=float)

    MaxIter = MaxFES // N
    FES     = 0
    C1 = 2.0; C3 = 3.0; sigma = 0.1; mu = 25.0

    X       = lb + (ub - lb) * np.random.rand(N, dim)
    fitness = np.array([func(X[i]) for i in range(N)]); FES += N

    best_idx     = np.argmin(fitness)
    X_G          = X[best_idx].copy()
    best_fitness = fitness[best_idx]
    X_L          = X_G.copy()

    convergence_curve = np.zeros(MaxIter)

    for t in range(1, MaxIter + 1):
        C2   = 2.0 - t / MaxIter
        temp = np.random.rand() * 15.0 + 20.0
        p    = C1 / (sigma * math.sqrt(2 * math.pi)) * \
               math.exp(-((temp - mu) ** 2) / (2 * sigma ** 2))
        X_shade = (X_G + X_L) / 2.0

        if temp > 30.0:
            # ── Stage 1 Summer Resort (vectorized) ──────────────────────
            R          = np.random.rand(N, dim)
            X_resort   = X + C2 * R * (X_shade - X)

            # ── Stage 2 Competition (vectorized) ─────────────────────────
            perm       = np.random.permutation(N)
            self_mask  = perm == np.arange(N)
            perm[self_mask] = (perm[self_mask] + 1) % N
            X_comp     = X - X[perm] + X_shade

            # Coin flip per individual
            coin       = np.random.rand(N) < 0.5
            X_new      = np.where(coin[:, None], X_resort, X_comp)
        else:
            # ── Stage 3 Foraging (vectorized) ────────────────────────────
            denom = abs(fitness[best_idx]) if fitness[best_idx] != 0.0 else 1e-10
            Q     = C3 * np.random.rand(N) * (fitness / denom)
            R     = np.random.rand(N, dim)
            X_shred = X - X_G * p * R           # large food
            X_eat   = (1.0 - p) * X + p * X_G  # small food
            X_new   = np.where((Q > (C3 + 1.0) / 2.0)[:, None], X_shred, X_eat)

        X_new = np.clip(X_new, lb, ub)

        # ── Batch evaluate ────────────────────────────────────────────────
        new_fitness = np.array([func(X_new[i]) for i in range(N)])
        FES += N

        # ── Greedy selection (vectorized) ─────────────────────────────────
        imp = new_fitness < fitness
        X[imp]       = X_new[imp]
        fitness[imp] = new_fitness[imp]

        # ── Update bests ──────────────────────────────────────────────────
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


if __name__ == "__main__":
    def sphere(x): return float(np.sum(x ** 2))
    bf, _, cv = COA(sphere, -100, 100, 30)
    print(f"COA  | Best: {bf:.6e}")