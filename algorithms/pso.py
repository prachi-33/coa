"""PSO — Fully vectorized (was already mostly vectorized)."""

import numpy as np


def PSO(func, lb, ub, dim, N=30, MaxFES=60000, w=0.7, c1=1.5, c2=1.5):
    lb = np.array([lb] * dim) if np.isscalar(lb) else np.array(lb, dtype=float)
    ub = np.array([ub] * dim) if np.isscalar(ub) else np.array(ub, dtype=float)

    MaxIter = MaxFES // N
    FES     = 0
    V_max   = 0.2 * (ub - lb)

    X        = lb + (ub - lb) * np.random.rand(N, dim)
    V        = np.zeros((N, dim))
    fitness  = np.array([func(X[i]) for i in range(N)]); FES += N

    pBest    = X.copy()
    pBestFit = fitness.copy()
    gBest_idx = np.argmin(fitness)
    gBest    = X[gBest_idx].copy()
    gBestFit = fitness[gBest_idx]

    convergence_curve = np.zeros(MaxIter)

    for t in range(MaxIter):
        w_t  = w - (w - 0.4) * t / MaxIter
        R1   = np.random.rand(N, dim)
        R2   = np.random.rand(N, dim)
        V    = w_t * V + c1 * R1 * (pBest - X) + c2 * R2 * (gBest - X)
        V    = np.clip(V, -V_max, V_max)
        X    = np.clip(X + V, lb, ub)

        # Batch evaluate
        new_fitness = np.array([func(X[i]) for i in range(N)]); FES += N

        improved   = new_fitness < pBestFit
        pBest[improved]    = X[improved].copy()
        pBestFit[improved] = new_fitness[improved]

        best_now = np.argmin(pBestFit)
        if pBestFit[best_now] < gBestFit:
            gBestFit = pBestFit[best_now]
            gBest    = pBest[best_now].copy()

        convergence_curve[t] = gBestFit
        if FES >= MaxFES:
            convergence_curve[t + 1:] = gBestFit
            break

    return gBestFit, gBest, convergence_curve


if __name__ == "__main__":
    def sphere(x): return float(np.sum(x ** 2))
    bf, _, _ = PSO(sphere, -100, 100, 30)
    print(f"PSO  | Best: {bf:.6e}")
