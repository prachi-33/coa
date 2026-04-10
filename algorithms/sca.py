"""SCA — Fully vectorized (batch sine/cosine update)."""

import numpy as np


def SCA(func, lb, ub, dim, N=30, MaxFES=60000):
    lb = np.array([lb] * dim) if np.isscalar(lb) else np.array(lb, dtype=float)
    ub = np.array([ub] * dim) if np.isscalar(ub) else np.array(ub, dtype=float)

    MaxIter = MaxFES // N
    FES     = 0
    a       = 2.0

    X       = lb + (ub - lb) * np.random.rand(N, dim)
    fitness = np.array([func(X[i]) for i in range(N)]); FES += N

    best_idx = np.argmin(fitness)
    dest_pos = X[best_idx].copy()
    dest_fit = fitness[best_idx]

    convergence_curve = np.zeros(MaxIter)

    for t in range(MaxIter):
        r1 = a - a * t / MaxIter               # scalar
        r2 = 2 * np.pi * np.random.rand(N, dim)
        r3 = 2 * np.random.rand(N, dim)
        r4 = np.random.rand(N, dim)

        diff = np.abs(r3 * dest_pos - X)
        X_sin = X + r1 * np.sin(r2) * diff
        X_cos = X + r1 * np.cos(r2) * diff
        X     = np.clip(np.where(r4 < 0.5, X_sin, X_cos), lb, ub)

        new_fitness = np.array([func(X[i]) for i in range(N)]); FES += N
        fitness     = new_fitness

        cur_best = np.argmin(fitness)
        if fitness[cur_best] < dest_fit:
            dest_fit = fitness[cur_best]
            dest_pos = X[cur_best].copy()

        convergence_curve[t] = dest_fit
        if FES >= MaxFES:
            convergence_curve[t + 1:] = dest_fit
            break

    return dest_fit, dest_pos, convergence_curve


if __name__ == "__main__":
    def sphere(x): return float(np.sum(x ** 2))
    bf, _, _ = SCA(sphere, -100, 100, 30)
    print(f"SCA  | Best: {bf:.6e}")
