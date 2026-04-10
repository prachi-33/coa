"""SSA — Vectorized (batch leader + follower update)."""

import numpy as np


def SSA(func, lb, ub, dim, N=30, MaxFES=60000):
    lb = np.array([lb] * dim) if np.isscalar(lb) else np.array(lb, dtype=float)
    ub = np.array([ub] * dim) if np.isscalar(ub) else np.array(ub, dtype=float)

    MaxIter = MaxFES // N
    FES     = 0
    n_lead  = N // 2

    X       = lb + (ub - lb) * np.random.rand(N, dim)
    fitness = np.array([func(X[i]) for i in range(N)]); FES += N

    best_idx = np.argmin(fitness)
    food_pos = X[best_idx].copy()
    food_fit = fitness[best_idx]

    convergence_curve = np.zeros(MaxIter)

    for t in range(MaxIter):
        c1 = 2 * np.exp(-((4 * t / MaxIter) ** 2))

        # ── Leaders (vectorised) ──────────────────────────────────────────
        c2 = np.random.rand(n_lead, dim)
        c3 = np.random.rand(n_lead, dim)
        step = c1 * ((ub - lb) * c2 + lb)
        X[:n_lead] = np.where(c3 >= 0.5,
                              food_pos + step,
                              food_pos - step)

        # ── Followers (vectorised average of neighbours) ──────────────────
        X[n_lead:] = 0.5 * (X[n_lead:] + X[n_lead - 1:-1])

        X = np.clip(X, lb, ub)

        # ── Batch evaluate ────────────────────────────────────────────────
        new_fitness = np.array([func(X[i]) for i in range(N)]); FES += N
        fitness     = new_fitness

        cur_best = np.argmin(fitness)
        if fitness[cur_best] < food_fit:
            food_fit = fitness[cur_best]
            food_pos = X[cur_best].copy()

        convergence_curve[t] = food_fit
        if FES >= MaxFES:
            convergence_curve[t + 1:] = food_fit
            break

    return food_fit, food_pos, convergence_curve


if __name__ == "__main__":
    def sphere(x): return float(np.sum(x ** 2))
    bf, _, _ = SSA(sphere, -100, 100, 30)
    print(f"SSA  | Best: {bf:.6e}")
