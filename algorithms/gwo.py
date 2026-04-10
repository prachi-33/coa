"""GWO — Vectorized (batch α/β/δ position update, no inner for-loop)."""

import numpy as np


def GWO(func, lb, ub, dim, N=30, MaxFES=60000):
    lb = np.array([lb] * dim) if np.isscalar(lb) else np.array(lb, dtype=float)
    ub = np.array([ub] * dim) if np.isscalar(ub) else np.array(ub, dtype=float)

    MaxIter = MaxFES // N
    FES     = 0

    X       = lb + (ub - lb) * np.random.rand(N, dim)
    fitness = np.array([func(X[i]) for i in range(N)]); FES += N

    idx = np.argsort(fitness)
    alpha_pos, alpha_fit = X[idx[0]].copy(), fitness[idx[0]]
    beta_pos,  beta_fit  = X[idx[1]].copy(), fitness[idx[1]]
    delta_pos, delta_fit = X[idx[2]].copy(), fitness[idx[2]]

    convergence_curve = np.zeros(MaxIter)

    for t in range(MaxIter):
        a = 2.0 - 2.0 * t / MaxIter

        # ── Vectorized: all N wolves updated at once ──────────────────────
        r1, r2 = np.random.rand(N, dim), np.random.rand(N, dim)
        A1 = 2 * a * r1 - a;  C1 = 2 * np.random.rand(N, dim)
        X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - X)

        r1, r2 = np.random.rand(N, dim), np.random.rand(N, dim)
        A2 = 2 * a * r1 - a;  C2 = 2 * np.random.rand(N, dim)
        X2 = beta_pos  - A2 * np.abs(C2 * beta_pos  - X)

        r1, r2 = np.random.rand(N, dim), np.random.rand(N, dim)
        A3 = 2 * a * r1 - a;  C3 = 2 * np.random.rand(N, dim)
        X3 = delta_pos - A3 * np.abs(C3 * delta_pos - X)

        X = np.clip((X1 + X2 + X3) / 3.0, lb, ub)

        # ── Batch evaluate ────────────────────────────────────────────────
        new_fitness = np.array([func(X[i]) for i in range(N)]); FES += N

        # ── Update α, β, δ ────────────────────────────────────────────────
        for i in range(N):
            f = new_fitness[i]
            if f < alpha_fit:
                delta_pos, delta_fit = beta_pos.copy(), beta_fit
                beta_pos,  beta_fit  = alpha_pos.copy(), alpha_fit
                alpha_pos, alpha_fit = X[i].copy(), f
            elif f < beta_fit:
                delta_pos, delta_fit = beta_pos.copy(), beta_fit
                beta_pos,  beta_fit  = X[i].copy(), f
            elif f < delta_fit:
                delta_pos, delta_fit = X[i].copy(), f

        convergence_curve[t] = alpha_fit
        if FES >= MaxFES:
            convergence_curve[t + 1:] = alpha_fit
            break

    return alpha_fit, alpha_pos, convergence_curve


if __name__ == "__main__":
    def sphere(x): return float(np.sum(x ** 2))
    bf, _, _ = GWO(sphere, -100, 100, 30)
    print(f"GWO  | Best: {bf:.6e}")
