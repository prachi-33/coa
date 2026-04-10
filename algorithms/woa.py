"""WOA — Vectorized (batch shrink/spiral/random update, no inner for-loop)."""

import numpy as np


def WOA(func, lb, ub, dim, N=30, MaxFES=60000):
    lb = np.array([lb] * dim) if np.isscalar(lb) else np.array(lb, dtype=float)
    ub = np.array([ub] * dim) if np.isscalar(ub) else np.array(ub, dtype=float)

    MaxIter = MaxFES // N
    FES     = 0

    X       = lb + (ub - lb) * np.random.rand(N, dim)
    fitness = np.array([func(X[i]) for i in range(N)]); FES += N

    best_idx = np.argmin(fitness)
    best_pos = X[best_idx].copy()
    best_fit = fitness[best_idx]

    convergence_curve = np.zeros(MaxIter)

    for t in range(MaxIter):
        a  = 2.0 - 2.0 * t / MaxIter
        a2 = -1.0 - t / MaxIter

        r  = np.random.rand(N)                          # (N,)
        p  = np.random.rand(N)                          # (N,)
        A  = 2 * a * np.random.rand(N, dim) - a        # (N, dim)
        C  = 2 * np.random.rand(N, dim)                 # (N, dim)
        l  = (a2 - 1) * np.random.rand(N) + 1          # (N,)
        b  = 1.0

        abs_A_mean = np.abs(A).mean(axis=1)             # (N,)

        # ── Spiral (bubble-net) update ────────────────────────────────────
        D_spiral = np.abs(best_pos - X)
        X_spiral = D_spiral * np.exp(b * l[:, None]) * \
                   np.cos(2 * np.pi * l[:, None]) + best_pos

        # ── Shrinking encircling ──────────────────────────────────────────
        D_shrink = np.abs(C * best_pos - X)
        X_shrink = best_pos - A * D_shrink

        # ── Random search (exploration) ───────────────────────────────────
        rand_idx = np.random.randint(0, N, size=N)
        X_rand   = X[rand_idx]
        D_rand   = np.abs(C * X_rand - X)
        X_random = X_rand - A * D_rand

        # ── Rule selection (vectorised) ───────────────────────────────────
        # p < 0.5: shrink or random;  p >= 0.5: spiral
        use_shrink = (p < 0.5) & (abs_A_mean < 1)
        use_random = (p < 0.5) & (abs_A_mean >= 1)
        use_spiral = p >= 0.5

        X_new = np.where(use_shrink[:, None], X_shrink,
                np.where(use_random[:, None], X_random,
                         X_spiral))
        X_new = np.clip(X_new, lb, ub)

        new_fitness = np.array([func(X_new[i]) for i in range(N)]); FES += N
        fitness     = new_fitness
        X           = X_new

        cur_best = np.argmin(fitness)
        if fitness[cur_best] < best_fit:
            best_fit = fitness[cur_best]
            best_pos = X[cur_best].copy()

        convergence_curve[t] = best_fit
        if FES >= MaxFES:
            convergence_curve[t + 1:] = best_fit
            break

    return best_fit, best_pos, convergence_curve


if __name__ == "__main__":
    def sphere(x): return float(np.sum(x ** 2))
    bf, _, _ = WOA(sphere, -100, 100, 30)
    print(f"WOA  | Best: {bf:.6e}")
