"""MPA — Vectorized (batch phase 1/2/3 and FADs effect)."""

import numpy as np
import math


def MPA(func, lb, ub, dim, N=30, MaxFES=60000):
    lb = np.array([lb] * dim) if np.isscalar(lb) else np.array(lb, dtype=float)
    ub = np.array([ub] * dim) if np.isscalar(ub) else np.array(ub, dtype=float)

    MaxIter = MaxFES // N
    FES     = 0
    P       = 0.5
    n_half  = N // 2

    X       = lb + (ub - lb) * np.random.rand(N, dim)
    fitness = np.array([func(X[i]) for i in range(N)]); FES += N

    best_idx = np.argmin(fitness)
    best_pos = X[best_idx].copy()
    best_fit = fitness[best_idx]
    Elite    = np.tile(best_pos, (N, 1))

    convergence_curve = np.zeros(MaxIter)

    for t in range(MaxIter):
        CF = (1.0 - t / MaxIter) ** (2.0 * t / MaxIter)

        RL = 0.05 * _levy_batch(N, dim)
        RB = np.random.randn(N, dim)

        if t < MaxIter // 3:
            # Phase 1: all use Levy
            X_new = X + P * np.random.rand(N, dim) * (
                Elite - np.random.rand(N, dim) * X
            ) * RL

        elif t < 2 * MaxIter // 3:
            # Phase 2: first half Levy, second half Brownian
            X_lev = X[:n_half] + P * np.random.rand(n_half, dim) * (
                Elite[:n_half] - np.random.rand(n_half, dim) * X[:n_half]
            ) * RL[:n_half]
            X_bro = Elite[n_half:] + P * CF * (
                Elite[n_half:] - np.random.rand(N - n_half, dim) * X[n_half:]
            ) * RB[n_half:]
            X_new = np.vstack([X_lev, X_bro])

        else:
            # Phase 3: all use Brownian
            X_new = Elite + P * CF * (
                Elite - np.random.rand(N, dim) * X
            ) * RB

        X_new = np.clip(X_new, lb, ub)
        new_fitness = np.array([func(X_new[i]) for i in range(N)]); FES += N

        imp = new_fitness < fitness
        X[imp]       = X_new[imp]
        fitness[imp] = new_fitness[imp]

        # ── FADs effect (vectorised) ──────────────────────────────────────
        fads_mask = np.random.rand(N) < P
        r1 = np.random.randint(0, N, N)
        r2 = np.random.randint(0, N, N)
        X_fad_A = X + CF * (lb + np.random.rand(N, dim) * (ub - lb))
        X_fad_B = X + (np.random.rand(N, dim) - 0.5) * 2 * (X[r1] - X[r2])
        X_fad   = np.clip(
            np.where(fads_mask[:, None], X_fad_A, X_fad_B), lb, ub
        )
        fad_fitness = np.array([func(X_fad[i]) for i in range(N)]); FES += N

        imp2 = fad_fitness < fitness
        X[imp2]       = X_fad[imp2]
        fitness[imp2] = fad_fitness[imp2]

        cur_best = np.argmin(fitness)
        if fitness[cur_best] < best_fit:
            best_fit = fitness[cur_best]
            best_pos = X[cur_best].copy()
        Elite = np.tile(best_pos, (N, 1))

        convergence_curve[t] = best_fit
        if FES >= MaxFES:
            convergence_curve[t + 1:] = best_fit
            break

    return best_fit, best_pos, convergence_curve


def _levy_batch(N, dim, beta=1.5):
    num   = math.gamma(1 + beta) * np.sin(math.pi * beta / 2)
    den   = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (num / den) ** (1 / beta)
    u     = np.random.randn(N, dim) * sigma
    v     = np.abs(np.random.randn(N, dim)) + 1e-10
    return u / (v ** (1 / beta))


if __name__ == "__main__":
    def sphere(x): return float(np.sum(x ** 2))
    bf, _, _ = MPA(sphere, -100, 100, 30)
    print(f"MPA  | Best: {bf:.6e}")
