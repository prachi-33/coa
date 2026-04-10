"""HHO — Vectorized (batch escape/besiege phases, no inner for-loop)."""

import numpy as np
import math


def HHO(func, lb, ub, dim, N=30, MaxFES=60000):
    lb = np.array([lb] * dim) if np.isscalar(lb) else np.array(lb, dtype=float)
    ub = np.array([ub] * dim) if np.isscalar(ub) else np.array(ub, dtype=float)

    MaxIter = MaxFES // N
    FES     = 0

    X       = lb + (ub - lb) * np.random.rand(N, dim)
    fitness = np.array([func(X[i]) for i in range(N)]); FES += N

    best_idx     = np.argmin(fitness)
    rabbit_pos   = X[best_idx].copy()
    rabbit_fit   = fitness[best_idx]

    convergence_curve = np.zeros(MaxIter)

    for t in range(MaxIter):
        E0 = 2 * np.random.rand() - 1
        E  = 2 * E0 * (1.0 - t / MaxIter)
        X_mean = X.mean(axis=0)

        for i in range(N):
            r = np.random.rand()
            
            if abs(E) >= 1:
                # ── Exploration ──────────────────────────────────────────────
                if r >= 0.5:
                    rand_idx = np.random.randint(0, N)
                    X[i] = X[rand_idx] - np.random.rand() * np.abs(
                        X[rand_idx] - 2 * np.random.rand() * X[i]
                    )
                else:
                    X[i] = (rabbit_pos - X_mean) - np.random.rand() * (
                        lb + np.random.rand(dim) * (ub - lb)
                    )
            else:
                # ── Exploitation ─────────────────────────────────────────────
                J = 2 * (1 - np.random.rand())
                
                if r >= 0.5 and abs(E) >= 0.5:
                    # Soft besiege
                    X[i] = rabbit_pos - E * np.abs(J * rabbit_pos - X[i])
                elif r >= 0.5 and abs(E) < 0.5:
                    # Hard besiege
                    X[i] = rabbit_pos - E * np.abs(rabbit_pos - X[i])
                elif r < 0.5 and abs(E) >= 0.5:
                    # Soft besiege with rapid dives
                    Y = rabbit_pos - E * np.abs(J * rabbit_pos - X[i])
                    Y = np.clip(Y, lb, ub)
                    Fy = func(Y); FES += 1
                    
                    if Fy < fitness[i]:
                        X[i] = Y
                        fitness[i] = Fy
                    else:
                        Z = Y + np.random.rand(dim) * _levy_step(dim)
                        Z = np.clip(Z, lb, ub)
                        Fz = func(Z); FES += 1
                        if Fz < fitness[i]:
                            X[i] = Z
                            fitness[i] = Fz
                            
                else:
                    # Hard besiege with rapid dives
                    Y = rabbit_pos - E * np.abs(J * rabbit_pos - X_mean)
                    Y = np.clip(Y, lb, ub)
                    Fy = func(Y); FES += 1
                    
                    if Fy < fitness[i]:
                        X[i] = Y
                        fitness[i] = Fy
                    else:
                        Z = Y + np.random.rand(dim) * _levy_step(dim)
                        Z = np.clip(Z, lb, ub)
                        Fz = func(Z); FES += 1
                        if Fz < fitness[i]:
                            X[i] = Z
                            fitness[i] = Fz

            # Regular evaluation
            X[i] = np.clip(X[i], lb, ub)
            
            if FES < MaxFES:
                f = func(X[i]); FES += 1
                if f < fitness[i]:
                    fitness[i] = f
                if f < rabbit_fit:
                    rabbit_fit = f
                    rabbit_pos = X[i].copy()

        convergence_curve[t] = rabbit_fit
        if FES >= MaxFES:
            convergence_curve[t + 1:] = rabbit_fit
            break

    return rabbit_fit, rabbit_pos, convergence_curve


def _levy_step(dim, beta=1.5):
    """Generates a single Lévy step vector."""
    num   = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    den   = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (num / den) ** (1 / beta)
    u     = np.random.randn(dim) * sigma
    v     = np.abs(np.random.randn(dim)) + 1e-10
    return u / (v ** (1 / beta))


if __name__ == "__main__":
    def sphere(x): return float(np.sum(x ** 2))
    bf, _, _ = HHO(sphere, -100, 100, 30)
    print(f"HHO  | Best: {bf:.6e}")
