"""
utils/helpers.py
================
Miscellaneous helper functions shared across the project.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── File helpers ─────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> str:
    """Create directory if it does not exist; return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def list_result_files(directory: str, extension: str = ".npy"):
    """Return sorted list of result files in a directory."""
    return sorted(
        f for f in os.listdir(directory) if f.endswith(extension)
    )


# ── Maths helpers ────────────────────────────────────────────────────────────

def levy_flight(dim: int, beta: float = 1.5) -> np.ndarray:
    """
    Generate a Lévy flight step vector of length `dim`.
    Uses the Mantegna algorithm.
    """
    import math
    num   = math.gamma(1 + beta) * np.sin(math.pi * beta / 2)
    den   = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (num / den) ** (1 / beta)
    u     = np.random.randn(dim) * sigma
    v     = np.abs(np.random.randn(dim)) + 1e-10
    return u / (v ** (1 / beta))


def clip_bounds(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Simple boundary clamping."""
    return np.clip(x, lb, ub)


def reflect_bounds(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Reflective boundary handling to avoid edge pile-up."""
    x = x.copy()
    for j in range(len(x)):
        while x[j] < lb[j] or x[j] > ub[j]:
            if x[j] < lb[j]:
                x[j] = 2 * lb[j] - x[j]
            if x[j] > ub[j]:
                x[j] = 2 * ub[j] - x[j]
    return x


# ── Statistics helpers ────────────────────────────────────────────────────────

def compute_stats(fitnesses: np.ndarray) -> dict:
    """
    Compute descriptive statistics from an array of fitness values.

    Returns dict with keys: mean, std, best, worst, median.
    """
    return {
        "mean":   float(np.mean(fitnesses)),
        "std":    float(np.std(fitnesses)),
        "best":   float(np.min(fitnesses)),
        "worst":  float(np.max(fitnesses)),
        "median": float(np.median(fitnesses)),
    }


def format_sci(value: float, precision: int = 2) -> str:
    """Format a float in scientific notation, e.g. 1.23e-05."""
    return f"{value:.{precision}e}"


# ── Plot helpers ──────────────────────────────────────────────────────────────

def save_convergence(curves: dict, title: str, out_path: str):
    """
    Save a convergence plot.

    Parameters
    ----------
    curves   : {algo_name: np.ndarray}   mean convergence curves
    title    : plot title string
    out_path : absolute path for the output .png
    """
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(7, 4))
    for idx, (name, curve) in enumerate(curves.items()):
        ax.semilogy(curve, label=name, color=colors[idx % 10], linewidth=1.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness (log scale)")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_boxplot(data: dict, title: str, out_path: str):
    """
    Save a box plot comparing algorithms.

    Parameters
    ----------
    data     : {algo_name: np.ndarray of fitness values}
    title    : plot title
    out_path : absolute path for the output .png
    """
    labels = list(data.keys())
    values = list(data.values())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(values, labels=labels, patch_artist=True,
               medianprops=dict(color="red", linewidth=2))
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Best Fitness")
    ax.set_yscale("symlog")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
