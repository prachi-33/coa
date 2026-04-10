"""
Generate all paper figures that don't come from the benchmark runs.
Run this once before compiling the LaTeX paper.

Figures generated:
  paper/figures/c2_decay_comparison.png   ← Linear vs Exponential C2 decay
  paper/figures/c3_dynamic_comparison.png ← Static vs Dynamic C3
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       13,
    "axes.titlesize":  14,
    "axes.labelsize":  13,
    "legend.fontsize": 12,
    "lines.linewidth": 2.2,
    "grid.alpha":      0.3,
    "figure.dpi":      180,
})

T = 1000                        # iteration horizon
t = np.arange(T + 1)
ratio = t / T                   # normalised [0, 1]

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — C2 Decay Comparison
# ─────────────────────────────────────────────────────────────────────────────
k = 3.0
C2_linear = 2.0 - ratio * 2.0          # base COA: C2 = 2 - t/T_max
C2_exp    = 2.0 * np.exp(-k * ratio)   # MCOA: C2 = 2*exp(-k*t/T_max)

fig, ax = plt.subplots(figsize=(7.5, 4.5))

ax.plot(ratio, C2_linear, color="#E07B54", linestyle="--",
        label=r"Linear decay — Base COA: $C_2 = 2 - t/T_{\max}$")
ax.plot(ratio, C2_exp,    color="#3A86FF", linestyle="-",
        label=r"Exponential decay — MCOA: $C_2 = 2e^{-k\,t/T_{\max}}$, $k=3$")

# Mark the 50 % crossover points
ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
ax.annotate("50% of range", xy=(0.25, 1.0), xytext=(0.30, 1.15),
            fontsize=10, color="#E07B54",
            arrowprops=dict(arrowstyle="->", color="#E07B54", lw=1.2))
ax.annotate("50% of range", xy=(np.log(2)/k, 1.0), xytext=(0.35, 0.65),
            fontsize=10, color="#3A86FF",
            arrowprops=dict(arrowstyle="->", color="#3A86FF", lw=1.2))

ax.fill_between(ratio, C2_linear, C2_exp,
                where=(C2_exp > C2_linear), alpha=0.08, color="#3A86FF",
                label="Extended exploration region (MCOA advantage)")

ax.set_xlabel(r"Normalised Iteration $t / T_{\max}$")
ax.set_ylabel(r"Control Parameter $C_2$")
ax.set_title(r"Comparison of $C_2$ Decay Strategies")
ax.set_xlim(0, 1); ax.set_ylim(-0.05, 2.1)
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(True)

fig.tight_layout()
path1 = os.path.join(OUT_DIR, "c2_decay_comparison.png")
fig.savefig(path1, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {path1}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — C3 Dynamic vs Static Comparison
# ─────────────────────────────────────────────────────────────────────────────
C3_static  = np.full_like(ratio, 3.0)
C3_dynamic = 3.0 * (1 - ratio)         # MCOA dynamic C3

fig, ax = plt.subplots(figsize=(7.5, 4.5))

ax.plot(ratio, C3_static,  color="#E07B54", linestyle="--",
        label=r"Static $C_3 = 3$ — Base COA")
ax.plot(ratio, C3_dynamic, color="#3A86FF", linestyle="-",
        label=r"Dynamic $C_3 = 3(1 - t/T_{\max})$ — MCOA")

ax.fill_between(ratio, C3_dynamic, C3_static,
                alpha=0.08, color="#E07B54",
                label="Wasted granularity in base COA")

ax.annotate("Fine-grained near end\n(micro-exploitation)", 
            xy=(0.90, 0.30), xytext=(0.60, 0.70),
            fontsize=10, color="#3A86FF",
            arrowprops=dict(arrowstyle="->", color="#3A86FF", lw=1.2))

ax.set_xlabel(r"Normalised Iteration $t / T_{\max}$")
ax.set_ylabel(r"Food Factor $C_3$")
ax.set_title(r"Comparison of $C_3$ Strategies: Static vs.\ Dynamic")
ax.set_xlim(0, 1); ax.set_ylim(-0.1, 3.4)
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(True)

fig.tight_layout()
path2 = os.path.join(OUT_DIR, "c3_dynamic_comparison.png")
fig.savefig(path2, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {path2}")

print("\nAll paper figures generated successfully!")
print(f"Location: {OUT_DIR}")
