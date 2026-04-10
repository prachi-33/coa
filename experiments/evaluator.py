"""
experiments/evaluator.py
========================
Reads raw .npy result files, computes statistics (mean, std, best, worst,
median), generates convergence plots, box plots, and LaTeX-ready tables.
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for servers
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.logger import get_logger

logger = get_logger(__name__)

ALGO_ORDER = ["MCOA", "COA", "PSO", "GWO", "HHO", "SSA", "WOA", "SCA", "MPA"]
SUITES = {
    "CEC2014": (config.RAW_CEC2014, config.CEC2014_FUNCTIONS),
    "CEC2017": (config.RAW_CEC2017, config.CEC2017_FUNCTIONS),
}


class Evaluator:
    """Load raw results → compute stats → save tables & plots."""

    def process_all(self):
        for suite, (raw_dir, func_ids) in SUITES.items():
            for dim in config.DIMENSIONS:
                logger.info(f"[EVAL] {suite} D={dim}")
                self._process_suite(suite, raw_dir, func_ids, dim)

    # ── Per-suite processing ──────────────────────────────────────────────

    def _process_suite(self, suite, raw_dir, func_ids, dim):
        # Collect stats for every algo × function
        stats = {}   # stats[algo][fid] = {"mean", "std", "best", "worst", "median"}

        for algo in ALGO_ORDER:
            stats[algo] = {}
            for fid in func_ids:
                path = os.path.join(raw_dir, f"{algo}_F{fid}_D{dim}.npy")
                if not os.path.exists(path):
                    continue
                data     = np.load(path)
                fitnesses = data[:, 0]
                stats[algo][fid] = {
                    "mean":   fitnesses.mean(),
                    "std":    fitnesses.std(),
                    "best":   fitnesses.min(),
                    "worst":  fitnesses.max(),
                    "median": np.median(fitnesses),
                    "curve":  data[:, 2:].mean(axis=0),   # mean convergence
                }

        self._save_table(stats, suite, dim, func_ids)
        self._save_convergence_plots(stats, suite, dim, func_ids)
        self._save_boxplots(stats, suite, dim, func_ids)

    # ── LaTeX / CSV table ─────────────────────────────────────────────────

    def _save_table(self, stats, suite, dim, func_ids):
        tag  = f"{suite}_D{dim}"
        csv_path = os.path.join(config.TABLES_DIR, f"{tag}_results.csv")
        tex_path = os.path.join(config.TABLES_DIR, f"{tag}_results.tex")

        header = "Func," + ",".join(
            f"{a}_mean,{a}_std" for a in ALGO_ORDER
        )
        rows = [header]

        for fid in func_ids:
            row = [f"F{fid}"]
            for algo in ALGO_ORDER:
                s = stats[algo].get(fid)
                if s:
                    row += [f"{s['mean']:.4e}", f"{s['std']:.4e}"]
                else:
                    row += ["N/A", "N/A"]
            rows.append(",".join(row))

        with open(csv_path, "w") as f:
            f.write("\n".join(rows))
        logger.info(f"  Table saved → {os.path.basename(csv_path)}")

        # Simple LaTeX table
        with open(tex_path, "w") as f:
            cols = "l" + "rr" * len(ALGO_ORDER)
            f.write(f"\\begin{{tabular}}{{{cols}}}\n\\hline\n")
            f.write("Func & " + " & ".join(
                f"\\multicolumn{{2}}{{c}}{{{a}}}" for a in ALGO_ORDER
            ) + " \\\\\n")
            f.write(" & " + " & ".join(
                "Mean & Std" for _ in ALGO_ORDER
            ) + " \\\\\n\\hline\n")
            for fid in func_ids:
                row = [f"$f_{{{fid}}}$"]
                for algo in ALGO_ORDER:
                    s = stats[algo].get(fid)
                    if s:
                        row += [f"{s['mean']:.2e}", f"{s['std']:.2e}"]
                    else:
                        row += ["--", "--"]
                f.write(" & ".join(row) + " \\\\\n")
            f.write("\\hline\n\\end{tabular}\n")
        logger.info(f"  LaTeX saved → {os.path.basename(tex_path)}")

    # ── Convergence Plots ─────────────────────────────────────────────────

    def _save_convergence_plots(self, stats, suite, dim, func_ids):
        colors = plt.cm.tab10.colors
        for fid in func_ids:
            fig, ax = plt.subplots(figsize=(7, 4))
            plotted = False
            for idx, algo in enumerate(ALGO_ORDER):
                s = stats[algo].get(fid)
                if s is None:
                    continue
                ax.semilogy(s["curve"], label=algo, color=colors[idx % 10], linewidth=1.5)
                plotted = True
            if not plotted:
                plt.close(fig)
                continue
            ax.set_title(f"{suite} F{fid} (D={dim}) — Convergence", fontsize=11)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Best Fitness (log scale)")
            ax.legend(fontsize=7, ncol=3, loc="upper right")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            out = os.path.join(config.CONV_DIR, f"{suite}_F{fid}_D{dim}_conv.png")
            fig.savefig(out, dpi=150)
            plt.close(fig)
        logger.info(f"  Convergence plots saved → {config.CONV_DIR}")

    # ── Box Plots ─────────────────────────────────────────────────────────

    def _save_boxplots(self, stats, suite, dim, func_ids):
        for fid in func_ids:
            data   = []
            labels = []
            for algo in ALGO_ORDER:
                # reload raw for box plot
                path = os.path.join(
                    config.RAW_CEC2014 if "2014" in suite else config.RAW_CEC2017,
                    f"{algo}_F{fid}_D{dim}.npy",
                )
                if not os.path.exists(path):
                    continue
                data.append(np.load(path)[:, 0])
                labels.append(algo)
            if not data:
                continue

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.boxplot(data, labels=labels, patch_artist=True,
                       medianprops=dict(color="red", linewidth=2))
            ax.set_title(f"{suite} F{fid} (D={dim}) — Box Plot", fontsize=11)
            ax.set_ylabel("Best Fitness")
            ax.set_yscale("symlog")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            out = os.path.join(config.BOX_DIR, f"{suite}_F{fid}_D{dim}_box.png")
            fig.savefig(out, dpi=150)
            plt.close(fig)
        logger.info(f"  Box plots saved → {config.BOX_DIR}")


if __name__ == "__main__":
    Evaluator().process_all()
