"""
statistics/wilcoxon.py
======================
Pairwise Wilcoxon signed-rank tests comparing every algorithm against
the REFERENCE_ALGO (MCOA by default) defined in config.py.

Output:
    results/processed/rankings/{suite}_D{dim}_wilcoxon.csv
    results/processed/rankings/{suite}_D{dim}_wilcoxon.tex
    results/processed/rankings/{suite}_D{dim}_rankings.csv  (Friedman rank)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from scipy.stats import wilcoxon, friedmanchisquare, rankdata
    _SCIPY = True
except ImportError:
    _SCIPY = False

ALGO_ORDER = ["MCOA", "COA", "PSO", "GWO", "HHO", "SSA", "WOA", "SCA", "MPA"]
SUITES = {
    "CEC2014": (config.RAW_CEC2014, config.CEC2014_FUNCTIONS),
    "CEC2017": (config.RAW_CEC2017, config.CEC2017_FUNCTIONS),
}


class WilcoxonTest:

    def run_all(self):
        if not _SCIPY:
            logger.warning("scipy not installed — skipping statistical tests.")
            return
        for suite, (raw_dir, func_ids) in SUITES.items():
            for dim in config.DIMENSIONS:
                logger.info(f"[STAT] {suite} D={dim}")
                self._run_suite(suite, raw_dir, func_ids, dim)

    def _run_suite(self, suite, raw_dir, func_ids, dim):
        ref  = config.REFERENCE_ALGO
        rows = []
        header = ["Func"] + [a for a in ALGO_ORDER if a != ref] + ["W+", "W-", "W="]

        # ---- Collect fitness matrices ------------------------------------
        fitness_data = {}    # algo -> list of (fid, fits)
        for algo in ALGO_ORDER:
            fitness_data[algo] = {}
            for fid in func_ids:
                path = os.path.join(raw_dir, f"{algo}_F{fid}_D{dim}.npy")
                if os.path.exists(path):
                    fitness_data[algo][fid] = np.load(path)[:, 0]

        # ---- Pairwise Wilcoxon for each function ------------------------
        summary = {a: {"win": 0, "loss": 0, "tie": 0}
                   for a in ALGO_ORDER if a != ref}

        csv_rows = ["Func," + ",".join(
            f"{a}_stat,{a}_p,{a}_result" for a in ALGO_ORDER if a != ref
        )]

        for fid in func_ids:
            ref_fits = fitness_data[ref].get(fid)
            cols = [f"F{fid}"]
            for algo in ALGO_ORDER:
                if algo == ref:
                    continue
                algo_fits = fitness_data[algo].get(fid)
                if ref_fits is None or algo_fits is None:
                    cols += ["N/A", "N/A", "N/A"]
                    continue

                diff = ref_fits - algo_fits
                if np.all(diff == 0):
                    stat, p, result = 0.0, 1.0, "="
                else:
                    stat, p = wilcoxon(ref_fits, algo_fits,
                                       alternative="two-sided",
                                       zero_method="zsplit")
                    if p < config.ALPHA:
                        result = "+" if ref_fits.mean() < algo_fits.mean() else "-"
                    else:
                        result = "="

                summary[algo][{"+" : "win", "-": "loss", "=": "tie"}[result]] += 1
                cols += [f"{stat:.4f}", f"{p:.4f}", result]

            csv_rows.append(",".join(str(c) for c in cols))

        # ---- Summary row ------------------------------------------------
        w_row = ["W/L/T"]
        for algo in ALGO_ORDER:
            if algo == ref:
                continue
            s = summary[algo]
            w_row += [f"{s['win']}/{s['loss']}/{s['tie']}", "", ""]
        csv_rows.append(",".join(w_row))

        tag = f"{suite}_D{dim}"
        csv_path = os.path.join(config.RANKINGS_DIR, f"{tag}_wilcoxon.csv")
        with open(csv_path, "w") as f:
            f.write("\n".join(csv_rows))
        logger.info(f"  Wilcoxon saved → {os.path.basename(csv_path)}")

        # ---- Friedman rankings ------------------------------------------
        self._friedman_ranks(fitness_data, func_ids, suite, dim)

    def _friedman_ranks(self, fitness_data, func_ids, suite, dim):
        """Compute average Friedman ranking across all functions."""
        rank_sums = {a: 0.0 for a in ALGO_ORDER}
        count     = 0

        for fid in func_ids:
            means = []
            valid_algos = []
            for algo in ALGO_ORDER:
                d = fitness_data[algo].get(fid)
                if d is not None:
                    means.append(d.mean())
                    valid_algos.append(algo)
            if not means:
                continue
            ranks = rankdata(means)
            for algo, r in zip(valid_algos, ranks):
                rank_sums[algo] += r
            count += 1

        if count == 0:
            return

        avg_ranks = {a: rank_sums[a] / count for a in ALGO_ORDER}
        sorted_algos = sorted(avg_ranks, key=lambda a: avg_ranks[a])

        tag      = f"{suite}_D{dim}"
        csv_path = os.path.join(config.RANKINGS_DIR, f"{tag}_rankings.csv")
        with open(csv_path, "w") as f:
            f.write("Rank,Algorithm,AvgFriedmanRank\n")
            for rank_pos, algo in enumerate(sorted_algos, start=1):
                f.write(f"{rank_pos},{algo},{avg_ranks[algo]:.4f}\n")
        logger.info(f"  Friedman ranks saved → {os.path.basename(csv_path)}")


if __name__ == "__main__":
    WilcoxonTest().run_all()
