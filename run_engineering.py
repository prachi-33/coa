"""
run_engineering.py
==================
Dedicated runner for the 5 engineering optimisation problems in the
prescribed order:

  F1  Tension/Compression Spring   (3D)
  F2  Pressure Vessel              (4D)
  F3  Welded Beam                  (4D)
  F4  Speed Reducer                (7D)
  F5  Rolling Element Bearing     (10D)

Settings  : 50 independent runs, 60 000 FEs per run
Algorithms: COA (base) + MCOA (modified)

Results are stored as .npy files in:
    results/raw/engineering/{ALGO}_F{id}_DEng.npy

Each file shape: (N_RUNS, 2 + MaxIter)
    col 0        : best fitness for that run
    col 1        : RNG seed used
    cols 2..end  : mean convergence curve

After all runs the script calls the Evaluator to compute statistics,
build convergence plots, box plots and CSV / LaTeX tables.

Usage
-----
    python run_engineering.py           # full 50 runs × 60k FEs
    python run_engineering.py --force   # delete old results and re-run
"""

import os
import sys
import time
import glob
import argparse
import numpy as np

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config
from utils.logger import get_logger
from algorithms.coa.coa_base     import COA
from algorithms.coa.coa_modified import MCOA
from benchmarks.engineering import get_function, get_problem_name, _ENGINEERING_PROBLEMS

logger = get_logger("engineering_runner")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration (inherits global defaults; override here if needed)
# ─────────────────────────────────────────────────────────────────────────────
N_RUNS    = config.N_RUNS_ENG    # 50 runs for engineering
MAX_FES   = config.MAX_FES_ENG   # 60,000 FEs per run
POPULATION= config.POPULATION    # 30
SEED      = config.SEED          # 42
RAW_DIR   = config.RAW_ENGINEERING

# Function IDs to run — in desired order
FUNC_IDS  = [1, 2, 3, 4, 5]   # Spring → PV → WB → SR → Bearing

# Algorithms enabled
from algorithms.pso import PSO
from algorithms.gwo import GWO
from algorithms.hho import HHO
from algorithms.ssa import SSA
from algorithms.woa import WOA
from algorithms.sca import SCA
from algorithms.mpa import MPA

ALGOS = {}
if config.RUN_COA:          ALGOS["COA"]  = (COA,  config.COA_PARAMS)
if config.RUN_COA_MODIFIED: ALGOS["MCOA"] = (MCOA, config.MCOA_PARAMS)
if config.RUN_PSO:          ALGOS["PSO"]  = (PSO,  config.PSO_PARAMS)
if config.RUN_GWO:          ALGOS["GWO"]  = (GWO,  config.GWO_PARAMS)
if config.RUN_HHO:          ALGOS["HHO"]  = (HHO,  config.HHO_PARAMS)
if config.RUN_SSA:          ALGOS["SSA"]  = (SSA,  config.SSA_PARAMS)
if config.RUN_WOA:          ALGOS["WOA"]  = (WOA,  config.WOA_PARAMS)
if config.RUN_SCA:          ALGOS["SCA"]  = (SCA,  config.SCA_PARAMS)
if config.RUN_MPA:          ALGOS["MPA"]  = (MPA,  config.MPA_PARAMS)

# -----------------------------------------------------------------------------
# Banner
# -----------------------------------------------------------------------------
def print_banner():
    logger.info("=" * 65)
    logger.info("  Engineering Optimisation Benchmark")
    logger.info("  Problem order: Spring -> PV -> WeldedBeam -> SpeedReducer -> Bearing")
    logger.info(f"  Algorithms    : {list(ALGOS.keys())}")
    logger.info(f"  Runs per algo : {N_RUNS}")
    logger.info(f"  Max FES       : {MAX_FES:,}")
    logger.info(f"  Population    : {POPULATION}")
    logger.info(f"  Output dir    : {RAW_DIR}")
    logger.info("=" * 65)


# -----------------------------------------------------------------------------
# Stale-result cleanup
# -----------------------------------------------------------------------------
def purge_old_results():
    """Remove .npy files from a previous run (e.g. after problem reordering)."""
    pattern = os.path.join(RAW_DIR, "*.npy")
    files   = glob.glob(pattern)
    if not files:
        logger.info("No old engineering results to remove.")
        return
    for f in files:
        os.remove(f)
        logger.info(f"  Removed: {os.path.basename(f)}")
    logger.info(f"Removed {len(files)} stale result file(s).")


# -----------------------------------------------------------------------------
# Core run logic
# -----------------------------------------------------------------------------
def run_one(algo_name, algo_fn, algo_params, fn, fid):
    """Run N_RUNS trials of algo_fn on fn and save the .npy file."""

    MaxIter  = algo_params["MaxFES"] // algo_params["N"]
    results  = np.zeros((N_RUNS, 2 + MaxIter))
    out_path = os.path.join(RAW_DIR, f"{algo_name}_F{fid}_DEng.npy")

    if os.path.exists(out_path):
        logger.info(f"  [SKIP] {os.path.basename(out_path)} already exists - skipping.")
        return

    logger.info(
        f"  >> {algo_name:5s} | F{fid} {get_problem_name(fid)}"
        f"  | dim={fn.dim} | {N_RUNS} runs x {MAX_FES:,} FEs"
    )
    t0 = time.time()

    for run in range(N_RUNS):
        seed = SEED * 1000 + fid * 100 + run
        np.random.seed(seed)

        best_fit, _, curve = algo_fn(
            func=fn,
            lb=fn.lb,
            ub=fn.ub,
            dim=fn.dim,
            **algo_params,
        )

        # Pad / trim convergence curve to MaxIter length
        c = np.full(MaxIter, best_fit)
        c[: len(curve)] = curve[:MaxIter]

        results[run, 0] = best_fit
        results[run, 1] = seed
        results[run, 2:] = c

        if (run + 1) % 10 == 0 or run == 0:
            logger.info(
                f"     run {run+1:>2}/{N_RUNS}  "
                f"best={best_fit:.6e}  "
                f"elapsed={time.time()-t0:.1f}s"
            )

    elapsed = time.time() - t0
    np.save(out_path, results)

    fits = results[:, 0]
    logger.info(
        f"  OK {os.path.basename(out_path)}"
        f"  mean={fits.mean():.4e}  std={fits.std():.4e}"
        f"  best={fits.min():.4e}  [{elapsed:.1f}s]"
    )


def run_all_engineering(force: bool = False):
    print_banner()

    if force:
        logger.info("\n[--force] Removing old engineering results...")
        purge_old_results()

    total_combos = len(FUNC_IDS) * len(ALGOS)
    done = 0

    for fid in FUNC_IDS:
        fn_name = get_problem_name(fid)
        logger.info(f"\n" + "-"*65)
        logger.info(f"  Problem F{fid}: {fn_name}")
        logger.info("-"*65)

        try:
            fn = get_function(fid)
        except Exception as exc:
            logger.error(f"  Could not instantiate F{fid}: {exc}")
            continue

        for algo_name, (algo_fn, algo_params) in ALGOS.items():
            run_one(algo_name, algo_fn, algo_params, fn, fid)
            done += 1
            logger.info(f"  Progress: {done}/{total_combos} algo×problem combinations done.")

    logger.info("\n" + "=" * 65)
    logger.info("  All engineering runs complete.")
    logger.info("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing (stats + plots)
# ─────────────────────────────────────────────────────────────────────────────
def post_process():
    """Run the evaluator on engineering results only."""
    logger.info("\n[POST-PROCESS] Computing statistics, tables, and plots...")
    from experiments.evaluator import Evaluator, SUITES
    import config
    ev = Evaluator()
    raw_dir   = config.RAW_ENGINEERING
    func_ids  = config.ENGINEERING_FUNCTIONS
    ev._process_suite("Engineering", raw_dir, func_ids, "Eng")
    logger.info("  Post-processing complete.")
    logger.info(f"  Tables → {config.TABLES_DIR}")
    logger.info(f"  Plots  → {config.CONV_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run engineering optimisation benchmark (50 runs × 60k FEs)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete all existing engineering .npy results and re-run from scratch.",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Skip evaluator / plot generation after runs.",
    )
    args = parser.parse_args()

    np.random.seed(SEED)
    run_all_engineering(force=args.force)

    if not args.no_postprocess:
        post_process()
