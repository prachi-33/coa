"""
experiments/runner.py
=====================
Parallel experiment runner using multiprocessing.

Each (algo, func_id, dim, suite) task runs in its own worker process,
so multiple tasks execute simultaneously (N_JOBS workers in config.py).

Raw result layout: results/raw/{suite}/{algo}_F{id}_D{dim}.npy
  shape: (N_RUNS, 2 + MaxIter)
  cols : [best_fitness, seed, *convergence_curve]
"""

import os
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── Algorithm name → (module, function_name) ─────────────────────────────────
_ALGO_MAP = {
    "COA":  ("algorithms.coa.coa_base",     "COA"),
    "MCOA": ("algorithms.coa.coa_modified", "MCOA"),
    "PSO":  ("algorithms.pso",              "PSO"),
    "GWO":  ("algorithms.gwo",              "GWO"),
    "HHO":  ("algorithms.hho",              "HHO"),
    "SSA":  ("algorithms.ssa",              "SSA"),
    "WOA":  ("algorithms.woa",              "WOA"),
    "SCA":  ("algorithms.sca",              "SCA"),
    "MPA":  ("algorithms.mpa",              "MPA"),
}

_ALGO_FLAGS = {
    "COA":  config.RUN_COA,
    "MCOA": config.RUN_COA_MODIFIED,
    "PSO":  config.RUN_PSO,
    "GWO":  config.RUN_GWO,
    "HHO":  config.RUN_HHO,
    "SSA":  config.RUN_SSA,
    "WOA":  config.RUN_WOA,
    "SCA":  config.RUN_SCA,
    "MPA":  config.RUN_MPA,
}

_ALGO_PARAMS = {
    "COA":  config.COA_PARAMS,
    "MCOA": config.MCOA_PARAMS,
    "PSO":  config.PSO_PARAMS,
    "GWO":  config.GWO_PARAMS,
    "HHO":  config.HHO_PARAMS,
    "SSA":  config.SSA_PARAMS,
    "WOA":  config.WOA_PARAMS,
    "SCA":  config.SCA_PARAMS,
    "MPA":  config.MPA_PARAMS,
}

_SUITE_CFG = {
    "CEC2014": {
        "module":   "benchmarks.cec2014",
        "getter":   "get_function",
        "func_ids": config.CEC2014_FUNCTIONS,
        "raw_dir":  config.RAW_CEC2014,
    },
    "CEC2017": {
        "module":   "benchmarks.cec2017",
        "getter":   "get_function",
        "func_ids": config.CEC2017_FUNCTIONS,
        "raw_dir":  config.RAW_CEC2017,
    },
}


# =============================================================================
#  Worker function  (must be top-level so multiprocessing can pickle it)
# =============================================================================

def _run_task(task: tuple):
    """
    Execute one (algo_name, suite_name, func_id, dim) experiment.
    Imports are done inside the worker to avoid pickling issues.
    """
    import importlib, sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config, numpy as np

    algo_name, suite_name, func_id, dim = task

    suite_cfg = _SUITE_CFG[suite_name]
    out_path  = os.path.join(suite_cfg["raw_dir"],
                             f"{algo_name}_F{func_id}_D{dim}.npy")

    if os.path.exists(out_path):
        return f"[SKIP] {os.path.basename(out_path)}"

    # Reconstruct benchmark function
    try:
        bench_mod  = importlib.import_module(suite_cfg["module"])
        bench_fn   = getattr(bench_mod, suite_cfg["getter"])
        fn         = bench_fn(func_id, ndim=dim)
    except Exception as e:
        return f"[ERROR] {suite_name} F{func_id} D{dim}: {e}"

    # Reconstruct algorithm function
    mod_path, fn_name = _ALGO_MAP[algo_name]
    algo_mod = importlib.import_module(mod_path)
    algo_fn  = getattr(algo_mod, fn_name)
    params   = _ALGO_PARAMS[algo_name]

    MaxIter  = params["MaxFES"] // params["N"]
    results  = np.zeros((config.N_RUNS, 2 + MaxIter))

    t0 = time.time()
    for run in range(config.N_RUNS):
        seed = config.SEED * 1000 + func_id * 100 + run
        np.random.seed(seed)

        best_fit, _, curve = algo_fn(
            func=fn, lb=fn.lb, ub=fn.ub, dim=dim, **params
        )
        c = np.full(MaxIter, best_fit)
        c[: len(curve)] = curve
        results[run, 0] = best_fit
        results[run, 1] = seed
        results[run, 2:] = c

    np.save(out_path, results)
    elapsed = time.time() - t0
    return (f"[DONE] {algo_name:5s} | {suite_name} F{func_id:02d} D{dim:2d} | "
            f"mean={results[:, 0].mean():.4e} | {elapsed:.1f}s")


# =============================================================================
#  ExperimentRunner
# =============================================================================

class ExperimentRunner:

    def run_all(self):
        from utils.logger import get_logger
        logger = get_logger(__name__)

        # Build task list
        tasks = []
        for suite_name, suite_cfg in _SUITE_CFG.items():
            for dim in config.DIMENSIONS:
                for fid in suite_cfg["func_ids"]:
                    for algo_name, enabled in _ALGO_FLAGS.items():
                        if not enabled:
                            continue
                        out = os.path.join(suite_cfg["raw_dir"],
                                           f"{algo_name}_F{fid}_D{dim}.npy")
                        if not os.path.exists(out):
                            tasks.append((algo_name, suite_name, fid, dim))

        if not tasks:
            logger.info("All results already exist. Nothing to run.")
            return

        n_jobs = min(config.N_JOBS, len(tasks))
        logger.info(f"Running {len(tasks)} tasks with {n_jobs} parallel workers...")

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_run_task, t): t for t in tasks}
            for future in as_completed(futures):
                try:
                    msg = future.result()
                    logger.info(msg)
                except Exception as exc:
                    task = futures[future]
                    logger.error(f"[FAIL] {task}: {exc}")


if __name__ == "__main__":
    ExperimentRunner().run_all()
