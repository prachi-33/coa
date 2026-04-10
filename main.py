# =============================================================================
# main.py — Entry Point
# Runs the full experiment pipeline: benchmark → optimize → evaluate → report
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import config
from experiments.runner    import ExperimentRunner
from experiments.evaluator import Evaluator
from statistics.wilcoxon   import WilcoxonTest
from utils.logger          import get_logger

logger = get_logger(__name__)


def main():
    np.random.seed(config.SEED)

    logger.info("=" * 60)
    logger.info("  COA Benchmark Experiment Suite — Starting")
    logger.info(f"  Dimensions : {config.DIMENSIONS}")
    logger.info(f"  Runs       : {config.N_RUNS}")
    logger.info(f"  Max FES    : {config.MAX_FES}")
    logger.info(f"  Population : {config.POPULATION}")
    logger.info("=" * 60)

    # ── 1. Run all algorithms on all benchmark functions ──────────────────
    logger.info("\n[PHASE 1] Running experiments...")
    runner = ExperimentRunner()
    runner.run_all()

    # ── 2. Evaluate / aggregate raw results ───────────────────────────────
    logger.info("\n[PHASE 2] Evaluating results...")
    evaluator = Evaluator()
    evaluator.process_all()

    # ── 3. Statistical tests ──────────────────────────────────────────────
    logger.info("\n[PHASE 3] Running statistical tests...")
    wt = WilcoxonTest()
    wt.run_all()

    logger.info("\n" + "=" * 60)
    logger.info("  Experiment complete.")
    logger.info(f"  Tables     → {config.TABLES_DIR}")
    logger.info(f"  Rankings   → {config.RANKINGS_DIR}")
    logger.info(f"  Conv plots → {config.CONV_DIR}")
    logger.info(f"  Box plots  → {config.BOX_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
