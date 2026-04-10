# =============================================================================
# config.py — Central Configuration  (VERY IMPORTANT)
# All parameters for algorithms, benchmarks, and experiments live here.
# =============================================================================

import os
import multiprocessing

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED    = 42
VERBOSE = True

# ── Parallelism ───────────────────────────────────────────────────────────────
# Number of parallel workers for the experiment runner.
# Set to 1 to run sequentially (for debugging).
N_JOBS = max(1, multiprocessing.cpu_count() - 1)

# ── Problem Dimensions ────────────────────────────────────────────────────────
DIMENSIONS = [10, 30]

# ── Benchmark Suites ──────────────────────────────────────────────────────────
CEC2014_FUNCTIONS     = list(range(1, 31))   # f1 – f30
CEC2017_FUNCTIONS     = list(range(1, 30))   # f1 – f29
ENGINEERING_FUNCTIONS = list(range(1, 6))    # 5 problems: Spring, PV, WeldedBeam, SpeedReducer, Bearing
LOWER_BOUND = -100.0
UPPER_BOUND =  100.0

# ── Experiment Settings ───────────────────────────────────────────────────────
N_RUNS     = 30          # Independent runs per algorithm per function (CEC benchmarks)
N_RUNS_ENG = 50          # Independent runs for engineering problems
POPULATION = 30          # Swarm / population size
MAX_FES    = 10_000      # Max FES per run for CEC benchmarks
MAX_FES_ENG= 60_000      # Max FES per run for engineering problems (harder, more budget)
# Recommended: MAX_FES = POPULATION * 1000  (e.g. 30000 for D=10, 100000 for D=30)

# ── Algorithm Flags (set False to skip) ───────────────────────────────────────
RUN_COA          = True
RUN_COA_MODIFIED = True
RUN_PSO          = True
RUN_GWO          = True
RUN_HHO          = True
RUN_SSA          = True
RUN_WOA          = True
RUN_SCA          = True
RUN_MPA          = True

# ── Algorithm Parameters ──────────────────────────────────────────────────────
COA_PARAMS  = dict(N=POPULATION, MaxFES=MAX_FES)
MCOA_PARAMS = dict(N=POPULATION, MaxFES=MAX_FES)
PSO_PARAMS  = dict(N=POPULATION, MaxFES=MAX_FES, w=0.7, c1=1.5, c2=1.5)
GWO_PARAMS  = dict(N=POPULATION, MaxFES=MAX_FES)
HHO_PARAMS  = dict(N=POPULATION, MaxFES=MAX_FES)
SSA_PARAMS  = dict(N=POPULATION, MaxFES=MAX_FES)
WOA_PARAMS  = dict(N=POPULATION, MaxFES=MAX_FES)
SCA_PARAMS  = dict(N=POPULATION, MaxFES=MAX_FES)
MPA_PARAMS  = dict(N=POPULATION, MaxFES=MAX_FES)

# ── Statistical Test Settings ─────────────────────────────────────────────────
ALPHA          = 0.05      # Significance level for Wilcoxon test
REFERENCE_ALGO = "MCOA"    # Algorithm all others are compared against

# ── Output Paths  (VERY IMPORTANT) ───────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
RAW_CEC2014     = os.path.join(RESULTS_DIR, "raw",       "cec2014")
RAW_CEC2017     = os.path.join(RESULTS_DIR, "raw",       "cec2017")
RAW_ENGINEERING = os.path.join(RESULTS_DIR, "raw",       "engineering")
TABLES_DIR      = os.path.join(RESULTS_DIR, "processed", "tables")
RANKINGS_DIR    = os.path.join(RESULTS_DIR, "processed", "rankings")
CONV_DIR        = os.path.join(RESULTS_DIR, "plots",     "convergence")
BOX_DIR         = os.path.join(RESULTS_DIR, "plots",     "boxplots")

# Auto-create all output directories
for _d in [RAW_CEC2014, RAW_CEC2017, RAW_ENGINEERING,
           TABLES_DIR, RANKINGS_DIR, CONV_DIR, BOX_DIR]:
    os.makedirs(_d, exist_ok=True)
