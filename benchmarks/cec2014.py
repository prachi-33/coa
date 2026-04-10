"""
Benchmarks — CEC-2014 wrapper via opfunu
=========================================
Uses the opfunu library (pip install opfunu) which provides standardised
CEC-2014 functions with correct evaluation counts and shift/rotation data.

Usage:
    from benchmarks.cec2014 import get_function, get_all_functions
    fn = get_function(func_id=1, ndim=30)
    fitness = fn.evaluate(x)    # or fn(x)
"""

import numpy as np

try:
    import opfunu
    _OPFUNU = True
except ImportError:
    _OPFUNU = False


# ── Public API ────────────────────────────────────────────────────────────────

def get_function(func_id: int, ndim: int = 30):
    """
    Return a callable CEC-2014 function object.

    Parameters
    ----------
    func_id : int   1 … 30
    ndim    : int   10 or 30 (paper standard)

    Returns
    -------
    fn : callable wrapper with attributes
         fn.lb, fn.ub, fn.dim, fn.name, fn(x)->float
    """
    if not _OPFUNU:
        raise ImportError(
            "opfunu is required for CEC benchmarks.\n"
            "Install via:  pip install opfunu"
        )
    assert 1 <= func_id <= 30, "CEC-2014: func_id must be in [1, 30]"

    # opfunu naming: CEC_2014_F{id}
    cls_name = f"F{func_id}2014"
    cls      = getattr(opfunu.cec_based, cls_name)
    fn_obj   = cls(ndim=ndim)

    return _FuncWrapper(fn_obj, func_id, ndim, suite="CEC2014")


def get_all_functions(ndim: int = 30):
    """
    Return a list of all 30 CEC-2014 function wrappers.
    """
    return [get_function(i, ndim) for i in range(1, 31)]


# ── Internal wrapper ──────────────────────────────────────────────────────────

class _FuncWrapper:
    """Thin wrapper that standardises the opfunu API for this project."""

    def __init__(self, fn_obj, func_id, ndim, suite):
        self._fn    = fn_obj
        self.id     = func_id
        self.dim    = ndim
        self.suite  = suite
        self.name   = f"{suite}_F{func_id}"
        self.lb     = fn_obj.lb[0] if hasattr(fn_obj, "lb") else -100.0
        self.ub     = fn_obj.ub[0] if hasattr(fn_obj, "ub") else  100.0

    def __call__(self, x: np.ndarray) -> float:
        return float(self._fn.evaluate(np.asarray(x, dtype=float)))

    def __repr__(self):
        return f"<{self.name} dim={self.dim}>"


# ── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    fn = get_function(1, ndim=30)
    x  = np.zeros(30)
    print(f"{fn.name}: f(0) = {fn(x):.6e}   lb={fn.lb}  ub={fn.ub}")
    print("CEC-2014 wrapper OK.")
