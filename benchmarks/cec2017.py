"""
Benchmarks — CEC-2017 wrapper via opfunu
=========================================
Uses the opfunu library (pip install opfunu) which provides standardised
CEC-2017 functions with correct evaluation counts and shift/rotation data.

Usage:
    from benchmarks.cec2017 import get_function, get_all_functions
    fn = get_function(func_id=4, ndim=30)
    fitness = fn(x)
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
    Return a callable CEC-2017 function object.

    Parameters
    ----------
    func_id : int   1 … 29  (f2 is excluded in the official competition)
    ndim    : int   10, 30, 50, or 100

    Returns
    -------
    fn : callable wrapper with fn.lb, fn.ub, fn.dim, fn.name, fn(x)->float
    """
    if not _OPFUNU:
        raise ImportError(
            "opfunu is required for CEC benchmarks.\n"
            "Install via:  pip install opfunu"
        )
    assert 1 <= func_id <= 29, "CEC-2017: func_id must be in [1, 29]"

    cls_name = f"F{func_id}2017"
    cls      = getattr(opfunu.cec_based, cls_name)
    fn_obj   = cls(ndim=ndim)

    return _FuncWrapper(fn_obj, func_id, ndim, suite="CEC2017")


def get_all_functions(ndim: int = 30):
    """
    Return a list of all 29 CEC-2017 function wrappers.
    """
    return [get_function(i, ndim) for i in range(1, 30)]


# ── Internal wrapper ──────────────────────────────────────────────────────────

class _FuncWrapper:

    def __init__(self, fn_obj, func_id, ndim, suite):
        self._fn   = fn_obj
        self.id    = func_id
        self.dim   = ndim
        self.suite = suite
        self.name  = f"{suite}_F{func_id}"
        self.lb    = fn_obj.lb[0] if hasattr(fn_obj, "lb") else -100.0
        self.ub    = fn_obj.ub[0] if hasattr(fn_obj, "ub") else  100.0

    def __call__(self, x: np.ndarray) -> float:
        return float(self._fn.evaluate(np.asarray(x, dtype=float)))

    def __repr__(self):
        return f"<{self.name} dim={self.dim}>"


# ── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    fn = get_function(1, ndim=30)
    x  = np.zeros(30)
    print(f"{fn.name}: f(0) = {fn(x):.6e}   lb={fn.lb}  ub={fn.ub}")
    print("CEC-2017 wrapper OK.")
