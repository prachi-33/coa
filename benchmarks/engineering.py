import numpy as np

class EngineeringProblem:
    def __init__(self, name, dim, lb, ub):
        self.name = name
        self.dim = dim
        # Ensure lb and ub are numpy arrays
        self.lb = np.array(lb) if isinstance(lb, (list, tuple)) else np.full(dim, lb)
        self.ub = np.array(ub) if isinstance(ub, (list, tuple)) else np.full(dim, ub)
        self.suite = "Engineering"

    def evaluate(self, x):
        raise NotImplementedError

    def __call__(self, x):
        # Clip x to bounds just to be safe, though optimizer should handle this
        x = np.clip(x, self.lb, self.ub)
        return float(self.evaluate(np.asarray(x, dtype=float)))
    
    def __repr__(self):
        return f"<{self.name} dim={self.dim}>"

# ─────────────────────────────────────────────────────────────────────────────
# F1 — Tension/Compression Spring  (3D)
# Minimise weight of a tension/compression spring subject to 4 constraints.
# Variables: d (wire diameter), D (coil diameter), N (number of active coils)
# Reference: Arora (1989), Belegundu (1982)
# ─────────────────────────────────────────────────────────────────────────────
class TensionCompressionSpring(EngineeringProblem):
    def __init__(self):
        super().__init__("TensionCompressionSpring", 3,
                         lb=[0.05, 0.25, 2.0],
                         ub=[2.0,  1.3,  15.0])
        
    def evaluate(self, x):
        d, D, N = x[0], x[1], x[2]
        
        obj = (N + 2) * D * d**2
        
        g = np.zeros(4)
        g[0] = 1.0 - (D**3 * N) / (71785.0 * d**4)
        g[1] = (4 * D**2 - d * D) / (12566 * (D * d**3 - d**4)) + 1 / (5108 * d**2) - 1.0
        g[2] = 1.0 - (140.45 * d) / (D**2 * N)
        g[3] = (d + D) / 1.5 - 1.0
        
        penalty = 1e15 * np.sum(np.maximum(0, g)**2)
        return obj + penalty

# ─────────────────────────────────────────────────────────────────────────────
# F2 — Pressure Vessel  (4D)
# Minimise total cost (material + forming + welding) of a cylindrical pressure
# vessel capped with hemispherical heads.
# Variables: Ts (shell thickness), Th (head thickness), R (radius), L (length)
# Reference: Kannan & Kramer (1994)
# ─────────────────────────────────────────────────────────────────────────────
class PressureVessel(EngineeringProblem):
    def __init__(self):
        super().__init__("PressureVessel", 4,
                         lb=[0.0,  0.0,  10.0, 10.0],
                         ub=[99.0, 99.0, 200.0, 200.0])
        
    def evaluate(self, x):
        Ts, Th, R, L = x[0], x[1], x[2], x[3]
        
        obj = 0.6224 * Ts * R * L + 1.7781 * Th * R**2 \
            + 3.1661 * Ts**2 * L + 19.84 * Ts**2 * R
        
        g = np.zeros(4)
        g[0] = -Ts + 0.0193 * R
        g[1] = -Th + 0.00954 * R
        g[2] = -np.pi * R**2 * L - (4/3) * np.pi * R**3 + 1296000.0
        g[3] = L - 240.0
        
        penalty = 1e15 * np.sum(np.maximum(0, g)**2)
        return obj + penalty

# ─────────────────────────────────────────────────────────────────────────────
# F3 — Welded Beam  (4D)
# Minimise fabrication cost of a welded beam satisfying constraints on shear
# stress, normal stress, buckling load, and end deflection.
# Variables: h (weld thickness), l (weld length), t (bar depth), b (bar width)
# Reference: Ragsdell & Phillips (1976)
# ─────────────────────────────────────────────────────────────────────────────
class WeldedBeam(EngineeringProblem):
    def __init__(self):
        super().__init__("WeldedBeam", 4, 
                         lb=[0.1, 0.1, 0.1, 0.1], 
                         ub=[2.0, 10.0, 10.0, 2.0])
        
    def evaluate(self, x):
        h, l, t, b = x[0], x[1], x[2], x[3]
        
        P = 6000.0
        L = 14.0
        E = 30e6
        G = 12e6
        tau_max   = 13600.0
        sigma_max = 30000.0
        delta_max = 0.25
        
        # Shear stress
        M  = P * (L + l / 2.0)
        R  = np.sqrt(l**2 / 4.0 + ((h + t) / 2.0)**2)
        J  = 2 * (np.sqrt(2) * h * l * (l**2 / 12.0 + ((h + t) / 2.0)**2))
        tau1 = P / (np.sqrt(2) * h * l)
        tau2 = M * R / J
        tau  = np.sqrt(tau1**2 + 2 * tau1 * tau2 * l / (2 * R) + tau2**2)
        
        sigma = 6 * P * L / (b * t**2)
        delta = 6 * P * L**3 / (E * b * t**3)
        Pc    = 4.013 * E * np.sqrt(t**2 * b**6 / 36.0) / L**2 \
                * (1 - t / (2 * L) * np.sqrt(E / (4 * G)))
        
        # Objective: fabrication cost
        obj = 1.10471 * h**2 * l + 0.04811 * t * b * (14.0 + l)
        
        # Constraints (≤ 0 form)
        g = np.zeros(7)
        g[0] = tau   - tau_max
        g[1] = sigma - sigma_max
        g[2] = h - b
        g[3] = 0.10471 * h**2 + 0.04811 * t * b * (14.0 + l) - 5.0
        g[4] = 0.125 - h
        g[5] = delta - delta_max
        g[6] = P - Pc
        
        penalty = 1e15 * np.sum(np.maximum(0, g)**2)
        return obj + penalty

# ─────────────────────────────────────────────────────────────────────────────
# F4 — Speed Reducer  (7D)
# Minimise the weight of a speed reducer satisfying 11 mechanical constraints.
# Variables: x1 (face width), x2 (module of teeth), x3 (number of teeth on
#   pinion), x4 (shaft 1 length), x5 (shaft 2 length), x6 (shaft 1 diameter),
#   x7 (shaft 2 diameter)
# Reference: Golinski (1970)
# ─────────────────────────────────────────────────────────────────────────────
class SpeedReducer(EngineeringProblem):
    def __init__(self):
        super().__init__("SpeedReducer", 7,
                         lb=[2.6, 0.7, 17.0, 7.3, 7.8, 2.9, 5.0],
                         ub=[3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5])
        
    def evaluate(self, x):
        x1, x2, x3, x4, x5, x6, x7 = x
        
        obj = 0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934) \
              - 1.508 * x1 * (x6**2 + x7**2) + 7.4777 * (x6**3 + x7**3) \
              + 0.7854 * (x4 * x6**2 + x5 * x7**2)
              
        g = np.zeros(11)
        g[0]  = 27.0 / (x1 * x2**2 * x3) - 1.0
        g[1]  = 397.5 / (x1 * x2**2 * x3**2) - 1.0
        g[2]  = 1.93 * x4**3 / (x2 * x3 * x6**4) - 1.0
        g[3]  = 1.93 * x5**3 / (x2 * x3 * x7**4) - 1.0
        g[4]  = np.sqrt((745.0 * x4 / (x2 * x3))**2 + 16.9e6) / (110.0 * x6**3) - 1.0
        g[5]  = np.sqrt((745.0 * x5 / (x2 * x3))**2 + 157.5e6) / (85.0 * x7**3) - 1.0
        g[6]  = x2 * x3 / 40.0 - 1.0
        g[7]  = 5.0 * x2 / x1 - 1.0
        g[8]  = x1 / (12.0 * x2) - 1.0
        g[9]  = (1.5 * x6 + 1.9) / x4 - 1.0
        g[10] = (1.1 * x7 + 1.9) / x5 - 1.0
        
        penalty = 1e15 * np.sum(np.maximum(0, g)**2)
        return obj + penalty

# ─────────────────────────────────────────────────────────────────────────────
# F5 — Rolling Element Bearing (REB)  (10D)
# Maximise the dynamic load-carrying capacity C of a rolling element bearing
# (converted to minimisation by returning −C) subject to 9 constraints.
# Variables (10):
#   x1  Dm   – pitch circle diameter of ball set   [125, 150] mm
#   x2  Db   – ball diameter                       [  0,  40] mm  (x2 ≤ 0.5*Dm)
#   x3  Z    – number of balls                     [  4,  50]
#   x4  fi   – inner-raceway curvature radius ratio [ 0.515, 0.6]
#   x5  fo   – outer-raceway curvature radius ratio [ 0.515, 0.6]
#   x6  ε    – contact angle (rad)                 [  0, π/2]   (used as cosα)
#   x7  Bw   – bearing width                       [  B_min, B_max]
#   x8  ri   – inner-groove radius ratio           [0.515, 0.6]
#   x9  ro   – outer-groove radius ratio           [0.515, 0.6]
#   x10 Kd   – diametral clearance ratio           [  0, 0.1]
#
# Simplified continuous formulation widely used in metaheuristic benchmarking
# (Gupta & Dangayach; Savsani et al. 2010).
# ─────────────────────────────────────────────────────────────────────────────
class RollingElementBearing(EngineeringProblem):
    """
    10-dimensional Rolling Element Bearing design optimisation.

    Objective  : Maximise dynamic load capacity C  → minimise -C
    Constraints: 9 inequality constraints (g ≤ 0).
    Penalty    : static exterior penalty with coefficient 1e10.

    Variable bounds follow Savsani et al. (2010) / Gupta & Dangayach (2012):
      x1  Dm  in [125, 150]
      x2  Db  in [  0,  40]    (continuous; further restricted by g1: Db ≤ 0.5*Dm)
      x3  Z   in [  4,  50]    (treated as continuous for simplicity)
      x4  fi  in [0.515, 0.6]
      x5  fo  in [0.515, 0.6]
      x6  : cosine of contact angle, in [0.5, 0.97]
      x7  : Bw/Dm ratio fraction in [0.15, 0.45]
      x8  Kd* in [0.0, 0.1]   expansion clearance ratio (not used in C below)
      x9, x10 : auxiliary proportional variables in [0, 1]  (penalty only)
    """

    # Published known best: C ≈ 82 695.6 kN  (objective ≈ -82 695.6)
    KNOWN_BEST = -82_695.6

    def __init__(self):
        lb = [125.0,  0.0,   4.0, 0.515, 0.515, 0.50, 0.15, 0.0, 0.0, 0.0]
        ub = [150.0, 40.0,  50.0, 0.600, 0.600, 0.97, 0.45, 0.1, 1.0, 1.0]
        super().__init__("RollingElementBearing", 10, lb, ub)

    def evaluate(self, x):
        Dm   = x[0]   # pitch circle diameter (mm)
        Db   = x[1]   # ball diameter (mm)
        Z    = x[2]   # number of balls (continuous)
        fi   = x[3]   # inner raceway curvature ratio
        fo   = x[4]   # outer raceway curvature ratio
        cA   = x[5]   # cos(contact angle)
        bw_r = x[6]   # Bw/Dm ratio
        Kd   = x[7]   # diametral clearance ratio
        # x[8], x[9] — auxiliary (penalised by constraints)

        # Derived quantities
        Bw = bw_r * Dm          # bearing width (mm)
        sA = np.sqrt(max(0.0, 1.0 - cA**2))   # sin(contact angle)

        # ── Dynamic load capacity  (Lundberg-Palmgren, simplified) ────────
        # C = fc * Z^(2/3) * Db^1.8   for Db ≤ 25.4 mm
        #   = fc * Z^(2/3) * 3.647 * Db^1.4   for Db > 25.4 mm
        # fc depends on geometry:  fc = 37.91*(1 + 1.04*((1-gamma)/(1+gamma))^1.72
        #                              * (fi*(2*fo-1)/(fo*(2*fi-1)))^0.41)^0.3
        #                              * (gamma^0.3*(1-gamma)^1.39/(1+gamma)^(1/3))
        #                              * (2*fi/(2*fi-1))^0.41
        gamma = Db * cA / Dm

        # Protect against domain errors
        dfi = 2.0 * fi - 1.0
        dfo = 2.0 * fo - 1.0
        if dfi <= 0 or dfo <= 0 or (1 + gamma) == 0 or (1 - gamma) <= 0:
            return 1e15

        inner_ratio = (fi * dfo) / (fo * dfi)
        fc = (37.91
              * (1 + 1.04 * ((1 - gamma) / (1 + gamma))**1.72
                 * inner_ratio**0.41)**0.3
              * (gamma**0.3 * (1 - gamma)**1.39 / (1 + gamma)**(1/3))
              * (2 * fi / dfi)**0.41)

        if Db <= 25.4:
            C = fc * Z**(2/3) * Db**1.8
        else:
            C = fc * Z**(2/3) * 3.647 * Db**1.4

        # ── Constraints (g ≤ 0) ──────────────────────────────────────────
        g = np.zeros(9)
        # g1: Db < 0.5*Dm
        g[0] = Db - 0.5 * Dm
        # g2: Db > 0  (already enforced by lb, but keep for robustness)
        g[1] = -Db
        # g3: Z ≥ 4
        g[2] = 4.0 - Z
        # g4: Bw ≥ 0.075*Dm  (minimum width)
        g[3] = 0.075 * Dm - Bw
        # g5: Kd ≤ (Dm*0.002)  clearance limit
        g[4] = Kd - 0.002
        # g6: fi ≥ 0.515
        g[5] = 0.515 - fi
        # g7: fo ≥ 0.515
        g[6] = 0.515 - fo
        # g8: x[8] ≤ 0.5
        g[7] = x[8] - 0.5
        # g9: x[9] ≤ 0.5
        g[8] = x[9] - 0.5

        penalty = 1e10 * np.sum(np.maximum(0.0, g)**2)

        # Minimise -C  (we want to maximise C)
        return float(-C + penalty)


# ─────────────────────────────────────────────────────────────────────────────
# Registry — order defines function IDs 1..5
# F1: TensionCompressionSpring  (3D)
# F2: PressureVessel            (4D)
# F3: WeldedBeam                (4D)
# F4: SpeedReducer              (7D)
# F5: RollingElementBearing    (10D)
# ─────────────────────────────────────────────────────────────────────────────
_ENGINEERING_PROBLEMS = [
    TensionCompressionSpring,   # F1  3D
    PressureVessel,             # F2  4D
    WeldedBeam,                 # F3  4D
    SpeedReducer,               # F4  7D
    RollingElementBearing,      # F5 10D
]

_PROBLEM_NAMES = {
    1: "Spring (3D)",
    2: "PressureVessel (4D)",
    3: "WeldedBeam (4D)",
    4: "SpeedReducer (7D)",
    5: "Bearing (10D)",
}

def get_function(func_id: int, ndim: int = None):
    """Return the engineering problem instance for the given function ID (1-based)."""
    if 1 <= func_id <= len(_ENGINEERING_PROBLEMS):
        return _ENGINEERING_PROBLEMS[func_id - 1]()
    raise ValueError(
        f"Engineering function ID must be between 1 and {len(_ENGINEERING_PROBLEMS)}, got {func_id}"
    )

def get_all_functions(ndim: int = None):
    return [get_function(i) for i in range(1, len(_ENGINEERING_PROBLEMS) + 1)]

def get_problem_name(func_id: int) -> str:
    return _PROBLEM_NAMES.get(func_id, f"F{func_id}")


if __name__ == "__main__":
    print(f"{'ID':<4} {'Name':<30} {'Dim':<5} {'f(mid)':>15}")
    print("-" * 58)
    for i in range(1, len(_ENGINEERING_PROBLEMS) + 1):
        fn = get_function(i)
        x  = np.array([(lb + ub) / 2 for lb, ub in zip(fn.lb, fn.ub)])
        val = fn(x)
        print(f"F{i:<3} {fn.name:<30} {fn.dim:<5} {val:>15.6e}")
