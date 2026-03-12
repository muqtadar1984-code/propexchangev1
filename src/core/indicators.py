"""
PATENT ELEMENT E/E1: Technical Indicators Engine
"""

import math
import numpy as np
from collections import deque


class TechnicalIndicatorsEngine:
    """
    PATENT ELEMENT E/E1: Five technical indicators computed from sensor data.

    SHF  — Structural Health Factor      — quadratic penalty
    ESF  — Environmental Stability Factor — piecewise linear
    USS  — Usage Stress Score             — power-law wear
    PDP  — Predictive Deterioration Penalty — sigmoid aging
    CI   — Confidence Index               — exponential decay on variance

    All outputs ∈ [0, 1] where 1 = optimal.
    Weights and thresholds are configurable parameters (not hardcoded constants)
    to support claim breadth per patent attorney guidance.
    """

    def __init__(self, config=None):
        cfg = config or {}
        # SHF params
        self.shf_exponent = cfg.get("shf_exponent", 2.0)
        self.shf_vib_weight = cfg.get("shf_vib_weight", 0.6)
        self.shf_strain_weight = cfg.get("shf_strain_weight", 0.4)
        # ESF params — widened safe zone for normal maritime/tropical climate
        self.tau1 = cfg.get("esf_tau1", 0.40)      # no penalty below 40%
        self.tau2 = cfg.get("esf_tau2", 0.80)      # linear degradation zone
        self.epsilon = cfg.get("esf_epsilon", 0.55) # floor at severe conditions
        # USS params
        self.gamma = cfg.get("uss_gamma", 2.5)
        # PDP params — Maintenance-Adjusted Effective Age + Asymptotic Floor
        self.alpha = cfg.get("pdp_alpha", 8.0)          # sigmoid steepness
        self.theta = cfg.get("pdp_theta", 0.65)         # sigmoid midpoint
        self.max_life = cfg.get("pdp_max_life", 120)    # max useful life (years)
        self.pdp_floor = cfg.get("pdp_floor", 0.40)     # minimum residual (40%)
        self.maintenance_max = cfg.get("pdp_maint_max", 0.55)  # max age reduction
        # CI params
        self.lambda_c = cfg.get("ci_lambda", 3.0)
        # History
        self.history = {"vibration": deque(maxlen=60),
                        "environment": deque(maxlen=60),
                        "usage": deque(maxlen=60)}

    def calculate_shf(self, vibration, strain):
        s = self.shf_vib_weight * vibration + self.shf_strain_weight * strain
        s = np.clip(s, 0, 1)
        shf = 1.0 - (s ** self.shf_exponent)
        self.history["vibration"].append(s)
        return shf

    def calculate_esf(self, moisture, temperature, air_quality):
        e = 0.4 * moisture + 0.3 * temperature + 0.3 * air_quality
        e = np.clip(e, 0, 1)
        if e < self.tau1:
            esf = 1.0
        elif e <= self.tau2:
            frac = (e - self.tau1) / (self.tau2 - self.tau1)
            esf = 1.0 - frac * (1.0 - self.epsilon)
        else:
            esf = self.epsilon * 0.8
        self.history["environment"].append(e)
        return esf

    def calculate_uss(self, occupancy, electrical_load):
        """
        USAGE STRESS SCORE (USS) — Threshold-gated power-law with dead zone.

        Normal building operations (u < 0.6) do NOT penalize valuation.
        Only sustained heavy usage (u > 0.6) causes wear that affects value.
        """
        u = 0.5 * occupancy + 0.5 * electrical_load
        u = np.clip(u, 0, 1)

        dead_zone = 0.6  # No penalty below 60% usage
        if u <= dead_zone:
            uss = 1.0
        else:
            excess = (u - dead_zone) / (1.0 - dead_zone)
            uss = 1.0 - (excess ** self.gamma)

        self.history["usage"].append(u)
        return uss

    def calculate_pdp(self, property_age_years, shf=None, esf=None, uss=None):
        """
        PREDICTIVE DETERIORATION PENALTY (PDP)
        Maintenance-Adjusted Effective Age + Asymptotic Floor.

        Steps:
        1. Compute Maintenance Factor from live sensor indicators
        2. Compute Effective Age (industry-standard Marshall & Swift concept)
        3. Apply sigmoid with asymptotic floor
        """
        max_life = self.max_life

        _shf = shf if shf is not None else 0.5
        _esf = esf if esf is not None else 0.5
        _uss = uss if uss is not None else 0.5

        maintenance_factor = (0.40 * _shf + 0.35 * _esf + 0.25 * _uss)
        maintenance_factor = np.clip(maintenance_factor, 0.0, 1.0)

        age_reduction = maintenance_factor * self.maintenance_max
        effective_age = property_age_years * (1.0 - age_reduction)
        effective_age = max(0.0, effective_age)

        p = np.clip(effective_age / max_life, 0, 1)
        try:
            raw_sigmoid = 1.0 / (1.0 + math.exp(self.alpha * (p - self.theta)))
        except OverflowError:
            raw_sigmoid = 0.0

        pdp = self.pdp_floor + (1.0 - self.pdp_floor) * raw_sigmoid

        return pdp, {
            "chronological_age": property_age_years,
            "maintenance_factor": round(maintenance_factor, 4),
            "effective_age": round(effective_age, 1),
            "age_reduction_pct": round(age_reduction * 100, 1),
            "floor": self.pdp_floor,
        }

    def calculate_ci(self):
        if len(self.history["vibration"]) < 5:
            return 1.0
        stds = [np.std(list(h)) for h in self.history.values() if len(h) >= 5]
        sigma_avg = np.mean(stds) if stds else 0.0
        return math.exp(-self.lambda_c * sigma_avg)
