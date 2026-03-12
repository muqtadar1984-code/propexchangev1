"""
PATENT ELEMENT F/F1/F2/F2.1: Property Valuation Engine
"""

from collections import deque


class PropertyValuationEngine:
    """
    PATENT ELEMENT F/F2: RTPMV = MV × SHF × ESF × USS × PDP × CI

    Element F2.1: MV is the baseline market value.
    - Default MV = Government assessed valuation (static, from county records)
    - User may override MV manually (the override replaces govt valuation)
    - Land value is preserved separately (location-based, non-depreciating)
    - Structure value = MV - Land value (subject to technical factor adjustment)
    """

    def __init__(self, govt_valuation, land_value, manual_override=None):
        self.govt_valuation = govt_valuation
        self.base_mv = manual_override if manual_override else govt_valuation
        self.land_value = min(land_value, self.base_mv * 0.95)
        self.structure_value = self.base_mv - self.land_value
        self.is_override = manual_override is not None
        self.history = deque(maxlen=1000)

    def update_mv(self, new_mv):
        """Update base MV (manual override)."""
        self.base_mv = new_mv
        self.is_override = True
        self.land_value = min(self.land_value, new_mv * 0.95)
        self.structure_value = new_mv - self.land_value

    def calculate_rtpmv(self, shf, esf, uss, pdp, ci):
        health = shf * esf * uss * pdp * ci
        adjusted_structure = self.structure_value * health
        rtpmv = self.land_value + adjusted_structure
        self.history.append(rtpmv)
        return rtpmv, health
