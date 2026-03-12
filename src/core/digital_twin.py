"""
PATENT ELEMENT D/D1: Digital Twin Model
"""

import copy
from collections import deque
from datetime import datetime

from config.buildings import ASHRAE_BUILDINGS


class DigitalTwinModel:
    """
    PATENT ELEMENT D/D1: Continuously updated digital twin reflecting current
    physical state, historical behavior, and environmental context.
    Initialized from ASHRAE building metadata.
    """

    def __init__(self, building_key):
        bld = ASHRAE_BUILDINGS[building_key]
        self.property_id = building_key
        self.name = bld["name"]
        self.primary_use = bld["primary_use"]
        self.square_feet = bld["square_feet"]
        self.year_built = bld["year_built"]
        self.floor_count = bld["floor_count"]
        self.property_age = datetime.now().year - bld["year_built"]

        self.state = {
            "structural_integrity": 1.0,
            "environmental_condition": 1.0,
            "usage_intensity": 0.0,
            "deterioration_level": 0.0,
            "overall_health": 1.0,
        }
        self.state_history = deque(maxlen=500)

    def update(self, sensors, indicators):
        """Update twin state from latest sensor + indicator data."""
        self.state["structural_integrity"] = indicators.get("SHF", 1.0)
        self.state["environmental_condition"] = indicators.get("ESF", 1.0)
        self.state["usage_intensity"] = sensors.get("occupancy", 0)
        self.state["deterioration_level"] = 1.0 - indicators.get("PDP", 1.0)
        self.state["overall_health"] = indicators.get("health_factor", 1.0)
        self.state_history.append(copy.deepcopy(self.state))
