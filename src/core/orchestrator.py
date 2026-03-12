"""
Building Orchestrator — ties all systems together per building.
"""

from config.buildings import ASHRAE_BUILDINGS
from src.data.sensor_engine import ASHRAESensorEngine
from src.data.preprocessor import DataPreprocessor
from src.core.digital_twin import DigitalTwinModel
from src.core.indicators import TechnicalIndicatorsEngine
from src.core.valuation import PropertyValuationEngine
from src.utils.hash_chain import HashChainVerification


class BuildingOrchestrator:
    """Orchestrates sensor → preprocess → twin → indicators → valuation → hash
    for a single building."""

    def __init__(self, building_key, manual_mv_override=None):
        bld = ASHRAE_BUILDINGS[building_key]
        self.building_key = building_key
        self.sensor_engine = ASHRAESensorEngine(building_key)
        self.preprocessor = DataPreprocessor()
        self.twin = DigitalTwinModel(building_key)
        self.indicators = TechnicalIndicatorsEngine()
        self.valuation = PropertyValuationEngine(
            bld["govt_valuation"], bld["land_value"], manual_mv_override
        )
        self.verification = HashChainVerification()
        self.cycle_count = 0

    def process_cycle(self, scenario_overrides=None):
        """Run one valuation cycle. Returns complete cycle data."""
        # 1. Sensor reading (Element B)
        raw = self.sensor_engine.get_sensor_reading(scenario_overrides)
        sensors = raw["sensors"]

        # 2. Preprocess (Element C)
        processed = {}
        for k, v in sensors.items():
            processed[k] = self.preprocessor.process(k, v)

        # 3. Technical indicators (Element E)
        shf = self.indicators.calculate_shf(processed["vibration"], processed["strain"])
        esf = self.indicators.calculate_esf(processed["moisture"], processed["temperature"],
                                             processed["air_quality"])
        uss = self.indicators.calculate_uss(processed["occupancy"], processed["electrical_load"])
        pdp, pdp_detail = self.indicators.calculate_pdp(
            self.twin.property_age, shf=shf, esf=esf, uss=uss
        )
        ci = self.indicators.calculate_ci()

        ind = {"SHF": shf, "ESF": esf, "USS": uss, "PDP": pdp, "CI": ci}

        # 4. RTPMV (Element F)
        rtpmv, health = self.valuation.calculate_rtpmv(shf, esf, uss, pdp, ci)
        val = {
            "rtpmv": rtpmv,
            "health_factor": health,
            "base_mv": self.valuation.base_mv,
            "govt_valuation": self.valuation.govt_valuation,
            "land_value": self.valuation.land_value,
            "structure_adjusted": self.valuation.structure_value * health,
            "is_override": self.valuation.is_override,
            "pdp_detail": pdp_detail,
        }

        # 5. Update twin (Element D)
        ind_with_health = {**ind, "health_factor": health}
        self.twin.update(sensors, ind_with_health)

        # 6. Hash-chain record (Element G)
        block = self.verification.add_record(
            self.building_key, sensors, ind, val, raw["timestamp"]
        )
        token = self.verification.generate_token(
            rtpmv, health, ci, self.building_key, raw["timestamp"]
        )

        self.cycle_count += 1

        return {
            "timestamp": raw["timestamp"],
            "hour": raw["hour"],
            "building_key": self.building_key,
            "raw_sensors": sensors,
            "processed_sensors": processed,
            "ashrae_meters": raw["ashrae_meters"],
            "weather": raw["weather"],
            "indicators": ind,
            "valuation": val,
            "verification": {
                "block_index": block["block_index"],
                "block_hash": block["block_hash"][:16] + "...",
                "prev_hash": block["prev_hash"][:16] + "...",
                "chain_length": len(self.verification.chain),
                "token_id": token["token_id"],
            },
        }
