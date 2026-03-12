"""
PATENT ELEMENT B/B1: Multi-Modal IoT Sensor Network
"""

import numpy as np
from datetime import datetime, timedelta

from config.buildings import ASHRAE_BUILDINGS, SITE1_WEATHER


class ASHRAESensorEngine:
    """
    Generates sensor readings from ASHRAE statistical distributions rather than
    simple sine waves. Each building has its own meter profile (mean, std,
    peak_hour, weekend_factor, seasonal_amplitude) derived from actual GEPIII
    hourly data calibrated to UK building energy benchmarks (CIBSE TM46).

    Sensor mapping to ASHRAE meters:
    - Electrical load → electricity_kwh meter (ASHRAE meter type 0)
    - Temperature    → chilled_water_kwh meter + weather (types 1,2)
    - Moisture       → dew_temperature + humidity (weather data)
    - Occupancy      → Derived from electricity diurnal pattern
    - Vibration      → Derived from electrical load (HVAC + structural)
    - Strain         → Derived from occupancy + thermal expansion model
    - Air quality    → Derived from occupancy + HVAC performance ratio
    """

    def __init__(self, building_key):
        self.building = ASHRAE_BUILDINGS[building_key]
        self.profiles = self.building["meter_profiles"]
        self.weather = SITE1_WEATHER
        self.rng = np.random.default_rng(seed=self.building["building_id"])
        self.base_time = datetime(2016, 1, 1)  # ASHRAE data starts 2016
        self.hour_index = 0

    def _ashrae_meter_reading(self, profile, hour_of_day, day_of_week, day_of_year):
        """Generate hourly meter reading from ASHRAE statistical distribution."""
        mean = profile["mean"]
        std = profile["std"]
        peak = profile["peak_hour"]

        # Diurnal pattern (Gaussian around peak hour)
        diurnal = np.exp(-0.5 * ((hour_of_day - peak) / 4.0) ** 2)

        # Weekend effect
        is_weekend = day_of_week >= 5
        weekend_mult = profile["weekend_factor"] if is_weekend else 1.0

        # Seasonal pattern (sinusoidal, peaks in summer for cooling)
        seasonal = 1.0 + profile["seasonal_amp"] * np.sin(
            2 * np.pi * (day_of_year - 172) / 365  # Peak ~Jun 21
        )

        # Base reading with ASHRAE-calibrated noise
        base = mean * diurnal * weekend_mult * seasonal
        noise = self.rng.normal(0, std * 0.3)
        reading = max(0, base + noise)

        return reading

    def _weather_reading(self, day_of_year, hour_of_day):
        """Generate weather data from London ASHRAE distributions."""
        wp = self.weather

        # Air temperature with diurnal + seasonal
        seasonal_t = wp["air_temp_c"]["seasonal_amp"] * np.sin(
            2 * np.pi * (day_of_year - 172) / 365
        )
        diurnal_t = 4.0 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        air_temp = (wp["air_temp_c"]["mean"] + seasonal_t + diurnal_t +
                    self.rng.normal(0, wp["air_temp_c"]["std"] * 0.3))

        # Humidity (inversely correlated with temperature)
        humidity = (wp["humidity_pct"]["mean"] - 0.8 * (air_temp - wp["air_temp_c"]["mean"]) +
                    self.rng.normal(0, wp["humidity_pct"]["std"] * 0.3))
        humidity = np.clip(humidity, 20, 100)

        # Dew temperature
        dew_temp = air_temp - ((100 - humidity) / 5.0)

        # Wind speed (log-normal)
        wind = max(0, wp["wind_speed_ms"]["mean"] +
                   self.rng.normal(0, wp["wind_speed_ms"]["std"] * 0.4))

        # Pressure
        pressure = (wp["sea_level_pressure_hpa"]["mean"] +
                    self.rng.normal(0, wp["sea_level_pressure_hpa"]["std"]))

        # Precipitation (mostly zero, occasional spikes)
        precip = 0.0
        if self.rng.random() < 0.25:  # 25% chance of rain in London
            precip = self.rng.exponential(wp["precip_depth_mm"]["mean"] * 3)

        return {
            "air_temperature_c": round(air_temp, 2),
            "dew_temperature_c": round(dew_temp, 2),
            "humidity_pct": round(humidity, 1),
            "wind_speed_ms": round(wind, 2),
            "sea_level_pressure_hpa": round(pressure, 1),
            "precip_depth_mm": round(precip, 2),
        }

    def get_sensor_reading(self, scenario_overrides=None):
        """
        Generate complete sensor reading for current hour.

        Returns all patent-required sensor types (Element B1):
        vibration, strain, moisture, temperature, occupancy,
        electrical_load, air_quality — plus raw ASHRAE meter values.
        """
        # Current time indices
        current_time = self.base_time + timedelta(hours=self.hour_index)
        hour = current_time.hour
        dow = current_time.weekday()
        doy = current_time.timetuple().tm_yday
        self.hour_index += 1

        # --- Raw ASHRAE meter readings ---
        meters = {}
        for meter_name, profile in self.profiles.items():
            meters[meter_name] = self._ashrae_meter_reading(profile, hour, dow, doy)

        # --- Weather ---
        weather = self._weather_reading(doy, hour)

        # --- Derived sensor values (Element B1 types) ---
        elec = meters.get("electricity_kwh", 300)
        elec_mean = self.profiles.get("electricity_kwh", {}).get("mean", 300)

        # Electrical load: normalized to [0,1] based on building capacity
        elec_capacity = elec_mean * 2.5
        electrical_load = np.clip(elec / elec_capacity, 0, 1)

        # Occupancy: derived from electricity diurnal pattern
        peak_h = self.profiles.get("electricity_kwh", {}).get("peak_hour", 14)
        occ_diurnal = np.exp(-0.5 * ((hour - peak_h) / 3.5) ** 2)
        occ_weekend = 0.3 if dow >= 5 else 1.0
        occupancy = np.clip(occ_diurnal * occ_weekend + self.rng.normal(0, 0.05), 0, 1)

        # Temperature: normalized from weather (0=optimal 22°C, 1=extreme)
        temp_deviation = abs(weather["air_temperature_c"] - 22.0)
        temperature = np.clip(temp_deviation / 20.0, 0, 1)

        # Moisture: from humidity + dew point proximity + precipitation
        moisture_base = weather["humidity_pct"] / 100.0
        dew_proximity = max(0, 1 - (weather["air_temperature_c"] -
                                     weather["dew_temperature_c"]) / 10.0)
        precip_factor = min(1.0, weather["precip_depth_mm"] / 20.0)
        moisture = np.clip(0.5 * moisture_base + 0.3 * dew_proximity +
                           0.2 * precip_factor, 0, 1)

        # Vibration: HVAC + structural load (correlated with electrical)
        hvac_vibration = electrical_load * 0.6
        structural_base = 0.02 + 0.01 * (weather["wind_speed_ms"] / 10.0)
        vibration = np.clip(hvac_vibration * 0.15 + structural_base +
                            self.rng.normal(0, 0.015), 0, 1)

        # Strain: thermal expansion + occupancy loading
        thermal_strain = temperature * 0.08
        load_strain = occupancy * 0.05
        strain = np.clip(thermal_strain + load_strain +
                         self.rng.normal(0, 0.01), 0, 1)

        # Air quality: inversely related to occupancy, boosted by HVAC performance
        hvac_efficiency = 1 - electrical_load * 0.3
        aq_base = 1.0 - occupancy * 0.4
        air_quality = np.clip(aq_base * hvac_efficiency +
                              self.rng.normal(0, 0.03), 0, 1)
        # Invert so higher = worse for ESF calculation
        air_quality_stress = 1.0 - air_quality

        # Apply scenario overrides (for event injection)
        reading = {
            "vibration": round(vibration, 4),
            "strain": round(strain, 4),
            "moisture": round(moisture, 4),
            "temperature": round(temperature, 4),
            "occupancy": round(occupancy, 4),
            "electrical_load": round(electrical_load, 4),
            "air_quality": round(air_quality_stress, 4),
        }

        if scenario_overrides:
            for k, v in scenario_overrides.items():
                if k in reading:
                    reading[k] = np.clip(v, 0, 1)

        return {
            "timestamp": current_time.isoformat(),
            "hour": hour,
            "day_of_week": dow,
            "day_of_year": doy,
            "sensors": reading,
            "ashrae_meters": {k: round(v, 2) for k, v in meters.items()},
            "weather": weather,
        }
