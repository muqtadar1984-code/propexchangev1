# ═══════════════════════════════════════════════════════════════════════════════
# ASHRAE BUILDING DATABASE — Malaysia Properties (Kuala Lumpur & Selangor)
# ═══════════════════════════════════════════════════════════════════════════════
#
# ASHRAE GEPIII metadata schema applied to Malaysian commercial properties.
# Statistical profiles (mean, std, peak_hour, weekend_factor, seasonal_amp)
# calibrated to ASHRAE tropical climate benchmarks (Climate Zone 0A / Af).
#
# Malaysia Energy Context: Tropical equatorial climate — no heating required.
# Dominant energy loads: chilled water (district cooling / in-building chillers),
# electricity (lighting, IT, elevators). No gas_heating or steam meters.
#
# Valuations: Government appraisal values in MYR (Malaysian Ringgit).
# Properties aligned to frontend dashboard PROPERTIES array (same 5 assets).
# ═══════════════════════════════════════════════════════════════════════════════

ASHRAE_BUILDINGS = {
    "MY-KL-OFF-KLCC": {
        "building_id": 101, "site_id": 1,
        "name": "KLCC Tower",
        "primary_use": "Grade A Office",
        "square_feet": 125_000, "year_built": 1998, "floor_count": 48,
        "latitude": 3.1578, "longitude": 101.7123,
        "address": "Kuala Lumpur City Centre, 50088 Kuala Lumpur",
        "climate_zone": "0A — Very Hot Humid (Af — Tropical Rainforest)",
        "meters": {"electricity": 0, "chilled_water": 1},
        "govt_valuation": 185_000_000,
        "land_value": 75_000_000,
        "meter_profiles": {
            "electricity_kwh": {"mean": 3480.5, "std": 620.4, "peak_hour": 14,
                                "weekend_factor": 0.22, "seasonal_amp": 0.04},
            "chilled_water_kwh": {"mean": 1920.8, "std": 380.5, "peak_hour": 13,
                                   "weekend_factor": 0.18, "seasonal_amp": 0.06},
        },
    },
    "MY-SA-IDC-AXIS": {
        "building_id": 102, "site_id": 2,
        "name": "Axis Shah Alam DC",
        "primary_use": "Industrial / Data Centre",
        "square_feet": 48_000, "year_built": 2016, "floor_count": 4,
        "latitude": 3.0851, "longitude": 101.5325,
        "address": "Shah Alam, Selangor 40150",
        "climate_zone": "0A — Very Hot Humid (Af — Tropical Rainforest)",
        "meters": {"electricity": 0, "chilled_water": 1},
        "govt_valuation": 62_000_000,
        "land_value": 22_000_000,
        "meter_profiles": {
            "electricity_kwh": {"mean": 8650.2, "std": 920.8, "peak_hour": 12,
                                "weekend_factor": 0.95, "seasonal_amp": 0.02},
            "chilled_water_kwh": {"mean": 3280.4, "std": 480.2, "peak_hour": 12,
                                   "weekend_factor": 0.93, "seasonal_amp": 0.02},
        },
    },
    "MY-KL-HC-ALAQAR": {
        "building_id": 103, "site_id": 1,
        "name": "Al-Aqar Medical Hub",
        "primary_use": "Healthcare",
        "square_feet": 32_000, "year_built": 2007, "floor_count": 12,
        "latitude": 3.1502, "longitude": 101.6235,
        "address": "Damansara, 47500 Petaling Jaya, Selangor",
        "climate_zone": "0A — Very Hot Humid (Af — Tropical Rainforest)",
        "meters": {"electricity": 0, "chilled_water": 1},
        "govt_valuation": 41_000_000,
        "land_value": 16_000_000,
        "meter_profiles": {
            "electricity_kwh": {"mean": 1820.6, "std": 310.8, "peak_hour": 11,
                                "weekend_factor": 0.88, "seasonal_amp": 0.03},
            "chilled_water_kwh": {"mean": 820.4, "std": 180.5, "peak_hour": 10,
                                   "weekend_factor": 0.85, "seasonal_amp": 0.03},
        },
    },
    "MY-KL-RET-PAV": {
        "building_id": 104, "site_id": 1,
        "name": "Pavilion Retail Arcade",
        "primary_use": "Retail",
        "square_feet": 89_000, "year_built": 2007, "floor_count": 7,
        "latitude": 3.1489, "longitude": 101.7132,
        "address": "168 Jalan Bukit Bintang, 55100 Kuala Lumpur",
        "climate_zone": "0A — Very Hot Humid (Af — Tropical Rainforest)",
        "meters": {"electricity": 0, "chilled_water": 1},
        "govt_valuation": 137_000_000,
        "land_value": 58_000_000,
        "meter_profiles": {
            "electricity_kwh": {"mean": 2750.4, "std": 510.6, "peak_hour": 16,
                                "weekend_factor": 1.35, "seasonal_amp": 0.05},
            "chilled_water_kwh": {"mean": 1480.2, "std": 310.4, "peak_hour": 15,
                                   "weekend_factor": 1.30, "seasonal_amp": 0.05},
        },
    },
    "MY-SJ-LOG-SUN": {
        "building_id": 105, "site_id": 3,
        "name": "Sunway Logistics Park",
        "primary_use": "Logistics / Industrial",
        "square_feet": 71_000, "year_built": 2014, "floor_count": 5,
        "latitude": 3.0588, "longitude": 101.5841,
        "address": "Subang Jaya, 47500 Selangor",
        "climate_zone": "0A — Very Hot Humid (Af — Tropical Rainforest)",
        "meters": {"electricity": 0, "chilled_water": 1},
        "govt_valuation": 88_000_000,
        "land_value": 34_000_000,
        "meter_profiles": {
            "electricity_kwh": {"mean": 840.8, "std": 195.4, "peak_hour": 10,
                                "weekend_factor": 0.40, "seasonal_amp": 0.03},
            "chilled_water_kwh": {"mean": 415.6, "std": 110.2, "peak_hour": 11,
                                   "weekend_factor": 0.38, "seasonal_amp": 0.03},
        },
    },
}

# Kuala Lumpur weather profile (ASHRAE tropical Af — equatorial)
# Source: ASHRAE Climate Zone 0A, calibrated to Subang (WMKK) station data.
# No meaningful seasonality — consistently hot and humid year-round.
SITE1_WEATHER = {
    "air_temp_c":             {"mean": 27.2, "std": 1.8, "seasonal_amp": 1.2},
    "dew_temp_c":             {"mean": 23.5, "std": 1.4},
    "humidity_pct":           {"mean": 82.0, "std": 7.5},
    "wind_speed_ms":          {"mean": 2.1,  "std": 1.2},
    "sea_level_pressure_hpa": {"mean": 1009.8, "std": 3.2},
    "precip_depth_mm":        {"mean": 7.5,  "std": 14.2},
}
