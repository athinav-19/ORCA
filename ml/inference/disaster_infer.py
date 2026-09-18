"""
disaster_infer.py - Real-Time Disaster & Hazard Inference Adapter
SIH 2026 Problem Statement SIH26176

Provides low-latency, thread-safe inference for the Disaster Prediction Model
predicting future hazard probability 24 hours in advance without target leakage.
"""

import os
import math
import joblib
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

from ml.data_ingestion.bathymetry_geography import get_geographic_features
from ml.feature_engineering.disaster_features import DISASTER_FEATURE_COLUMNS

MODEL_PATH = os.path.join("ml", "models", "disaster", "best_model.joblib")
_CACHED_MODEL = None


def get_disaster_model():
    """Loads and caches the trained disaster hazard model via ModelRegistry."""
    global _CACHED_MODEL
    try:
        from ml.model_registry import ModelRegistry
        reg_model = ModelRegistry.get_model("disaster")
        if reg_model is not None:
            _CACHED_MODEL = reg_model
            return _CACHED_MODEL
    except Exception:
        pass

    if _CACHED_MODEL is None and os.path.exists(MODEL_PATH):
        try:
            _CACHED_MODEL = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"[DisasterInfer Error] Model loading failed: {e}")
            _CACHED_MODEL = None
    return _CACHED_MODEL


def predict_marine_disaster(
    latitude: float,
    longitude: float,
    wind_speed_kmh: float = 20.0,
    surface_pressure_hpa: float = 1011.0,
    pressure_tendency_6h: float = 0.0,
    sst_c: float = 28.5,
    olr_wm2: float = 240.0,
    rainfall_rate_mmh: float = 0.0,
    wave_height_m: float = 1.2,
    storm_speed_kmh: float = 0.0,
    storm_dir_deg: float = 0.0,
    observation_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Evaluates real-time 24h future marine hazard probability and severity using the trained ML model.
    """
    model = get_disaster_model()
    dt = observation_time or datetime.now()
    m = dt.month
    month_sin = math.sin(2 * math.pi * m / 12.0)
    month_cos = math.cos(2 * math.pi * m / 12.0)
    is_cyclone_season = 1 if m in (4, 5, 10, 11, 12) else 0

    rad = math.radians(storm_dir_deg)
    u_wind = round(-wind_speed_kmh * math.sin(rad), 2)
    v_wind = round(-wind_speed_kmh * math.cos(rad), 2)

    geo = get_geographic_features(latitude, longitude)
    dist_coast = geo["distance_to_coast_nm"]
    depth_m = geo["bathymetry_depth_m"]
    sst_anomaly = round(sst_c - 28.5, 2)

    # In operational inference, derive realistic physical tendencies if not provided
    wind_tendency_6h = 0.0
    pressure_tendency_12h = pressure_tendency_6h * 1.8
    wind_tendency_12h = wind_tendency_6h * 1.8

    if wind_speed_kmh >= 45.0 and pressure_tendency_6h == 0.0:
        pressure_tendency_6h = -max(1.0, (wind_speed_kmh - 35.0) / 8.0)
        pressure_tendency_12h = pressure_tendency_6h * 1.8
        wind_tendency_6h = round((wind_speed_kmh - 35.0) * 0.25, 1)
        wind_tendency_12h = wind_tendency_6h * 1.8

    feature_dict = {
        "latitude": latitude,
        "longitude": longitude,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "is_cyclone_season": is_cyclone_season,
        "surface_pressure_hpa": surface_pressure_hpa,
        "wind_speed_kmh": wind_speed_kmh,
        "u_wind": u_wind,
        "v_wind": v_wind,
        "storm_speed_kmh": storm_speed_kmh,
        "storm_dir_deg": storm_dir_deg,
        "pressure_tendency_6h": pressure_tendency_6h,
        "wind_tendency_6h": wind_tendency_6h,
        "pressure_tendency_12h": pressure_tendency_12h,
        "wind_tendency_12h": wind_tendency_12h,
        "distance_to_coast_nm": dist_coast,
        "bathymetry_depth_m": depth_m,
        "sst_c": sst_c,
        "sst_anomaly_c": sst_anomaly,
    }

    features = [feature_dict[col] for col in DISASTER_FEATURE_COLUMNS]

    if model is not None:
        try:
            X = np.array(features).reshape(1, -1)
            prob = float(model.predict_proba(X)[0, 1])
            is_hazard = bool(prob >= 0.50)
        except Exception as e:
            return {
                "status": "UNAVAILABLE",
                "model_available": False,
                "message": f"Prediction unavailable because required data/model is unavailable: {e}",
                "hazard_probability": None,
                "is_hazard": None,
                "hazard_class": "PREDICTION_UNAVAILABLE",
                "severity_level": "UNKNOWN",
                "prediction_horizon_hours": 24,
                "confidence_score": 0.0,
                "primary_drivers": ["Inference computation failed"],
                "provenance": "ORCA Disaster ML Model (UNAVAILABLE)",
            }
    else:
        return {
            "status": "UNAVAILABLE",
            "model_available": False,
            "message": "Prediction unavailable because required data/model is unavailable.",
            "hazard_probability": None,
            "is_hazard": None,
            "hazard_class": "PREDICTION_UNAVAILABLE",
            "severity_level": "UNKNOWN",
            "prediction_horizon_hours": 24,
            "confidence_score": 0.0,
            "primary_drivers": ["Disaster ML model artifact offline"],
            "provenance": "ORCA Disaster ML Model (UNAVAILABLE)",
        }

    # IMD Cyclone & Squall Category Assignment based on projected severity
    if not is_hazard and prob < 0.40:
        hazard_class = "NORMAL_CONDITIONS"
        severity_level = "LEVEL_0_NORMAL"
    elif prob >= 0.85 or wind_speed_kmh >= 89.0:
        hazard_class = "VERY_SEVERE_CYCLONIC_STORM"
        severity_level = "LEVEL_4_CRITICAL"
    elif prob >= 0.70 or wind_speed_kmh >= 62.0:
        hazard_class = "CYCLONIC_STORM"
        severity_level = "LEVEL_3_SEVERE"
    elif is_hazard or prob >= 0.50:
        hazard_class = "DEPRESSION_OR_SQUALL_WATCH"
        severity_level = "LEVEL_2_WATCH"
    else:
        hazard_class = "WEATHER_ADVISORY"
        severity_level = "LEVEL_1_ADVISORY"

    primary_drivers = []
    if wind_speed_kmh >= 45.0:
        primary_drivers.append(f"Elevated sustained surface wind ({wind_speed_kmh:.1f} km/h)")
    if pressure_tendency_6h <= -2.0:
        primary_drivers.append(f"Barometric pressure drop ({pressure_tendency_6h:.1f} hPa/6h)")
    if sst_c >= 29.0:
        primary_drivers.append(f"High Sea Surface Temperature ({sst_c:.1f}°C latent heat)")
    if is_cyclone_season:
        primary_drivers.append("Active North Indian Ocean cyclone season")
    if not primary_drivers:
        primary_drivers.append("Stable maritime atmospheric pressure gradient")

    confidence = round(float(0.70 + 0.28 * abs(prob - 0.5) * 2), 2)

    return {
        "status": "SUCCESS",
        "model_available": True,
        "hazard_probability": round(prob, 4),
        "is_hazard": is_hazard,
        "hazard_class": hazard_class,
        "severity_level": severity_level,
        "prediction_horizon_hours": 24,
        "confidence_score": confidence,
        "primary_drivers": primary_drivers,
        "provenance": "ORCA Leakage-Free ML Cyclone Predictor (t+24h Horizon)",
    }
