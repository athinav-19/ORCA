"""
pfz_infer.py - Potential Fishing Zone (PFZ) Habitat Suitability Inference Adapter
SIH 2026 Problem Statement SIH26176

Evaluates biophysical habitat suitability for pelagic shoals based on INCOIS criteria
combining satellite SST/Chlorophyll observations with trained ML probabilities
predicting 48-hour frontal persistence.
"""

import os
import math
import joblib
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

from ml.data_ingestion.bathymetry_geography import get_geographic_features
from ml.feature_engineering.pfz_features import PFZ_FEATURE_COLS

MODEL_PATH = os.path.join("ml", "models", "pfz", "best_model.joblib")
_CACHED_MODEL = None


def get_pfz_model():
    """Loads and caches the trained PFZ model via ModelRegistry."""
    global _CACHED_MODEL
    try:
        from ml.model_registry import ModelRegistry
        reg_model = ModelRegistry.get_model("pfz")
        if reg_model is not None:
            _CACHED_MODEL = reg_model
            return _CACHED_MODEL
    except Exception:
        pass

    if _CACHED_MODEL is None and os.path.exists(MODEL_PATH):
        try:
            _CACHED_MODEL = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"[PFZInfer Error] Model loading failed: {e}")
            _CACHED_MODEL = None
    return _CACHED_MODEL


def predict_pfz_suitability(
    latitude: float,
    longitude: float,
    sst_c: Optional[float] = None,
    sst_gradient: Optional[float] = None,
    chlorophyll_a_mg_m3: Optional[float] = None,
    chlorophyll_gradient: Optional[float] = None,
    current_speed_knots: Optional[float] = None,
    wind_speed_kmh: Optional[float] = None,
    observation_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Predicts oceanographic habitat suitability and potential pelagic aggregation probability.
    Requires observed SST and Chlorophyll inputs. Returns DATA_UNAVAILABLE when missing.
    """
    model = get_pfz_model()

    # Reject missing / None / NaN inputs rather than fabricating fake oceanic conditions
    if sst_c is None or chlorophyll_a_mg_m3 is None:
        return {
            "status": "DATA_UNAVAILABLE",
            "model_available": model is not None,
            "message": "PFZ prediction unavailable: genuine SST and Chlorophyll-a satellite/in-situ observations are required.",
            "is_favorable": None,
            "pfz_probability": None,
            "confidence_score": 0.0,
            "recommended_action": "DATA_UNAVAILABLE",
            "data_classification": "RULE_EMULATION",
            "ground_truth_status": "PENDING_REAL_GROUND_TRUTH",
            "ethical_disclaimer": "Potential Fishing Zone model evaluates oceanographic fronts. NEVER represents guaranteed fish catch.",
        }

    try:
        sst_c = float(sst_c)
        chlorophyll_a_mg_m3 = float(chlorophyll_a_mg_m3)
        if math.isnan(sst_c) or math.isnan(chlorophyll_a_mg_m3):
            raise ValueError("NaN value in oceanographic inputs")
    except (TypeError, ValueError):
        return {
            "status": "DATA_UNAVAILABLE",
            "model_available": model is not None,
            "message": "PFZ prediction unavailable: invalid or NaN biophysical inputs.",
            "is_favorable": None,
            "pfz_probability": None,
            "confidence_score": 0.0,
            "recommended_action": "DATA_UNAVAILABLE",
            "data_classification": "RULE_EMULATION",
            "ground_truth_status": "PENDING_REAL_GROUND_TRUTH",
        }

    sst_gradient = 0.8 if (sst_gradient is None or math.isnan(sst_gradient)) else float(sst_gradient)
    chlorophyll_gradient = 0.3 if (chlorophyll_gradient is None or math.isnan(chlorophyll_gradient)) else float(chlorophyll_gradient)
    current_speed_knots = 1.2 if (current_speed_knots is None or math.isnan(current_speed_knots)) else float(current_speed_knots)
    wind_speed_kmh = 18.0 if (wind_speed_kmh is None or math.isnan(wind_speed_kmh)) else float(wind_speed_kmh)

    dt = observation_time or datetime.now()
    m = dt.month
    month_sin = math.sin(2 * math.pi * m / 12.0)
    month_cos = math.cos(2 * math.pi * m / 12.0)

    geo = get_geographic_features(latitude, longitude)
    depth_m = geo["bathymetry_depth_m"]
    dist_nm = geo["distance_to_coast_nm"]

    wind_stress = round(0.0013 * 1.2 * ((wind_speed_kmh / 3.6) ** 2), 4)
    upwelling_idx = round(sst_gradient * (wind_speed_kmh / 11.0), 2)
    sst_lag_24h = sst_c
    chl_lag_24h = chlorophyll_a_mg_m3

    feature_dict = {
        "latitude": latitude,
        "longitude": longitude,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "sst_c": sst_c,
        "sst_gradient": sst_gradient,
        "chlorophyll_a_mg_m3": chlorophyll_a_mg_m3,
        "chlorophyll_gradient": chlorophyll_gradient,
        "current_speed_knots": current_speed_knots,
        "wind_speed_kmh": wind_speed_kmh,
        "wind_stress": wind_stress,
        "bathymetry_depth_m": depth_m,
        "distance_to_coast_nm": dist_nm,
        "upwelling_index": upwelling_idx,
        "sst_lag_24h": sst_lag_24h,
        "chl_lag_24h": chl_lag_24h,
    }

    features = [feature_dict[col] for col in PFZ_FEATURE_COLS]

    if model is not None:
        try:
            X = np.array(features).reshape(1, -1)
            prob = float(model.predict_proba(X)[0, 1])
            is_fav = bool(prob >= 0.50)
        except Exception as e:
            return {
                "status": "UNAVAILABLE",
                "model_available": False,
                "message": f"Prediction unavailable because required data/model is unavailable: {e}",
                "pfz_probability": None,
                "habitat_suitability_index": None,
                "is_favorable": None,
                "confidence_score": 0.0,
                "recommended_action": "PREDICTION_UNAVAILABLE",
                "prediction_horizon_hours": 48,
                "biophysical_factors": {},
                "ethical_disclaimer": "PFZ Habitat Suitability Model (UNAVAILABLE)",
                "provenance": "ORCA PFZ Habitat Suitability Model (UNAVAILABLE)",
            }
    else:
        return {
            "status": "UNAVAILABLE",
            "model_available": False,
            "message": "Prediction unavailable because required data/model is unavailable.",
            "pfz_probability": None,
            "habitat_suitability_index": None,
            "is_favorable": None,
            "confidence_score": 0.0,
            "recommended_action": "PREDICTION_UNAVAILABLE",
            "prediction_horizon_hours": 48,
            "biophysical_factors": {},
            "ethical_disclaimer": "PFZ Habitat Suitability Model (UNAVAILABLE)",
            "provenance": "ORCA PFZ Habitat Suitability Model (UNAVAILABLE)",
        }

    hsi = round(prob, 3)
    conf = round(float(0.72 + 0.24 * abs(prob - 0.5) * 2), 2)

    if hsi >= 0.70:
        recommendation = "HIGH_HABITAT_SUITABILITY (Optimal Thermal & Chlorophyll Front Convergence)"
    elif hsi >= 0.45:
        recommendation = "MODERATE_SUITABILITY (Marginal Frontal Feature)"
    else:
        recommendation = "LOW_SUITABILITY (Uniform Water Mass / Dispersed Biomass)"

    return {
        "status": "SUCCESS",
        "model_available": True,
        "pfz_probability": hsi,
        "habitat_suitability_index": hsi,
        "is_favorable": is_fav,
        "confidence_score": conf,
        "recommended_action": recommendation,
        "prediction_horizon_hours": 48,
        "biophysical_factors": {
            "thermal_gradient_c_10km": sst_gradient,
            "chlorophyll_mg_m3": chlorophyll_a_mg_m3,
            "bathymetric_depth_m": depth_m,
            "distance_to_coast_nm": dist_nm,
            "upwelling_proxy": upwelling_idx,
        },
        "ethical_disclaimer": "Predicts favorable oceanographic habitat suitability. NEVER represents guaranteed fish presence.",
        "provenance": "ORCA PFZ Habitat Suitability Model (INCOIS Methodology)",
    }
