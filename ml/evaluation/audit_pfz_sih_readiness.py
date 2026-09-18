"""
audit_pfz_sih_readiness.py - Deep Audit & Stress Testing of ORCA PFZ Model
SIH 2026 Problem Statement SIH26176
"""

import os
import sys
import math
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Add root directory to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.model_registry import ModelRegistry
from ml.feature_engineering.pfz_features import (
    PFZ_FEATURE_COLS,
    build_pfz_dataset,
    get_pfz_splits,
)
from ml.inference.pfz_infer import predict_pfz_suitability


def run_pfz_audit():
    print("=" * 80)
    print("ORCA PFZ MODEL COMPREHENSIVE AUDIT & SIH READINESS CHECK")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Inspect Dataset & Target Derivation
    # -------------------------------------------------------------------------
    print("\n[CHECK 1 & 2] Verifying Target Derivation & Feature Availability Timestamps...")
    df = build_pfz_dataset(start_year=2005, end_year=2025)
    train_df, val_df, test_df = get_pfz_splits(df)

    print(f"  Total samples generated : {len(df)}")
    print(f"  Train samples (2005-2018): {len(train_df)}")
    print(f"  Val samples   (2019-2021): {len(val_df)}")
    print(f"  Test samples  (2022-2025): {len(test_df)}")

    # Check correlations between input features at time t and target at time t+1
    print("\n  Feature-to-Target Correlation Check (Checking for circular derivation / leakage):")
    max_corr = 0.0
    max_feat = ""
    correlations = {}
    for feat in PFZ_FEATURE_COLS:
        r = np.corrcoef(train_df[feat], train_df["target_pfz_persistence"])[0, 1]
        correlations[feat] = round(float(r), 4)
        if abs(r) > abs(max_corr):
            max_corr = r
            max_feat = feat
        print(f"    - {feat:<22}: r = {r:+.4f}")

    assert abs(max_corr) < 0.95, f"Data leakage detected! Feature '{max_feat}' has correlation {max_corr:.4f} >= 0.95"
    print(f"  [PASS] Highest continuous correlation is '{max_feat}' (r = {max_corr:.4f}), well below 0.95 boundary.")

    # Verify that target is NOT identical to current-step threshold
    current_step_favorable = (
        (train_df["sst_gradient"] >= 0.60) &
        (train_df["chlorophyll_a_mg_m3"] >= 0.55) &
        (train_df["bathymetry_depth_m"] >= 25.0) &
        (train_df["bathymetry_depth_m"] <= 850.0)
    ).astype(int)

    disagreements = (current_step_favorable != train_df["target_pfz_persistence"]).sum()
    disagreement_pct = (disagreements / len(train_df)) * 100.0
    print(f"  [PASS] Disagreements between current condition and future t+48h target: {disagreements} / {len(train_df)} ({disagreement_pct:.2f}% dynamic transitions).")
    assert disagreements > 0, "Target is identical to current step! Severe leakage!"

    # -------------------------------------------------------------------------
    # 2. Verify Chronological Separation
    # -------------------------------------------------------------------------
    print("\n[CHECK 3] Verifying Chronological Separation & Temporal Barrier...")
    max_train_date = train_df["datetime"].max()
    min_test_date = test_df["datetime"].min()
    print(f"  Max Train Date: {max_train_date}")
    print(f"  Min Test Date : {min_test_date}")
    assert min_test_date > max_train_date, "Test set overlaps chronologically with training set!"
    print(f"  [PASS] Strict chronological barrier confirmed: {min_test_date} > {max_train_date}.")

    # -------------------------------------------------------------------------
    # 3. Model Loading & Feature Order
    # -------------------------------------------------------------------------
    print("\n[CHECK 4] Model Loading & Feature Ordering Verification...")
    ModelRegistry.load_all_models()
    model = ModelRegistry.get_model("pfz")
    assert model is not None, "PFZ model failed to load from ModelRegistry!"
    features = ModelRegistry.get_features("pfz")
    assert features == PFZ_FEATURE_COLS, f"Feature ordering mismatch! {features} != {PFZ_FEATURE_COLS}"
    print(f"  [PASS] Model loaded successfully: {type(model).__name__}")
    print(f"  [PASS] Feature ordering verified (16 features in exact sequence): {features[:4]} ... {features[-3:]}")

    # -------------------------------------------------------------------------
    # 4. Five Specific Operational Inference Scenarios
    # -------------------------------------------------------------------------
    print("\n[CHECK 5 & 6] Executing 5 Specific Operational Inference Scenarios...")

    scenarios = [
        {
            "name": "Normal Conditions",
            "desc": "Calm coastal waters off Tuticorin (Lat 8.76, Lon 78.13)",
            "params": {
                "latitude": 8.76,
                "longitude": 78.13,
                "sst_c": 28.2,
                "sst_gradient": 0.45,
                "chlorophyll_a_mg_m3": 0.40,
                "chlorophyll_gradient": 0.15,
                "current_speed_knots": 0.8,
                "wind_speed_kmh": 14.0,
            }
        },
        {
            "name": "Favorable PFZ Conditions",
            "desc": "Kanyakumari Wadge Bank Upwelling Zone during Monsoon (Lat 7.80, Lon 77.20)",
            "params": {
                "latitude": 7.80,
                "longitude": 77.20,
                "sst_c": 26.8,
                "sst_gradient": 1.25,
                "chlorophyll_a_mg_m3": 1.95,
                "chlorophyll_gradient": 0.65,
                "current_speed_knots": 1.6,
                "wind_speed_kmh": 26.0,
                "observation_time": datetime(2026, 7, 15, 12, 0),
            }
        },
        {
            "name": "Unfavorable Conditions",
            "desc": "Deep Pelagic Abyssal Plain with low chlorophyll (Lat 15.00, Lon 67.00)",
            "params": {
                "latitude": 15.00,
                "longitude": 67.00,
                "sst_c": 29.5,
                "sst_gradient": 0.12,
                "chlorophyll_a_mg_m3": 0.09,
                "chlorophyll_gradient": 0.02,
                "current_speed_knots": 0.4,
                "wind_speed_kmh": 10.0,
            }
        },
        {
            "name": "Missing Data Conditions",
            "desc": "Simulated satellite sensor dropouts with None / NaN telemetry",
            "params": {
                "latitude": 13.15,
                "longitude": 80.55,
                "sst_c": None,
                "sst_gradient": None,
                "chlorophyll_a_mg_m3": None,
                "chlorophyll_gradient": None,
                "current_speed_knots": None,
                "wind_speed_kmh": None,
            }
        },
        {
            "name": "Unusual Coordinates",
            "desc": "Equatorial Open Ocean High-Seas Boundary (Lat 0.50, Lon 92.00)",
            "params": {
                "latitude": 0.50,
                "longitude": 92.00,
                "sst_c": 29.8,
                "sst_gradient": 0.20,
                "chlorophyll_a_mg_m3": 0.15,
                "chlorophyll_gradient": 0.05,
                "current_speed_knots": 0.6,
                "wind_speed_kmh": 12.0,
            }
        },
    ]

    inference_results = []
    for sc in scenarios:
        print(f"\n  Testing: {sc['name']} - {sc['desc']}")
        out = predict_pfz_suitability(**sc["params"])
        print(f"    Status       : {out.get('status')}")
        print(f"    PFZ Prob     : {out.get('pfz_probability')}")
        print(f"    Is Favorable : {out.get('is_favorable')}")
        print(f"    Confidence   : {out.get('confidence_score')}")
        print(f"    Action       : {out.get('recommended_action')}")

        prob = out.get("pfz_probability")
        conf = out.get("confidence_score")

        # Confirm no NaN / Inf
        assert prob is not None, f"Probability is None in {sc['name']}"
        assert not math.isnan(prob), f"Probability is NaN in {sc['name']}"
        assert not math.isinf(prob), f"Probability is Inf in {sc['name']}"
        assert 0.0 <= prob <= 1.0, f"Probability {prob} out of [0, 1] range in {sc['name']}"

        assert conf is not None, f"Confidence is None in {sc['name']}"
        assert not math.isnan(conf), f"Confidence is NaN in {sc['name']}"
        assert not math.isinf(conf), f"Confidence is Inf in {sc['name']}"
        assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of [0, 1] range in {sc['name']}"

        inference_results.append({
            "scenario": sc["name"],
            "probability": prob,
            "is_favorable": out.get("is_favorable"),
            "confidence": conf,
            "action": out.get("recommended_action"),
        })

    # Validate physical discrimination:
    fav_prob = inference_results[1]["probability"] # Favorable Wadge Bank
    unfav_prob = inference_results[2]["probability"] # Abyssal Plain
    print(f"\n  [Physical Discrimination Verification]:")
    print(f"    Favorable Wadge Bank Prob: {fav_prob:.4f}")
    print(f"    Unfavorable Abyssal Prob : {unfav_prob:.4f}")
    assert fav_prob > unfav_prob, f"Model failed physical discrimination: {fav_prob} <= {unfav_prob}"
    assert fav_prob >= 0.70, f"Favorable conditions expected >= 0.70, got {fav_prob}"
    assert unfav_prob <= 0.30, f"Unfavorable conditions expected <= 0.30, got {unfav_prob}"
    print(f"  [PASS] Physical discrimination confirmed (Delta = {fav_prob - unfav_prob:.4f}).")

    print("\n" + "=" * 80)
    print("ALL PFZ CHECKS COMPLETED AND VERIFIED SUCCESSFULLY [PASS]!")
    print("=" * 80)

    return {
        "dataset_size": len(df),
        "correlations": correlations,
        "disagreements": int(disagreements),
        "disagreement_pct": float(disagreement_pct),
        "inference_results": inference_results,
    }


if __name__ == "__main__":
    run_pfz_audit()
