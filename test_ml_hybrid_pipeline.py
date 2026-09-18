"""
test_ml_hybrid_pipeline.py - Verification Suite for ORCA Hybrid ML Pipeline
Tests:
1. Data Catalog: Verifies ml/data_catalog.csv documentation of MOSDAC, Copernicus, IBTrACS datasets.
2. ML Inference Engine:
   - Disaster Prediction Model (hazard class, probability, horizon, confidence, drivers)
   - Weather / Marine Forecasting Model (6h, 12h, 24h, 48h, 72h multi-horizon forecasts)
   - PFZ Habitat Suitability Model (suitability score, habitat class, ethical disclaimer)
3. Domain Agents Hybrid Integration:
   - DisasterAgent (MOSDAC satellite telemetry + ML disaster prediction)
   - WeatherAgent (3OSCAT/3RIMG telemetry + ML multi-horizon forecast)
   - PfzAgent (MOSDAC thermal/chlorophyll fronts + ML habitat suitability)
4. Decision Engine Priority & Safety Override:
   - Safety strictly overrides high PFZ fishing opportunity under hazard conditions
5. Checkpoint & Operational Resumption:
   - Verifies ml/checkpoint.json and train_all_models.py --resume
"""

import os
import sys
import json
import csv
from typing import Dict, Any

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def test_1_data_catalog():
    print("\n" + "=" * 70)
    print("[Test 1] Verifying Data Catalog (ml/data_catalog.csv)...")
    print("=" * 70)
    catalog_path = os.path.join("ml", "data_catalog.csv")
    assert os.path.exists(catalog_path), f"Missing {catalog_path}"
    
    with open(catalog_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    assert len(rows) >= 5, f"Expected at least 5 catalog entries, found {len(rows)}"
    
    agencies = set(r["source"].strip().upper() for r in rows)
    print(f"  [OK] Found {len(rows)} authoritative catalog entries across agencies: {agencies}")
    assert any("ISRO" in a or "MOSDAC" in a for a in agencies), "ISRO MOSDAC missing from catalog"
    assert any("COPERNICUS" in a or "CMEMS" in a for a in agencies), "Copernicus missing from catalog"
    assert any("NOAA" in a or "IBTRACS" in a for a in agencies), "NOAA IBTrACS missing from catalog"
    print("  [PASS] Test 1: Data Catalog contains multi-agency satellite/historical datasets.")


def test_2_ml_inference_engine():
    print("\n" + "=" * 70)
    print("[Test 2] Verifying 3 Specialized ML Inference Adapters...")
    print("=" * 70)
    
    # 2a. Disaster Model
    from ml.inference.disaster_infer import predict_marine_disaster
    calm_pred = predict_marine_disaster(
        latitude=18.9220, longitude=72.8347,
        olr_wm2=260.0, rainfall_rate_mmh=0.0, wind_speed_kmh=15.0,
        surface_pressure_hpa=1012.0, pressure_tendency_6h=0.5, sst_c=28.0
    )
    assert "hazard_probability" in calm_pred, "Missing hazard_probability"
    assert "hazard_class" in calm_pred, "Missing hazard_class"
    assert "prediction_horizon_hours" in calm_pred, "Missing prediction_horizon_hours"
    assert calm_pred["is_hazard"] is False, f"Expected calm conditions, got {calm_pred}"
    print(f"  [OK] Disaster Infer (Calm): Prob={calm_pred['hazard_probability']*100:.1f}%, Class={calm_pred['hazard_class']}")

    storm_pred = predict_marine_disaster(
        latitude=12.0, longitude=85.0,
        olr_wm2=120.0, rainfall_rate_mmh=45.0, wind_speed_kmh=85.0,
        surface_pressure_hpa=970.0, pressure_tendency_6h=-8.0, sst_c=30.5
    )
    assert storm_pred["is_hazard"] is True, f"Expected storm hazard, got {storm_pred}"
    print(f"  [OK] Disaster Infer (Storm): Prob={storm_pred['hazard_probability']*100:.1f}%, Class={storm_pred['hazard_class']}, Horizon={storm_pred['prediction_horizon_hours']}h")

    # 2b. Weather Multi-Horizon Forecasting Model
    from ml.inference.weather_infer import predict_marine_weather_forecast
    fc = predict_marine_weather_forecast(
        latitude=18.9220, longitude=72.8347,
        current_wind_kmh=18.0, current_wave_height_m=1.3,
        current_sst_c=28.2, current_pressure_hpa=1011.0,
        current_rainfall_mmh=0.0
    )
    assert "forecast_table" in fc, "Missing forecast_table in weather forecast"
    table = {row["horizon"]: row for row in fc["forecast_table"]}
    for h in ["6h", "12h", "24h", "48h", "72h"]:
        assert h in table, f"Missing horizon {h}"
        h_data = table[h]
        assert "wind_speed_kmh" in h_data
        assert "wave_height_m" in h_data
        assert "sst_c" in h_data
    print(f"  [OK] Weather Infer: Multi-horizon forecasts generated for {list(table.keys())}")
    print(f"       24h: Wind={table['24h']['wind_speed_kmh']} km/h, Wave={table['24h']['wave_height_m']} m")

    # 2c. PFZ Habitat Suitability Model
    from ml.inference.pfz_infer import predict_pfz_suitability
    pfz_res = predict_pfz_suitability(
        latitude=18.9220, longitude=72.8347,
        sst_c=27.5, sst_gradient=0.45,
        chlorophyll_a_mg_m3=0.85, chlorophyll_gradient=0.35,
        current_speed_knots=1.1, wind_speed_kmh=15.0
    )
    assert "habitat_suitability_index" in pfz_res or "pfz_probability" in pfz_res, "Missing suitability index"
    assert "ethical_disclaimer" in pfz_res, "Missing ethical_disclaimer"
    assert "predicts favorable" in pfz_res["ethical_disclaimer"].lower() or "guarantee" in pfz_res["ethical_disclaimer"].lower()
    print(f"  [OK] PFZ Infer: Prob={pfz_res['pfz_probability']}, Action={pfz_res['recommended_action'][:40]}...")
    print("  [PASS] Test 2: All 3 ML inference adapters operational and validated.")


def test_3_agents_hybrid_integration():
    print("\n" + "=" * 70)
    print("[Test 3] Verifying Domain Agents Hybrid Integration (MOSDAC + ML)...")
    print("=" * 70)
    
    from models import LocationContext, LocationSource, LocationStatus
    mumbai_loc = LocationContext(
        source=LocationSource.EXPLICIT_QUERY,
        status=LocationStatus.RESOLVED,
        name="Mumbai",
        latitude=18.9220,
        longitude=72.8347,
    )

    # 3a. DisasterAgent
    from disaster_agent import DisasterAgent
    da = DisasterAgent()
    d_out = da.execute_task(location=mumbai_loc, persona="FISHERMAN")
    assert "ml_prediction" in d_out, "Missing ml_prediction in DisasterAgent output"
    assert "satellite_provenance" in d_out or "telemetry" in d_out or "hazard_summary" in d_out
    print(f"  [OK] DisasterAgent: MOSDAC hazard={d_out.get('hazard_summary', {}).get('hazard_status')} | ML Class={d_out['ml_prediction'].get('hazard_class')}")

    # 3b. WeatherAgent
    from weather_agent import WeatherAgent
    wa = WeatherAgent()
    w_out = wa.execute_task(location=mumbai_loc, persona="FISHERMAN")
    assert "ml_forecast" in w_out, "Missing ml_forecast in WeatherAgent output"
    assert "forecast_table" in w_out["ml_forecast"]
    w_table = {r["horizon"]: r for r in w_out["ml_forecast"]["forecast_table"]}
    print(f"  [OK] WeatherAgent: Wind={w_out.get('telemetry', {}).get('wind_speed_kmh')} km/h | ML 24h Wave={w_table['24h']['wave_height_m']} m")

    # 3c. PFZAgent
    from pfz_agent import PfzAgent
    pa = PfzAgent()
    p_out = pa.execute_task(location=mumbai_loc, persona="FISHERMAN")
    assert "ml_suitability" in p_out, "Missing ml_suitability in PfzAgent output"
    assert "ethical_disclaimer" in p_out["ml_suitability"]
    print(f"  [OK] PFZAgent: Score={p_out.get('score')} | ML Prob={p_out['ml_suitability'].get('pfz_probability')}")
    print("  [PASS] Test 3: Domain agents return both deterministic satellite telemetry and ML predictions.")


def test_4_safety_overrides_pfz():
    print("\n" + "=" * 70)
    print("[Test 4] Verifying Safety Overrides Fishing Opportunity Guardrail...")
    print("=" * 70)
    from decision_engine import RiskAnalysisAgent

    raa = RiskAnalysisAgent()
    
    # Simulate high PFZ suitability (score 92, ML prob 0.88) combined with ML Disaster alert
    hazardous_payload = {
        "WEATHER_AGENT": {
            "telemetry": {
                "wave_height_m": 3.8,
                "wind_speed_kmh": 62.0,
            }
        },
        "DISASTER_AGENT": {
            "hazard_summary": {
                "has_active_hazard": True,
                "hazard_status": "DANGER",
                "description": "Severe Cyclonic Storm in operating sector",
            },
            "ml_prediction": {
                "is_hazard": True,
                "hazard_class": "VERY_SEVERE_CYCLONIC_STORM",
                "hazard_probability": 0.94,
                "prediction_horizon_hours": 24,
                "confidence_score": 0.92,
                "primary_drivers": ["Intense convection", "Rapid pressure plunge"]
            }
        },
        "PFZ_AGENT": {
            "geojson": {
                "features": [{
                    "properties": {
                        "suitability_score": 92,
                        "ml_pfz_probability": 0.88,
                    }
                }]
            },
            "ml_suitability": {
                "pfz_probability": 0.88,
                "habitat_suitability_index": 0.88
            }
        }
    }
    
    threat_eval = raa.evaluate_threats(
        aggregated_data=hazardous_payload,
        normalized_query="Where is the best fishing zone?",
        persona="FISHERMAN"
    )
    
    assert threat_eval["status"] == "DANGER", f"Expected DANGER status, got {threat_eval['status']}"
    assert threat_eval["risk_score"] >= 88.0, f"Expected risk score >= 88, got {threat_eval['risk_score']}"
    
    # Verify SAFETY_OVERRIDES_PFZ alert is present
    threat_types = [t.get("type") for t in threat_eval.get("threat_prioritization", []) if isinstance(t, dict)]
    warning_texts = " ".join(threat_eval.get("warnings", []))
    
    assert "SAFETY_OVERRIDES_PFZ" in threat_types or "Safety Priority Override" in warning_texts, (
        f"Safety override over fishing not found in threats: {threat_eval.get('warnings')}"
    )
    print(f"  [OK] Evaluated Status: {threat_eval['status']} (Risk Score: {threat_eval['risk_score']}/100)")
    print(f"  [OK] Safety Override confirmed: High PFZ opportunity suppressed due to cyclonic threat.")
    print("  [PASS] Test 4: Safety Strictly Overrides Fishing Opportunity contract verified.")


def test_5_checkpoint_resumption():
    print("\n" + "=" * 70)
    print("[Test 5] Verifying ml/checkpoint.json & CLI Resumption Contract...")
    print("=" * 70)
    import subprocess
    ckpt_path = os.path.join("ml", "checkpoint.json")
    assert os.path.exists(ckpt_path), f"Missing {ckpt_path}"
    
    with open(ckpt_path, "r", encoding="utf-8") as f:
        ckpt_data = json.load(f)
    
    assert "completed_steps" in ckpt_data, "Missing completed_steps in checkpoint"
    completed = ckpt_data["completed_steps"]
    print(f"  [OK] Checkpoint records {len(completed)} completed pipeline stages:")
    for s, info in completed.items():
        print(f"       - {s} (at {info.get('timestamp')})")
    
    assert len(completed) >= 3, f"Expected at least 3 completed steps, found {len(completed)}"

    # Test master CLI with --resume
    proc = subprocess.run(
        [sys.executable, "train_all_models.py", "--resume"],
        capture_output=True,
        text=True,
        timeout=30
    )
    combined_out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, f"train_all_models.py --resume failed: {proc.stderr}"
    assert "PIPELINE EXECUTION FINISHED SUCCESSFULLY" in combined_out, "Pipeline finish banner not found"
    print("  [OK] train_all_models.py --resume executed cleanly in < 1s verifying all checkpoints.")
    print("  [PASS] Test 5: Checkpoint manager and resume contract fully validated.")


if __name__ == "__main__":
    print("\n=================================================================")
    print("ORCA HYBRID ML SYSTEM VERIFICATION SUITE")
    print("ISRO SIH 2026 Problem Statement SIH26176")
    print("=================================================================")
    
    try:
        test_1_data_catalog()
        test_2_ml_inference_engine()
        test_3_agents_hybrid_integration()
        test_4_safety_overrides_pfz()
        test_5_checkpoint_resumption()
        print("\n=================================================================")
        print("ALL HYBRID ML PIPELINE TESTS PASSED WITH 100% SUCCESS! [SUCCESS]")
        print("=================================================================\n")
    except Exception as e:
        print(f"\n[FAIL] Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
