"""
test_api_trained_models_integration.py - Full-Stack End-to-End API Test Suite
SIH 2026 Problem Statement SIH26176

Tests the complete flow across the 5 mandatory user queries:
User query -> Language Layer -> Manager -> Specialized Agent -> ML Model -> Risk Analysis -> Reasoning Agent -> Final Response

1. "Will the weather be safe tomorrow?"
2. "Is there a cyclone near Chennai?"
3. "Where is the best fishing zone?"
4. "Show the wave conditions near Mumbai."
5. "Plan a safe route from Tuticorin to Sri Lanka."
"""

import os
import sys
import json
from pathlib import Path

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from server import execute_orca_core
from ml.model_registry import ModelRegistry


def run_api_tests():
    print("=" * 80)
    print("ORCA TRAINED MODEL API INTEGRATION TEST SUITE")
    print("=" * 80)

    # Verify model registry startup status
    ModelRegistry.load_all_models()
    ModelRegistry.print_startup_status()

    test_queries = [
        {
            "id": 1,
            "query": "Will the weather be safe tomorrow?",
            "lat": 8.76, "lon": 78.13,
            "expected_agent": "WEATHER_AGENT",
            "expected_ml_key": "ml_forecast",
            "intent": "Weather Forecasting",
        },
        {
            "id": 2,
            "query": "Is there a cyclone near Chennai?",
            "lat": 13.08, "lon": 80.27,
            "expected_agent": "DISASTER_AGENT",
            "expected_ml_key": "ml_disaster",
            "intent": "Disaster & Cyclone Hazard",
        },
        {
            "id": 3,
            "query": "Where is the best fishing zone?",
            "lat": 8.76, "lon": 78.13,
            "expected_agent": "PFZ_AGENT",
            "expected_ml_key": "ml_pfz",
            "intent": "PFZ Habitat Suitability",
        },
        {
            "id": 4,
            "query": "Show the wave conditions near Mumbai.",
            "lat": 18.92, "lon": 72.83,
            "expected_agent": "WEATHER_AGENT",
            "expected_ml_key": "ml_forecast",
            "intent": "Marine Observation & Forecast",
        },
        {
            "id": 5,
            "query": "Plan a safe route from Tuticorin to Sri Lanka.",
            "lat": 8.76, "lon": 78.13,
            "expected_agent": "GIS_AGENT",
            "expected_ml_key": None,
            "intent": "IMBL Border Security & Landmasking",
        },
    ]

    all_passed = True

    for t in test_queries:
        print("\n" + "=" * 80)
        print(f"[TEST {t['id']}/5] Query: '{t['query']}'")
        print(f"  - Target Intent : {t['intent']}")
        print(f"  - Coordinate Ref: ({t['lat']}, {t['lon']})")
        print("=" * 80)

        res = execute_orca_core(
            query=t["query"],
            lat=t["lat"],
            lon=t["lon"],
            persona="FISHERMAN",
            language="en",
        )

        assert res is not None, "Response cannot be None"
        status = res.get("status")
        chat_text = res.get("chat_text") or res.get("native_advisory_text") or ""
        agents_used = res.get("agents_used") or []
        telemetry_prov = res.get("telemetry_provenance") or {}
        risk_data = res.get("risk_assessment") or {}
        advisory_dict = res.get("advisory") or {}
        key_advisories = advisory_dict.get("key_advisories", [])

        print(f"  - Server Status   : {status}")
        print(f"  - Threat Status   : {res.get('threat_status') or risk_data.get('status')}")
        print(f"  - Risk Score      : {res.get('risk_score')}")
        print(f"  - Advisory Text   : {chat_text[:110]}...")

        # Check telemetry provenance tags
        print("  - Key Advisories  :")
        for ka in key_advisories[:3]:
            print(f"      * {ka}")

        print("  - Telemetry Provenance Breakdown:")
        for k, v in list(telemetry_prov.items())[:4]:
            print(f"      * {k:<20}: {v.get('value')} [{v.get('source')}]")

        # Specific Query Verifications
        if t["id"] == 1:
            # Weather query: check forecast active & ML table
            weather_out = res.get("WEATHER_AGENT") or {}
            ml_fc = weather_out.get("ml_forecast") or {}
            has_fc = len(ml_fc.get("forecast_table", [])) > 0
            print(f"  [CHECK] WeatherAgent invoked with Multi-Horizon ML forecast: {has_fc}")
            assert has_fc, "WeatherAgent should produce multi-horizon forecast table"

        elif t["id"] == 2:
            # Cyclone query: check disaster model
            disaster_out = res.get("DISASTER_AGENT") or {}
            ml_dis = disaster_out.get("ml_disaster") or {}
            prob = ml_dis.get("hazard_probability")
            print(f"  [CHECK] DisasterAgent invoked with 24h hazard prob: {prob}")
            assert prob is not None, "DisasterAgent should produce 24h hazard probability"

        elif t["id"] == 3:
            # PFZ query: check habitat suitability
            pfz_out = res.get("PFZ_AGENT") or {}
            ml_p = pfz_out.get("ml_pfz") or pfz_out.get("ml_suitability") or {}
            hsi = ml_p.get("habitat_suitability_index") or ml_p.get("pfz_probability")
            print(f"  [CHECK] PFZAgent invoked with 48h persistence index: {hsi}")
            assert hsi is not None, "PFZAgent should produce habitat suitability index"

        elif t["id"] == 4:
            # Mumbai wave conditions query
            loc_ctx = res.get("location_context") or {}
            loc_name = loc_ctx.get("name") if isinstance(loc_ctx, dict) else getattr(loc_ctx, "location_name", "")
            print(f"  [CHECK] Mumbai location resolved cleanly: {loc_name}")
            assert "mumbai" in str(loc_name).lower(), "Location must be Mumbai"

        elif t["id"] == 5:
            # Tuticorin to Sri Lanka: IMBL border check
            threat_st = res.get("threat_status") or risk_data.get("status")
            imbl_v = res.get("imbl_violation") or risk_data.get("metrics", {}).get("imbl_violation")
            print(f"  [CHECK] Sri Lanka route IMBL violation detected: {imbl_v} | Threat: {threat_st}")
            assert threat_st == "DANGER", "Sri Lanka cross-border transit must evaluate to DANGER"

        print(f"  [RESULT] TEST {t['id']} PASSED SUCCESSFULLY.")

    print("\n" + "=" * 80)
    print("ALL 5 FULL-STACK END-TO-END API TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 80)


if __name__ == "__main__":
    run_api_tests()

