"""
test_pfz_real_model.py - Test Suite for ORCA PFZ Model & Production Integration
Covers all 14 mandatory test cases specified in the PFZ ML Master Task.
"""

import os
import json
import math
import numpy as np

from ml.inference.pfz_infer import predict_pfz_suitability, get_pfz_model
from ml.feature_engineering.pfz_features import PFZ_FEATURE_COLS
from gis_agent import GisAgent
from pfz_agent import PfzAgent


def test_1_model_loads():
    """1. Model loads successfully and exposes prediction interface."""
    model = get_pfz_model()
    assert model is not None, "PFZ model failed to load from disk / ModelRegistry"
    assert hasattr(model, "predict_proba"), "Loaded PFZ model must support predict_proba"
    assert hasattr(model, "predict"), "Loaded PFZ model must support predict"


def test_2_feature_schema_matches():
    """2. Feature schema in metadata/feature_list matches PFZ_FEATURE_COLS exactly."""
    feature_file = os.path.join("ml", "models", "pfz", "feature_list.json")
    assert os.path.exists(feature_file), "feature_list.json not found"
    with open(feature_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "features" in data, "feature_list.json must define 'features'"
    assert data["features"] == PFZ_FEATURE_COLS, (
        f"Feature list mismatch:\nExpected: {PFZ_FEATURE_COLS}\nFound: {data['features']}"
    )
    assert len(data["features"]) == 16, f"Expected 16 features, got {len(data['features'])}"


def test_3_inference_works():
    """3. Inference succeeds with valid inputs and returns expected structure."""
    res = predict_pfz_suitability(
        latitude=15.35,
        longitude=73.40,
        sst_c=28.2,
        sst_gradient=0.75,
        chlorophyll_a_mg_m3=0.68,
        chlorophyll_gradient=0.28,
        current_speed_knots=1.2,
        wind_speed_kmh=15.0,
    )
    assert res["status"] == "SUCCESS", f"Inference status should be SUCCESS, got: {res['status']}"
    assert res["model_available"] is True
    assert isinstance(res["is_favorable"], bool)
    assert 0.0 <= res["pfz_probability"] <= 1.0
    assert 0.0 <= res["confidence_score"] <= 1.0
    assert res["prediction_horizon_hours"] == 48


def test_4_missing_sst_fails_safely():
    """4. Missing SST fails safely with DATA_UNAVAILABLE without crashing."""
    res = predict_pfz_suitability(
        latitude=15.35,
        longitude=73.40,
        sst_c=None,
        chlorophyll_a_mg_m3=0.68,
    )
    assert res["status"] == "DATA_UNAVAILABLE"
    assert res["is_favorable"] is None
    assert res["pfz_probability"] is None
    assert "required" in res["message"].lower() or "unavailable" in res["message"].lower()


def test_5_missing_chlorophyll_fails_safely():
    """5. Missing Chlorophyll-a fails safely with DATA_UNAVAILABLE."""
    res = predict_pfz_suitability(
        latitude=15.35,
        longitude=73.40,
        sst_c=28.5,
        chlorophyll_a_mg_m3=None,
    )
    assert res["status"] == "DATA_UNAVAILABLE"
    assert res["is_favorable"] is None
    assert res["pfz_probability"] is None


def test_6_missing_required_feature_fails_safely():
    """6. Invalid or NaN biophysical inputs fail safely with DATA_UNAVAILABLE."""
    res_nan = predict_pfz_suitability(
        latitude=15.35,
        longitude=73.40,
        sst_c=float("nan"),
        chlorophyll_a_mg_m3=0.5,
    )
    assert res_nan["status"] == "DATA_UNAVAILABLE"
    assert res_nan["is_favorable"] is None


def test_7_no_fake_default_environmental_values():
    """7. Omitting inputs does NOT silently substitute fake values."""
    res_empty = predict_pfz_suitability(latitude=12.0, longitude=74.0)
    assert res_empty["status"] == "DATA_UNAVAILABLE", "Must not fabricate default SST/Chl values"
    assert res_empty["pfz_probability"] is None
    assert res_empty["recommended_action"] == "DATA_UNAVAILABLE"


def test_8_output_probability_bounded():
    """8. Output probability is strictly bounded in [0.0, 1.0]."""
    test_points = [
        (20.85, 70.35, 27.5, 0.95),  # Veraval upwelling zone
        (15.00, 67.00, 29.5, 0.12),  # Central Arabian Sea Abyssal plain
        (9.80, 75.80, 27.8, 1.10),   # Kochi Wadge Bank
        (13.15, 80.55, 29.0, 0.45),  # Chennai Deep Shelf
    ]
    for lat, lon, sst, chl in test_points:
        res = predict_pfz_suitability(latitude=lat, longitude=lon, sst_c=sst, chlorophyll_a_mg_m3=chl)
        assert res["status"] == "SUCCESS"
        prob = res["pfz_probability"]
        assert 0.0 <= prob <= 1.0, f"Probability {prob} out of bounds for point ({lat}, {lon})"


def test_9_output_contains_provenance():
    """9. Output contains explicit provenance and binding ethical disclaimer."""
    res = predict_pfz_suitability(latitude=10.0, longitude=76.0, sst_c=28.0, chlorophyll_a_mg_m3=0.7)
    assert "provenance" in res
    assert "ethical_disclaimer" in res
    assert "NEVER represents guaranteed fish presence" in res["ethical_disclaimer"]


def test_10_model_version_and_status_reported():
    """10. Model metadata explicitly reports classification and ground-truth status."""
    meta_path = os.path.join("ml", "models", "pfz", "metadata.json")
    assert os.path.exists(meta_path), "metadata.json missing"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["data_classification"] == "RULE_EMULATION"
    assert meta["status"] == "PENDING_REAL_GROUND_TRUTH"
    assert "scientific_basis" in meta
    assert "ethical_disclaimer" in meta


def test_11_spatial_filtering_works():
    """11. Spatial coordinate resolution works for recognized ports and catches unknown locations."""
    agent = PfzAgent()
    res_valid = agent._resolve_port("Kochi")
    assert res_valid[0] is not None and res_valid[1] is not None
    assert abs(res_valid[0] - 9.9312) < 0.1

    res_unknown = agent.execute_task(location="RandomNonExistentPlaceXYZ", time_frame="today")
    assert res_unknown["status"] == "LOCATION_REQUIRED"


def test_12_sri_lanka_land_areas_rejected():
    """12. Sri Lanka inland points are rejected as non-marine land area."""
    gis = GisAgent()
    # Central Sri Lanka inland point (Kandy / central highlands)
    is_marine_land = gis.is_over_land(7.2906, 80.6337)
    assert is_marine_land is True, "Sri Lanka inland coordinate (7.29, 80.63) must be recognized as land"


def test_13_eez_filtering_works():
    """13. Indian EEZ boundary correctly filters domestic waters vs international high seas."""
    gis = GisAgent()
    # Point inside Indian EEZ (off Mumbai coast)
    in_eez = gis.is_in_indian_eez(18.9, 71.5)
    assert in_eez is True, "Mumbai offshore (18.9N, 71.5E) must be inside Indian EEZ"

    # Point far outside Indian EEZ (Southern Ocean / Atlantic)
    out_eez = gis.is_in_indian_eez(-35.0, 15.0)
    assert out_eez is False, "South Atlantic point (-35.0N, 15.0E) must be outside Indian EEZ"


def test_14_api_integration_works():
    """14. PFZAgent.execute_task integration produces valid GeoJSON with ML suitability."""
    agent = PfzAgent()
    res = agent.execute_task(
        location="Kochi",
        time_frame="today",
        expected_format="GEOJSON_POLYGONS",
    )
    assert "status" in res
    assert "geojson" in res
    assert res["geojson"]["type"] == "FeatureCollection"
    assert "ethical_disclaimer" in res or "advisory" in res
    if res["status"] in ["RECOMMENDED", "MARGINAL"]:
        assert len(res["geojson"]["features"]) > 0
        feat = res["geojson"]["features"][0]
        assert feat["geometry"]["type"] == "Polygon"
        assert "suitability_score" in feat["properties"]


if __name__ == "__main__":
    tests = [
        test_1_model_loads,
        test_2_feature_schema_matches,
        test_3_inference_works,
        test_4_missing_sst_fails_safely,
        test_5_missing_chlorophyll_fails_safely,
        test_6_missing_required_feature_fails_safely,
        test_7_no_fake_default_environmental_values,
        test_8_output_probability_bounded,
        test_9_output_contains_provenance,
        test_10_model_version_and_status_reported,
        test_11_spatial_filtering_works,
        test_12_sri_lanka_land_areas_rejected,
        test_13_eez_filtering_works,
        test_14_api_integration_works,
    ]
    passed = 0
    print("==================================================")
    print("RUNNING 14 PFZ MODEL & PRODUCTION INTEGRATION TESTS")
    print("==================================================")
    for i, t in enumerate(tests, 1):
        try:
            t()
            print(f"[{i:2d}/14] {t.__name__}: PASSED")
            passed += 1
        except Exception as e:
            print(f"[{i:2d}/14] {t.__name__}: FAILED -> {e}")
    print("==================================================")
    print(f"RESULTS: {passed}/14 tests passed.")
    if passed == 14:
        print("ALL 14 PFZ INTEGRATION TESTS PASSED SUCCESSFULLY.")
    else:
        raise SystemExit(1)
