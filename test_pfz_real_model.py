"""
test_pfz_real_model.py - Test Suite for ORCA PFZ Model & Production Integration
Covers all 14 mandatory test cases specified in the PFZ ML Master Task.
"""

from tests.test_pfz_real_model import *

if __name__ == "__main__":
    import tests.test_pfz_real_model as m
    tests = [
        m.test_1_model_loads,
        m.test_2_feature_schema_matches,
        m.test_3_inference_works,
        m.test_4_missing_sst_fails_safely,
        m.test_5_missing_chlorophyll_fails_safely,
        m.test_6_missing_required_feature_fails_safely,
        m.test_7_no_fake_default_environmental_values,
        m.test_8_output_probability_bounded,
        m.test_9_output_contains_provenance,
        m.test_10_model_version_and_status_reported,
        m.test_11_spatial_filtering_works,
        m.test_12_sri_lanka_land_areas_rejected,
        m.test_13_eez_filtering_works,
        m.test_14_api_integration_works,
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

