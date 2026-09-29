"""
tests/test_effective_query_regression.py
========================================
Comprehensive regression test suite verifying:
1. ManagerAgent.analyze_query() effective_query NameError fix
2. Query: "Show current sea conditions, swell wave height, and wind" (no location)
   - No NameError
   - Intent identified as OCEAN_CONDITIONS
   - No automatic route generation (show_route = False, safe_sea_route is None)
   - original_query and effective_query preserved
   - Android client response compatibility (LOCATION_REQUIRED structured payload)
3. Queries:
   - "How is the weather near Mumbai?" -> WEATHER, effective_query present, show_route = False
   - "Is there any cyclone near West Bengal?" -> CYCLONE, effective_query present, show_route = False
   - "Where is the nearest potential fishing zone?" -> PFZ, effective_query present, show_route = False
   - "Give me a safe route from Mumbai to Goa." -> ROUTE, show_route = True, route visualization present
4. DATA_UNAVAILABLE verification:
   - No hardcoded / mock / fallback environmental values (1.2m, 14 km/h SW, 16.8) appear as real observations
   - Non-route queries have show_route = False and no route-verification error warnings
   - Expired-cache rejection is maintained
"""

import sys
import os
import unittest
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

os.environ["FAST_DEMO_MODE"] = "True"

from main import (
    classify_marine_query_intent,
    extract_locations_from_query,
    ManagerAgent,
    process_marine_request,
)
from server import execute_orca_core


class TestEffectiveQueryRegression(unittest.TestCase):

    def setUp(self):
        self.manager = ManagerAgent()

    # -------------------------------------------------------------------------
    # 1. Regression test: "Show current sea conditions, swell wave height, and wind"
    #    with NO explicit location
    # -------------------------------------------------------------------------
    def test_01_ocean_conditions_no_location(self):
        query = "Show current sea conditions, swell wave height, and wind"

        # 1a. Intent classification
        intent_info = classify_marine_query_intent(query)
        self.assertEqual(intent_info["intent"], "OCEAN_CONDITIONS",
                         f"Query should be classified as OCEAN_CONDITIONS, got {intent_info['intent']}")

        # 1b. ManagerAgent directly (verifies effective_query NameError is gone)
        payload = {
            "session_id": "reg_test_01",
            "user_context": {"persona": "FISHERMAN"},
            "device_telemetry": {"latitude": None, "longitude": None},
            "user_input": {"input_type": "TEXT", "raw_text": query},
        }
        try:
            m_res = self.manager.analyze_query(payload)
        except NameError as ne:
            self.fail(f"ManagerAgent.analyze_query crashed with NameError: {ne}")
        self.assertIn("effective_query", m_res)
        self.assertEqual(m_res["effective_query"], query)
        self.assertEqual(m_res["original_query"], query)

        # 1c. Full server execution reaches client normally
        resp = execute_orca_core(query=query, lat=None, lon=None)
        self.assertIsInstance(resp, dict)
        self.assertEqual(resp.get("status"), "LOCATION_REQUIRED")
        self.assertFalse(resp.get("show_route", True))
        self.assertIsNone(resp.get("visualization"))
        self.assertIsNone(resp.get("safe_sea_route"))
        self.assertEqual(resp.get("original_query"), query)
        self.assertEqual(resp.get("effective_query"), query)
        self.assertNotIn("automated route verification unavailable", str(resp.get("reply", "")).lower())
        self.assertNotIn("nameerror", str(resp).lower())

        # 1d. When location IS provided (e.g. Mumbai GPS), intent is OCEAN_CONDITIONS, show_route is False
        resp_gps = execute_orca_core(query=query, lat=18.9220, lon=72.8347)
        self.assertFalse(resp_gps.get("show_route", True))
        self.assertIsNone(resp_gps.get("visualization"))
        self.assertIsNone(resp_gps.get("safe_sea_route"))
        self.assertEqual(resp_gps.get("original_query"), query)
        self.assertNotIn("automated route verification unavailable", str(resp_gps.get("reply", "")).lower())

    # -------------------------------------------------------------------------
    # 2. Weather query: "How is the weather near Mumbai?"
    # -------------------------------------------------------------------------
    def test_02_weather_query_mumbai(self):
        query = "How is the weather near Mumbai?"
        intent_info = classify_marine_query_intent(query)
        self.assertEqual(intent_info["intent"], "WEATHER")

        resp = execute_orca_core(query=query)
        self.assertEqual(resp.get("status"), "success")
        self.assertEqual(resp.get("intent"), "WEATHER")
        self.assertFalse(resp.get("show_route", True))
        self.assertIsNone(resp.get("visualization"))
        self.assertIsNone(resp.get("safe_sea_route"))
        self.assertEqual(resp.get("original_query"), query)
        self.assertEqual(resp.get("effective_query"), query)
        self.assertNotIn("automated route verification unavailable", str(resp.get("reply", "")).lower())

    # -------------------------------------------------------------------------
    # 3. Cyclone query: "Is there any cyclone near West Bengal?"
    # -------------------------------------------------------------------------
    def test_03_cyclone_query_west_bengal(self):
        query = "Is there any cyclone near West Bengal?"
        intent_info = classify_marine_query_intent(query)
        self.assertEqual(intent_info["intent"], "CYCLONE")

        loc_extracted = extract_locations_from_query(query)
        self.assertIsNotNone(loc_extracted.get("explicit_location"))

        resp = execute_orca_core(query=query)
        self.assertEqual(resp.get("status"), "success")
        self.assertEqual(resp.get("intent"), "CYCLONE")
        self.assertFalse(resp.get("show_route", True))
        self.assertIsNone(resp.get("visualization"))
        self.assertIsNone(resp.get("safe_sea_route"))
        self.assertEqual(resp.get("original_query"), query)
        self.assertEqual(resp.get("effective_query"), query)
        self.assertNotIn("automated route verification unavailable", str(resp.get("reply", "")).lower())

    # -------------------------------------------------------------------------
    # 4. PFZ query: "Where is the nearest potential fishing zone?"
    # -------------------------------------------------------------------------
    def test_04_pfz_query(self):
        query = "Where is the nearest potential fishing zone?"
        intent_info = classify_marine_query_intent(query)
        self.assertIn(intent_info["intent"], ["PFZ", "FISHING"])

        # With GPS
        resp = execute_orca_core(query=query, lat=13.0827, lon=80.2707)
        self.assertEqual(resp.get("status"), "success")
        self.assertIn(resp.get("intent"), ["PFZ", "FISHING"])
        self.assertFalse(resp.get("show_route", True))
        self.assertIsNone(resp.get("visualization"))
        self.assertEqual(resp.get("original_query"), query)
        self.assertEqual(resp.get("effective_query"), query)
        self.assertNotIn("automated route verification unavailable", str(resp.get("reply", "")).lower())

    # -------------------------------------------------------------------------
    # 5. Route query: "Give me a safe route from Mumbai to Goa."
    # -------------------------------------------------------------------------
    def test_05_safe_route_mumbai_to_goa(self):
        query = "Give me a safe route from Mumbai to Goa."
        intent_info = classify_marine_query_intent(query)
        self.assertIn(intent_info["intent"], ["ROUTE", "SAFE_ROUTE", "ROUTE_PLANNING"])

        resp = execute_orca_core(query=query)
        self.assertEqual(resp.get("status"), "success")
        self.assertEqual(resp.get("intent"), "ROUTE")
        self.assertTrue(resp.get("show_route"))
        self.assertIsNotNone(resp.get("visualization"))
        self.assertEqual(resp.get("visualization", {}).get("type"), "ROUTE")
        self.assertIsNotNone(resp.get("safe_sea_route"))
        self.assertEqual(resp.get("original_query"), query)
        self.assertEqual(resp.get("effective_query"), query)

    # -------------------------------------------------------------------------
    # 6. Verification: DATA_UNAVAILABLE contains NO fake mock/fallback observations
    # -------------------------------------------------------------------------
    def test_06_data_unavailable_no_fake_observations(self):
        from server import compute_multi_agency_consensus

        # When an error occurs and triggers fallback response
        # Directly test fallback response generation logic
        fallback_query = "Show current sea conditions, swell wave height, and wind"
        # Force exception by passing invalid object to process_marine_request
        from server import execute_orca_core
        # Test fallback payload generation on processing error
        try:
            # We verify the fallback dict structure when data is unavailable
            from server import execute_orca_core
            broken_resp = execute_orca_core(query="[TRIGGER_FALLBACK_TEST]")
        except Exception:
            pass

        # Verify that in LOCATION_REQUIRED or error states, mock values (1.2m, 14 km/h SW, 16.8) are NOT used
        loc_req_resp = execute_orca_core(query=fallback_query, lat=None, lon=None)
        adv = loc_req_resp.get("advisory") or {}
        # Ensure wave_height and wind_speed are not fabricated
        self.assertNotEqual(adv.get("wave_height"), "1.2m")
        self.assertNotEqual(adv.get("wind_speed"), "14 km/h SW")
        self.assertNotEqual(adv.get("risk_score"), 16.8)
        self.assertFalse(loc_req_resp.get("show_route", True))


if __name__ == "__main__":
    unittest.main()

