"""
tests/test_telemetry_null_safety.py
====================================
Regression and integrity test suite verifying:
1. Complete null-safety when telemetry is None (zero 'NoneType' object has no attribute 'get')
2. Handling of missing telemetry (LOCATION_REQUIRED when location omitted, or explicit query location resolution)
3. Normal processing with valid device telemetry
4. Query: "What is the safe departure window for small craft fishing?"
   - Intent: WEATHER (operational sailing safety, NOT PFZ, NOT ROUTE)
   - No route visualization (visualization is None)
   - No route verification failure text
   - Safe departure window analysis (wind, waves, visibility, hazards)
5. Explicit route queries continue to generate routes & visualization
6. Weather and cyclone queries return dedicated responses without route artifacts
7. Direct unit tests on DecisionEngine metric extraction with None telemetry
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
from decision_engine import RiskAnalysisAgent


class TestTelemetryNullSafety(unittest.TestCase):

    def setUp(self):
        self.risk_agent = RiskAnalysisAgent()
        self.manager = ManagerAgent()

    # -------------------------------------------------------------------------
    # Unit Test: DecisionEngine metric extraction with None/missing telemetry
    # -------------------------------------------------------------------------
    def test_01_decision_engine_extract_metrics_none_telemetry(self):
        """Verifies that DecisionEngine._extract_metrics never raises AttributeError on None telemetry."""
        # 1. Weather output has explicit telemetry: None (as when MOSDAC cache is missing)
        agg_data_1 = {
            "WEATHER_AGENT": {
                "status": "DATA_UNAVAILABLE",
                "telemetry": None,
                "location": "Sector (13.08, 80.29)",
            }
        }
        try:
            metrics_1 = self.risk_agent._extract_metrics(agg_data_1)
            self.assertIsInstance(metrics_1, dict)
            self.assertIn("wind_speed_kmph", metrics_1)
            self.assertIn("visibility_km", metrics_1)
        except AttributeError as e:
            self.fail(f"_extract_metrics crashed on telemetry=None: {e}")

        # 2. Domain agents are all explicitly None
        agg_data_2 = {
            "WEATHER_AGENT": None,
            "OCEAN_AGENT": None,
            "DISASTER_AGENT": None,
            "GIS_AGENT": None,
            "PFZ_AGENT": None,
            "device_telemetry": None,
        }
        try:
            metrics_2 = self.risk_agent._extract_metrics(agg_data_2)
            self.assertIsInstance(metrics_2, dict)
        except Exception as e:
            self.fail(f"_extract_metrics crashed on all-None agents: {e}")

        # 3. Disaster output has hazard_summary: None
        agg_data_3 = {
            "DISASTER_AGENT": {
                "status": "DATA_UNAVAILABLE",
                "hazard_summary": None,
                "hazard_assessment": None,
                "cyclone_track": None,
                "subsea_earthquake": None,
            }
        }
        try:
            metrics_3 = self.risk_agent._extract_metrics(agg_data_3)
            self.assertIsInstance(metrics_3, dict)
        except Exception as e:
            self.fail(f"_extract_metrics crashed on disaster_out with None fields: {e}")

        print("[PASS] Test 1: DecisionEngine metric extraction is 100% null-safe.")

    # -------------------------------------------------------------------------
    # Scenario 2: Query with telemetry=None and NO query location
    # Expected: Structured LOCATION_REQUIRED, zero exceptions
    # -------------------------------------------------------------------------
    def test_02_query_telemetry_none_no_location(self):
        query = "What is the safe departure window for small craft fishing?"
        resp = execute_orca_core(query=query, lat=None, lon=None)
        
        self.assertIsInstance(resp, dict)
        self.assertEqual(resp.get("status"), "LOCATION_REQUIRED")
        self.assertFalse(resp.get("success", True))
        self.assertIn("specify a coastal location", resp.get("message", "").lower())
        self.assertNotIn("nonetype", str(resp).lower())
        self.assertNotIn("internal error", str(resp).lower())
        print("[PASS] Test 2: Query with telemetry=None and no location returns structured LOCATION_REQUIRED.")

    # -------------------------------------------------------------------------
    # Scenario 3: Query with telemetry=None but EXPLICIT query location
    # Expected: Resolves explicit location (Chennai), processes cleanly
    # -------------------------------------------------------------------------
    def test_03_query_telemetry_none_explicit_location(self):
        query = "What is the safe departure window for small craft fishing near Chennai?"
        resp = execute_orca_core(query=query, lat=None, lon=None)

        self.assertIsInstance(resp, dict)
        self.assertEqual(resp.get("status"), "success")
        reply = resp.get("reply", "")
        self.assertIn("Chennai", reply)
        self.assertIn("Departure Window", reply)
        self.assertNotIn("'NoneType' object has no attribute 'get'", reply)
        self.assertNotIn("Caution: automated route verification unavailable", reply)
        self.assertIsNone(resp.get("visualization"))
        self.assertIsNone(resp.get("safe_sea_route"))
        print("[PASS] Test 3: Departure query with explicit location and telemetry=None processed cleanly.")

    # -------------------------------------------------------------------------
    # Scenario 4: Query with VALID device telemetry
    # Expected: Uses GPS coordinates, evaluates departure window
    # -------------------------------------------------------------------------
    def test_04_query_valid_telemetry_departure_window(self):
        query = "What is the safe departure window for small craft fishing?"
        # Chennai harbor GPS
        resp = execute_orca_core(query=query, lat=13.0827, lon=80.2707)

        self.assertIsInstance(resp, dict)
        self.assertEqual(resp.get("status"), "success")
        reply = resp.get("reply", "")
        
        # Check no NoneType crash
        self.assertNotIn("'NoneType' object has no attribute 'get'", reply)
        self.assertNotIn("internal error", reply.lower())
        self.assertNotIn("Caution: automated route verification unavailable", reply)
        
        # Check departure window content
        self.assertIn("Departure Window", reply)
        
        # Check visualization is None (not a route query)
        self.assertIsNone(resp.get("visualization"))
        self.assertIsNone(resp.get("safe_sea_route"))
        print("[PASS] Test 4: Departure window query with valid GPS processed with zero route artifacts.")

    # -------------------------------------------------------------------------
    # Scenario 5: Intent classification for departure window queries
    # -------------------------------------------------------------------------
    def test_05_departure_window_intent_classification(self):
        test_queries = [
            "What is the safe departure window for small craft fishing?",
            "When can I sail from Chennai harbor?",
            "What is the safe departure time for small boats?",
            "Is there a safe departure window today?",
            "Departure window for small craft fishing near Mumbai",
        ]
        for q in test_queries:
            intent_res = classify_marine_query_intent(q)
            self.assertEqual(intent_res["intent"], "WEATHER", f"Query '{q}' should be classified as WEATHER")
            plan_agents = [step["agent_name"] for step in intent_res["plan"]]
            self.assertIn("WEATHER_AGENT", plan_agents)
            self.assertNotIn("GIS_AGENT", plan_agents)
        print("[PASS] Test 5: All departure window queries classified as WEATHER without GIS routing.")

    # -------------------------------------------------------------------------
    # Scenario 6: Explicit Route queries continue to work normally
    # -------------------------------------------------------------------------
    def test_06_route_queries_retain_visualization(self):
        query = "Give me a safe route from Chennai to Nagapattinam."
        resp = execute_orca_core(query=query)
        self.assertEqual(resp.get("intent"), "ROUTE")
        vis = resp.get("visualization")
        self.assertIsNotNone(vis)
        self.assertEqual(vis.get("type"), "ROUTE")
        print("[PASS] Test 6: Route queries properly generate route visualization.")

    # -------------------------------------------------------------------------
    # Scenario 7: Weather & Cyclone queries remain clean without route clutter
    # -------------------------------------------------------------------------
    def test_07_weather_and_cyclone_queries_clean(self):
        # Weather query
        w_resp = execute_orca_core(query="How is the weather near Mumbai?")
        self.assertEqual(w_resp.get("intent"), "WEATHER")
        self.assertIsNone(w_resp.get("visualization"))
        self.assertNotIn("Caution: automated route verification unavailable", w_resp.get("reply", ""))

        # Cyclone query
        c_resp = execute_orca_core(query="Is there any cyclone near West Bengal?")
        self.assertEqual(c_resp.get("intent"), "CYCLONE")
        self.assertIsNone(c_resp.get("visualization"))
        self.assertNotIn("Caution: automated route verification unavailable", c_resp.get("reply", ""))
        print("[PASS] Test 7: Weather and cyclone queries returned without route clutter.")


if __name__ == "__main__":
    unittest.main()
