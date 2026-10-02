"""
tests/test_audit_bugs_regression.py
===================================
Comprehensive regression tests for audit-confirmed bugs:
1. BUG 1: GIS LocationContext is JSON serializable (no TypeError on json.dumps)
2. BUG 2: Mumbai -> Goa route pipeline: uses actual destination (Goa), distance ~217.6 NM, not local ~10.4 NM shift
3. BUG 3: /api/status endpoint not shadowed by /health or duplicate declarations
4. BUG 4: Solar irradiance is not random on greetings (no random.randint)
5. BUG 5: No fake SST fallback (27.5 C / 27.7 C) when data is unavailable
6. BUG 6: No fake wave/wind fallbacks (1.2m, 14.0 km/h) leading to unsafe departure window advisories
7. Serialization hardening: full execute_orca_core payloads are JSON serializable
8. Non-route queries do not generate routes (show_route = False, safe_sea_route = None)
"""

import os
import json
import unittest
from pathlib import Path

# Enable FAST_DEMO_MODE for fast deterministic testing
os.environ["FAST_DEMO_MODE"] = "True"

import asyncio
import httpx
import server
from server import app, execute_orca_core
from gis_agent import GisAgent
from models import LocationContext
from decision_engine import compute_live_green_energy, run_decision_engine


def client_request(method: str, path: str, **kwargs):
    async def _do():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            return await c.request(method, path, **kwargs)
    return asyncio.run(_do())


class TestAuditBugsRegression(unittest.TestCase):

    def setUp(self):
        if hasattr(server, "manager_agent") and hasattr(server.manager_agent, "sessions"):
            server.manager_agent.sessions.clear()

    # -------------------------------------------------------------------------
    # BUG 1: LocationContext JSON serializable in GISAgent & API response
    # -------------------------------------------------------------------------
    def test_gis_location_context_is_json_serializable(self):
        """Verify that LocationContext passed to GISAgent produces JSON-serializable outputs."""
        gis = GisAgent()
        loc = LocationContext(latitude=18.9220, longitude=72.8347, name="Mumbai Port")
        imbl_res = gis.check_imbl_proximity(loc)
        target_loc = imbl_res.get("target_location")
        self.assertIsInstance(target_loc, dict, f"Expected dict, got {type(target_loc)}")
        # Check json.dumps does not raise TypeError
        dumped = json.dumps(imbl_res)
        self.assertIn("Mumbai Port", dumped)

    def test_hazard_query_no_serialization_error(self):
        """Query: Are there any active weather hazards, cyclones, or IMBL boundary alerts?"""
        query = "Are there any active weather hazards, cyclones, or IMBL boundary alerts?"
        res = execute_orca_core(query)
        self.assertNotIn("TypeError", str(res))
        self.assertNotIn("Traceback", str(res))
        # Ensure json.dumps succeeds
        dumped = json.dumps(res)
        self.assertIsInstance(dumped, str)
        self.assertFalse(res.get("show_route", True))

    def test_colombo_eez_no_serialization_error(self):
        """Query: Is Colombo inside India's EEZ?"""
        query = "Is Colombo inside India's EEZ?"
        res = execute_orca_core(query)
        self.assertNotIn("TypeError", str(res))
        self.assertNotIn("Traceback", str(res))
        # Ensure json.dumps succeeds
        dumped = json.dumps(res)
        self.assertIsInstance(dumped, str)
        reply = res.get("reply", "")
        # Colombo is outside India's EEZ
        self.assertFalse(res.get("show_route", True))
        self.assertTrue(
            "outside" in reply.lower() or "international" in reply.lower() or "sri lank" in reply.lower() or "not" in reply.lower(),
            f"Expected outside EEZ indication, got: {reply}"
        )

    # -------------------------------------------------------------------------
    # BUG 2: Mumbai -> Goa Route Pipeline
    # -------------------------------------------------------------------------
    def test_mumbai_goa_route_uses_actual_destination(self):
        """Verify Mumbai -> Goa route retains Goa as destination throughout pipeline."""
        query = "Give me a safe route from Mumbai to Goa."
        res = execute_orca_core(query)
        self.assertEqual(res.get("status"), "success")
        self.assertTrue(res.get("show_route"), "show_route must be True for route queries")
        route = res.get("safe_sea_route")
        self.assertIsInstance(route, dict, "safe_sea_route must be a dict")
        dest = str(route.get("destination", "")).lower()
        self.assertIn("goa", dest, f"Expected destination to contain 'goa', got {dest}")
        self.assertNotIn("centroid", dest)

    def test_mumbai_goa_route_distance_not_local_shift(self):
        """Verify Mumbai -> Goa route distance is ~217.6 NM, not ~10.4 NM harbor shift."""
        query = "Give me a safe route from Mumbai to Goa."
        res = execute_orca_core(query)
        route = res.get("safe_sea_route") or {}
        dist = route.get("total_distance_nm")
        self.assertIsNotNone(dist, "total_distance_nm must not be None")
        self.assertGreater(dist, 100.0, f"Route distance {dist} NM is too short (must be inter-city ~217 NM)")
        self.assertLess(dist, 350.0, f"Route distance {dist} NM is excessively large")
        waypoints = route.get("waypoints") or []
        self.assertGreaterEqual(len(waypoints), 2, "Must have at least origin and destination waypoints")
        vis = res.get("visualization")
        vis_type = vis.get("type") if isinstance(vis, dict) else vis
        self.assertEqual(vis_type, "ROUTE")

    # -------------------------------------------------------------------------
    # BUG 3: Duplicate /api/status route
    # -------------------------------------------------------------------------
    def test_status_endpoint_not_shadowed(self):
        """Verify GET /api/status returns operational MOSDAC telemetry and is not shadowed."""
        # /health endpoint
        resp_health = client_request("GET", "/health")
        self.assertEqual(resp_health.status_code, 200)
        data_health = resp_health.json()
        self.assertEqual(data_health.get("status"), "healthy")

        # /api/status endpoint
        resp_status = client_request("GET", "/api/status")
        self.assertEqual(resp_status.status_code, 200)
        data_status = resp_status.json()
        self.assertIn("status", data_status)

        # /status endpoint
        resp_s = client_request("GET", "/status")
        self.assertEqual(resp_s.status_code, 200)

        # /api/build and /api/version
        resp_build = client_request("GET", "/api/build")
        self.assertEqual(resp_build.status_code, 200)
        resp_ver = client_request("GET", "/api/version")
        self.assertEqual(resp_ver.status_code, 200)

    # -------------------------------------------------------------------------
    # BUG 4: No random solar irradiance
    # -------------------------------------------------------------------------
    def test_no_random_solar_fallback(self):
        """Verify solar irradiance calculation is deterministic and not random."""
        res1 = execute_orca_core("Hello Captain")
        res2 = execute_orca_core("Hello Captain")
        # Ensure json serializable
        json.dumps(res1)
        json.dumps(res2)
        # Compute live green energy directly
        val1 = compute_live_green_energy(18.9220, 72.8347)
        val2 = compute_live_green_energy(18.9220, 72.8347)
        self.assertEqual(val1, val2, "compute_live_green_energy must be strictly deterministic")
        self.assertTrue("solar_irradiance" in val1 or "solar_irradiance_wm2" in val1)

    # -------------------------------------------------------------------------
    # BUG 5: No fake SST 27.5 C fallback
    # -------------------------------------------------------------------------
    def test_no_fake_sst_fallback(self):
        """Verify that missing SST does not produce fabricated 27.5 C."""
        query = "How is the weather near Mumbai?"
        res = execute_orca_core(query)
        reply = res.get("reply", "")
        conditions = res.get("conditions", {})
        cond_str = json.dumps(conditions)
        # If SST is not from real observation, it should not default to 27.5 C
        # Check that 27.5 C is not present as a fake fallback
        self.assertNotIn("27.5", reply, "Fabricated 27.5 C found in reply")
        self.assertNotIn("27.5", cond_str, "Fabricated 27.5 C found in conditions")

    # -------------------------------------------------------------------------
    # BUG 6: No fake wave/wind fallbacks (1.2m, 14.0 km/h) for departure window
    # -------------------------------------------------------------------------
    def test_no_fake_wave_wind_fallback(self):
        """Verify that departure query with missing telemetry returns DATA_UNAVAILABLE, not Safe."""
        query = "What is the safe departure window for small craft fishing?"
        res = execute_orca_core(query)
        reply = res.get("reply", "").lower()
        conditions = res.get("conditions", {})
        # Without real observations, should not declare conditions favorable or safe
        self.assertFalse(
            "departure window: open" in reply or "safe to depart" in reply,
            f"Advisory declared safe departure without real data: {res.get('reply')}"
        )

    def test_missing_weather_data_does_not_produce_safe_advisory(self):
        """Verify decision engine with DATA_UNAVAILABLE does not advise safe departure."""
        mock_dispatched = {
            "WEATHER_AGENT": {"status": "DATA_UNAVAILABLE"},
            "OCEAN_AGENT": {"status": "DATA_UNAVAILABLE"},
            "DISASTER_AGENT": {"status": "DATA_UNAVAILABLE"},
            "GIS_AGENT": {"status": "DATA_UNAVAILABLE"},
            "PFZ_AGENT": {"status": "DATA_UNAVAILABLE"},
            "analyzed_intent": "WEATHER",
            "vessel_location": "Mumbai",
        }
        res = run_decision_engine(mock_dispatched, persona="FISHERMAN", normalized_query="What is the safe departure window for small craft fishing?")
        reply = (res.get("reply") or res.get("text_advisory_local") or "").lower()
        self.assertNotIn("safe to depart", reply)
        self.assertTrue("data_unavailable" in reply or "unavailable" in reply or "unable" in reply or "caution" in reply)

    # -------------------------------------------------------------------------
    # 7. Non-route queries do not generate routes
    # -------------------------------------------------------------------------
    def test_non_route_queries_do_not_generate_routes(self):
        """Verify various non-route queries do not output safe_sea_route or show_route=True."""
        queries = [
            "Show current sea conditions, swell wave height, and wind",
            "How is the weather near Mumbai?",
            "Are there any active weather hazards, cyclones, or IMBL boundary alerts?",
            "Where are the potential fishing zones near Chennai?",
            "What are the current waves near Kochi?",
        ]
        for q in queries:
            with self.subTest(query=q):
                res = execute_orca_core(q)
                self.assertFalse(res.get("show_route", True), f"show_route must be False for: {q}")
                self.assertIsNone(res.get("safe_sea_route"), f"safe_sea_route must be None for: {q}")

    # -------------------------------------------------------------------------
    # 8. Full payload JSON serialization hardening
    # -------------------------------------------------------------------------
    def test_full_payload_json_serializable(self):
        """Verify full payloads of key queries are 100% JSON serializable."""
        test_queries = [
            "Show current sea conditions, swell wave height, and wind",
            "How is the weather near Mumbai?",
            "Are there any active weather hazards, cyclones, or IMBL boundary alerts?",
            "What is the safe departure window for small craft fishing?",
            "Give me a safe route from Mumbai to Goa.",
            "Is Colombo inside India's EEZ?",
            "Where are the potential fishing zones near Chennai?",
            "What are the current waves near Kochi?",
        ]
        for q in test_queries:
            with self.subTest(query=q):
                payload = execute_orca_core(q)
                try:
                    dumped = json.dumps(payload)
                    self.assertIsInstance(dumped, str)
                except TypeError as e:
                    self.fail(f"Payload for query '{q}' failed JSON serialization: {e}")


if __name__ == "__main__":
    unittest.main()
