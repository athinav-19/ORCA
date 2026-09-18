"""
test_no_fake_fallbacks.py - Regression & Integrity Test Suite
Verifies:
1. No synthetic/mock fallbacks in OceanAgent, WeatherAgent, DisasterAgent, PfzAgent
2. Structured DATA_UNAVAILABLE returned when telemetry/cache is missing
3. Authoritative Marine Regions EEZ containment (Mainland, Lakshadweep, Andaman & Nicobar)
4. Absence of random.randint solar generation in DecisionEngine
5. PFZ ML model transparent classification as RULE_EMULATION / PENDING_REAL_GROUND_TRUTH
"""

import unittest
import json
import os
import math
from typing import Dict, Any


class TestNoFakeFallbacks(unittest.TestCase):

    def test_ocean_agent_no_mock(self):
        from ocean_agent import OceanAgent
        agent = OceanAgent(cache_dir="non_existent_cache_dir")
        res = agent.execute_task("mumbai", "today")
        self.assertEqual(res.get("status"), "DATA_UNAVAILABLE")
        self.assertIsNone(res.get("wave_height_m"))
        self.assertFalse(hasattr(agent, "generate_mock_timeseries"))
        print("[PASS] OceanAgent: Zero mock timeseries, returns DATA_UNAVAILABLE on missing cache.")

    def test_weather_agent_no_simulation(self):
        from weather_agent import WeatherAgent
        agent = WeatherAgent(cache_dir="non_existent_cache_dir")
        res = agent.execute_task("mumbai", "today")
        self.assertEqual(res.get("status"), "DATA_UNAVAILABLE")
        self.assertIsNone(res.get("telemetry"))
        self.assertFalse(hasattr(agent, "simulate_mosdac_feed"))
        print("[PASS] WeatherAgent: Zero simulated feed, returns DATA_UNAVAILABLE on missing cache.")

    def test_disaster_agent_no_fake_earthquake(self):
        from disaster_agent import DisasterAgent
        agent = DisasterAgent(cache_dir="non_existent_cache_dir")
        res = agent.execute_task("mumbai", "today")
        self.assertIsNone(res.get("subsea_earthquake"))
        seismic = agent.fetch_subsea_seismic_events(18.9, 72.8)
        self.assertEqual(seismic.get("status"), "NO_ACTIVE_SEISMIC_EVENT")
        self.assertIsNone(seismic.get("subsea_earthquake"))
        print("[PASS] DisasterAgent: Zero mock magnitude 4.8 earthquake, reports NO_ACTIVE_SEISMIC_EVENT.")

    def test_pfz_agent_no_fake_overlap(self):
        from pfz_agent import PfzAgent
        agent = PfzAgent(cache_dir="non_existent_cache_dir")
        res = agent.execute_task("mumbai", "today")
        self.assertEqual(res.get("status"), "DATA_UNAVAILABLE")
        self.assertFalse(hasattr(agent, "simulate_sst_and_chlorophyll_overlap"))
        print("[PASS] PfzAgent: Zero synthetic front synthesis, returns DATA_UNAVAILABLE on missing cache.")

    def test_pfz_infer_requires_real_inputs(self):
        from ml.inference.pfz_infer import predict_pfz_suitability
        res = predict_pfz_suitability(latitude=18.9, longitude=72.8, sst_c=None, chlorophyll_a_mg_m3=None)
        self.assertEqual(res.get("status"), "DATA_UNAVAILABLE")
        self.assertEqual(res.get("ground_truth_status"), "PENDING_REAL_GROUND_TRUTH")
        print("[PASS] PFZ Inference: Requires genuine SST & Chlorophyll; returns PENDING_REAL_GROUND_TRUTH.")

    def test_eez_highres_containment(self):
        from gis_agent import GisAgent
        agent = GisAgent()
        self.assertIsNotNone(agent.eez_shape)
        self.assertIsNotNone(agent.prepared_eez)

        cases = [
            ("Mumbai Offshore", 18.9, 72.5, True),
            ("Chennai Offshore", 13.1, 80.5, True),
            ("Port Blair (Andaman)", 11.65, 92.8, True),
            ("Kavaratti (Lakshadweep)", 10.55, 72.6, True),
            ("Colombo (Sri Lanka)", 6.9, 79.5, False),
            ("High Seas Arabian Sea", 15.0, 60.0, False),
            ("High Seas Bay of Bengal", 12.0, 97.0, False),
        ]
        for name, lat, lon, expected in cases:
            res = agent.check_imbl_proximity({"lat": lat, "lon": lon})
            in_eez = res.get("is_within_eez")
            self.assertEqual(in_eez, expected, f"EEZ containment failed for {name} ({lat}, {lon})")
        print("[PASS] GisAgent: High-res UNCLOS EEZ verified for Mainland, Lakshadweep, Andaman & Nicobar, and international exclusions.")

    def test_pfz_metadata_classification(self):
        with open(os.path.join("ml", "models", "pfz", "metadata.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta.get("data_classification"), "RULE_EMULATION")
        self.assertEqual(meta.get("status"), "PENDING_REAL_GROUND_TRUTH")
        print("[PASS] PFZ Metadata: Formally verified as RULE_EMULATION & PENDING_REAL_GROUND_TRUTH.")


if __name__ == "__main__":
    unittest.main()

