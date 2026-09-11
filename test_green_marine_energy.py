"""
test_green_marine_energy.py
Verification test suite for ORCA Green Marine Energy & Sustainability Telemetry.
Validates:
1. compute_live_green_energy() returns complete, diurnal-aware metrics.
2. FastAPI /health and /api/green-energy endpoints include non-null green_marine_energy.
3. Models and Kotlin field compatibility (zero_emission_hours, solar_insolation_wm2, fuel_saved_liters, carbon_offset_kg).
4. decision_engine.py populates both extended_zero_emission_hours and zero_emission_hours.
"""

import sys
import os
import json
from datetime import datetime, timezone, timedelta

print("=" * 75)
print("[Test 1] Testing server.compute_live_green_energy() directly...")
from server import compute_live_green_energy

green_data = compute_live_green_energy(lat=8.7642, lon=78.1348)
print("Returned green_marine_energy payload:")
print(json.dumps(green_data, indent=2))

required_fields = [
    "solar_insolation_wm2",
    "zero_emission_hours",
    "extended_zero_emission_hours",
    "fuel_saved_liters",
    "carbon_offset_kg",
    "daily_solar_yield_kwh_m2",
    "effective_solar_recharge_kw",
    "solar_assisted_range_nm",
    "is_daylight",
    "is_nighttime",
    "diurnal_cycle",
    "current_hour_ist",
    "advisory",
]

for field in required_fields:
    assert field in green_data, f"Missing required field '{field}' in green_data!"
    assert green_data[field] is not None, f"Field '{field}' must not be None!"

IST = timezone(timedelta(hours=5, minutes=30))
now_ist = datetime.now(IST)
hour_ist = now_ist.hour + now_ist.minute / 60.0
is_day = 6.0 <= hour_ist < 18.5

print(f"\nCurrent IST Time: {now_ist.strftime('%Y-%m-%d %H:%M:%S')} (Decimal: {hour_ist:.2f}h) | Expected Daytime: {is_day}")
assert green_data["is_daylight"] == is_day, f"is_daylight mismatch: expected {is_day}, got {green_data['is_daylight']}"

if is_day:
    assert green_data["solar_insolation_wm2"] > 0, "solar_insolation_wm2 should be > 0 during daytime!"
    assert green_data["zero_emission_hours"] > 0, "zero_emission_hours should be > 0 during daytime!"
    assert green_data["fuel_saved_liters"] > 0, "fuel_saved_liters should be > 0 during daytime!"
    assert green_data["carbon_offset_kg"] > 0, "carbon_offset_kg should be > 0 during daytime!"
    print(f"  [PASS] Daytime Solar Insolation: {green_data['solar_insolation_wm2']} W/m²")
    print(f"  [PASS] Daytime Zero Emission Hours: {green_data['zero_emission_hours']} hrs")
    print(f"  [PASS] Daytime Fuel Saved: {green_data['fuel_saved_liters']} L")
    print(f"  [PASS] Daytime Carbon Offset: {green_data['carbon_offset_kg']} kg CO2")
else:
    assert green_data["solar_insolation_wm2"] == 0.0, "solar_insolation_wm2 should be 0.0 at night!"
    print("  [PASS] Nighttime zero solar insolation verified.")

# Step 2: Test FastAPI /health, /api/health, and /api/green-energy endpoints
print("\n" + "=" * 75)
print("[Test 2] Testing FastAPI /health, /api/health, and /api/green-energy endpoints...")
import asyncio
import httpx
from server import app

async def test_endpoints():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 2a. GET /health
        res_health = await client.get("/health")
        assert res_health.status_code == 200, f"/health failed with status {res_health.status_code}"
        body_health = res_health.json()
        assert "green_marine_energy" in body_health, "Missing green_marine_energy in /health response!"
        assert body_health["green_marine_energy"] is not None, "green_marine_energy in /health must NOT be null!"
        print("  [PASS] GET /health returns HTTP 200 with non-null green_marine_energy:")
        print(f"         Solar: {body_health['green_marine_energy']['solar_insolation_wm2']} W/m² | Zero-Emission: {body_health['green_marine_energy']['zero_emission_hours']}h")

        # 2b. GET /health with lat/lon coordinates
        res_health_coords = await client.get("/health?lat=13.0827&lon=80.2707")
        assert res_health_coords.status_code == 200
        body_coords = res_health_coords.json()
        assert body_coords["green_marine_energy"]["solar_insolation_wm2"] is not None
        print("  [PASS] GET /health?lat=...&lon=... handles coordinate query parameters cleanly.")

        # 2c. GET /api/green-energy
        res_green = await client.get("/api/green-energy")
        assert res_green.status_code == 200
        body_green = res_green.json()
        assert body_green["status"] == "success"
        assert "green_marine_energy" in body_green
        print("  [PASS] GET /api/green-energy returns HTTP 200 with green_marine_energy.")

asyncio.run(test_endpoints())

# Step 3: Test decision_engine.py output
print("\n" + "=" * 75)
print("[Test 3] Testing decision_engine.py green_marine_energy output...")
from decision_engine import run_decision_engine

sample_agent_outputs = {
    "is_daylight": True,
    "hour": 11.5,
    "map_status": "SAFE",
    "status": "SAFE",
}
dec_out = run_decision_engine(sample_agent_outputs, persona="FISHERMAN")
assert "green_marine_energy" in dec_out, "Missing green_marine_energy in Decision Engine output!"
g_dec = dec_out["green_marine_energy"]

assert "zero_emission_hours" in g_dec, "Missing zero_emission_hours in Decision Engine output!"
assert "extended_zero_emission_hours" in g_dec, "Missing extended_zero_emission_hours in Decision Engine output!"
assert g_dec["zero_emission_hours"] == g_dec["extended_zero_emission_hours"], "zero_emission_hours must match extended_zero_emission_hours!"
assert g_dec["fuel_saved_liters"] > 0, f"Expected fuel_saved_liters > 0 in daylight, got {g_dec['fuel_saved_liters']}"
assert g_dec["carbon_offset_kg"] > 0, f"Expected carbon_offset_kg > 0 in daylight, got {g_dec['carbon_offset_kg']}"

print("  [PASS] Decision engine green_marine_energy matches Kotlin models:")
print(f"         Solar: {g_dec['solar_insolation_wm2']} W/m²")
print(f"         Zero Emission: {g_dec['zero_emission_hours']} hrs")
print(f"         Fuel Saved: {g_dec['fuel_saved_liters']} L")
print(f"         Carbon Offset: {g_dec['carbon_offset_kg']} kg CO2")

print("\n" + "=" * 75)
print(">>> ALL GREEN MARINE ENERGY TESTS PASSED WITH 100% SUCCESS! <<<")
print("=" * 75 + "\n")
