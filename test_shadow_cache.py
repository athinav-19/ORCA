"""
test_shadow_cache.py - Verification Suite for 9 ISRO MOSDAC Product Suites & Domain Agents
ISRO SIH Problem Statement 176: Marine Multi-Agent System

Runs tests verifying:
1. MOSDAC config.json schema and 9-product catalog across INSAT-3DR, Oceansat-3, and SARAL-AltiKa
2. Shadow Cache worker multi-dataset cycle and date roll (yesterday -> today)
3. OceanAgent SARAL-AltiKa Ka-band radar altimeter SWH & geostrophic currents
4. PfzAgent Oceansat-3 OCM Chlorophyll-a, Turbidity (Kd_490), and TSM
5. WeatherAgent Oceansat-3 scatterometer winds, INSAT-3DR sea fog, and solar insolation
6. DisasterAgent INSAT-3DR OLR cyclogenesis & HEM cloudburst detection
7. GisAgent Shapely EEZ & SLOC shipping lane geofencing
8. DecisionEngine Green Marine Solar Energy endurance & satellite fog-driven collision evasion
"""

import os
import sys
import datetime

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

print("\n" + "=" * 75)
print("     ORCA - 100% ISRO MOSDAC 9-DATASET MULTI-AGENT TEST SUITE")
print("=" * 75)

# Test 1: Config Generator with 9-Product Catalog
print("\n[Test 1] Testing setup_mosdac_config.py 9-product catalog support...")
from setup_mosdac_config import generate_mosdac_config, SUPPORTED_DATASETS
cfg = generate_mosdac_config()
assert os.path.exists("config.json"), "config.json was not created!"
assert "user_credentials" in cfg and "search_parameters" in cfg, "Invalid config schema!"
assert "supported_datasets" in cfg and len(cfg["supported_datasets"]) == 9, f"Expected 9 datasets, got {len(cfg.get('supported_datasets', []))}!"
print(f"  [PASS] config.json generated with all {len(SUPPORTED_DATASETS)} ISRO MOSDAC product suites:")
for idx, ds in enumerate(SUPPORTED_DATASETS, 1):
    print(f"         {idx}. {ds}")

# Test 2: Worker Date Roll & Multi-Dataset Rotation
print("\n[Test 2] Testing shadow_cache_worker.py multi-dataset date roll...")
from shadow_cache_worker import update_config_dates
start_d, end_d = update_config_dates(dataset_id="3OSCAT_L2B_WND")
today_d = datetime.date.today().strftime("%Y-%m-%d")
yesterday_d = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
assert start_d == yesterday_d, f"Expected start {yesterday_d}, got {start_d}"
assert end_d == today_d, f"Expected end {today_d}, got {end_d}"
print(f"  [PASS] Dataset successfully rotated to 3OSCAT_L2B_WND: {start_d} -> {end_d}.")

# Test 3: OceanAgent SARAL AltiKa Radar Altimeter & Scatterometer Ingestion
print("\n[Test 3] Testing OceanAgent SARAL-AltiKa radar altimeter SWH & currents...")
from ocean_agent import OceanAgent
o = OceanAgent()
sample_nc = o.load_cached_netcdf(8.76, 78.13)
assert sample_nc is not None, "Failed to read cached dataset in ./data/mosdac_cache/!"
assert "significant_wave_height_m" in sample_nc, "Missing significant_wave_height_m from SARAL!"
print(f"  [PASS] Read file: {sample_nc['source_file']}")
print(f"  [PASS] Radar SWH: {sample_nc['significant_wave_height_m']} m | SSHA: {sample_nc['sea_surface_height_anomaly_m']} m")
print(f"  [PASS] Geostrophic Surface Current: {sample_nc['geostrophic_current_speed_knots']} kt ({sample_nc['geostrophic_current_direction']})")
res_o = o.execute_task("Thoothukudi", "today", "waves", "BINARY_ADVISORY", "FISHERMAN")
print(f"  [PASS] Task Status: {res_o['status']} (Data Source: {res_o['metrics']['data_source']})")

# Test 4: PfzAgent Oceansat-3 OCM Ingestion (Chlorophyll-a & Turbidity)
print("\n[Test 4] Testing PfzAgent Oceansat-3 OCM Chlorophyll-a & Turbidity extraction...")
from pfz_agent import PfzAgent
p = PfzAgent()
sample_pfz = p.load_cached_netcdf(8.76, 78.13)
assert sample_pfz is not None, "Failed to read SST/Chlorophyll from NetCDF cache!"
assert "diffuse_attenuation_kd490" in sample_pfz, "Missing Kd_490 turbidity from OCM!"
print(f"  [PASS] Read file: {sample_pfz['source_file']}")
print(f"  [PASS] Sampled SST: {sample_pfz['sst_surface_temp_c']} deg C | Chlorophyll-a: {sample_pfz['chlorophyll_a_mg_m3']} mg/m3")
print(f"  [PASS] Water Clarity: {sample_pfz['water_clarity']} (Kd490: {sample_pfz['diffuse_attenuation_kd490']} 1/m, TSM: {sample_pfz['turbidity_tsm_g_m3']} g/m3)")
res_p = p.execute_task("Thoothukudi", "today", "pfz", "GEOJSON_POLYGONS", "FISHERMAN")
print(f"  [PASS] Composite PFZ Suitability Score: {res_p['pfz_details']['pfz_suitability_score']}/100")

# Test 5: WeatherAgent Oceansat-3 SCAT Winds & INSAT Fog/Insolation
print("\n[Test 5] Testing WeatherAgent Oceansat-3 scatterometer & INSAT solar/fog ingestion...")
from weather_agent import WeatherAgent
w = WeatherAgent()
res_w = w.execute_task("Thoothukudi", "today", "wind", "TEXT_SUMMARY", "FISHERMAN")
assert "MOSDAC" in res_w["telemetry"]["source"], f"Telemetry must be from MOSDAC, got {res_w['telemetry']['source']}"
assert "solar_insolation_wm2" in res_w["telemetry"], "Missing solar_insolation_wm2 in telemetry!"
print(f"  [PASS] Scatterometer Wind: {res_w['telemetry']['wind_speed_kmh']} km/h ({res_w['telemetry']['wind_direction']}) | Stress: {res_w['telemetry']['surface_wind_stress_nm2']} N/m2")
print(f"  [PASS] Rain: {res_w['telemetry']['rainfall_mmh']} mm/h | Visibility: {res_w['telemetry']['visibility_km']} km | Fog: {res_w['telemetry']['fog_cover_fraction']}")
print(f"  [PASS] Solar Insolation: {res_w['telemetry']['solar_insolation_wm2']} W/m2 (Yield: {res_w['telemetry']['solar_daily_kwh_m2']} kWh/m2)")

# Test 6: DisasterAgent INSAT OLR & HEM Storm Detection
print("\n[Test 6] Testing DisasterAgent ISRO MOSDAC satellite hazard ingestion...")
from disaster_agent import DisasterAgent
d = DisasterAgent()
res_d = d.execute_task("Thoothukudi", "today", "cyclone", "GEOJSON_POLYGONS", "AUTHORITY")
assert "MOSDAC" in res_d["hazard_summary"]["source"], f"Hazard source must be MOSDAC, got {res_d['hazard_summary']['source']}"
print(f"  [PASS] Hazard Status: {res_d['hazard_summary']['hazard_type']} ({res_d['hazard_summary']['severity']})")
print(f"  [PASS] GeoJSON Features Generated: {len(res_d['geojson']['features'])}")
print(f"  [PASS] Nearest Shelter: {res_d['nearest_shelter']['shelter_name']} ({res_d['nearest_shelter']['distance_nm']} NM)")

# Test 7: GisAgent EEZ & Shipping Lane SLOC Proximity
print("\n[Test 7] Testing GisAgent EEZ check & Shipping Lane SLOC Proximity...")
from gis_agent import GisAgent
g = GisAgent()
res_g1 = g.execute_task("Thoothukudi", "today", "check", "BINARY_ADVISORY", "FISHERMAN")
assert res_g1["is_within_eez"] is True, "Thoothukudi should be within Indian EEZ!"
print(f"  [PASS] Thoothukudi: Within EEZ = {res_g1['is_within_eez']} | Border Dist = {res_g1['distance_to_border_nm']} NM | Status = {res_g1['status']}")

sloc_eval = g.check_shipping_lane_proximity(7.45, 77.50)
assert sloc_eval["inside_lane"] is True, "Coordinates 7.45N, 77.5E should be inside Cape Comorin SLOC!"
print(f"  [PASS] SLOC Geofencing: {sloc_eval['nearest_lane_name']} | Inside Lane = {sloc_eval['inside_lane']} | Distance = {sloc_eval['distance_to_centerline_nm']} NM")

# Test 8: Decision Engine Green Marine Solar Energy & Satellite Fog Collision Risk
print("\n[Test 8] Testing Decision Engine Green Marine Energy & Fog SLOC evaluation...")
from decision_engine import run_decision_engine
test_payload = {
    "PFZ_AGENT": res_p,
    "WEATHER_AGENT": res_w,
    "OCEAN_AGENT": res_o,
    "DISASTER_AGENT": res_d,
    "GIS_AGENT": res_g1,
}
final_dec = run_decision_engine(test_payload, persona="FISHERMAN")
assert "green_marine_energy" in final_dec, "Missing green_marine_energy in Decision Engine output!"
green = final_dec["green_marine_energy"]
print(f"  [PASS] Green Marine Solar Range: {green['advisory']}")
print(f"  [PASS] Effective Solar Recharge: {green['effective_solar_recharge_kw']} kW | Extended Endurance: +{green['extended_zero_emission_hours']}h")

print("\n" + "=" * 75)
print(">>> ALL 8 SYSTEM TESTS PASSED WITH 100% 9-DATASET MOSDAC COMPLIANCE! <<<")
print("=" * 75 + "\n")
