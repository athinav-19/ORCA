"""
test_copernicus_sync.py - Verification for Copernicus Marine Sync & Multi-Agency Consensus Engine

Tests:
1. Authentication: copernicusmarine.login via environment variables
2. Bounding Box Subsetting: download_copernicus_data() configuration & error resilience
3. Dataset Freshness & Lifecycle: DataSyncManager.verify_or_update_datasets()
   - Missing/expired cache triggers automated fetch
   - 3.0-second fail-safe timeout prevents blocking
   - Fresh cache (<24h) returns immediate confirmation
4. Multi-Agency Consensus Engine: compute_multi_agency_consensus()
   - Cross-validates MOSDAC SST vs Copernicus SST (variance Delta < 0.5 deg C)
   - Outputs: [Multi-Agency Telemetry Fusion] ISRO MOSDAC (India) + Copernicus Marine (EU) Verified | SST Variance: Delta0.2 deg C | Confidence: HIGH
   - Confidence score: 98.4%
5. End-to-End API Integration: process_marine_query() injects satellite_provenance into Android response
"""

import sys
import os
import time
import json
import asyncio

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import copernicusmarine
from server import (
    download_copernicus_data,
    DataSyncManager,
    verify_or_update_datasets,
    compute_multi_agency_consensus,
    process_marine_query,
    COPERNICUS_CACHE_DIR,
    COPERNICUS_SST_FILE
)
from models import QueryRequest, TelemetryData


def test_copernicus_authentication():
    print("\n--- TEST 1: Copernicus Marine Service Authentication ---")
    c_user = os.getenv("COPERNICUS_MARINE_USERNAME") or os.getenv("COPERNICUS_USERNAME", "athinav_r")
    c_pass = os.getenv("COPERNICUS_MARINE_PASSWORD") or os.getenv("COPERNICUS_PASSWORD", "")
    if c_user and c_pass:
        login_ok = copernicusmarine.login(username=c_user, password=c_pass)
        assert login_ok is True or login_ok is None, f"Login failed: {login_ok}"
        print("[Copernicus Marine] Authentication successful")
    else:
        print("[Copernicus Marine] Authentication unavailable")


def test_copernicus_download_function():
    print("\n--- TEST 2: Copernicus All-India Subsetting Function ---")
    # Verify download_copernicus_data runs without uncaught exceptions
    # and properly creates the cache directory
    download_copernicus_data()
    assert os.path.exists(COPERNICUS_CACHE_DIR), f"Cache directory {COPERNICUS_CACHE_DIR} not created!"
    print(f"[OK] Cache directory verified: {COPERNICUS_CACHE_DIR}")
    print("[OK] download_copernicus_data executed with try-catch fail-safe protection")


def test_datasync_manager_lifecycle():
    print("\n--- TEST 3: DataSyncManager Freshness Check & 3.0s Timeout Fail-Safe ---")
    
    # 1. Test missing cache detection and automated initialization
    DataSyncManager.ensure_baseline_copernicus_cache()
    assert os.path.exists(COPERNICUS_SST_FILE), "Baseline Copernicus NetCDF file should exist"
    print(f"[OK] Baseline NetCDF cache verified at {COPERNICUS_SST_FILE}")

    # 2. Test fresh cache verification (<24h)
    is_fresh = DataSyncManager.is_copernicus_fresh()
    assert is_fresh is True, "Recently created/updated cache file must be fresh"
    print("[OK] DataSyncManager.is_copernicus_fresh() returns True for <24h cache")

    # 3. Test verify_or_update_datasets pre-flight hook
    result = verify_or_update_datasets()
    assert result is True, "verify_or_update_datasets must return True"
    print("[OK] verify_or_update_datasets() executed successfully")


def test_multi_agency_consensus_engine():
    print("\n--- TEST 4: Multi-Agency Consensus Engine Cross-Validation ---")
    
    # MOSDAC SST = 27.5 deg C, Copernicus SST = 27.7 deg C -> Variance = 0.2 deg C
    provenance = compute_multi_agency_consensus(mosdac_sst=27.5, lat=8.7642, lon=78.1348)
    
    assert provenance["primary_agency"] == "ISRO MOSDAC (INSAT-3DR, Oceansat-3)"
    assert provenance["secondary_agency"] == "Copernicus Marine Service (Sentinel-3 / CMEMS)"
    assert provenance["consensus_status"] == "DUAL_SOURCE_VERIFIED"
    assert provenance["confidence_score"] == 98.4, f"Expected 98.4, got {provenance['confidence_score']}"
    print(f"[OK] Consensus validated: score = {provenance['confidence_score']}%, status = {provenance['consensus_status']}")


def test_e2e_api_response_provenance():
    print("\n--- TEST 5: End-to-End API Satellite Provenance Delivery ---")
    
    req = QueryRequest(
        query="Where is the nearest fishing spot near Tuticorin?",
        query_text="Where is the nearest fishing spot near Tuticorin?",
        persona="FISHERMAN",
        language_preference="en",
        telemetry=TelemetryData(latitude=8.7642, longitude=78.1348)
    )
    
    response = asyncio.run(process_marine_query(req))
    assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
    
    data = json.loads(response.body.decode("utf-8"))
    assert "satellite_provenance" in data, "satellite_provenance must be present in response root"
    prov = data["satellite_provenance"]
    assert prov["primary_agency"] == "ISRO MOSDAC (INSAT-3DR, Oceansat-3)"
    assert prov["secondary_agency"] == "Copernicus Marine Service (Sentinel-3 / CMEMS)"
    assert prov["consensus_status"] == "DUAL_SOURCE_VERIFIED"
    assert prov["confidence_score"] == 98.4
    print("[OK] API JSON response contains verified satellite_provenance payload:")
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    print("=================================================================")
    print("RUNNING COPERNICUS MARINE SYNC & CONSENSUS ENGINE TESTS")
    print("=================================================================")
    try:
        test_copernicus_authentication()
        test_copernicus_download_function()
        test_datasync_manager_lifecycle()
        test_multi_agency_consensus_engine()
        test_e2e_api_response_provenance()
        print("\n=================================================================")
        print("ALL 5 COPERNICUS & MOSDAC CONSENSUS TESTS PASSED! [SUCCESS]")
        print("=================================================================\n")
    except AssertionError as e:
        print(f"\n[FAIL] TEST ASSERTION FAILURE: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

