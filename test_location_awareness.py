"""
test_location_awareness.py - Comprehensive Location Awareness Test Suite for ORCA
Verifies:
1. Mumbai weather -> lat ~18.9, lon ~72.8, no Tuticorin references.
2. Mumbai cyclone -> Mumbai/Arabian Sea, not Gulf of Mannar.
3. Mumbai PFZ -> PFZ off Mumbai coast.
4. Chennai ocean -> lat ~13.0, lon ~80.2.
5. "Cyclone near me" + GPS (18.9220, 72.8347) -> resolves to Mumbai.
6. Kochi weather -> lat ~9.9, lon ~76.2.
7. Route from Mumbai to Goa -> origin: Mumbai, dest: Goa.
8. Route from Tuticorin to Sri Lanka -> evaluated only when explicit.
9. Gujarati query "મુંબઈ માં હવામાન કેવું છે?" -> resolves to Mumbai.
10. Tamil query "மும்பையில் வானிலை எப்படி இருக்கிறது?" -> resolves to Mumbai.
11. Edge case: No location + No GPS -> returns LOCATION_REQUIRED.
12. Edge case: Explicit Mumbai + Tuticorin GPS -> explicit Mumbai strictly supersedes GPS.
"""

import os
import sys
import json
from typing import Dict, Any

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from main import process_marine_request
from server import execute_orca_core

results = []

def run_test(test_id: int, description: str, query: str, telemetry: Dict[str, Any], validator) -> bool:
    print(f"\n[{test_id}] Running: {description}")
    print(f"    Query: '{query}' | Telemetry: {telemetry}")
    
    req_data = {
        "session_id": f"test_loc_{test_id}",
        "user_context": {"persona": "FISHERMAN"},
        "device_telemetry": telemetry,
        "user_input": {
            "input_type": "TEXT",
            "raw_text": query,
        }
    }
    
    try:
        resp = process_marine_request(req_data)
        passed, detail = validator(resp)
        status_str = "PASS" if passed else "FAIL"
        print(f"    Result: [{status_str}] - {detail}")
        results.append((test_id, description, status_str, detail))
        return passed
    except Exception as e:
        print(f"    Result: [ERROR] - {e}")
        results.append((test_id, description, "ERROR", str(e)))
        return False


def contains_tuticorin_bias(text: str) -> bool:
    t_lower = text.lower()
    return "thoothukudi" in t_lower or "tuticorin" in t_lower or "gulf of mannar" in t_lower


# -------------------------------------------------------------
# Test 1: Mumbai weather
# -------------------------------------------------------------
def validate_t1(resp):
    lc = resp.get("location_context", {})
    name = lc.get("name", "")
    lat = lc.get("latitude")
    lon = lc.get("longitude")
    adv = resp.get("native_advisory_text", "")
    
    if name != "Mumbai":
        return False, f"Expected location Mumbai, got '{name}'"
    if lat is None or abs(lat - 18.9220) > 0.5:
        return False, f"Expected lat ~18.92, got {lat}"
    if lon is None or abs(lon - 72.8347) > 0.5:
        return False, f"Expected lon ~72.83, got {lon}"
    if contains_tuticorin_bias(adv):
        return False, f"Response contains Tuticorin bias: {adv[:100]}"
    return True, f"Resolved {name} ({lat:.4f}, {lon:.4f}) without Tuticorin bias"

run_test(1, "Mumbai weather query", "How is the weather in Mumbai today?", {"latitude": None, "longitude": None}, validate_t1)


# -------------------------------------------------------------
# Test 2: Mumbai cyclone
# -------------------------------------------------------------
def validate_t2(resp):
    lc = resp.get("location_context", {})
    name = lc.get("name", "")
    adv = resp.get("native_advisory_text", "")
    
    if name != "Mumbai":
        return False, f"Expected location Mumbai, got '{name}'"
    if contains_tuticorin_bias(adv):
        return False, f"Advisory contains Thoothukudi/Mannar bias: {adv[:100]}"
    if "mumbai" not in adv.lower():
        return False, f"Advisory does not mention Mumbai: {adv[:100]}"
    return True, f"Resolved cyclone analysis for Mumbai without Mannar bias"

run_test(2, "Mumbai cyclone query", "Is there any cyclone warning near Mumbai?", {"latitude": None, "longitude": None}, validate_t2)


# -------------------------------------------------------------
# Test 3: Mumbai PFZ
# -------------------------------------------------------------
def validate_t3(resp):
    lc = resp.get("location_context", {})
    name = lc.get("name", "")
    adv = resp.get("native_advisory_text", "")
    rec = resp.get("recommended_coordinates", "")
    
    if name != "Mumbai":
        return False, f"Expected location Mumbai, got '{name}'"
    if contains_tuticorin_bias(adv):
        return False, f"PFZ advisory contains Tuticorin bias: {adv[:100]}"
    if "9.0932" in adv or "78.3218" in adv:
        return False, f"PFZ advisory contains hardcoded Tuticorin PFZ coordinates 9.0932, 78.3218!"
    return True, f"PFZ evaluated off Mumbai ({rec}) without Tuticorin coordinates"

run_test(3, "Mumbai PFZ query", "Where is the best fishing zone off Mumbai?", {"latitude": None, "longitude": None}, validate_t3)


# -------------------------------------------------------------
# Test 4: Chennai ocean
# -------------------------------------------------------------
def validate_t4(resp):
    lc = resp.get("location_context", {})
    name = lc.get("name", "")
    lat = lc.get("latitude")
    lon = lc.get("longitude")
    adv = resp.get("native_advisory_text", "")
    
    if name != "Chennai":
        return False, f"Expected location Chennai, got '{name}'"
    if lat is None or abs(lat - 13.0827) > 0.5:
        return False, f"Expected lat ~13.08, got {lat}"
    if contains_tuticorin_bias(adv):
        return False, f"Contains Tuticorin bias: {adv[:100]}"
    return True, f"Resolved Chennai ({lat:.4f}, {lon:.4f}) successfully"

run_test(4, "Chennai ocean state query", "What is the wave height and ocean current at Chennai?", {"latitude": None, "longitude": None}, validate_t4)


# -------------------------------------------------------------
# Test 5: "Cyclone near me" with GPS (18.9220, 72.8347)
# -------------------------------------------------------------
def validate_t5(resp):
    lc = resp.get("location_context", {})
    source = lc.get("source")
    name = lc.get("name", "")
    lat = lc.get("latitude")
    adv = resp.get("native_advisory_text", "")
    
    if source != "DEVICE_GPS":
        return False, f"Expected source DEVICE_GPS, got {source}"
    if lat is None or abs(lat - 18.9220) > 0.1:
        return False, f"Expected GPS lat ~18.9220, got {lat}"
    if contains_tuticorin_bias(adv):
        return False, f"Contains Tuticorin bias: {adv[:100]}"
    return True, f"Resolved 'near me' to device GPS sector {name} ({lat:.4f})"

run_test(5, "'Cyclone near me' with Mumbai GPS", "Is there any cyclone near me?", {"latitude": 18.9220, "longitude": 72.8347}, validate_t5)


# -------------------------------------------------------------
# Test 6: Kochi weather
# -------------------------------------------------------------
def validate_t6(resp):
    lc = resp.get("location_context", {})
    name = lc.get("name", "")
    lat = lc.get("latitude")
    lon = lc.get("longitude")
    adv = resp.get("native_advisory_text", "")
    
    if name != "Kochi":
        return False, f"Expected location Kochi, got '{name}'"
    if lat is None or abs(lat - 9.9312) > 0.5:
        return False, f"Expected lat ~9.93, got {lat}"
    if contains_tuticorin_bias(adv):
        return False, f"Contains Tuticorin bias: {adv[:100]}"
    return True, f"Resolved Kochi ({lat:.4f}, {lon:.4f}) successfully"

run_test(6, "Kochi weather query", "Weather forecast for Kochi port", {"latitude": None, "longitude": None}, validate_t6)


# -------------------------------------------------------------
# Test 7: Route from Mumbai to Goa
# -------------------------------------------------------------
def validate_t7(resp):
    orig = resp.get("origin", {}) or {}
    dest = resp.get("destination", {}) or {}
    orig_name = orig.get("name", "")
    dest_name = dest.get("name", "")
    adv = resp.get("native_advisory_text", "")
    
    if "mumbai" not in orig_name.lower():
        return False, f"Expected origin Mumbai, got '{orig_name}'"
    if "goa" not in dest_name.lower():
        return False, f"Expected destination Goa, got '{dest_name}'"
    if contains_tuticorin_bias(adv):
        return False, f"Route contains Tuticorin bias: {adv[:100]}"
    return True, f"Route planned from {orig_name} ({orig.get('latitude')}) to {dest_name} ({dest.get('latitude')})"

run_test(7, "Route from Mumbai to Goa", "Plan a safe passage from Mumbai to Goa", {"latitude": None, "longitude": None}, validate_t7)


# -------------------------------------------------------------
# Test 8: Route from Tuticorin to Sri Lanka
# -------------------------------------------------------------
def validate_t8(resp):
    orig = resp.get("origin", {}) or {}
    dest = resp.get("destination", {}) or {}
    orig_name = orig.get("name", "")
    dest_name = dest.get("name", "")
    status = resp.get("map_status", "")
    
    if "tuticorin" not in orig_name.lower() and "thoothukudi" not in orig_name.lower():
        return False, f"Expected origin Tuticorin, got '{orig_name}'"
    if "sri lanka" not in dest_name.lower() and "colombo" not in dest_name.lower():
        return False, f"Expected destination Sri Lanka, got '{dest_name}'"
    if status not in ("NO-GO", "DANGER", "CAUTION"):
        return False, f"Expected IMBL boundary restriction NO-GO/CAUTION, got '{status}'"
    return True, f"Route Tuticorin to Sri Lanka evaluated IMBL border restriction correctly ({status})"

run_test(8, "Route Tuticorin to Sri Lanka (Explicit)", "Plan navigation from Tuticorin to Sri Lanka", {"latitude": None, "longitude": None}, validate_t8)


# -------------------------------------------------------------
# Test 9: Gujarati query for Mumbai
# -------------------------------------------------------------
def validate_t9(resp):
    lc = resp.get("location_context", {})
    name = lc.get("name", "")
    eng_q = resp.get("english_query", "")
    
    if name != "Mumbai":
        return False, f"Expected location Mumbai, got '{name}' (English: {eng_q})"
    return True, f"Gujarati query correctly resolved to {name} via translation/gazetteer"

run_test(9, "Gujarati Mumbai weather query", "મુંબઈ માં હવામાન કેવું છે?", {"latitude": None, "longitude": None}, validate_t9)


# -------------------------------------------------------------
# Test 10: Tamil query for Mumbai
# -------------------------------------------------------------
def validate_t10(resp):
    lc = resp.get("location_context", {})
    name = lc.get("name", "")
    eng_q = resp.get("english_query", "")
    
    if name != "Mumbai":
        return False, f"Expected location Mumbai, got '{name}' (English: {eng_q})"
    return True, f"Tamil query correctly resolved to {name} via translation/gazetteer"

run_test(10, "Tamil Mumbai weather query", "மும்பையில் வானிலை எப்படி இருக்கிறது?", {"latitude": None, "longitude": None}, validate_t10)


# -------------------------------------------------------------
# Test 11: Edge Case - No Location + No GPS -> LOCATION_REQUIRED
# -------------------------------------------------------------
def validate_t11(resp):
    status = resp.get("status")
    msg = resp.get("message", "")
    supp = resp.get("supported_locations", [])
    
    if status != "LOCATION_REQUIRED":
        return False, f"Expected status LOCATION_REQUIRED, got '{status}'"
    if not supp or "Mumbai" not in supp:
        return False, f"Expected supported_locations list, got {supp}"
    return True, f"Returned structured LOCATION_REQUIRED state with {len(supp)} supported locations"

run_test(11, "No Location & No GPS -> LOCATION_REQUIRED", "How is the weather today?", {"latitude": None, "longitude": None}, validate_t11)


# -------------------------------------------------------------
# Test 12: Edge Case - Explicit Mumbai supersedes Tuticorin GPS
# -------------------------------------------------------------
def validate_t12(resp):
    lc = resp.get("location_context", {})
    source = lc.get("source")
    name = lc.get("name", "")
    lat = lc.get("latitude")
    
    if source != "EXPLICIT_QUERY":
        return False, f"Expected source EXPLICIT_QUERY, got {source}"
    if name != "Mumbai":
        return False, f"Expected Mumbai to override GPS, got '{name}'"
    if lat is None or abs(lat - 18.9220) > 0.5:
        return False, f"Expected Mumbai latitude ~18.9220, got {lat}"
    return True, f"Explicit Mumbai strictly superseded device Tuticorin GPS"

run_test(12, "Explicit Mumbai overrides GPS", "Check weather in Mumbai", {"latitude": 8.7642, "longitude": 78.1348}, validate_t12)


print("\n" + "=" * 70)
print("             ORCA LOCATION AWARENESS TEST SUMMARY")
print("=" * 70)
all_passed = True
for tid, desc, status, detail in results:
    badge = "✅" if status == "PASS" else "❌"
    if status != "PASS":
        all_passed = False
    print(f"  {badge} [{tid:2d}] {status:4s} | {desc[:42]:<42} | {detail}")
print("=" * 70)

if all_passed:
    print("\n🎉 ALL 12 LOCATION AWARENESS TESTS PASSED SUCCESSFULLY!")
    sys.exit(0)
else:
    print("\n⚠️ SOME TESTS FAILED. CHECK DETAILS ABOVE.")
    sys.exit(1)

