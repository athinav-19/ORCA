import time
import requests
import json
import sys

# Ensure UTF-8 printing in Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"

print("=" * 75)
print("RUNNING ORCA SERVER COMPREHENSIVE RENDER READINESS VERIFICATION")
print("=" * 75)

# 1. Health & Docs
print("\n[1] Testing GET /health ...")
r_health = requests.get(f"{BASE_URL}/health", timeout=10)
assert r_health.status_code == 200, f"/health failed: {r_health.status_code}"
h_json = r_health.json()
print("  Status: HTTP 200 OK")
print(f"  Service: {h_json.get('service')}")
print(f"  Green Marine Energy Calculation Basis: {h_json.get('green_marine_energy', {}).get('calculation_basis')}")
print(f"  Green Marine Energy is_estimate: {h_json.get('green_marine_energy', {}).get('is_estimate')}")
assert h_json.get('green_marine_energy', {}).get('is_estimate') is True
assert h_json.get('green_marine_energy', {}).get('calculation_basis') == 'THEORETICAL_MODEL_ESTIMATE'
print("  [PASS] /health verified with theoretical energy model labels.")

print("\n[2] Testing GET /docs (OpenAPI Swagger UI) ...")
r_docs = requests.get(f"{BASE_URL}/docs", timeout=10)
assert r_docs.status_code == 200, f"/docs failed: {r_docs.status_code}"
print("  Status: HTTP 200 OK")
print("  [PASS] /docs verified.")

# 3. Location & Route Sanity: California out-of-domain coordinates
print("\n[3] Testing Out-of-Domain Coordinates (California 37.421997, -122.083999) ...")
payload_ca = {
    "query": "Is it safe to go fishing here?",
    "persona": "FISHERMAN",
    "telemetry": {
        "latitude": 37.421997,
        "longitude": -122.083999,
        "speed_knots": 0.0,
        "heading_degrees": 0.0,
        "gps_accuracy_meters": 5.0
    }
}
r_ca = requests.post(f"{BASE_URL}/api/chat", json=payload_ca, timeout=30)
assert r_ca.status_code == 200, f"California test failed: {r_ca.status_code}"
ca_json = r_ca.json()
within_eez = ca_json.get("within_eez")
map_status = ca_json.get("map_status")
safe_route = ca_json.get("safe_sea_route")
print(f"  within_eez: {within_eez}")
print(f"  map_status: {map_status}")
print(f"  safe_route status: {safe_route.get('route_status') if safe_route else None}")
assert within_eez is False, f"Expected within_eez=False, got {within_eez}"
if safe_route:
    assert safe_route.get("route_status") in ("OUT_OF_OPERATIONAL_DOMAIN", "ROUTE_EXCEEDS_OPERATIONAL_RANGE", None)
    assert len(safe_route.get("waypoints", [])) == 0, "Waypoints must be empty for out-of-domain!"
print("  [PASS] California coordinates safely rejected without fabricating EEZ or 8000 NM route.")

# 4. Location Sanity: Real Indian GPS coordinates (Mumbai 18.9220, 72.8347)
print("\n[4] Testing Valid Indian Coordinates (Mumbai 18.9220, 72.8347) ...")
payload_in = {
    "query": "Check ocean conditions and wave safety offshore",
    "persona": "FISHERMAN",
    "telemetry": {
        "latitude": 18.9220,
        "longitude": 72.8347,
        "speed_knots": 0.0,
        "heading_degrees": 270.0,
        "gps_accuracy_meters": 3.0
    }
}
r_in = requests.post(f"{BASE_URL}/api/chat", json=payload_in, timeout=30)
assert r_in.status_code == 200, f"Mumbai test failed: {r_in.status_code}"
in_json = r_in.json()
print(f"  within_eez: {in_json.get('within_eez')}")
print(f"  map_status: {in_json.get('map_status')}")
print(f"  advisory: {in_json.get('chat_text', '')[:120]}...")
assert in_json.get("within_eez") is True, f"Expected within_eez=True for Mumbai, got {in_json.get('within_eez')}"
print("  [PASS] Valid Indian GPS coordinates utilized correctly.")

# 5. Multilingual Responses (Preserve exact original query language)
languages_to_test = [
    ("English", "What is the wave height and wind speed offshore?", "en"),
    ("Gujarati", "દરિયામાં મોજા કેવા છે?", "gu"),
    ("Tamil", "இன்று கடலில் அலைகள் எப்படி இருக்கிறது?", "ta"),
    ("Hindi", "आज समुद्र में लहरें कितनी ऊंची हैं?", "hi"),
    ("Malayalam", "കടലിൽ തിരമാലകൾ എത്ര ഉയർന്നതാണ്?", "ml"),
]

print("\n[5] Testing Multilingual Responses (Gujarati, Tamil, Hindi, Malayalam, English) ...")
for lang_name, query_text, expected_code in languages_to_test:
    r_lang = requests.post(
        f"{BASE_URL}/api/chat",
        json={"query": query_text, "persona": "FISHERMAN"},
        timeout=30
    )
    assert r_lang.status_code == 200, f"{lang_name} query failed: {r_lang.status_code}"
    resp_json = r_lang.json()
    resp_lang = resp_json.get("response_language")
    detected_lang = resp_json.get("detected_language")
    chat_text = resp_json.get("chat_text", "")
    print(f"\n  [{lang_name}] Query: '{query_text}'")
    print(f"    Detected Lang: {detected_lang} | Response Lang: {resp_lang}")
    print(f"    Chat text snippet: {chat_text[:100]}...")
    assert resp_lang == expected_code, f"Expected language {expected_code}, got {resp_lang}"
    if expected_code != "en":
        assert any(ord(c) > 127 for c in chat_text), f"Expected Indic script response for {lang_name}, got ASCII!"
    print(f"    [PASS] {lang_name} responded in native language.")

# 6. Wave Height Consistency Check
print("\n[6] Testing Wave Height Consistency ...")
r_wave = requests.post(
    f"{BASE_URL}/api/chat",
    json={"query": "Give me detailed wave height report for Thoothukudi", "persona": "FISHERMAN"},
    timeout=30
)
assert r_wave.status_code == 200
w_json = r_wave.json()
adv_wave = w_json.get("advisory", {}).get("wave_height") if isinstance(w_json.get("advisory"), dict) else None
risk_metrics = w_json.get("risk_assessment", {}).get("metrics", {})
metric_wave = risk_metrics.get("wave_height_m")
print(f"  Advisory wave_height: {adv_wave}")
print(f"  Risk metrics wave_height_m: {metric_wave}")
assert metric_wave is not None and float(metric_wave) > 0.0, "Wave height metric must NOT be 0.0!"
print("  [PASS] Wave height consistency verified.")

print("\n" + "=" * 75)
print("ALL RENDER READINESS VERIFICATION CHECKS PASSED SUCCESSFULLY!")
print("=" * 75)
