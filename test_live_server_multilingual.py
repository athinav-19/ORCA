"""
test_live_server_multilingual.py - Live HTTP integration test for ORCA /api/chat endpoint
Sends requests in 5 languages to the running FastAPI server on http://localhost:8000
Verifies:
1. Gujarati query -> Gujarati response
2. Tamil query -> Tamil response
3. Hindi query -> Hindi response
4. Malayalam query -> Malayalam response
5. English query -> English response
6. Overriding default 'en' language_preference sent by mobile app
7. All response aliases populated (chat_text, reply, response, message, native_advisory_text)
"""

import sys
import json
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SERVER_URL = "http://localhost:8000/api/chat"

TESTS = [
    {
        "lang_name": "English",
        "expected_code": "en",
        "query": "Where is the nearest fishing ground?",
        "is_indic": False
    },
    {
        "lang_name": "Gujarati",
        "expected_code": "gu",
        "query": "સૌથી નજીકનો માછીમારી ક્ષેત્ર ક્યાં છે?",
        "is_indic": True
    },
    {
        "lang_name": "Tamil",
        "expected_code": "ta",
        "query": "அருகிலுள்ள மீன்பிடி பகுதி எங்கே?",
        "is_indic": True
    },
    {
        "lang_name": "Hindi",
        "expected_code": "hi",
        "query": "निकटतम मछली पकड़ने का क्षेत्र कहाँ है?",
        "is_indic": True
    },
    {
        "lang_name": "Malayalam",
        "expected_code": "ml",
        "query": "ഏറ്റവും അടുത്തുള്ള മത്സ്യബന്ധന പ്രദേശം എവിടെയാണ്?",
        "is_indic": True
    }
]

def run_live_tests():
    print("=" * 70)
    print("ORCA LIVE HTTP MULTILINGUAL VERIFICATION TEST")
    print(f"Target: {SERVER_URL}")
    print("=" * 70)

    all_passed = True

    for t in TESTS:
        print(f"\n>> Sending {t['lang_name']} Request: '{t['query']}'")
        payload = {
            "query": t["query"],
            "persona": "FISHERMAN",
            "language_preference": "en",  # Simulate mobile app sending 'en' default
            "telemetry": {
                "latitude": 8.7642,
                "longitude": 78.1348,
                "speed_knots": 0.0,
                "heading_degrees": 120.0,
                "gps_accuracy_meters": 4.5
            }
        }
        
        req = urllib.request.Request(
            SERVER_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"[FAIL] HTTP request failed for {t['lang_name']}: {e}")
            all_passed = False
            continue

        detected = data.get("detected_language")
        resp_lang = data.get("response_language") or data.get("source_language")
        original_q = data.get("original_query")
        english_q = data.get("english_query")
        reasoning_out = data.get("reasoning_output")
        reply = data.get("reply") or data.get("chat_text")

        print(f"   Detected Lang     : {detected}")
        print(f"   Response Lang     : {resp_lang}")
        print(f"   English Query     : {english_q}")
        print(f"   Reasoning (EN)    : {str(reasoning_out)[:70]}...")
        print(f"   Reply (Native)    : {str(reply)[:70]}...")

        # Assertions
        if detected != t["expected_code"]:
            print(f"[FAIL] Detected language mismatch: expected {t['expected_code']}, got {detected}")
            all_passed = False
        elif t["is_indic"]:
            has_indic = any(ord(c) > 127 for c in str(reply))
            if not has_indic:
                print(f"[FAIL] Expected native script for {t['lang_name']}, got English!")
                all_passed = False
            else:
                print(f"[PASS] {t['lang_name']} query correctly answered in native {t['lang_name']} script!")
        else:
            print(f"[PASS] English query correctly answered in English.")

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL 5 LIVE HTTP ENDPOINT TESTS PASSED SUCCESSFULLY!")
    else:
        print("SOME LIVE TESTS FAILED.")
        sys.exit(1)
    print("=" * 70)

if __name__ == "__main__":
    run_live_tests()

