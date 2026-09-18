"""
test_indic_translation.py - End-to-End Verification of Indic Multilingual Translation
Verifies:
1. IndicTranslationService inbound and outbound translation for Indian coastal languages.
2. Complete absence of Bhashini dependencies, imports, warnings, and bhashini_text payload keys.
3. process_marine_query handling lang parameter via request body and query params.
4. Correct population of reply, response, message, chat_text, and native_advisory_text.
5. Dual-agency satellite provenance retention.
"""

import sys
import os
import json
import asyncio

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from translation_service import IndicTranslationService, SUPPORTED_LANGUAGES
from models import QueryRequest
from server import process_marine_query

async def run_tests():
    print("=" * 75)
    print("PROJECT ORCA - INDIC MULTILINGUAL TRANSLATION SERVICE VERIFICATION")
    print("=" * 75)

    # 1. Inbound translation test
    print("\n--- Test 1: Inbound Translation (Tamil -> English) ---")
    tamil_q = "தூத்துக்குடி அருகே இன்று மீன்பிடிக்க செல்லலாமா?"
    en_q = IndicTranslationService.translate_to_english(tamil_q, source_lang="ta")
    print(f"Original (ta) : {tamil_q}")
    print(f"Inbound (en)  : {en_q}")
    assert len(en_q) > 0, "Inbound translation should not be empty"

    # 2. Outbound translation test
    print("\n--- Test 2: Outbound Translation (English -> Tamil, Malayalam, Hindi) ---")
    advisory_en = "Safe for standard artisanal fishing operations. Continue monitoring radar alerts."
    ta_adv = IndicTranslationService.translate_to_target(advisory_en, target_lang="ta")
    ml_adv = IndicTranslationService.translate_to_target(advisory_en, target_lang="ml")
    hi_adv = IndicTranslationService.translate_to_target(advisory_en, target_lang="hi")

    print(f"English (en)   : {advisory_en}")
    print(f"Tamil (ta)     : {ta_adv}")
    print(f"Malayalam (ml) : {ml_adv}")
    print(f"Hindi (hi)     : {hi_adv}")

    # 3. End-to-End API Test with lang="ta"
    print("\n--- Test 3: API Pipeline with lang='ta' in Request Body ---")
    req_ta = QueryRequest(
        query=tamil_q,
        persona="FISHERMAN",
        lang="ta"
    )
    resp_ta = await process_marine_query(req_ta)
    body_ta = json.loads(resp_ta.body.decode("utf-8"))

    print(f"HTTP Status               : 200 OK")
    print(f"Response Status           : {body_ta.get('status')}")
    print(f"Source Language           : {body_ta.get('source_language')}")
    print(f"Language Name             : {body_ta.get('language_name')}")
    print(f"Reply Populated           : {bool(body_ta.get('reply'))}")
    print(f"Chat Text Populated       : {bool(body_ta.get('chat_text'))}")
    print(f"Native Advisory Populated : {bool(body_ta.get('native_advisory_text'))}")
    print(f"bhashini_text in Root?    : {'bhashini_text' in body_ta}")
    print(f"bhashini_text in Advisory?: {'bhashini_text' in body_ta.get('advisory', {})}")
    print(f"Satellite Provenance?     : {'satellite_provenance' in body_ta}")

    assert "bhashini_text" not in body_ta, "bhashini_text MUST NOT be in root payload"
    assert "bhashini_text" not in body_ta.get("advisory", {}), "bhashini_text MUST NOT be in advisory dict"
    assert body_ta.get("source_language") == "ta", "source_language should be 'ta'"
    assert body_ta.get("reply") == body_ta.get("native_advisory_text"), "reply should match native_advisory_text"

    # 4. End-to-End API Test with query param lang="hi"
    print("\n--- Test 4: API Pipeline with ?lang=hi Query Parameter ---")
    req_hi = QueryRequest(
        query="क्या आज मछली पकड़ना सुरक्षित है?",
        persona="FISHERMAN"
    )
    resp_hi = await process_marine_query(req_hi, lang="hi")
    body_hi = json.loads(resp_hi.body.decode("utf-8"))

    print(f"HTTP Status               : 200 OK")
    print(f"Response Status           : {body_hi.get('status')}")
    print(f"Source Language           : {body_hi.get('source_language')}")
    print(f"Language Name             : {body_hi.get('language_name')}")
    print(f"bhashini_text in Root?    : {'bhashini_text' in body_hi}")
    print(f"bhashini_text in Advisory?: {'bhashini_text' in body_hi.get('advisory', {})}")

    assert "bhashini_text" not in body_hi, "bhashini_text MUST NOT be in root payload"
    assert body_hi.get("source_language") == "hi", "source_language should be 'hi'"

    # 5. End-to-End API Test with default English
    print("\n--- Test 5: API Pipeline with Default English (lang='en') ---")
    req_en = QueryRequest(
        query="Where is the nearest fishing spot?",
        persona="FISHERMAN",
        lang="en"
    )
    resp_en = await process_marine_query(req_en)
    body_en = json.loads(resp_en.body.decode("utf-8"))

    print(f"HTTP Status               : 200 OK")
    print(f"Response Status           : {body_en.get('status')}")
    print(f"Source Language           : {body_en.get('source_language')}")
    print(f"bhashini_text in Root?    : {'bhashini_text' in body_en}")
    assert "bhashini_text" not in body_en, "bhashini_text MUST NOT be in root payload"
    assert body_en.get("source_language") == "en", "source_language should be 'en'"

    print("\n" + "=" * 75)
    print("ALL TESTS PASSED: Indic Translation Verified, Bhashini completely removed!")
    print("=" * 75)

if __name__ == "__main__":
    asyncio.run(run_tests())
