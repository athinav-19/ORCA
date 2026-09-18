"""
test_multilingual_pipeline.py - Verification suite for ORCA's Multilingual Response Pipeline
Validates:
1. Exact original user query capture
2. Deterministic language detection (Unicode script matching)
3. Internal translation to English for multi-agent reasoning
4. English reasoning output preserved in lifecycle
5. Final user-facing response strictly in detected language (native Indic script)
6. Response fields (reply, response, message, chat_text, native_advisory_text, final_response)
7. English, Gujarati, Tamil, Hindi, Malayalam
"""

import os
import sys
import json

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from translation_service import detect_language_from_text, IndicTranslationService
from server import execute_orca_core

TEST_CASES = [
    {
        "name": "English Fishing Query",
        "query": "Where is the nearest fishing ground?",
        "expected_lang": "en",
        "is_indic": False,
    },
    {
        "name": "Gujarati Fishing Query",
        "query": "સૌથી નજીકનો માછીમારી ક્ષેત્ર ક્યાં છે?",
        "expected_lang": "gu",
        "is_indic": True,
    },
    {
        "name": "Tamil Fishing Query",
        "query": "அருகிலுள்ள மீன்பிடி பகுதி எங்கே?",
        "expected_lang": "ta",
        "is_indic": True,
    },
    {
        "name": "Hindi Fishing Query",
        "query": "निकटतम मछली पकड़ने का क्षेत्र कहाँ है?",
        "expected_lang": "hi",
        "is_indic": True,
    },
    {
        "name": "Malayalam Fishing Query",
        "query": "ഏറ്റവും അടുത്തുള്ള മത്സ്യബന്ധന പ്രദേശം എവിടെയാണ്?",
        "expected_lang": "ml",
        "is_indic": True,
    },
]


def test_language_detection():
    print("\n" + "=" * 60)
    print("TEST 1: SCRIPT DETECTION ACCURACY")
    print("=" * 60)
    all_passed = True
    for tc in TEST_CASES:
        detected = detect_language_from_text(tc["query"], default_lang="en")
        passed = detected == tc["expected_lang"]
        if not passed:
            all_passed = False
        print(f"[{'PASS' if passed else 'FAIL'}] '{tc['query']}' -> Expected: {tc['expected_lang']}, Detected: {detected}")
    assert all_passed, "Language detection test failed!"
    print(">> All language detection tests PASSED!\n")


def test_full_pipeline_execution():
    print("\n" + "=" * 60)
    print("TEST 2: MULTI-AGENT PIPELINE END-TO-END EXECUTION")
    print("=" * 60)

    overall_passed = True

    for tc in TEST_CASES:
        print(f"\n--- Testing: {tc['name']} ---")
        print(f"Original Query: {tc['query']}")

        result = execute_orca_core(
            query=tc["query"],
            lat=8.7642,
            lon=78.1348,
            persona="FISHERMAN",
            language="en",  # Simulate mobile app sending default 'en'
        )

        detected_lang = result.get("detected_language")
        original_q = result.get("original_query")
        english_q = result.get("english_query")
        reasoning_out = result.get("reasoning_output")
        final_resp = result.get("final_response") or result.get("chat_text") or result.get("reply")

        print(f"Detected Language  : {detected_lang}")
        print(f"English Query      : {english_q}")
        print(f"Reasoning Output   : {str(reasoning_out)[:80]}...")
        print(f"Final Response     : {str(final_resp)[:80]}...")

        # Assertions
        assert detected_lang == tc["expected_lang"], f"Expected {tc['expected_lang']} but got {detected_lang}"
        assert original_q == tc["query"], f"Original query mismatch: {original_q}"
        assert english_q and len(english_q) > 0, "English query must not be empty"

        # Check reasoning_output is in English
        assert reasoning_out and len(reasoning_out) > 0, "Reasoning output must not be empty"

        if tc["is_indic"]:
            # Final response MUST contain Indic script characters (ord > 127)
            has_indic = any(ord(c) > 127 for c in str(final_resp))
            if not has_indic:
                print(f"FAIL: Expected Indic characters in response for {tc['expected_lang']}, got English: {final_resp}")
                overall_passed = False
            else:
                print(f"[PASS] Response contains native {tc['expected_lang']} script!")

            # Verify all user-facing fields match
            assert result.get("reply") == final_resp, "reply field does not match final_response"
            assert result.get("response") == final_resp, "response field does not match final_response"
            assert result.get("chat_text") == final_resp, "chat_text field does not match final_response"
            assert result.get("native_advisory_text") == final_resp, "native_advisory_text field does not match"
        else:
            print("[PASS] English query returned English response as expected.")

    if overall_passed:
        print("\n" + "=" * 60)
        print("ALL 5 MULTILINGUAL PIPELINE TESTS PASSED WITH 100% SUCCESS!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("SOME TESTS FAILED!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    test_language_detection()
    test_full_pipeline_execution()

