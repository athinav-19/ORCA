"""
test_gemini_synthesis_prompt.py - Verification of ORCA Marine Copilot Synthesis Prompt
Checks:
1. SYSTEM_PROMPT contains the exact required structure and instructions.
2. Advisory output is strictly 2 paragraphs.
3. Paragraph 1 opens naturally with Situational Brief / route analysis confirmation (no robotic coordinates).
4. Paragraph 2 integrates destination coords, why chosen, bearing/direction/distance, avoidance rationale, and MOSDAC+Copernicus validation with wave/wind.
5. No internal agent names ("Oceanographic Agent", "Spatial Routing Agent", "Meteorological Agent").
6. No robotic coordinate readout at start, no raw JSON, no final directive at the end.
7. Full API pipeline returns 200 with populated reply, message, response, chat_text, native_advisory_text.
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

from models import QueryRequest
import server
from server import process_marine_query, synthesize_copilot_advisory, SYSTEM_PROMPT

async def run_verification():
    print("=" * 80)
    print("ORCA MARINE COPILOT - GEMINI SYNTHESIS ENGINE PROMPT VERIFICATION")
    print("=" * 80)

    # 1. Verify SYSTEM_PROMPT Structure
    print("\n--- Test 1: Verify SYSTEM_PROMPT Template Structure ---")
    assert "You are the ORCA Marine Copilot" in SYSTEM_PROMPT, "Prompt missing persona identity"
    assert "Situational Brief:" in SYSTEM_PROMPT, "Prompt missing Situational Brief"
    assert "Integrated Navigational Reasoning:" in SYSTEM_PROMPT, "Prompt missing Navigational Reasoning"
    assert "Oceanographic Agent" in SYSTEM_PROMPT and "Spatial Routing Agent" in SYSTEM_PROMPT and "Meteorological Agent" in SYSTEM_PROMPT, "Prompt missing forbidden agent instructions"
    assert "ISRO MOSDAC" in SYSTEM_PROMPT and "Copernicus Marine Service" in SYSTEM_PROMPT, "Prompt missing satellite validation instructions"
    print("✓ SYSTEM_PROMPT matches exact required prompt template!")

    # 2. Test Direct Synthesis Function
    print("\n--- Test 2: Test synthesize_copilot_advisory Function ---")
    mock_payload = {
        "user_persona": "FISHERMAN",
        "threat_status": "SAFE",
        "primary_geographic_target": {
            "feature_type": "PFZ Aggregation Hotspot",
            "target_lat": 9.0932,
            "target_lon": 78.3218,
            "distance_nm": 22.6,
            "bearing_deg": 29.1,
            "cardinal_direction": "NNE",
            "cardinal_direction_full": "North-Northeast",
            "relative_vector": "22.6 NM along Bearing 029° NNE",
            "landmark_reference": "Thoothukudi Outer Harbor",
            "details": {
                "likely_catch": ["Yellowfin Tuna", "Mackerel", "Sardine"],
                "sst_c": 27.5
            }
        },
        "risk_assessment": {
            "status": "SAFE",
            "risk_score": 16.8,
            "metrics": {
                "wave_height_m": 1.2,
                "wind_speed_kmph": 14.0
            }
        }
    }

    advisory = synthesize_copilot_advisory(mock_payload)
    print("\n--- Generated Advisory (English) ---")
    print(advisory)
    print("------------------------------------\n")

    paragraphs = [p.strip() for p in advisory.split("\n\n") if p.strip()]
    assert len(paragraphs) == 2, f"Expected exactly 2 paragraphs, got {len(paragraphs)}"

    p1, p2 = paragraphs[0], paragraphs[1]

    # Paragraph 1 checks
    assert p1.startswith("Route analysis complete."), "Paragraph 1 must begin with 'Route analysis complete.'"
    assert not p1.startswith("8.7642") and not p1.startswith("Latitude"), "Paragraph 1 must not begin with robotic coordinate readout"
    print("✓ Paragraph 1 (Situational Brief) passes all checks!")

    # Paragraph 2 checks
    assert "Latitude 9.0932° N, Longitude 78.3218° E" in p2, "Paragraph 2 must state exact target coords"
    assert "27.5°C" in p2, "Paragraph 2 must mention SST"
    assert "Yellowfin Tuna" in p2, "Paragraph 2 must mention likely catch species"
    assert "029° NNE" in p2 or "29" in p2, "Paragraph 2 must state bearing"
    assert "North-Northeast" in p2 or "NNE" in p2, "Paragraph 2 must state direction"
    assert "22.6 nautical miles" in p2, "Paragraph 2 must state distance"
    assert "IMBL" in p2, "Paragraph 2 must mention IMBL clearance"
    assert "shipping fairways" in p2 or "shipping lanes" in p2, "Paragraph 2 must mention shipping lanes/fairways"
    assert "ISRO MOSDAC" in p2 and "Copernicus" in p2, "Paragraph 2 must mention ISRO MOSDAC and Copernicus telemetry"
    assert "1.2 meters" in p2, "Paragraph 2 must mention wave height"
    assert "14 km/h" in p2, "Paragraph 2 must mention wind speed"
    print("✓ Paragraph 2 (Integrated Navigational Reasoning) passes all checks!")

    # Forbidden checks
    forbidden_agents = ["Oceanographic Agent", "Spatial Routing Agent", "Meteorological Agent"]
    for agent in forbidden_agents:
        assert agent not in advisory, f"Advisory MUST NOT contain '{agent}'"
    print("✓ Forbidden internal agent names are completely absent!")

    # Format checks: No raw JSON, no final recommendation directive
    assert not advisory.strip().startswith("{") and not advisory.strip().endswith("}"), "Advisory must not be raw JSON"
    assert not advisory.strip().endswith("IMMEDIATE ACTION:"), "Advisory must not end with a robotic directive tag"
    print("✓ Output format contains no raw JSON and no trailing directives!")

    # 3. Test Full API Pipeline (English)
    print("\n--- Test 3: API Pipeline /query (English) ---")
    req_en = QueryRequest(
        query="Where is the nearest fishing spot?",
        persona="FISHERMAN",
        lang="en"
    )
    resp_en = await process_marine_query(req_en)
    body_en = json.loads(resp_en.body.decode("utf-8"))

    reply_text = body_en.get("reply", "")
    assert reply_text, "reply must be populated"
    assert body_en.get("chat_text") == reply_text, "chat_text must match reply"
    assert body_en.get("message") == reply_text, "message must match reply"
    assert body_en.get("response") == reply_text, "response must match reply"
    assert body_en.get("native_advisory_text") == reply_text, "native_advisory_text must match reply"

    api_paras = [p.strip() for p in reply_text.split("\n\n") if p.strip()]
    assert len(api_paras) == 2, f"API reply should have 2 paragraphs, got {len(api_paras)}"
    for agent in forbidden_agents:
        assert agent not in reply_text, f"API reply MUST NOT contain '{agent}'"
    print("✓ API Pipeline (English) verified with 2-paragraph copilot briefing!")

    # 4. Test Full API Pipeline (Tamil Localization)
    print("\n--- Test 4: API Pipeline /query (Tamil Localization) ---")
    req_ta = QueryRequest(
        query="தூத்துக்குடி அருகே இன்று மீன்பிடிக்க செல்லலாமா?",
        persona="FISHERMAN",
        lang="ta"
    )
    resp_ta = await process_marine_query(req_ta)
    body_ta = json.loads(resp_ta.body.decode("utf-8"))

    ta_reply = body_ta.get("reply", "")
    print("Tamil Localized Advisory:\n" + ta_reply[:200] + "...")
    assert len(ta_reply) > 50, "Tamil localized advisory should be populated"
    for agent in forbidden_agents:
        assert agent not in ta_reply, f"Tamil reply MUST NOT contain '{agent}'"
    print("✓ API Pipeline (Tamil) verified with localized copilot briefing!")

    print("\n" + "=" * 80)
    print("ALL VERIFICATIONS PASSED: Gemini Synthesis Engine Prompt Fully Refined!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_verification())
