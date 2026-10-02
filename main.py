"""
main.py - ORCA Agentic Orchestrator & Manager Agent
ISRO SIH Problem Statement 176: Marine Multi-Agent System

Architectural Capabilities:
1. Multi-turn conversational memory resolving context & co-references across turns.
2. Source language tracking (ISO 639-1) for multilingual pipeline coordination.
3. Stakeholder persona classification (FISHERMAN, RESEARCHER, AUTHORITY, MARITIME_OPERATOR).
4. Granular Agentic Execution Plan with task instructions & expected output formats.
5. Urgency assessment and resilient dynamic model fallbacks with rate-limiting backoff.
"""

import os
import json
import re
import sys
import time
import datetime
import warnings
from typing import Dict, Any, List, Optional, Union, Tuple
from dotenv import load_dotenv
from translation_service import IndicTranslationService, detect_language_from_text, SUPPORTED_LANGUAGES
from models import (
    StakeholderPersona,
    PERSONA_ALIASES,
    PERSONA_BADGES,
    resolve_persona,
    classify_persona_intent,
    LocationSource,
    LocationStatus,
    LocationContext,
    OrcaResponse,
)
from gis_agent import GisAgent

# Suppress SDK deprecation warnings for clean console output
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from google.api_core.exceptions import ResourceExhausted
except ImportError:
    class ResourceExhausted(Exception):
        pass

import google.generativeai as genai

# Ensure console supports UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Load environment variables
load_dotenv()

# Instant Fast-Path / Deterministic Mode (Zero-Wait)
FAST_DEMO_MODE = os.getenv("FAST_DEMO_MODE", "true").lower() in ("true", "1", "yes")

# Configure Google Generative AI API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Permitted enumerations for validation
ALLOWED_AGENTS = {
    "WEATHER_AGENT",
    "DISASTER_AGENT",
    "PFZ_AGENT",
    "GIS_AGENT",
    "OCEAN_AGENT",
}

ALLOWED_PERSONAS = {
    "FISHERMAN",
    "MARITIME_AUTHORITY",
    "AUTHORITY",
    "DISASTER_MANAGEMENT",
    "RESEARCHER",
    "MARITIME_OPERATOR",
    "UNKNOWN",
}

ALLOWED_FORMATS = {
    "BINARY_ADVISORY",
    "GEOJSON_POLYGONS",
    "RAW_TIMESERIES",
    "TEXT_SUMMARY",
}

ALLOWED_URGENCIES = {"LOW", "MEDIUM", "HIGH"}


def classify_marine_query_intent(query_text: str) -> Dict[str, Any]:
    """
    Deterministic rule-based maritime intent classification and dynamic agent execution planner.
    Accurately isolates canonical intents:
    - WEATHER: wind, rain, temperature, pressure, fog, gusts, atmospheric forecast
    - OCEAN_CONDITIONS: SST, ocean thermal/salinity properties, general sea conditions
    - WAVES: wave height, swell direction, sea roughness, wave period
    - CURRENT: ocean currents, geostrophic vectors, drift
    - PFZ: potential fishing zone coordinates, thermal-chlorophyll front alignment
    - FISHING: fishing grounds, target catch (tuna, mackerel, sardine), pelagic aggregation
    - CYCLONE: tropical cyclones, depressions, storm track, landfall, storm surge
    - DISASTER: tsunamis, marine emergencies, distress, coastal hazards
    - SAFE_ROUTE: clear passage between origin and destination avoiding hazards
    - ROUTE_PLANNING: passage planning, navigation corridors, fairways, waypoints
    - MARITIME_BOUNDARY: EEZ perimeter, international border clearance, Sri Lanka IMBL
    - EEZ: sovereign economic zone jurisdiction and compliance
    - LOCATION: geographic sector coordinates, port lookup
    - GENERAL_MARINE: broad coastal marine status
    - MULTI_FACTOR_MARINE: multi-domain maritime inquiries
    - UNKNOWN: ambiguous or unclassifiable queries
    - GREETING: conversational greetings and capability questions
    """
    q = (query_text or "").lower().strip()
    # Normalize common speech-to-text transcription errors in coastal maritime context
    # e.g., "wheat there in mumbai" -> "weather in mumbai", "wheat" -> "weather"
    q = re.sub(r"\bwheat\s+there\b", "weather", q)
    q = re.sub(r"\bwheat\b(?=.*\b(?:port|coast|sea|near|mumbai|chennai|kochi|goa|harbor|harbour|weather|condition|waves)\b)", "weather", q)
    q = re.sub(r"\b(how\s+is\s+the|what\s+is\s+the)\s+wheat\b", r"\1 weather", q)

    # 0. Conversational greeting check
    greeting_words = {
        "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
        "vanakkam", "namaste", "who are you", "what can you do", "help", "how are you"
    }
    if q in greeting_words or any(q.startswith(g + " ") for g in ["hi", "hello", "hey", "vanakkam", "namaste"]):
        if not any(k in q for k in ["cyclone", "weather", "fish", "route", "sail", "wave", "wind", "storm", "eez"]):
            return {
                "intent": "GREETING",
                "urgency": "LOW",
                "plan": []
            }

    # 1. MARITIME_BOUNDARY / EEZ (Prioritize jurisdictional / border questions)
    eez_boundary_keywords = [
        "eez", "exclusive economic zone", "boundary", "border", "imbl", "international waters",
        "inside india", "in india's eez", "sovereign waters", "jurisdiction", "sri lanka border",
        "cross border", "border clearance"
    ]
    is_boundary_inquiry = any(k in q for k in eez_boundary_keywords)
    has_explicit_route = (
        ("from " in q and " to " in q)
        or ("between " in q and " and " in q)
        or any(k in q for k in ["route to", "navigate to", "passage to", "sail to", "sail from", "plan route", "safe route", "safe passage"])
    )

    if is_boundary_inquiry and not has_explicit_route:
        sub_intent = "EEZ" if "eez" in q else "MARITIME_BOUNDARY"
        return {
            "intent": sub_intent,
            "urgency": "MEDIUM",
            "plan": [
                {"agent_name": "GIS_AGENT", "task_instructions": "Verify UNCLOS EEZ boundary and international maritime border clearance.", "expected_output_format": "GEOJSON_POLYGONS"},
            ]
        }

    # 1.5 SAFE DEPARTURE WINDOW (Operational sailing weather)
    departure_window_keywords = [
        "departure window", "departure time", "safe departure", "departure timing",
        "when can i sail", "when to sail", "sailing window"
    ]
    if any(k in q for k in departure_window_keywords):
        return {
            "intent": "WEATHER",
            "urgency": "LOW",
            "plan": [
                {"agent_name": "WEATHER_AGENT", "task_instructions": "Check surface wind speed, gusts, visibility, and departure conditions.", "expected_output_format": "TEXT_SUMMARY"},
                {"agent_name": "OCEAN_AGENT", "task_instructions": "Check swell wave height and ocean surface state.", "expected_output_format": "TEXT_SUMMARY"},
                {"agent_name": "DISASTER_AGENT", "task_instructions": "Check active cyclone or squall warnings.", "expected_output_format": "TEXT_SUMMARY"},
            ]
        }

    # 2. ROUTE / SAFE_ROUTE / ROUTE_PLANNING
    is_route_passage = (
        has_explicit_route
        or ("to sri lanka" in q)
        or (re.search(r"\bnavigate\s+to\s+colombo\b|\bsail\s+to\s+colombo\b|\broute\s+to\s+colombo\b|\bpassage\s+to\s+colombo\b", q) is not None)
    )
    if is_route_passage:
        sub_intent = "SAFE_ROUTE" if "safe" in q else "ROUTE_PLANNING"
        locs = extract_locations_from_query(query_text)
        orig = locs.get("origin")
        dest = locs.get("destination")
        if orig and dest:
            gis_instr = f"Generate safe sea passage route from {orig} to {dest}."
        elif dest:
            gis_instr = f"Generate safe sea passage route to {dest}."
        else:
            gis_instr = "Check EEZ boundary, shipping lanes, and navigation corridor."
        return {
            "intent": sub_intent,
            "urgency": "MEDIUM",
            "plan": [
                {"agent_name": "GIS_AGENT", "task_instructions": gis_instr, "expected_output_format": "GEOJSON_POLYGONS"},
                {"agent_name": "WEATHER_AGENT", "task_instructions": "Check wind speed, wave height, and visibility along corridor.", "expected_output_format": "TEXT_SUMMARY"},
                {"agent_name": "OCEAN_AGENT", "task_instructions": "Check SWH altimetry and currents along route.", "expected_output_format": "TEXT_SUMMARY"},
                {"agent_name": "DISASTER_AGENT", "task_instructions": "Check cyclone track and storm hazards along route.", "expected_output_format": "TEXT_SUMMARY"},
            ]
        }

    # 3. CYCLONE & DISASTER
    cyclone_keywords = ["cyclone", "storm", "hurricane", "typhoon", "depression", "storm surge", "storm track"]
    disaster_keywords = ["tsunami", "hazard", "warning", "gale", "squall", "emergency", "sos", "distress", "sinking", "capsiz", "disaster"]
    if any(k in q for k in cyclone_keywords):
        return {
            "intent": "CYCLONE",
            "urgency": "HIGH",
            "plan": [
                {"agent_name": "DISASTER_AGENT", "task_instructions": "Check active tropical cyclones, depressions, and storm trajectory.", "expected_output_format": "TEXT_SUMMARY"},
                {"agent_name": "WEATHER_AGENT", "task_instructions": "Check sustained wind speeds, storm gusts, and pressure drop.", "expected_output_format": "TEXT_SUMMARY"},
            ]
        }
    if any(k in q for k in disaster_keywords):
        return {
            "intent": "DISASTER",
            "urgency": "HIGH",
            "plan": [
                {"agent_name": "DISASTER_AGENT", "task_instructions": "Check coastal disaster warnings, tsunamis, and severe weather hazards.", "expected_output_format": "TEXT_SUMMARY"},
                {"agent_name": "WEATHER_AGENT", "task_instructions": "Check extreme weather parameters and storm indicators.", "expected_output_format": "TEXT_SUMMARY"},
            ]
        }

    # 4. PFZ & FISHING
    pfz_keywords = ["pfz", "potential fishing zone", "potential fishing", "chlorophyll front", "sst front", "thermal front"]
    fishing_keywords = ["fish", "fishing", "catch", "tuna", "mackerel", "sardine", "seerfish", "hilsa", "shoal", "yield", "pelagic", "angler", "anglers"]
    if any(k in q for k in pfz_keywords):
        return {
            "intent": "PFZ",
            "urgency": "LOW",
            "plan": [
                {"agent_name": "PFZ_AGENT", "task_instructions": "Extract verified PFZ zones, SST gradients, and chlorophyll-a fronts.", "expected_output_format": "GEOJSON_POINTS"},
                {"agent_name": "OCEAN_AGENT", "task_instructions": "Check SST and oceanographic sea conditions at fishing sector.", "expected_output_format": "TEXT_SUMMARY"},
                {"agent_name": "WEATHER_AGENT", "task_instructions": "Check operational sea safety and wave heights.", "expected_output_format": "TEXT_SUMMARY"},
            ]
        }
    if any(k in q for k in fishing_keywords):
        return {
            "intent": "FISHING",
            "urgency": "LOW",
            "plan": [
                {"agent_name": "PFZ_AGENT", "task_instructions": "Check pelagic fish habitat suitability and PFZ grounds.", "expected_output_format": "GEOJSON_POINTS"},
                {"agent_name": "OCEAN_AGENT", "task_instructions": "Check oceanographic telemetry and sea surface temperature.", "expected_output_format": "TEXT_SUMMARY"},
                {"agent_name": "WEATHER_AGENT", "task_instructions": "Check marine weather and wave safety for small craft operations.", "expected_output_format": "TEXT_SUMMARY"},
            ]
        }

    # 5. WAVES & CURRENTS & OCEAN_CONDITIONS
    ocean_keywords = [
        "sea surface temperature", "sst", "ocean temperature", "sea condition",
        "sea conditions", "ocean condition", "ocean conditions", "ocean state",
        "marine condition", "marine conditions"
    ]
    wave_keywords = ["wave", "waves", "swell", "sea state", "rough sea", "calm sea", "wave height"]
    current_keywords = [
        "ocean current", "ocean currents", "tidal current", "tidal currents",
        "water current", "surface current", "current velocity", "current direction", "current speed",
        "geostrophic", "drift", "currents"
    ]

    # Evaluate ocean/sea conditions first
    if any(k in q for k in ocean_keywords):
        return {
            "intent": "OCEAN_CONDITIONS",
            "urgency": "LOW",
            "plan": [
                {"agent_name": "OCEAN_AGENT", "task_instructions": "Check sea surface temperature (SST) and physical oceanographic state.", "expected_output_format": "TEXT_SUMMARY"},
                {"agent_name": "WEATHER_AGENT", "task_instructions": "Check surface meteorological parameters.", "expected_output_format": "TEXT_SUMMARY"},
            ]
        }

    # Evaluate ocean water currents (distinguish from temporal adjective "current")
    is_water_current = (
        any(k in q for k in current_keywords)
        or bool(re.search(r"\bcurrent\b(?!\s+(?:sea|weather|condition|conditions|swell|wave|waves|wind|state|situation|telemetry|forecast|time|status))", q))
    )
    if is_water_current:
        return {
            "intent": "CURRENT",
            "urgency": "LOW",
            "plan": [
                {"agent_name": "OCEAN_AGENT", "task_instructions": "Check ocean current velocity and direction.", "expected_output_format": "TEXT_SUMMARY"},
            ]
        }

    if any(k in q for k in wave_keywords):
        return {
            "intent": "WAVES",
            "urgency": "LOW",
            "plan": [
                {"agent_name": "OCEAN_AGENT", "task_instructions": "Check significant wave height (SWH), swell direction, and sea state roughness.", "expected_output_format": "TEXT_SUMMARY"},
                {"agent_name": "WEATHER_AGENT", "task_instructions": "Check surface wind speed driving wind waves.", "expected_output_format": "TEXT_SUMMARY"},
            ]
        }

    # 6. WEATHER / METEOROLOGY
    weather_keywords = [
        "weather", "wind", "rain", "rainfall", "temperature", "forecast", "cloud",
        "precipitation", "gust", "fog", "visibility", "pressure", "barometer", "wheat there"
    ]
    if any(k in q for k in weather_keywords):
        return {
            "intent": "WEATHER",
            "urgency": "LOW",
            "plan": [
                {"agent_name": "WEATHER_AGENT", "task_instructions": "Check wind speed, gusts, rainfall, visibility, and atmospheric pressure.", "expected_output_format": "TEXT_SUMMARY"},
            ]
        }

    # 7. LOCATION
    location_keywords = ["coordinates of", "where is port", "port location", "find coordinates", "port of"]
    if any(k in q for k in location_keywords):
        return {
            "intent": "LOCATION",
            "urgency": "LOW",
            "plan": [
                {"agent_name": "GIS_AGENT", "task_instructions": "Resolve coastal port or maritime sector geographic coordinates.", "expected_output_format": "TEXT_SUMMARY"},
            ]
        }

    # 8. GENERAL_MARINE / UNKNOWN
    general_marine_keywords = ["sea", "ocean", "water", "marine", "sailing", "boating"]
    if any(k in q for k in general_marine_keywords):
        return {
            "intent": "GENERAL_MARINE",
            "urgency": "LOW",
            "plan": [
                {"agent_name": "WEATHER_AGENT", "task_instructions": "Check general coastal weather and wind.", "expected_output_format": "TEXT_SUMMARY"},
                {"agent_name": "OCEAN_AGENT", "task_instructions": "Check sea state roughness and wave height.", "expected_output_format": "TEXT_SUMMARY"},
            ]
        }

    return {
        "intent": "UNKNOWN",
        "urgency": "LOW",
        "plan": [
            {"agent_name": "WEATHER_AGENT", "task_instructions": "Check general maritime weather.", "expected_output_format": "TEXT_SUMMARY"},
            {"agent_name": "OCEAN_AGENT", "task_instructions": "Check wave height and sea conditions.", "expected_output_format": "TEXT_SUMMARY"},
        ]
    }


def normalize_intent_category(intent_str: str, query_str: str = "") -> str:
    """Normalizes arbitrary intent descriptions or LLM outputs to canonical categories."""
    combined = f"{intent_str} {query_str}".lower()
    if any(k in combined for k in ["safe route", "route to", "passage", "navigate to", "fairway", "corridor"]):
        return "SAFE_ROUTE" if "safe" in combined else "ROUTE_PLANNING"
    if any(k in combined for k in ["eez", "exclusive economic zone"]):
        return "EEZ"
    if any(k in combined for k in ["border", "boundary", "imbl", "jurisdiction"]):
        return "MARITIME_BOUNDARY"
    if any(k in combined for k in ["cyclone", "typhoon", "hurricane", "depression"]):
        return "CYCLONE"
    if any(k in combined for k in ["tsunami", "disaster", "hazard", "warning", "emergency", "squall"]):
        return "DISASTER"
    if any(k in combined for k in ["pfz", "potential fishing zone", "potential fishing"]):
        return "PFZ"
    if any(k in combined for k in ["fish", "fishing", "catch", "tuna", "mackerel", "sardine", "angler"]):
        return "FISHING"
    if any(k in combined for k in ["wave", "swell", "sea state", "rough sea"]):
        return "WAVES"
    if any(k in combined for k in ["current", "drift"]):
        return "CURRENT"
    if any(k in combined for k in ["sst", "sea surface temperature", "ocean condition"]):
        return "OCEAN_CONDITIONS"
    if any(k in combined for k in ["weather", "wind", "rain", "temperature", "forecast", "pressure"]):
        return "WEATHER"
    if any(k in combined for k in ["greet", "conversational", "hello", "hi", "welcome"]):
        return "GREETING"
    if any(k in combined for k in ["location", "coordinates", "where is"]):
        return "LOCATION"
    if any(k in combined for k in ["marine", "ocean", "sea"]):
        return "GENERAL_MARINE"
    return "UNKNOWN"


def extract_locations_from_query(query_text: str) -> Dict[str, Any]:
    """
    Extracts explicit location, route origin, and destination from query text.
    Returns:
    {
        "has_route": bool,
        "origin": Optional[str],
        "destination": Optional[str],
        "explicit_location": Optional[str],
        "is_self_location": bool
    }
    """
    if not query_text:
        return {"has_route": False, "origin": None, "destination": None, "explicit_location": None, "is_self_location": False}

    q_lower = query_text.lower()

    # Self-location expressions
    self_loc_patterns = [
        r"\bnear\s+me\b", r"\baround\s+me\b", r"\bhere\b", r"\bmy\s+location\b",
        r"\bcurrent\s+location\b", r"\bcurrent\s+position\b", r"\bmy\s+coordinates\b"
    ]
    is_self_loc = any(re.search(p, q_lower) for p in self_loc_patterns)

    # Route pattern 1: "from X to Y"
    m_route = re.search(r"\bfrom\s+([a-zA-Z\s\u0900-\u0D7F]+?)\s+to\s+([a-zA-Z\s\u0900-\u0D7F]+)", query_text, re.IGNORECASE)
    if m_route:
        raw_orig = m_route.group(1).strip()
        raw_dest = m_route.group(2).strip()
        raw_dest = re.sub(r"\b(and|with|is|safe|weather|tomorrow|today|please|now)\b.*$", "", raw_dest, flags=re.IGNORECASE).strip()
        raw_dest = re.sub(r"[?!.,;:]+$", "", raw_dest).strip()
        return {
            "has_route": True,
            "origin": raw_orig,
            "destination": raw_dest,
            "explicit_location": raw_orig,
            "is_self_location": is_self_loc,
        }

    # Route pattern 2: "between X and Y"
    m_between = re.search(r"\bbetween\s+([a-zA-Z\s\u0900-\u0D7F]+?)\s+and\s+([a-zA-Z\s\u0900-\u0D7F]+)", query_text, re.IGNORECASE)
    if m_between:
        raw_orig = m_between.group(1).strip()
        raw_dest = m_between.group(2).strip()
        raw_dest = re.sub(r"\b(is|safe|weather|tomorrow|today|please|now)\b.*$", "", raw_dest, flags=re.IGNORECASE).strip()
        raw_dest = re.sub(r"[?!.,;:]+$", "", raw_dest).strip()
        return {
            "has_route": True,
            "origin": raw_orig,
            "destination": raw_dest,
            "explicit_location": raw_orig,
            "is_self_location": is_self_loc,
        }

    # Route pattern 3: "navigate/sail/route to Y"
    m_to = re.search(r"\b(?:navigate|sail|route|passage|heading|bound)\s+to\s+([a-zA-Z\s\u0900-\u0D7F]+)", query_text, re.IGNORECASE)
    if m_to:
        raw_dest = m_to.group(1).strip()
        raw_dest = re.sub(r"\b(and|with|is|safe|weather|tomorrow|today|please|now)\b.*$", "", raw_dest, flags=re.IGNORECASE).strip()
        raw_dest = re.sub(r"[?!.,;:]+$", "", raw_dest).strip()
        return {
            "has_route": True,
            "origin": None,
            "destination": raw_dest,
            "explicit_location": raw_dest,
            "is_self_location": is_self_loc,
        }

    # Search for known ports / sectors in query text
    gis = GisAgent()
    known_keys = set(gis.COASTAL_SECTORS.keys()) | set(gis.GAZETTEER_ALIASES.keys())
    sorted_keys = sorted(known_keys, key=len, reverse=True)

    found_locs = []
    for k in sorted_keys:
        if re.search(r"[\u0900-\u0D7F]", k):
            if k in q_lower:
                found_locs.append(k)
        else:
            if re.search(rf"\b{re.escape(k)}\b", q_lower):
                found_locs.append(k)

    if found_locs:
        first_loc = found_locs[0]
        canon = gis.resolve_location(first_loc).get("name", first_loc.title())
        return {
            "has_route": False,
            "origin": None,
            "destination": None,
            "explicit_location": canon,
            "is_self_location": is_self_loc,
        }

    return {
        "has_route": False,
        "origin": None,
        "destination": None,
        "explicit_location": None,
        "is_self_location": is_self_loc,
    }


def resolve_location_context(
    query_text: str,
    device_telemetry: Optional[Dict[str, Any]] = None,
    gis_agent: Optional[Any] = None,
    previous_location: Optional[Any] = None,
) -> Tuple[LocationContext, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Enforces the strict location resolution contract:
    PRIORITY 1: Explicit user location in query (Mumbai, Chennai, Kochi, etc.)
    PRIORITY 2: Route origin and destination
    PRIORITY 3: Device GPS (only when 'near me' or location omitted)
    PRIORITY 3.5: Multi-turn conversational context inheritance (from previous turn)
    PRIORITY 4: None -> LOCATION_REQUIRED
    
    Returns: (location_context, origin_dict, destination_dict)
    """
    if gis_agent is None:
        gis_agent = GisAgent()

    loc_info = extract_locations_from_query(query_text)
    explicit_loc = loc_info.get("explicit_location")
    has_route = loc_info.get("has_route", False)
    raw_origin = loc_info.get("origin")
    raw_dest = loc_info.get("destination")
    is_self_loc = loc_info.get("is_self_location", False)

    # Check device GPS coordinates
    dev_lat = None
    dev_lon = None
    if isinstance(device_telemetry, dict):
        raw_lat = device_telemetry.get("latitude")
        raw_lon = device_telemetry.get("longitude")
        if raw_lat is not None and raw_lon is not None:
            try:
                dev_lat = float(raw_lat)
                dev_lon = float(raw_lon)
            except (ValueError, TypeError):
                pass

    origin_dict = None
    dest_dict = None

    # Priority 2: Route (if query indicates a passage between locations)
    if has_route:
        if raw_dest:
            dest_res = gis_agent.resolve_location(raw_dest)
            dest_dict = {
                "name": dest_res.get("name", raw_dest),
                "latitude": dest_res.get("lat"),
                "longitude": dest_res.get("lon"),
            }
        if raw_origin:
            orig_res = gis_agent.resolve_location(raw_origin)
            origin_dict = {
                "name": orig_res.get("name", raw_origin),
                "latitude": orig_res.get("lat"),
                "longitude": orig_res.get("lon"),
            }
        elif dev_lat is not None and dev_lon is not None:
            origin_dict = {
                "name": f"GPS ({dev_lat:.4f}, {dev_lon:.4f})",
                "latitude": dev_lat,
                "longitude": dev_lon,
            }

        primary_name = origin_dict.get("name") if origin_dict else (dest_dict.get("name") if dest_dict else "Route Corridor")
        primary_lat = origin_dict.get("latitude") if origin_dict else (dest_dict.get("latitude") if dest_dict else None)
        primary_lon = origin_dict.get("longitude") if origin_dict else (dest_dict.get("longitude") if dest_dict else None)

        lc = LocationContext(
            source=LocationSource.ROUTE,
            status=LocationStatus.RESOLVED if primary_lat is not None else LocationStatus.LOCATION_REQUIRED,
            name=primary_name,
            latitude=primary_lat,
            longitude=primary_lon,
            origin=origin_dict,
            destination=dest_dict,
        )
        return lc, origin_dict, dest_dict

    # Priority 1: Explicit User Location in Query (supersedes device GPS)
    if explicit_loc and not is_self_loc:
        res = gis_agent.resolve_location(explicit_loc)
        if not res.get("is_unknown"):
            lc = LocationContext(
                source=LocationSource.EXPLICIT_QUERY,
                status=LocationStatus.RESOLVED,
                name=res.get("name", explicit_loc),
                latitude=res.get("lat"),
                longitude=res.get("lon"),
            )
            return lc, None, None
        else:
            lc = LocationContext(
                source=LocationSource.EXPLICIT_QUERY,
                status=LocationStatus.LOCATION_REQUIRED,
                name=explicit_loc,
                latitude=None,
                longitude=None,
            )
            return lc, None, None

    # Priority 3: Device GPS (when 'near me' or omitted)
    if dev_lat is not None and dev_lon is not None:
        gps_name = f"GPS ({dev_lat:.4f}, {dev_lon:.4f})"
        nearest_sector = None
        min_dist = float("inf")
        import math
        for s_name, s_coords in gis_agent.COASTAL_SECTORS.items():
            d = math.hypot(dev_lat - s_coords["lat"], dev_lon - s_coords["lon"])
            if d < min_dist:
                min_dist = d
                nearest_sector = s_coords.get("name") or s_name.title()

        display_name = nearest_sector if min_dist < 1.0 else gps_name
        lc = LocationContext(
            source=LocationSource.DEVICE_GPS,
            status=LocationStatus.RESOLVED,
            name=display_name,
            latitude=dev_lat,
            longitude=dev_lon,
        )
        return lc, None, None

    # Priority 3.5: Multi-Turn Conversational Memory (Inherit location from previous turn)
    if previous_location:
        prev_name = (
            previous_location.get("name")
            if isinstance(previous_location, dict)
            else (previous_location[0] if isinstance(previous_location, list) and previous_location else str(previous_location))
        )
        if prev_name:
            res_prev = gis_agent.resolve_location(prev_name)
            if not res_prev.get("is_unknown"):
                lc = LocationContext(
                    source=LocationSource.USER_QUERY,
                    status=LocationStatus.RESOLVED,
                    name=res_prev.get("name", prev_name),
                    latitude=res_prev.get("lat"),
                    longitude=res_prev.get("lon"),
                )
                return lc, None, None

    # Priority 4: No location specified and no GPS available (Default = NONE -> LOCATION_REQUIRED)
    lc = LocationContext(
        source=LocationSource.DEFAULT,
        status=LocationStatus.LOCATION_REQUIRED,
        name=None,
        latitude=None,
        longitude=None,
    )
    return lc, None, None


class ManagerAgent:
    """
    ManagerAgent acts as the central Agentic Orchestrator for Project ORCA.
    Maintains multi-turn conversational history, detects stakeholder personas,
    tracks source languages, and outputs structured execution plans.
    """

    # Class-level cache for model resolution
    _cached_model = None
    _cached_model_name = None

    def __init__(self):
        # Session-indexed conversational memory {session_id: [turns]}
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}
        # Legacy single session history reference
        self.chat_history: List[Dict[str, Any]] = []

        # JSON output configuration
        self.generation_config = {"response_mime_type": "application/json"}

        # Automatic ISRO MOSDAC cache freshness check (non-blocking)
        try:
            from shadow_cache_worker import ensure_latest_mosdac_cache
            ensure_latest_mosdac_cache(max_age_hours=72.0, non_blocking=True)
        except Exception:
            pass

        # Comprehensive System Prompt for the ORCA Agentic Orchestrator
        self.system_prompt = (
            "You are the central Agentic Orchestrator for ORCA, a Marine Multi-Agent Intelligence "
            "System built for ISRO SIH Problem Statement 176.\n"
            "Your role is to orchestrate queries from diverse coastal stakeholders (artisanal fishermen, "
            "trawler captains, oceanographic researchers, coast guard/disaster authorities, and port operators).\n\n"
            "Key Responsibilities:\n"
            "1. CONVERSATIONAL MEMORY & CONTEXT RESOLUTION:\n"
            "   Review conversational history to resolve missing context, pronouns, or references across turns. "
            "For example, if turn 1 discusses 'Mumbai' and turn 2 asks 'is it safe to sail there tomorrow?', "
            "resolve 'there' to 'Mumbai'.\n\n"
            "2. STAKEHOLDER PERSONA CLASSIFICATION:\n"
            "   Classify the user into one of: 'FISHERMAN', 'MARITIME_AUTHORITY', 'DISASTER_MANAGEMENT', 'RESEARCHER', 'MARITIME_OPERATOR', 'UNKNOWN'.\n\n"
            "3. LANGUAGE IDENTIFICATION:\n"
            "   Identify the ISO 639-1 source language code (e.g., 'en' for English, 'ta' for Tamil, 'hi' for Hindi, "
            "'te' for Telugu, 'ml' for Malayalam, 'bn' for Bengali).\n\n"
            "4. GRANULAR AGENTIC EXECUTION PLAN:\n"
            "   Do not merely list agents. Produce a sequenced execution plan where each step targets one of the "
            "5 specialized downstream agents with clear, actionable task instructions and the expected output format:\n"
            "   - WEATHER_AGENT: Wind speed/direction, gusts, rainfall, visibility, squalls.\n"
            "   - DISASTER_AGENT: Cyclones, cyclone trajectory/movement/heading/landfall forecast, undersea earthquakes, tsunami warnings, storm surges, emergency shelter routing.\n"
            "   - PFZ_AGENT: Potential Fishing Zones, chlorophyll-a fronts, sea surface temperature anomalies, fish school aggregations, and species-specific habitat predictions (e.g. Tuna, Mackerel, Sardine, Seerfish, Hilsa, Trevally, Ribbonfish). When the user asks for a specific fish, explicitly specify the target species in task_instructions.\n"
            "   - GIS_AGENT: Spatial boundary checks, GPS coordinates, International Maritime Boundary Line (IMBL), EEZ, navigation routing.\n"
            "   - OCEAN_AGENT: Wave heights, ocean currents, tides, salinity, sea state roughness, SST.\n\n"
            "   Expected output formats:\n"
            "   - 'BINARY_ADVISORY': Safe/Unsafe or Go/No-Go binary decisions.\n"
            "   - 'GEOJSON_POLYGONS': Coordinate boundaries, PFZ polygons, exclusion zones.\n"
            "   - 'RAW_TIMESERIES': Numerical wave/wind forecasts, hourly data curves.\n"
            "   - 'TEXT_SUMMARY': Bulleted summaries, natural language advisory.\n\n"
            "5. URGENCY ASSESSMENT:\n"
            "   - 'HIGH': Imminent cyclones, rough seas warning, distress, border crossing alerts.\n"
            "   - 'MEDIUM': Changing weather, marginal wave conditions, precautionary notices.\n"
            "   - 'LOW': Routine fishing expeditions, research data retrieval, calm sea queries.\n\n"
            "6. CONVERSATIONAL & GREETING QUERIES:\n"
            "   If the user's input is a general greeting or introduction (e.g. 'hi', 'hello', 'good morning', 'vanakkam', 'namaste', 'who are you', 'help'):\n"
            "   Set 'analyzed_intent' to 'GREETING' or 'CONVERSATIONAL', set urgency_level to 'LOW', and provide an EMPTY execution plan []\n"
            "   because no downstream domain agents need to be dispatched.\n\n"
            "STRICT JSON OUTPUT SCHEMA:\n"
            "You must output ONLY a valid JSON object matching this exact schema:\n"
            "{\n"
            '  "analyzed_intent": "string",\n'
            '  "user_persona": "FISHERMAN | MARITIME_AUTHORITY | DISASTER_MANAGEMENT | RESEARCHER | MARITIME_OPERATOR | UNKNOWN",\n'
            '  "source_language_code": "ISO 639-1 code (e.g., en, ta, hi, te)",\n'
            '  "location_entities": ["string"],\n'
            '  "time_entities": ["string"],\n'
            '  "execution_plan": [\n'
            "    {\n"
            '      "agent_name": "WEATHER_AGENT | DISASTER_AGENT | PFZ_AGENT | GIS_AGENT | OCEAN_AGENT",\n'
            '      "task_instructions": "Specific instructions for what data the agent must fetch and process",\n'
            '      "expected_output_format": "BINARY_ADVISORY | GEOJSON_POLYGONS | RAW_TIMESERIES | TEXT_SUMMARY"\n'
            "    }\n"
            "  ],\n"
            '  "urgency_level": "LOW | MEDIUM | HIGH"\n'
            "}\n\n"
            "Rules:\n"
            "- Output raw JSON only. Do not include markdown formatting or commentary outside the JSON."
        )

        # Primary model initialization: gemini-1.5-flash-latest with fallback
        if ManagerAgent._cached_model is not None:
            self.model = ManagerAgent._cached_model
            self.model_name = ManagerAgent._cached_model_name
            return

        self.model_name = "gemini-1.5-flash-latest"
        self._init_model_with_fallback()

        ManagerAgent._cached_model = self.model
        ManagerAgent._cached_model_name = self.model_name

    def _init_model_with_fallback(self):
        """
        Initializes gemini-1.5-flash-latest. If unavailable or deprecated,
        dynamically falls back to active models supporting generateContent.
        """
        available_models = []
        if os.getenv("GEMINI_API_KEY"):
            try:
                available_models = [
                    m.name.replace("models/", "")
                    for m in genai.list_models()
                    if "generateContent" in m.supported_generation_methods
                ]
            except Exception:
                available_models = []

        candidates = [
            "gemma-4-26b-a4b-it",  # Preferred Gemma 4B active parameter model
            "gemma-4-31b-it",
            "gemini-1.5-flash",
            "gemini-2.0-flash",
            "gemini-flash-latest",
        ]

        # Prioritize available models
        prioritized = [c for c in candidates if c in available_models or not available_models]
        if not prioritized:
            prioritized = candidates

        for candidate in prioritized:
            try:
                self.model = genai.GenerativeModel(
                    model_name=candidate,
                    system_instruction=self.system_prompt,
                    generation_config=self.generation_config,
                )
                self.model_name = candidate
                return
            except Exception:
                continue

        # Default fallback
        self.model_name = "gemini-flash-latest"
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_prompt,
            generation_config=self.generation_config,
        )

    def _parse_mobile_payload(self, input_data: Any) -> Dict[str, Any]:
        """
        Parses either a JSON string, a dictionary matching the mobile schema,
        or a legacy plain string into a standardized mobile payload dictionary.
        Preserves original_query and creates effective_query following language processing.
        """
        if isinstance(input_data, str):
            trimmed = input_data.strip()
            if trimmed.startswith("{") and trimmed.endswith("}"):
                try:
                    parsed = json.loads(trimmed)
                    if isinstance(parsed, dict) and (
                        "session_id" in parsed
                        or "device_telemetry" in parsed
                        or "user_input" in parsed
                        or "query" in parsed
                        or "query_text" in parsed
                        or "raw_text" in parsed
                    ):
                        input_data = parsed
                except Exception:
                    pass

        if isinstance(input_data, dict):
            session_id = str(input_data.get("session_id", "sess_default"))
            client_timestamp = str(
                input_data.get(
                    "client_timestamp",
                    datetime.datetime.now(datetime.timezone.utc).isoformat(),
                )
            )
            user_context = input_data.get("user_context", {})
            if not isinstance(user_context, dict):
                user_context = {}
            raw_telemetry = input_data.get("device_telemetry")
            device_telemetry = raw_telemetry if isinstance(raw_telemetry, dict) else {}
            raw_user_input = input_data.get("user_input")
            user_input = raw_user_input if isinstance(raw_user_input, dict) else {}

            raw_text = (
                user_input.get("raw_text")
                or input_data.get("raw_text")
                or input_data.get("query")
                or input_data.get("query_text")
                or input_data.get("text")
                or ""
            )
            raw_audio_base64 = user_input.get("raw_audio_base64") or input_data.get("raw_audio_base64")
            input_type = str(
                user_input.get("input_type")
                or input_data.get("input_type")
                or ("TEXT" if raw_text else ("AUDIO" if raw_audio_base64 else "TEXT"))
            ).upper()

            raw_lat = device_telemetry.get("latitude") if device_telemetry else (input_data.get("latitude", input_data.get("lat")))
            raw_lon = device_telemetry.get("longitude") if device_telemetry else (input_data.get("longitude", input_data.get("lon")))
            lat = float(raw_lat) if raw_lat is not None else None
            lon = float(raw_lon) if raw_lon is not None else None

            # 1. Capture exact original query
            original_query = (
                user_input.get("original_query")
                or user_context.get("original_query")
                or input_data.get("original_query")
                or (str(raw_text).strip() if raw_text else "")
            ).strip()

            # 2. Determine effective_query via language processing / translation if required
            given_effective = (
                input_data.get("effective_query")
                or user_input.get("effective_query")
                or user_input.get("english_query")
                or input_data.get("english_query")
            )

            if given_effective and str(given_effective).strip():
                effective_query = str(given_effective).strip()
            elif original_query:
                detected_lang = detect_language_from_text(original_query)
                if detected_lang != "en":
                    try:
                        translated = IndicTranslationService.translate_to_english(original_query, detected_lang)
                        effective_query = str(translated).strip() if (translated and str(translated).strip()) else original_query
                    except Exception:
                        effective_query = original_query
                else:
                    effective_query = original_query
            elif input_type == "AUDIO" and raw_audio_base64:
                if lat is not None and lon is not None:
                    effective_query = f"Voice Request: Identify optimal fishing zones and maritime sea safety near coordinates ({lat:.4f}, {lon:.4f})"
                else:
                    effective_query = "Voice Request: Identify optimal fishing zones and maritime sea safety"
            else:
                if lat is not None and lon is not None:
                    effective_query = f"Where can I go fishing today near coordinates ({lat:.4f}, {lon:.4f}) and is it safe to sail?"
                else:
                    effective_query = "Where can I go fishing today and is it safe to sail?"

            if not original_query:
                original_query = effective_query

            return {
                "session_id": session_id,
                "client_timestamp": client_timestamp,
                "user_context": user_context,
                "device_telemetry": {
                    "latitude": lat,
                    "longitude": lon,
                    "gps_accuracy_meters": float(device_telemetry.get("gps_accuracy_meters", 4.5)) if device_telemetry.get("gps_accuracy_meters") is not None else 4.5,
                    "speed_knots": float(device_telemetry.get("speed_knots", 0.0)) if device_telemetry.get("speed_knots") is not None else 0.0,
                    "heading_degrees": float(device_telemetry.get("heading_degrees", 0.0)) if device_telemetry.get("heading_degrees") is not None else 0.0,
                },
                "user_input": {
                    "input_type": input_type,
                    "raw_text": raw_text or original_query,
                    "raw_audio_base64": raw_audio_base64,
                    "original_query": original_query,
                    "effective_query": effective_query,
                    "english_query": effective_query,
                },
                "original_query": original_query,
                "effective_query": effective_query,
                "query": original_query,
            }

        # Legacy plain string query
        query_str = str(input_data or "").strip()
        original_query = query_str
        if original_query:
            detected_lang = detect_language_from_text(original_query)
            if detected_lang != "en":
                try:
                    translated = IndicTranslationService.translate_to_english(original_query, detected_lang)
                    effective_query = str(translated).strip() if (translated and str(translated).strip()) else original_query
                except Exception:
                    effective_query = original_query
            else:
                effective_query = original_query
        else:
            effective_query = "Where can I go fishing today and is it safe to sail?"
            original_query = effective_query

        return {
            "session_id": "sess_default",
            "client_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "user_context": {"persona": "FISHERMAN"},
            "device_telemetry": {
                "latitude": None,
                "longitude": None,
                "gps_accuracy_meters": 4.5,
                "speed_knots": 0.0,
                "heading_degrees": 0.0,
            },
            "user_input": {
                "input_type": "TEXT",
                "raw_text": original_query,
                "raw_audio_base64": None,
                "original_query": original_query,
                "effective_query": effective_query,
                "english_query": effective_query,
            },
            "original_query": original_query,
            "effective_query": effective_query,
            "query": original_query,
        }

    def analyze_query(self, query: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Processes query or mobile app JSON payload with session-indexed multi-turn
        conversational memory, integrates device GPS telemetry, and returns the
        comprehensive Agentic Execution Plan.
        """
        mobile_payload = self._parse_mobile_payload(query)
        session_id = mobile_payload["session_id"]
        client_timestamp = mobile_payload["client_timestamp"]
        telemetry = mobile_payload["device_telemetry"]
        user_context = mobile_payload["user_context"]

        # Define effective_query and original_query BEFORE location extraction
        original_query = mobile_payload.get("original_query", "")
        effective_query = mobile_payload.get("effective_query", "")
        if not effective_query:
            effective_query = original_query if original_query else "Where can I go fishing today and is it safe to sail?"
        if not original_query:
            original_query = effective_query

        lat = telemetry.get("latitude") if isinstance(telemetry, dict) else None
        lon = telemetry.get("longitude") if isinstance(telemetry, dict) else None
        gps_str = f"{lat:.4f},{lon:.4f}" if (lat is not None and lon is not None) else None

        loc_extracted = extract_locations_from_query(effective_query)
        deterministic_locations = []
        if loc_extracted.get("explicit_location"):
            deterministic_locations = [loc_extracted["explicit_location"]]
        elif loc_extracted.get("has_route"):
            deterministic_locations = [l for l in [loc_extracted.get("origin"), loc_extracted.get("destination")] if l]
        elif gps_str:
            deterministic_locations = [gps_str]

        # Multi-turn conversational memory context & inheritance
        session_history = self.sessions.get(session_id, [])
        prev_loc = None
        prev_intent = None
        if session_history:
            prev_turn = session_history[-1]
            prev_loc = prev_turn.get("location") or (prev_turn.get("locations")[0] if prev_turn.get("locations") else None)
            prev_intent = prev_turn.get("intent")

        if not deterministic_locations and prev_loc:
            deterministic_locations = [prev_loc]

        deterministic_info = classify_marine_query_intent(effective_query)
        deterministic_intent = deterministic_info["intent"]
        deterministic_plan = deterministic_info["plan"]
        deterministic_urgency = deterministic_info["urgency"]

        # Temporal follow-up intent inheritance (e.g., "What about tomorrow?")
        from decision_engine import detect_temporal_intent
        temp_chk = detect_temporal_intent(effective_query)
        if temp_chk.get("has_temporal_intent") and deterministic_intent in ("WEATHER", "GENERAL_MARINE", "UNKNOWN"):
            if prev_intent and prev_intent not in ("GREETING", "UNKNOWN"):
                deterministic_intent = prev_intent
                inherited_info = classify_marine_query_intent(f"{prev_intent} {effective_query}")
                deterministic_plan = inherited_info["plan"]
                deterministic_urgency = inherited_info["urgency"]

        def record_turn(intent_val, locs_val):
            session_history.append(
                {
                    "query": effective_query,
                    "original_query": original_query,
                    "intent": intent_val,
                    "locations": locs_val,
                    "location": locs_val[0] if locs_val else None,
                }
            )
            self.sessions[session_id] = session_history[-5:]
            self.chat_history = self.sessions[session_id]

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or FAST_DEMO_MODE:
            record_turn(deterministic_intent, deterministic_locations)
            return {
                "session_id": session_id,
                "client_timestamp": client_timestamp,
                "device_telemetry": telemetry,
                "analyzed_intent": deterministic_intent,
                "user_persona": user_context.get("persona", "FISHERMAN"),
                "source_language_code": "en",
                "location_entities": deterministic_locations,
                "time_entities": ["today"],
                "execution_plan": deterministic_plan,
                "urgency_level": deterministic_urgency,
                "original_query": original_query,
                "effective_query": effective_query,
                "query": original_query or effective_query,
            }

        # Format session-specific conversational memory context
        history_text = ""
        if session_history:
            history_lines = [f"Conversational History for Session [{session_id}]:"]
            for idx, turn in enumerate(session_history, 1):
                prev_q = turn.get("query", "")
                prev_i = turn.get("intent", "")
                prev_ls = turn.get("locations", [])
                history_lines.append(
                    f"Turn {idx}: User asked: \"{prev_q}\" | Resolved Intent: \"{prev_i}\" | Locations: {prev_ls}"
                )
            history_text = "\n".join(history_lines) + "\n\n"

        app_persona = user_context.get("persona")
        persona_hint = f"Mobile Client Specified Persona: {app_persona}\n" if app_persona else ""

        gps_info_str = f"Latitude {lat:.4f}, Longitude {lon:.4f}" if (lat is not None and lon is not None) else "GPS Unavailable (None)"
        prompt = (
            f"{history_text}"
            f"Vessel Device Telemetry: {gps_info_str}, Speed: {telemetry['speed_knots']} knots, Heading: {telemetry['heading_degrees']} deg.\n"
            f"{persona_hint}"
            f"Current User Request: \"{effective_query}\"\n\n"
            f"Note: If the user mentions an explicit coastal location (e.g., 'Mumbai', 'Chennai', 'Kochi'), use that explicit location. If the user refers to 'here', 'current position', or 'near me' and GPS is available, use GPS coordinates '{gps_str}' as the location entity.\n"
            "Analyze the current query in light of telemetry & conversational history and output the JSON routing plan."
        )

        response = None
        if self.model and api_key:
            try:
                response = self.model.generate_content(
                    prompt,
                    request_options={"timeout": 15.0, "retry": None}
                )
            except Exception as e:
                err_str = str(e)
                if "504" in err_str or "timeout" in err_str.lower() or "deadline" in err_str.lower():
                    print(f"[ManagerAgent Notice] LLM query planning timeout (15s limit reached). Immediate fallback to deterministic multi-agent routing plan.")
                elif "429" in err_str or "quota" in err_str.lower():
                    print(f"[ManagerAgent Notice] Gemini quota/rate limit reached (429). Immediate fallback to deterministic multi-agent routing plan.")
                else:
                    print(f"[ManagerAgent Notice] Query planning fallback: {err_str[:120]}")
                record_turn(deterministic_intent, deterministic_locations)
                return {
                    "session_id": session_id,
                    "client_timestamp": client_timestamp,
                    "device_telemetry": telemetry,
                    "analyzed_intent": deterministic_intent,
                    "user_persona": user_context.get("persona", "FISHERMAN"),
                    "source_language_code": "en",
                    "location_entities": deterministic_locations,
                    "time_entities": ["today"],
                    "execution_plan": deterministic_plan,
                    "urgency_level": deterministic_urgency,
                }

        if response is None:
            record_turn(deterministic_intent, deterministic_locations)
            return {
                "session_id": session_id,
                "client_timestamp": client_timestamp,
                "device_telemetry": telemetry,
                "analyzed_intent": deterministic_intent,
                "user_persona": user_context.get("persona", "FISHERMAN"),
                "source_language_code": "en",
                "location_entities": deterministic_locations,
                "time_entities": ["today"],
                "execution_plan": deterministic_plan,
                "urgency_level": deterministic_urgency,
            }

        try:
            raw_text = response.text.strip()

            # Parse JSON: handle direct JSON, markdown code blocks, and nested object extraction
            parsed_data = None
            try:
                cand = json.loads(raw_text)
                if isinstance(cand, dict) and ("analyzed_intent" in cand or "execution_plan" in cand):
                    parsed_data = cand
            except Exception:
                pass

            if parsed_data is None:
                code_blocks = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw_text, re.IGNORECASE)
                for block in reversed(code_blocks):
                    try:
                        cand = json.loads(block.strip())
                        if isinstance(cand, dict) and ("analyzed_intent" in cand or "execution_plan" in cand):
                            parsed_data = cand
                            break
                    except Exception:
                        continue

            if parsed_data is None:
                end_idx = raw_text.rfind("}")
                if end_idx != -1:
                    start_indices = [i for i, ch in enumerate(raw_text) if ch == "{" and i < end_idx]
                    for start in start_indices:
                        candidate = raw_text[start : end_idx + 1].strip()
                        if '"analyzed_intent"' in candidate or '"execution_plan"' in candidate:
                            try:
                                cand = json.loads(candidate)
                                if isinstance(cand, dict) and ("analyzed_intent" in cand or "execution_plan" in cand):
                                    parsed_data = cand
                                    break
                            except Exception:
                                continue

            if parsed_data is None:
                raise json.JSONDecodeError(
                    "Could not extract valid JSON object from response", raw_text, 0
                )

            # Sanitize and validate fields
            raw_intent = str(parsed_data.get("analyzed_intent", deterministic_intent))
            intent = normalize_intent_category(raw_intent, effective_query)
            persona = str(parsed_data.get("user_persona", "UNKNOWN")).upper()
            if persona not in ALLOWED_PERSONAS:
                persona = "UNKNOWN"

            lang_code = str(parsed_data.get("source_language_code", "en")).lower()

            raw_locations = parsed_data.get("location_entities", [])
            location_entities = (
                [str(loc).strip() for loc in raw_locations if str(loc).strip()]
                if isinstance(raw_locations, list)
                else []
            )

            raw_times = parsed_data.get("time_entities", [])
            time_entities = (
                [str(t).strip() for t in raw_times if str(t).strip()]
                if isinstance(raw_times, list)
                else []
            )

            urgency = str(parsed_data.get("urgency_level", deterministic_urgency)).upper()
            if urgency not in ALLOWED_URGENCIES:
                urgency = deterministic_urgency

            # Validate execution plan
            raw_plan = parsed_data.get("execution_plan", [])
            validated_plan = []
            if isinstance(raw_plan, list):
                for item in raw_plan:
                    if isinstance(item, dict):
                        agent_name = str(item.get("agent_name", "")).upper()
                        if agent_name in ALLOWED_AGENTS:
                            output_fmt = str(
                                item.get("expected_output_format", "TEXT_SUMMARY")
                            ).upper()
                            if output_fmt not in ALLOWED_FORMATS:
                                output_fmt = "TEXT_SUMMARY"

                            validated_plan.append(
                                {
                                    "agent_name": agent_name,
                                    "task_instructions": str(
                                        item.get(
                                            "task_instructions",
                                            "Fetch and process marine data",
                                        )
                                    ),
                                    "expected_output_format": output_fmt,
                                }
                            )

            if not validated_plan:
                validated_plan = deterministic_plan

            # If persona was specified by mobile app user_context, prioritize it
            if app_persona and str(app_persona).upper() in ALLOWED_PERSONAS:
                persona = str(app_persona).upper()

            # Ensure location entities defaults to vessel GPS if empty
            if not location_entities:
                location_entities = [gps_str]

            final_result = {
                "session_id": session_id,
                "client_timestamp": client_timestamp,
                "device_telemetry": telemetry,
                "analyzed_intent": intent,
                "user_persona": persona,
                "source_language_code": lang_code,
                "location_entities": location_entities,
                "time_entities": time_entities,
                "execution_plan": validated_plan,
                "urgency_level": urgency,
                "original_query": original_query,
                "effective_query": effective_query,
                "query": original_query or effective_query,
            }

            # Update session-specific conversational memory
            session_history.append(
                {
                    "query": effective_query,
                    "original_query": original_query,
                    "intent": intent,
                    "locations": location_entities,
                    "location": location_entities[0] if location_entities else None,
                }
            )
            self.sessions[session_id] = session_history[-5:]
            self.chat_history = self.sessions[session_id]

            return final_result

        except json.JSONDecodeError as jde:
            print(f"[Error] Failed to parse model response as JSON: {jde}")
            return {
                "session_id": session_id,
                "client_timestamp": client_timestamp,
                "device_telemetry": telemetry,
                "analyzed_intent": "Error parsing JSON response",
                "user_persona": user_context.get("persona", "FISHERMAN"),
                "source_language_code": "en",
                "location_entities": [gps_str],
                "time_entities": [],
                "execution_plan": [],
                "urgency_level": "LOW",
                "error": f"Raw text could not be parsed: {jde}",
            }
        except Exception as e:
            print(f"[Error] Exception occurred during output processing: {e}")
            return {
                "analyzed_intent": "Error processing output",
                "user_persona": "UNKNOWN",
                "source_language_code": "en",
                "location_entities": [],
                "time_entities": [],
                "execution_plan": [],
                "urgency_level": "LOW",
                "error": str(e),
            }


def display_orchestration_card(result: Dict[str, Any]):
    """
    Renders a clean, highly structured, colorized terminal card for the orchestration plan.
    """
    intent = result.get("analyzed_intent", "N/A")
    persona = result.get("user_persona", "UNKNOWN")
    lang_code = result.get("source_language_code", "en")
    locations = result.get("location_entities", [])
    times = result.get("time_entities", [])
    urgency = result.get("urgency_level", "LOW")
    plan = result.get("execution_plan", [])

    # Urgency styling badge
    urgency_badges = {
        "HIGH": "🔴 HIGH URGENCY (Critical Marine Hazard / Alert)",
        "MEDIUM": "🟡 MEDIUM URGENCY (Changing Ocean Conditions)",
        "LOW": "🟢 LOW URGENCY (Routine Maritime Operations)",
    }
    urgency_str = urgency_badges.get(urgency, f"⚪ {urgency}")

    # Persona styling
    p_enum = resolve_persona(persona)
    if p_enum and p_enum.value in PERSONA_BADGES:
        persona_str = PERSONA_BADGES[p_enum.value]
    elif persona in PERSONA_BADGES:
        persona_str = PERSONA_BADGES[persona]
    else:
        persona_str = f"👤 {persona}"

    print("\n" + "=" * 70)
    print("             ORCA AGENTIC ORCHESTRATION REPORT")
    print("=" * 70)
    print(f"👤 Persona          : {persona_str}")
    print(f"🌐 Language Code    : {lang_code.upper()} (ISO 639-1)")
    print(f"🎯 Analyzed Intent  : {intent}")
    print(f"📍 Location Entities: {', '.join(locations) if locations else 'None identified'}")
    print(f"🕒 Time Entities    : {', '.join(times) if times else 'None identified'}")
    print(f"🚨 Urgency Level    : {urgency_str}")
    print("-" * 70)
    print("📋 AGENTIC EXECUTION PLAN:")
    print("──────────────────────────────────────────────────────────────────────")

    if plan:
        for idx, step in enumerate(plan, 1):
            agent = step.get("agent_name", "AGENT")
            task = step.get("task_instructions", "N/A")
            fmt = step.get("expected_output_format", "TEXT_SUMMARY")
            print(f"  Step {idx}. [{agent}]")
            print(f"     • Task Instructions     : {task}")
            print(f"     • Expected Output Format: {fmt}")
            if idx < len(plan):
                print()
    else:
        print("  None specified.")

    print("──────────────────────────────────────────────────────────────────────")
    print("\n[+] Structured Routing Payload (JSON):")
    print(json.dumps(result, indent=2))
    print("=" * 70 + "\n")


def process_marine_request(
    request_data: Dict[str, Any],
    manager: Optional[ManagerAgent] = None,
    bhashini: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Main Multi-Modal Routing Controller for Project ORCA (ISRO SIH 176).
    Handles:
    - Audio vs. Text input detection: is_voice = request_data["user_input"]["input_type"] == "AUDIO"
    - Regional language to English translation via IndicTranslationService for ManagerAgent reasoning
    - ManagerAgent orchestration & domain agent task dispatching
    - DecisionEngine synthesis (Risk, Safe Route, Collision Clearance, Solar Energy)
    - Reverse translation of English advisory to user's detected language
    - Standardized ORCA Multi-Modal JSON output packaging
    """
    if manager is None:
        manager = ManagerAgent()

    # Step 1: Check incoming JSON for input_type (ISRO SIH 176 architecture)
    user_input = request_data.get("user_input", {})
    # Strict check as specified: is_voice = request_data["user_input"]["input_type"] == "AUDIO"
    is_voice = user_input.get("input_type") == "AUDIO" if isinstance(user_input, dict) else False

    session_id = request_data.get("session_id", "sess_marine_ui")
    client_timestamp = request_data.get(
        "client_timestamp",
        datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    user_context = request_data.get("user_context", {})
    if not isinstance(user_context, dict):
        user_context = {}
    device_telemetry = request_data.get("device_telemetry", {})
    raw_persona = user_context.get("persona")
    resolved_persona_enum = resolve_persona(raw_persona)
    persona = resolved_persona_enum.value if resolved_persona_enum else None

    transcribed_text = None
    lat_val = device_telemetry.get("latitude") if isinstance(device_telemetry, dict) else None
    lon_val = device_telemetry.get("longitude") if isinstance(device_telemetry, dict) else None
    lat = float(lat_val) if lat_val is not None else None
    lon = float(lon_val) if lon_val is not None else None
    gps_str = f"{lat:.4f},{lon:.4f}" if (lat is not None and lon is not None) else None

    # Step 2: Inbound Language / Audio Processing
    original_user_query = (
        user_input.get("original_query")
        or user_context.get("original_query")
        or user_input.get("raw_text")
        or ""
    ).strip()

    if is_voice:
        raw_audio_base64 = user_input.get("raw_audio_base64", "")
        client_lang_pref = (
            user_input.get("source_language_code")
            or user_context.get("language_preference")
            or "ta"
        ).lower().strip()

        transcribed_text = user_input.get("raw_text", "")
        source_lang = detect_language_from_text(transcribed_text or original_user_query, default_lang=client_lang_pref)

        if source_lang != "en" and transcribed_text:
            english_text = IndicTranslationService.translate_to_english(transcribed_text, source_lang)
        else:
            english_text = transcribed_text or "Check marine conditions at current coordinates."
    else:
        raw_text = user_input.get("raw_text", "")
        client_lang_pref = (
            user_input.get("source_language_code")
            or user_context.get("language_preference")
            or "en"
        ).lower().strip()

        # Deterministic Indic script auto-detection directly from original query
        source_lang = detect_language_from_text(original_user_query or raw_text, default_lang=client_lang_pref)

        # Translate regional text to English for agent reasoning
        pre_english = user_input.get("english_query")
        if pre_english and all(ord(c) < 128 for c in pre_english):
            english_text = pre_english
        elif source_lang != "en" and raw_text:
            english_text = IndicTranslationService.translate_to_english(raw_text, source_lang)
        else:
            english_text = raw_text

    # Step 2.5: Conversational Greeting Fast-Path
    # Prevents simple greetings ("hi", "hello", "வணக்கம்") from triggering false wave hazards
    clean_eng_lower = english_text.lower().strip()
    is_greeting = clean_eng_lower in (
        "hi", "hello", "hey", "howdy", "vanakkam", "namaste", "good morning",
        "good afternoon", "good evening", "help", "who are you", "what can you do"
    ) or clean_eng_lower.startswith("hi ") or clean_eng_lower.startswith("hello ")

    if is_greeting:
        from decision_engine import compute_live_green_energy
        if resolved_persona_enum is None:
            resolved_persona_enum = classify_persona_intent(english_text) or StakeholderPersona.FISHERMAN
        persona = resolved_persona_enum.value
        user_context["persona"] = persona

        greeting_english = (
            "Hello Captain! Welcome to ORCA Marine Multi-Agent AI, powered by 9 ISRO MOSDAC satellite Earth Observation products.\n\n"
            "I am your real-time maritime intelligence co-pilot. Here is how I can assist your voyage:\n"
            "• 🐟 Potential Fishing Zones (PFZ): High-chlorophyll pelagic fronts & species guidance (Tuna, Mackerel, Sardines).\n"
            "• 🌊 Ocean State & Wave Safety: Real-time wave heights, swell direction, surface winds, and currents.\n"
            "• ⚠️ Disaster & Storm Alerts: Early warnings from satellite scatterometer and sounder telemetry.\n"
            "• 🧭 Safe Corridors: Route planning avoiding turbulent seas with full IMBL border clearance.\n"
            "• ☀️ Green Marine Energy: Solar-assisted zero-emission operational range calculation.\n\n"
            "How can I assist your voyage today?"
        )

        # Translate greeting back to user's native language
        if source_lang != "en":
            native_advisory = IndicTranslationService.translate_to_target(greeting_english, source_lang)
        else:
            native_advisory = greeting_english

        audio_payload_base64 = None

        return {
            "success": True,
            "session_id": session_id,
            "client_timestamp": client_timestamp,
            "device_telemetry": device_telemetry,
            "user_persona": persona,
            "input_type": "AUDIO" if is_voice else "TEXT",
            "transcribed_text": transcribed_text,
            "english_query": english_text,
            "source_language_code": source_lang,
            "is_conversational": True,
            "query_type": "CONVERSATIONAL",
            "map_status": "READY - SATELLITE ACTIVE",
            "recommended_coordinates": gps_str,
            "risk_assessment": {
                "risk_score": 0,
                "threat_level": "LOW",
                "threats": [],
                "status": "OPERATIONAL",
                "summary": "9 ISRO MOSDAC satellite feeds synchronized. Monitoring active."
            },
            "chat_text": native_advisory,
            "native_advisory_text": native_advisory,
            "audio_payload_base64": audio_payload_base64,
            "safe_sea_route": None,
            "green_marine_energy": compute_live_green_energy(lat=lat, lon=lon),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

    # Step 2.6: Security Rejection & Prompt Injection Shield
    from decision_engine import check_prompt_injection, check_temporal_bounds, check_emergency_sar, check_out_of_domain
    is_injection, injection_reason = check_prompt_injection(english_text)
    if is_injection:
        sec_msg = (
            "STATUS: SECURITY_REJECTION. Critical security warning: Adversarial prompt injection or safety protocol bypass attempt detected. "
            "Project ORCA maritime safety guardrails, IMBL border enforcement, and SOLAS life-safety protocols cannot be overridden or disabled. Query terminated."
        )
        return {
            "success": False,
            "error": "SECURITY_REJECTION",
            "session_id": session_id,
            "client_timestamp": client_timestamp,
            "device_telemetry": device_telemetry,
            "user_persona": persona or "UNKNOWN",
            "input_type": "AUDIO" if is_voice else "TEXT",
            "transcribed_text": transcribed_text,
            "english_query": english_text,
            "source_language_code": source_lang,
            "map_status": "SECURITY_REJECTION",
            "recommended_coordinates": gps_str,
            "risk_assessment": {
                "risk_score": 100.0,
                "threat_level": "CRITICAL",
                "threats": [f"SECURITY REJECTION: {injection_reason}"],
                "status": "SECURITY_REJECTION",
                "summary": "Critical security warning: Adversarial prompt injection attempt detected.",
            },
            "chat_text": sec_msg,
            "native_advisory_text": sec_msg,
            "audio_payload_base64": None,
            "safe_sea_route": None,
            "green_marine_energy": {
                "solar_irradiance_wm2": 0,
                "extended_zero_emission_hours": 0.0,
                "fuel_saved_liters": 0.0,
                "carbon_offset_kg": 0.0,
            },
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    # Step 2.7: Chronological Temporal Bounds & Forecast Intent Check (24-Hour Horizon Limit)
    from decision_engine import detect_temporal_intent
    temporal_info = detect_temporal_intent(english_text)
    user_context["is_future_query"] = temporal_info.get("is_future_query", False)
    if temporal_info["is_beyond_24h"]:
        pending_msg = (
            "STATUS: PENDING FORECAST. Real-time ISRO MOSDAC satellite Earth Observation cache provides "
            "near-real-time observations and short-term 24-hour nowcasting, and cannot predict conditions beyond 24 hours. "
            "The requested forecast horizon is unavailable. Please check back closer to your departure window for updated satellite passes and numerical forecast model runs."
        )
        return {
            "success": False,
            "error": "PENDING_FORECAST",
            "session_id": session_id,
            "client_timestamp": client_timestamp,
            "device_telemetry": device_telemetry,
            "user_persona": persona or "UNKNOWN",
            "input_type": "AUDIO" if is_voice else "TEXT",
            "transcribed_text": transcribed_text,
            "english_query": english_text,
            "source_language_code": source_lang,
            "map_status": "PENDING FORECAST",
            "recommended_coordinates": "",
            "risk_assessment": {
                "risk_score": "N/A",
                "threat_level": "PENDING_FORECAST",
                "threats": [f"PENDING FORECAST: {temporal_info['reason']}"],
                "status": "PENDING FORECAST",
                "summary": "Query intercepted: Horizon exceeds 24-hour ISRO MOSDAC satellite cache window.",
            },
            "chat_text": pending_msg,
            "native_advisory_text": pending_msg,
            "audio_payload_base64": None,
            "safe_sea_route": None,
            "green_marine_energy": {
                "solar_irradiance_wm2": 0,
                "extended_zero_emission_hours": 0.0,
                "fuel_saved_liters": 0.0,
                "carbon_offset_kg": 0.0,
            },
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    # Step 2.8: Out-of-Domain (OOD) Rejection Guardrail
    is_ood, ood_reason = check_out_of_domain(english_text)
    if is_ood:
        ood_msg = (
            "I am ORCA, a specialized maritime intelligence engine. "
            "I am programmed exclusively to assist with marine navigation, weather analysis, and coastal safety operations. "
            "I cannot process requests outside of this scope."
        )
        native_advisory = ood_msg
        if source_lang and source_lang != "en":
            try:
                from language_layer import LanguageLayer
                ll = LanguageLayer()
                native_advisory = ll.translate_advisory(ood_msg, source_lang)
            except Exception:
                native_advisory = ood_msg

        return {
            "success": False,
            "error": "OUT_OF_DOMAIN",
            "session_id": session_id,
            "client_timestamp": client_timestamp,
            "device_telemetry": device_telemetry,
            "user_persona": persona or "UNKNOWN",
            "input_type": "AUDIO" if is_voice else "TEXT",
            "transcribed_text": transcribed_text,
            "english_query": english_text,
            "source_language_code": source_lang,
            "map_status": "OUT_OF_DOMAIN",
            "recommended_coordinates": "",
            "risk_assessment": {
                "risk_score": "N/A",
                "threat_level": "OUT_OF_DOMAIN",
                "threats": [f"OUT OF DOMAIN: {ood_reason}"],
                "status": "OUT_OF_DOMAIN",
                "summary": "Query rejected: Request is outside the maritime domain.",
            },
            "chat_text": native_advisory,
            "native_advisory_text": native_advisory,
            "audio_payload_base64": None,
            "safe_sea_route": None,
            "green_marine_energy": {
                "solar_irradiance_wm2": 0,
                "extended_zero_emission_hours": 0.0,
                "fuel_saved_liters": 0.0,
                "carbon_offset_kg": 0.0,
            },
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    # Dynamic Persona Intent Classification if unspecified
    if resolved_persona_enum is None and english_text:
        resolved_persona_enum = classify_persona_intent(english_text)
        if resolved_persona_enum:
            persona = resolved_persona_enum.value
            user_context["persona"] = persona

    # Step 2.9: Strict Location Resolution Contract
    # Enforces: Explicit User Location > Route Origin/Dest > Device GPS > Multi-Turn Memory > Default (NONE -> LOCATION_REQUIRED)
    prev_loc_cand = None
    if manager and hasattr(manager, "sessions") and session_id in manager.sessions and manager.sessions[session_id]:
        last_turn = manager.sessions[session_id][-1]
        prev_loc_cand = last_turn.get("location") or (last_turn.get("locations")[0] if last_turn.get("locations") else None)

    location_context, origin_info, dest_info = resolve_location_context(
        english_text,
        device_telemetry=device_telemetry,
        previous_location=prev_loc_cand,
    )

    if location_context.status == LocationStatus.LOCATION_REQUIRED:
        loc_req_msg = (
            "Please specify a coastal location or port (e.g., Mumbai, Chennai, Kochi, Goa, Tuticorin, Visakhapatnam) or enable GPS."
        )
        if source_lang != "en":
            try:
                from language_layer import LanguageLayer
                ll = LanguageLayer()
                loc_req_msg_native = ll.translate_advisory(loc_req_msg, source_lang)
            except Exception:
                loc_req_msg_native = loc_req_msg
        else:
            loc_req_msg_native = loc_req_msg

        loc_req_intent_info = classify_marine_query_intent(english_text)
        loc_req_intent = loc_req_intent_info.get("intent", "GENERAL_MARINE")

        return {
            "success": False,
            "status": "LOCATION_REQUIRED",
            "message": loc_req_msg_native,
            "reply": loc_req_msg_native,
            "response": loc_req_msg_native,
            "intent": loc_req_intent,
            "analyzed_intent": loc_req_intent,
            "show_route": False,
            "supported_locations": [
                "Mumbai", "Chennai", "Kochi", "Goa", "Tuticorin", "Visakhapatnam",
                "Mangalore", "Kandla", "Porbandar", "Paradip", "Haldia", "Kolkata", "Kanyakumari", "Rameswaram"
            ],
            "session_id": session_id,
            "client_timestamp": client_timestamp,
            "device_telemetry": device_telemetry,
            "user_persona": persona or "FISHERMAN",
            "input_type": "AUDIO" if is_voice else "TEXT",
            "transcribed_text": transcribed_text,
            "english_query": english_text,
            "effective_query": english_text,
            "original_query": original_user_query or raw_text or english_text,
            "query": original_user_query or raw_text or english_text,
            "source_language_code": source_lang,
            "map_status": "LOCATION_REQUIRED",
            "recommended_coordinates": "",
            "risk_assessment": {
                "risk_score": 0,
                "threat_level": "LOCATION_REQUIRED",
                "threats": ["No coastal location or GPS coordinates provided."],
                "status": "LOCATION_REQUIRED",
                "summary": loc_req_msg,
            },
            "chat_text": loc_req_msg_native,
            "native_advisory_text": loc_req_msg_native,
            "audio_payload_base64": None,
            "safe_sea_route": None,
            "location_context": location_context.to_dict(),
            "origin": None,
            "destination": None,
            "green_marine_energy": {
                "solar_irradiance_wm2": 0,
                "extended_zero_emission_hours": 0.0,
                "fuel_saved_liters": 0.0,
                "carbon_offset_kg": 0.0,
            },
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    # Step 3: Pass English text to ManagerAgent
    manager_payload = {
        "session_id": session_id,
        "client_timestamp": client_timestamp,
        "user_context": user_context,
        "device_telemetry": device_telemetry,
        "original_query": original_user_query,
        "effective_query": english_text,
        "user_input": {
            "input_type": "TEXT",
            "raw_text": english_text,
            "original_query": original_user_query,
            "effective_query": english_text,
            "english_query": english_text,
        },
    }

    orchestration_result = manager.analyze_query(manager_payload)

    # If persona still unspecified, inspect ManagerAgent output
    if resolved_persona_enum is None:
        mgr_persona = orchestration_result.get("user_persona")
        resolved_persona_enum = resolve_persona(mgr_persona) or StakeholderPersona.FISHERMAN
        persona = resolved_persona_enum.value
        user_context["persona"] = persona

    # Step 4: Dispatch tasks to Domain Agents
    from ocean_agent import OceanAgent
    from weather_agent import WeatherAgent
    from disaster_agent import DisasterAgent
    from gis_agent import GisAgent
    from pfz_agent import PfzAgent
    from decision_engine import run_decision_engine

    agent_registry = {
        "OCEAN_AGENT": OceanAgent(),
        "WEATHER_AGENT": WeatherAgent(),
        "DISASTER_AGENT": DisasterAgent(),
        "GIS_AGENT": GisAgent(),
        "PFZ_AGENT": PfzAgent(),
    }

    locations = orchestration_result.get("location_entities", [])
    times = orchestration_result.get("time_entities", [])
    plan = orchestration_result.get("execution_plan", [])
    primary_time = times[0] if times else "today"

    dispatched_results = {
        "location_context": location_context.to_dict(),
        "primary_location": location_context.name,
    }

    if origin_info:
        dispatched_results["origin"] = origin_info
        dispatched_results["vessel_location"] = origin_info.get("name")
    else:
        dispatched_results["vessel_location"] = location_context.name or (f"{location_context.latitude:.4f},{location_context.longitude:.4f}" if location_context.latitude is not None else None)

    if dest_info:
        dispatched_results["destination"] = dest_info.get("name")
        dispatched_results["target_location"] = dest_info.get("name")

    analyzed_intent_upper = (orchestration_result.get("analyzed_intent") or "").upper()
    non_route_intents = (
        "MARITIME_BOUNDARY", "EEZ", "CYCLONE", "DISASTER", "WEATHER",
        "OCEAN", "OCEAN_CONDITIONS", "SEA_CONDITIONS", "PFZ", "FISHING", "GENERAL", "GENERAL_MARINE"
    )
    eng_lower = english_text.lower().strip()
    has_explicit_route_keyword = (
        any(k in eng_lower for k in ["safe route", "give me a route", "route from", "best route", "alternative route", "recommend a route", "sail from", "navigate to", "passage from"])
        or (re.search(r"\b(navigate|sail|route|passage)\s+to\s+colombo\b", eng_lower) is not None)
        or ("route" in eng_lower and "from " in eng_lower and " to " in eng_lower)
    )
    is_route_intent = (
        analyzed_intent_upper in ("ROUTE", "SAFE_ROUTE", "ROUTE_PLANNING")
        or (has_explicit_route_keyword and analyzed_intent_upper not in non_route_intents)
    )

    # Dynamically execute only domain agents specified in the execution plan
    for step in plan:
        ag_name = step.get("agent_name")
        t_instr = step.get("task_instructions", "")
        exp_fmt = step.get("expected_output_format", "TEXT_SUMMARY")

        if ag_name == "GIS_AGENT":
            orig_name = origin_info.get("name") if origin_info else (location_context.name or (f"{location_context.latitude:.4f},{location_context.longitude:.4f}" if location_context.latitude is not None else None))
            dest_name = dest_info.get("name") if dest_info else (dispatched_results.get("destination") or dispatched_results.get("target_location"))
            if (is_route_intent or (origin_info and dest_info)) and orig_name and dest_name:
                t_instr = f"Generate safe sea passage route from {orig_name} to {dest_name}."
            elif (is_route_intent or dest_info) and dest_name:
                t_instr = f"Generate safe sea passage route to {dest_name}."

        agent_instance = agent_registry.get(ag_name)
        if agent_instance and ag_name not in dispatched_results:
            try:
                out = agent_instance.execute_task(
                    location=location_context,
                    time_frame=primary_time,
                    task_instructions=t_instr,
                    expected_format=exp_fmt,
                    persona=persona,
                )
                dispatched_results[ag_name] = out
            except Exception as _ag_err:
                print(f"[Agent Execution Warning] {ag_name} task notice: {_ag_err}")
                p_lat = location_context.latitude or 18.9220
                p_lon = location_context.longitude or 72.8347
                p_name = location_context.name or "Operational Sector"
                dispatched_results[ag_name] = {
                    "agent": ag_name,
                    "location": {"name": p_name, "latitude": p_lat, "longitude": p_lon},
                    "status": "DATA_UNAVAILABLE",
                    "error": f"Agent execution error: {_ag_err}",
                    "summary": f"Data for {ag_name} is currently unavailable for {p_name}.",
                    "advisory": f"{ag_name} telemetry could not be retrieved. Exercise caution and verify with local maritime authorities.",
                    "is_live_satellite": False,
                    "metadata": {"source": "DATA_UNAVAILABLE", "error": str(_ag_err)},
                }

    # Step 5: DecisionEngine Final Synthesis
    normalized_query = english_text
    dispatched_results["normalized_query"] = normalized_query
    dispatched_results["user_query"] = english_text
    dispatched_results["original_query"] = raw_text or english_text
    dispatched_results["analyzed_intent"] = orchestration_result.get("analyzed_intent")
    dispatched_results["execution_plan"] = plan
    if "vessel_location" not in dispatched_results:
        dispatched_results["vessel_location"] = location_context.name or (f"{location_context.latitude:.4f},{location_context.longitude:.4f}" if location_context.latitude is not None else None)
    dispatched_results["device_telemetry"] = device_telemetry
    final_payload = run_decision_engine(
        dispatched_results,
        persona=persona,
        language_code=source_lang,
        normalized_query=normalized_query,
    )
    final_payload["analyzed_intent"] = orchestration_result.get("analyzed_intent")
    final_payload["execution_plan"] = plan
    final_payload["agents_used"] = [s.get("agent_name") for s in plan if s.get("agent_name")]

    # Step 6: Translate final English advisory back to user's detected language
    detected_lang = (
        orchestration_result.get("source_language_code")
        or source_lang
        or "en"
    ).lower()

    english_advisory = (
        final_payload.get("chat_text")
        or final_payload.get("native_advisory_text")
        or final_payload.get("text_advisory_local")
        or ""
    )
    if detected_lang != "en" and english_advisory:
        native_advisory = IndicTranslationService.translate_to_target(english_advisory, detected_lang)
    else:
        native_advisory = english_advisory

    final_payload["native_advisory_text"] = native_advisory
    final_payload["chat_text"] = native_advisory
    final_payload["text_advisory_local"] = native_advisory
    final_payload.pop("bhashini_text", None)

    # Step 7: Audio payload (set to None; zero-credential in-process pipeline)
    audio_payload_base64 = None
    final_payload["audio_payload_base64"] = audio_payload_base64

    # Step 8: Package final multi-modal payload matching ORCA schema
    analyzed_intent_upper = (orchestration_result.get("analyzed_intent") or "").upper()
    non_route_intents = (
        "MARITIME_BOUNDARY", "EEZ", "CYCLONE", "DISASTER", "WEATHER",
        "OCEAN", "OCEAN_CONDITIONS", "SEA_CONDITIONS", "PFZ", "FISHING", "GENERAL", "GENERAL_MARINE"
    )
    eng_lower = english_text.lower().strip()
    has_explicit_route_keyword = (
        any(k in eng_lower for k in ["safe route", "give me a route", "route from", "best route", "alternative route", "recommend a route", "sail from", "navigate to", "passage from"])
        or (re.search(r"\b(navigate|sail|route|passage)\s+to\s+colombo\b", eng_lower) is not None)
        or ("route" in eng_lower and "from " in eng_lower and " to " in eng_lower)
    )
    is_route_intent = (
        analyzed_intent_upper in ("ROUTE", "SAFE_ROUTE", "ROUTE_PLANNING")
        or (has_explicit_route_keyword and analyzed_intent_upper not in non_route_intents)
    )
    if not is_route_intent:
        final_payload["safe_sea_route"] = None
        if "alternative_route" in final_payload and isinstance(final_payload["alternative_route"], dict):
            final_payload["alternative_route"]["safe_sea_route"] = None

    final_payload["success"] = True
    final_payload["session_id"] = session_id
    final_payload["client_timestamp"] = client_timestamp
    final_payload["device_telemetry"] = device_telemetry
    final_payload["user_persona"] = persona
    final_payload["input_type"] = "AUDIO" if is_voice else "TEXT"
    final_payload["transcribed_text"] = transcribed_text
    final_payload["english_query"] = english_text
    final_payload["effective_query"] = english_text
    final_payload["source_language_code"] = detected_lang
    final_payload["detected_language"] = detected_lang
    final_payload["original_query"] = original_user_query or raw_text or english_text
    final_payload["query"] = original_user_query or raw_text or english_text
    final_payload["reasoning_output"] = english_advisory
    final_payload["final_response"] = native_advisory
    final_payload["reply"] = native_advisory
    final_payload["response"] = native_advisory
    final_payload["message"] = native_advisory
    final_payload["location_context"] = location_context.to_dict()
    final_payload["origin"] = origin_info
    final_payload["destination"] = dest_info
    final_payload["intent"] = orchestration_result.get("analyzed_intent")
    final_payload["show_route"] = is_route_intent
    final_payload["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return final_payload


if __name__ == "__main__":
    print("=" * 70)
    print("     ORCA - Agentic Orchestrator & Multilingual Marine Intelligence Controller")
    print("             ISRO SIH Problem Statement 176")
    print("=" * 70)

    if not os.getenv("GEMINI_API_KEY"):
        print("[!] NOTICE: GEMINI_API_KEY is not configured in your .env file.")
        print("    Please set GEMINI_API_KEY in .env to enable live LLM planning.\n")

    # Step 0: Pre-Flight Check: Verify latest ISRO MOSDAC satellite data before processing
    try:
        from shadow_cache_worker import ensure_latest_mosdac_cache
        print("[MOSDAC Pre-Flight Check] Verifying latest satellite files...")
        ensure_latest_mosdac_cache(max_age_hours=72.0, non_blocking=True)
    except Exception as cache_err:
        print(f"[MOSDAC Pre-Flight Notice] Cache check completed with note: {cache_err}")

    manager = ManagerAgent()
    print(f"[OK] ManagerAgent initialized with model: {manager.model_name}")

    print("[i] Memory active: Context & location co-references persist across queries.")
    print("[i] Multi-Modal Routing: Accepts natural text, JSON payloads, or audio.")
    print("[i] Enter your query below, or type 'exit' / 'quit' to end.\n")

    while True:
        try:
            raw_input_line = input("ORCA Marine Query > ").strip()

            if not raw_input_line:
                continue

            if raw_input_line.lower() in ("exit", "quit", "q"):
                print("\nShutting down ORCA Agentic Orchestrator. Fair winds and safe voyages!")
                break

            # Parse input as JSON if formatted as JSON payload, otherwise wrap text
            if raw_input_line.startswith("{") and raw_input_line.endswith("}"):
                try:
                    request_data = json.loads(raw_input_line)
                except Exception:
                    request_data = {
                        "session_id": "sess_cli",
                        "user_context": {"persona": None},
                        "device_telemetry": {"latitude": None, "longitude": None},
                        "user_input": {"input_type": "TEXT", "raw_text": raw_input_line}
                    }
            else:
                request_data = {
                    "session_id": "sess_cli",
                    "user_context": {"persona": None},
                    "device_telemetry": {"latitude": None, "longitude": None},
                    "user_input": {"input_type": "TEXT", "raw_text": raw_input_line}
                }

            # Check is_voice as required by SIH 176 architecture
            is_voice = request_data.get("user_input", {}).get("input_type") == "AUDIO"

            response = process_marine_request(request_data, manager=manager)

            print("\n" + "=" * 70)
            print("             ORCA MULTI-MODAL ROUTING REPORT")
            print("=" * 70)
            print(f"🎤 Voice Input Mode     : {is_voice}")
            if is_voice:
                print(f"🎙️ Transcribed Text     : {response.get('transcribed_text')}")
            print(f"🌐 Source Language      : {response.get('source_language_code', 'en').upper()}")
            print(f"📝 English Query        : {response.get('english_query')}")
            print(f"🛡️ Map Status           : {response.get('map_status')}")
            print(f"📍 Recommended Coords   : {response.get('recommended_coordinates')}")
            print(f"📊 Risk Assessment      : Score {response.get('risk_assessment', {}).get('risk_score', 'N/A')}/100")
            print(f"🗣️ Localized Advisory   :\n{response.get('native_advisory_text')}")
            if response.get("audio_payload_base64"):
                audio_len = len(response["audio_payload_base64"])
                print(f"🔊 Audio Payload (TTS)  : Present ({audio_len} chars Base64)")
            else:
                print("🔊 Audio Payload (TTS)  : null")
            print("=" * 70 + "\n")

            # Prevent rapid-fire queries
            time.sleep(1)

        except KeyboardInterrupt:
            print("\n\nSession interrupted by user. Exiting...")
            break
        except Exception as err:
            print(f"\n[Error] {err}\n")


def __getattr__(name):
    if name == "app":
        from server import app
        return app
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

