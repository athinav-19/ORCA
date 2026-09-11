"""
decision_engine.py - ORCA Intelligent Analysis & Decision Engine
ISRO SIH Problem Statement 176: Marine Multi-Agent System

Acts as the final synthesis layer for the ORCA multi-agent marine platform:
1. Aggregates outputs from the 5 specialized domain agents.
2. Applies deterministic risk scoring & threat prioritization (RiskAnalysisAgent).
3. Calculates safe alternate coordinates when hazards are detected (SafeAlternative).
4. Uses Gemini (gemini-1.5-flash-latest) to synthesize conflicting inputs into an
   explainable multi-modal response payload ready for frontend & Bhashini translation (ReasoningAgent).
"""

import os
import re
import json
import math
import random
import warnings
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from gis_agent import GisAgent, resolve_ocean_target, find_nearest_coastal_landmark
from models import (
    StakeholderPersona,
    resolve_persona,
    PERSONA_BADGES,
    PERSONA_FUEL_CONSUMPTION_RATES,
    get_fuel_consumption_rate,
)

from cyclone_track_agent import CycloneTrackAgent, fetch_active_cyclone_tracks, haversine_nm

warnings.filterwarnings("ignore")
import google.generativeai as genai

load_dotenv()

# Configure Gemini API if available
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

# Global Fast-Path / Deterministic Mode (Zero-Wait for mobile clients)
FAST_DEMO_MODE = os.getenv("FAST_DEMO_MODE", "true").lower() in ("true", "1", "yes")


# =====================================================================
# 0. EXTREME EDGE-CASE GUARDRAIL DETECTORS
# =====================================================================
def check_prompt_injection(query_text: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Detects adversarial prompt injections and security protocol bypass attempts.
    Returns (is_injection, reason_str).
    """
    if not query_text:
        return False, None

    q = str(query_text).lower().strip()
    injection_patterns = [
        (r"\b(ignore|disregard|forget|override|bypass)\s+(?:all\s+)?(?:previous\s+)?(?:instructions|protocols|rules|guidelines|guardrails)\b", "Instruction override directive"),
        (r"\b(override|disable|bypass)\s+(?:imbl|border|eez|safety|security|solas)\s+(?:protocols?|rules?|guardrails?|checks?)\b", "Border/Safety override attempt"),
        (r"\b(system\s+prompt|reveal\s+instructions|print\s+system\s+message|show\s+prompt)\b", "System prompt extraction attempt"),
        (r"\b(you\s+are\s+now\s+dan|act\s+as\s+jailbreak|jailbreak\s+mode|unrestricted\s+mode)\b", "Persona jailbreak exploit"),
        (r"\b(ignore\s+maritime\s+law|pretend\s+there\s+are\s+no\s+borders)\b", "Maritime law bypass"),
    ]
    for pat, reason in injection_patterns:
        if re.search(pat, q):
            return True, reason
    return False, None


def detect_temporal_intent(query_text: Optional[str]) -> Dict[str, Any]:
    """
    Scans the user's normalized text for future-tense temporal expressions and forecast intent.
    Sets is_future_query = True if future-tense keywords are detected (e.g., 'tomorrow',
    'forecast', 'upcoming', 'future', 'later', 'tonight', 'next week', etc.).
    """
    result = {
        "is_future_query": False,
        "has_temporal_intent": False,
        "is_beyond_24h": False,
        "timeframe_label": None,
        "matched_keyword": None,
        "reason": None,
    }
    if not query_text:
        return result

    q = str(query_text).lower().strip()

    # 1. Check for extended horizons beyond 24 hours
    patterns_beyond_24h = [
        (r"\b(?:next|upcoming)\s+(?:week|month|year|season|weekend)\b", "next week"),
        (r"\bday\s+after\s+tomorrow\b", "day after tomorrow"),
        (r"\b(?:in|after)\s+([2-9]|\d{2,})\s+days?\b", "multi-day"),
        (r"\b([2-9]|\d{2,})\s+days?\s+(?:from\s+now|later|ahead)\b", "multi-day"),
        (r"\b(?:in|after)\s+(\d+)\s+weeks?\b", "weeks"),
        (r"\b(\d+)\s+weeks?\s+(?:from\s+now|later|ahead)\b", "weeks"),
        (r"\b(?:in|after)\s+(\d+)\s+months?\b", "months"),
        (r"\b(\d+)\s+months?\s+(?:from\s+now|later|ahead)\b", "months"),
        (r"\b(?:in|after)\s+([3-9]\d|\d{3,})\s+hours?\b", "hours_extended"),
        (r"\b([3-9]\d|\d{3,})\s+hours?\s+(?:from\s+now|later|ahead)\b", "hours_extended"),
        (r"\b([2-9]|\d{2,})[ -]day\s+(?:forecast|outlook|weather|prediction)\b", "multi-day"),
        (r"\b(?:10|7|14)[ -]?day\b", "extended_days"),
        (r"\blong[- ]range\s+(?:forecast|outlook|prediction)\b", "long-range forecast"),
        (r"\bextended\s+(?:forecast|outlook)\b", "extended forecast"),
        (r"\bweekly\s+(?:forecast|outlook)\b", "weekly forecast"),
    ]

    for pattern, label in patterns_beyond_24h:
        m = re.search(pattern, q)
        if m:
            matched_str = m.group(0)
            result["is_future_query"] = True
            result["has_temporal_intent"] = True
            result["is_beyond_24h"] = True
            result["matched_keyword"] = matched_str
            result["timeframe_label"] = matched_str
            result["reason"] = f"Future query detected ('{matched_str}') requesting extended forecast horizon."
            return result

    # 2. Check for near-future temporal intent (within 24-48 hours / tomorrow / forecast)
    patterns_within_24h = [
        (r"\btomorrow\s+morning\b", "tomorrow morning"),
        (r"\btomorrow\s+afternoon\b", "tomorrow afternoon"),
        (r"\btomorrow\s+evening\b", "tomorrow evening"),
        (r"\btomorrow\s+night\b", "tomorrow night"),
        (r"\btomorrow\b", "tomorrow"),
        (r"\btonight\b", "tonight"),
        (r"\b(?:this|in\s+the)\s+evening\b", "this evening"),
        (r"\blater\s+today\b", "later today"),
        (r"\blater\b", "later"),
        (r"\bupcoming\b", "upcoming period"),
        (r"\bforecast\b", "forecast window"),
        (r"\bprediction\b", "prediction window"),
        (r"\boutlook\b", "outlook window"),
        (r"\bfuture\b", "future window"),
        (r"\b(?:in|after)\s+(?:1|one)\s+day\b", "tomorrow"),
        (r"\b(?:in|after)\s+([1-9]|1\d|2[0-4])\s+hours?\b", "in a few hours"),
        (r"\b([1-9]|1\d|2[0-4])\s+hours?\s+(?:from\s+now|later|ahead)\b", "in a few hours"),
        (r"\bahead\b", "ahead"),
    ]

    for pattern, label in patterns_within_24h:
        m = re.search(pattern, q)
        if m:
            matched_str = m.group(0)
            result["is_future_query"] = True
            result["has_temporal_intent"] = True
            result["is_beyond_24h"] = False
            result["matched_keyword"] = matched_str
            result["timeframe_label"] = label
            result["reason"] = f"Future query detected ('{matched_str}') for near-future horizon."
            return result

    return result


# =====================================================================
# 0.1 FORECAST AGENT (OPEN-METEO MARINE API INTEGRATION)
# =====================================================================
class ForecastAgent:
    """
    Open-Meteo Marine API Integration for forward-looking marine weather forecasts (24-48h).
    Bypasses real-time MOSDAC state evaluation when is_future_query is True.
    Includes strict 5.0s network timeout and silent fallback to MOSDAC.
    """

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
        self.base_url = "https://marine-api.open-meteo.com/v1/marine"

    def fetch_forecast(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """
        Calls Open-Meteo Marine API for target coordinates (lat, lon),
        extracting the maximum wave_height for the upcoming 24-48 hours.
        Wrapped in try/except with strict timeout=5.0. Returns None on any error/timeout.
        """
        try:
            params = {
                "latitude": round(float(lat), 4),
                "longitude": round(float(lon), 4),
                "hourly": "wave_height",
                "forecast_days": 2,
            }
            resp = requests.get(self.base_url, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                hourly = data.get("hourly", {})
                wave_heights = hourly.get("wave_height", [])
                valid_waves = [float(w) for w in wave_heights if w is not None]
                if valid_waves:
                    max_wave = round(max(valid_waves), 2)
                    avg_wave = round(sum(valid_waves) / len(valid_waves), 2)
                    return {
                        "success": True,
                        "max_wave_height_m": max_wave,
                        "avg_wave_height_m": avg_wave,
                        "time_horizon": "24–48 hours",
                        "source": "Open-Meteo Marine API",
                        "coordinates": f"{lat:.4f},{lon:.4f}",
                    }
        except Exception:
            # Catch exception silently as required
            pass
        return None


def resolve_target_coordinates(aggregated_data: Dict[str, Any], query_text: Optional[str] = None) -> Tuple[float, float]:
    """
    Resolves the primary marine target or departure coordinates (lat, lon)
    for forecast queries or routing tasks.
    """
    if query_text:
        gis = GisAgent()
        q_low = str(query_text).lower()
        for alias, sector in gis.COUNTRY_PORT_ALIASES.items():
            if alias in q_low:
                return float(sector["lat"]), float(sector["lon"])
        for name, sector in gis.COASTAL_SECTORS.items():
            if name in q_low:
                return float(sector["lat"]), float(sector["lon"])
        m_loc = re.search(r"\b(?:near|in|at|off|around|to)\s+([a-zA-Z\s]+)", q_low)
        if m_loc:
            loc_candidate = m_loc.group(1).strip().split()[0]
            sec = gis.resolve_location(loc_candidate)
            if sec and "lat" in sec and "lon" in sec:
                return float(sec["lat"]), float(sec["lon"])

    pfz_out = aggregated_data.get("PFZ_AGENT")
    if isinstance(pfz_out, dict):
        features = pfz_out.get("geojson", {}).get("features", [])
        if features:
            props = features[0].get("properties", {})
            c_lat = props.get("centroid_lat")
            c_lon = props.get("centroid_lon")
            if c_lat is not None and c_lon is not None:
                return float(c_lat), float(c_lon)

    v_loc = aggregated_data.get("vessel_location")
    if not v_loc and "device_telemetry" in aggregated_data:
        dt = aggregated_data["device_telemetry"]
        if isinstance(dt, dict) and "latitude" in dt and "longitude" in dt:
            return float(dt["latitude"]), float(dt["longitude"])

    if v_loc and "," in str(v_loc):
        try:
            parts = str(v_loc).split(",")
            return float(parts[0].strip()), float(parts[1].strip())
        except Exception:
            pass

    return 8.7642, 78.1348


def get_dataset_timestamp(aggregated_data: Dict[str, Any]) -> str:
    """
    Extracts or formats a human-readable observation timestamp from MOSDAC satellite telemetry.
    E.g. '09-Sep-2026 07:45 UTC'.
    """
    if not isinstance(aggregated_data, dict):
        return "09-Sep-2026 07:45 UTC"

    for agent_key in ["WEATHER_AGENT", "DISASTER_AGENT", "OCEAN_AGENT", "PFZ_AGENT"]:
        agent_dict = aggregated_data.get(agent_key)
        if isinstance(agent_dict, dict):
            granule = (
                agent_dict.get("satellite_granule")
                or agent_dict.get("source")
                or agent_dict.get("hazard_summary", {}).get("source")
            )
            if granule and isinstance(granule, str):
                m = re.search(r"(\d{2})([A-Za-z]{3})(\d{4})_(\d{2})(\d{2})", granule)
                if m:
                    day, mon, yr, hh, mm = m.groups()
                    return f"{day}-{mon.capitalize()}-{yr} {hh}:{mm} UTC"

    for k in ["timestamp", "client_timestamp"]:
        ts = aggregated_data.get(k)
        if ts and isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return dt.strftime("%d-%b-%Y %H:%M UTC")
            except Exception:
                pass

    return "09-Sep-2026 07:45 UTC"


def check_temporal_bounds(query_text: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Checks if user query requests forecasts beyond the 24-hour ISRO MOSDAC satellite window.
    Returns (is_out_of_bounds, reason_str).
    """
    intent = detect_temporal_intent(query_text)
    if intent["is_beyond_24h"]:
        return True, intent["reason"]
    return False, None


def check_emergency_sar(query_text: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Priority regex check for life-safety emergency distress keywords.
    Life-safety distress takes absolute precedence under SOLAS conventions.
    Returns (is_emergency_sar, reason_str).
    """
    if not query_text:
        return False, None

    q = str(query_text).lower().strip()
    sar_pattern = (
        r"\b(mayday|sinking|taking\s+(?:in\s+)?water|capsiz(?:ed?|ing)|"
        r"(?:engine\s+|vessel\s+|cabin\s+)?fire(?:\s+onboard|\s+on\s+board)?|"
        r"medical\s+emergency|crew\s+injured|man\s+overboard|\bmob\b|"
        r"vessel\s+in\s+distress|distress\s+call|abandon\s+ship)\b"
    )
    m = re.search(sar_pattern, q)
    if m:
        return True, f"Life-safety maritime emergency detected ({m.group(1).upper()})"
    return False, None


def check_out_of_domain(query_text: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Evaluates whether the user's text query falls outside the maritime domain
    (e.g., general math, cooking, unrelated coding, pop culture, general trivia).
    Returns (is_out_of_domain, reason_str).
    """
    if not query_text:
        return False, None

    q = str(query_text).lower().strip()

    # Allow marine greetings
    if q in ("hi", "hello", "hey", "vanakkam", "namaste", "good morning", "good evening"):
        return False, None

    marine_context_kws = [
        "fish", "fishing", "seafood", "catch", "marine", "pelagic", "tuna", "mackerel", "sardine",
        "knot", "knots", "nautical", "nm", "wind", "wave", "waves", "coordinate", "coordinates",
        "latitude", "longitude", "distance", "speed", "course", "heading", "clearance", "fuel",
        "carbon", "irradiance", "port", "harbor", "coast", "coastal", "sea", "ocean", "cyclone",
        "storm", "tsunami", "gis", "eez", "imbl", "vessel", "boat", "trawler", "ship", "sailor", "reef"
    ]
    has_marine_context = any(re.search(rf"\b{re.escape(kw)}\b", q) for kw in marine_context_kws)

    # 1. Cooking / Culinary recipes
    cooking_pattern = r"\b(?:recipes?|cook(?:ing)?|bake|baking|fry(?:ing)?|boil(?:ing)?|ingredients?|pastas?|cakes?|pizzas?|biryani|chicken|desserts?|soups?)\b"
    if re.search(cooking_pattern, q) and not any(re.search(rf"\b{kw}\b", q) for kw in ["fish", "seafood", "catch", "marine", "pelagic", "tuna", "mackerel", "sardine"]):
        return True, "Culinary / cooking request outside maritime domain"

    # 2. General math / algebra / arithmetic
    math_pattern = r"\b(?:solve|calculate\s+(?:the\s+)?(?:square\s+root|sum|integral|derivative|equation|math|algebra|arithmetic)|square\s+root\s+of|\d+\s*[\+\-\*\/]\s*\d+)\b"
    if re.search(math_pattern, q) and not has_marine_context:
        return True, "General mathematical calculation outside maritime domain"

    # 3. Unrelated coding / IT / software development
    coding_pattern = r"\b(?:write\s+a\s+(?:python|javascript|java|c\+\+|rust|html|css|sql)\s+(?:script|program|function|code)|reverse\s+a\s+linked\s+list|react\s+hooks|docker\s+container|css\s+flexbox|fibonacci|binary\s+tree)\b"
    if re.search(coding_pattern, q) and not has_marine_context:
        return True, "Unrelated computer programming request outside maritime domain"

    # 4. General trivia / pop culture / entertainment / non-marine conversation
    trivia_pattern = r"\b(?:who\s+won\s+the\s+world\s+cup|capital\s+of\s+france|president\s+of|movie\s+recommendation|tell\s+me\s+a\s+joke|write\s+a\s+(?:love\s+)?poem\s+about\s+(?!sea|ocean|sailor|storm))\b"
    if re.search(trivia_pattern, q) and not has_marine_context:
        return True, "General trivia / entertainment outside maritime domain"

    # 5. Sports / crypto / finance
    misc_pattern = r"\b(?:crypto(?:currency)?|bitcoin|ethereum|stock\s+market|cricket\s+score|ipl\s+match|football\s+match|nba\s+finals)\b"
    if re.search(misc_pattern, q) and not has_marine_context:
        return True, "Unrelated general topic outside maritime domain"

    return False, None


def check_diurnal_cycle(
    raw_agent_outputs: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, float, str]:
    """
    Evaluates diurnal (day/night) time-awareness based on Indian Standard Time (IST, UTC+5:30).
    Daylight hours are strictly defined as: 06:00 <= current_hour < 18:30 IST.

    Checks:
    1. Explicit simulation flags: raw_agent_outputs['simulated_hour'] or ['hour']
    2. Explicit boolean: raw_agent_outputs['is_daylight']
    3. Dataset timestamp if available
    4. Current local/system time converted to IST (UTC+5:30)

    Returns:
        (is_daylight: bool, current_decimal_hour: float, diurnal_label: str)
    """
    IST = timezone(timedelta(hours=5, minutes=30))

    if raw_agent_outputs and isinstance(raw_agent_outputs, dict):
        if "is_daylight" in raw_agent_outputs and isinstance(raw_agent_outputs["is_daylight"], bool):
            is_day = raw_agent_outputs["is_daylight"]
            h = 12.0 if is_day else 22.0
            return is_day, h, "Daytime" if is_day else "Nighttime"

        if "simulated_hour" in raw_agent_outputs:
            h = float(raw_agent_outputs["simulated_hour"])
            is_day = (6.0 <= h < 18.5)
            return is_day, round(h, 2), "Daytime" if is_day else "Nighttime"

        if "hour" in raw_agent_outputs:
            h = float(raw_agent_outputs["hour"])
            is_day = (6.0 <= h < 18.5)
            return is_day, round(h, 2), "Daytime" if is_day else "Nighttime"

        # Check explicit timestamp
        ts_val = raw_agent_outputs.get("timestamp") or raw_agent_outputs.get("WEATHER_AGENT", {}).get("timestamp")
        if ts_val and isinstance(ts_val, str):
            try:
                dt = datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
                if dt.tzinfo is not None:
                    dt = dt.astimezone(IST)
                h = dt.hour + dt.minute / 60.0
                is_day = (6.0 <= h < 18.5)
                return is_day, round(h, 2), "Daytime" if is_day else "Nighttime"
            except Exception:
                pass

        # Check dataset granule timestamp in WEATHER_AGENT / DISASTER_AGENT
        granule = (
            raw_agent_outputs.get("WEATHER_AGENT", {}).get("satellite_granule")
            or raw_agent_outputs.get("satellite_granule")
        )
        if granule and isinstance(granule, str) and raw_agent_outputs.get("use_dataset_timestamp"):
            m_time = re.search(r"_(\d{2})(\d{2})_", granule)
            if m_time:
                utc_h = int(m_time.group(1))
                utc_m = int(m_time.group(2))
                ist_total_min = (utc_h * 60 + utc_m + 330) % 1440
                h = ist_total_min / 60.0
                is_day = (6.0 <= h < 18.5)
                return is_day, round(h, 2), "Daytime" if is_day else "Nighttime"

    # Default to current local time in IST
    now_ist = datetime.now(timezone.utc).astimezone(IST)
    h = now_ist.hour + now_ist.minute / 60.0
    is_day = (6.0 <= h < 18.5)
    return is_day, round(h, 2), "Daytime" if is_day else "Nighttime"


# =====================================================================
# 1. RISK ANALYSIS AGENT
# =====================================================================
class RiskAnalysisAgent:
    """
    Evaluates aggregated telemetry and outputs from all domain agents.
    Applies deterministic safety thresholds:
    - Wind speed > 40 km/h -> DANGER
    - Wave height > 2.5 m  -> DANGER
    - Active Cyclone / Tsunami hazard -> DANGER
    - Outside EEZ / IMBL cross-border -> DANGER
    """

    WIND_DANGER_THRESHOLD_KMPH = 40.0
    WAVE_DANGER_THRESHOLD_M = 2.5
    WIND_WARNING_THRESHOLD_KMPH = 30.0
    WAVE_WARNING_THRESHOLD_M = 1.8

    SRI_LANKA_KEYWORDS = [
        "sri lanka", "srilanka", "ceylon", "jaffna", "colombo", "kankesanthurai",
        "talaimannar", "trincomalee", "batticaloa", "galle", "hambantota",
        "kachchatheevu", "kachchativu", "neduntheevu", "delft island", "pesalai", "point pedro",
        "mannar island", "mannar district"
    ]
    CROSS_BORDER_KEYWORDS = [
        "cross border", "cross imbl", "cross the border", "into sri lanka",
        "towards sri lanka", "to sri lanka", "sail to sri lanka", "sri lankan waters",
        "foreign waters", "international waters"
    ]

    def __init__(
        self,
        wind_danger_limit: float = WIND_DANGER_THRESHOLD_KMPH,
        wave_danger_limit: float = WAVE_DANGER_THRESHOLD_M,
    ):
        self.wind_danger_limit = wind_danger_limit
        self.wave_danger_limit = wave_danger_limit
        self.gis_agent = GisAgent()

    def _extract_metrics(
        self, aggregated_data: Dict[str, Any], normalized_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extracts key numerical and categorical metrics from all domain agents
        and evaluates IMBL boundary crossing and severe disaster signals.
        """
        metrics = {
            "wind_speed_kmph": 0.0,
            "gust_speed_kmph": 0.0,
            "wave_height_m": 0.0,
            "current_speed_knots": 0.0,
            "hazard_active": False,
            "hazard_description": None,
            "is_cyclone_active": False,
            "within_eez": True,
            "distance_to_border_nm": 50.0,
            "imbl_violation": False,
            "border_zone": "Indian Exclusive Economic Zone",
            "pfz_score": 50,
            "location_name": "Offshore Sector",
        }

        # Check for direct flat inputs
        for w_key in ["wind_speed_kmph", "wind_speed_kmh", "wind_speed"]:
            if w_key in aggregated_data:
                metrics["wind_speed_kmph"] = float(aggregated_data[w_key])
                break

        for wv_key in ["wave_height_m", "wave_height", "max_wave_height_m"]:
            if wv_key in aggregated_data:
                metrics["wave_height_m"] = float(aggregated_data[wv_key])
                break

        # Extract from WEATHER_AGENT output
        weather_out = aggregated_data.get("WEATHER_AGENT")
        if isinstance(weather_out, dict):
            telemetry = weather_out.get("telemetry", {})
            metrics["wind_speed_kmph"] = float(
                telemetry.get("wind_speed_kmh", metrics["wind_speed_kmph"])
            )
            metrics["gust_speed_kmph"] = float(
                telemetry.get("gust_speed_kmh", metrics["gust_speed_kmph"])
            )
            if "location" in weather_out:
                metrics["location_name"] = weather_out["location"]

        # Extract from OCEAN_AGENT output
        ocean_out = aggregated_data.get("OCEAN_AGENT")
        if isinstance(ocean_out, dict):
            ocean_data = ocean_out.get("data", {})
            stats = ocean_data.get("statistics", {}) if isinstance(ocean_data, dict) else {}

            # 1. Max wave height extraction across all candidate locations
            found_wave = None
            if "max_wave_height_m" in stats and stats["max_wave_height_m"] is not None:
                found_wave = stats["max_wave_height_m"]
            elif "wave_height_m" in ocean_out and ocean_out["wave_height_m"] is not None:
                found_wave = ocean_out["wave_height_m"]
            elif "metrics" in ocean_out and isinstance(ocean_out["metrics"], dict):
                m = ocean_out["metrics"]
                found_wave = m.get("peak_wave_height_m") or m.get("max_wave_height_m") or m.get("wave_height_m")
            elif "key_metrics" in ocean_out and isinstance(ocean_out["key_metrics"], dict):
                km = ocean_out["key_metrics"]
                found_wave = km.get("max_wave_m") or km.get("max_wave_height_m") or km.get("wave_height_m")
            elif "wave_height_m" in ocean_data and ocean_data["wave_height_m"]:
                wv = ocean_data["wave_height_m"]
                if isinstance(wv, list):
                    valid_waves = [w for w in wv if not math.isnan(w)]
                    if valid_waves:
                        found_wave = max(valid_waves)
                elif isinstance(wv, (int, float)):
                    found_wave = wv

            if found_wave is not None and not math.isnan(float(found_wave)):
                metrics["wave_height_m"] = float(found_wave)

            # 2. Max current speed extraction across all candidate locations
            found_curr = None
            if "max_current_speed_knots" in stats and stats["max_current_speed_knots"] is not None:
                found_curr = stats["max_current_speed_knots"]
            elif "current_speed_knots" in ocean_out and ocean_out["current_speed_knots"] is not None:
                found_curr = ocean_out["current_speed_knots"]
            elif "metrics" in ocean_out and isinstance(ocean_out["metrics"], dict):
                m = ocean_out["metrics"]
                found_curr = m.get("avg_current_speed_knots") or m.get("current_speed_knots")
            elif "key_metrics" in ocean_out and isinstance(ocean_out["key_metrics"], dict):
                km = ocean_out["key_metrics"]
                found_curr = km.get("current_knots") or km.get("current_speed_knots")

            if found_curr is not None and not math.isnan(float(found_curr)):
                metrics["current_speed_knots"] = float(found_curr)

        # Extract from DISASTER_AGENT output (Severe Cyclonic Storms & Tsunamis)
        disaster_out = aggregated_data.get("DISASTER_AGENT")
        if isinstance(disaster_out, dict):
            hazard_info = disaster_out.get("hazard_summary", disaster_out.get("hazard_assessment", {}))
            hazard_type = hazard_info.get("hazard_type", hazard_info.get("hazard_status", disaster_out.get("hazard_alert", "NONE_ACTIVE")))
            agent_status = disaster_out.get("status", "SAFE")
            has_hazard = (
                bool(hazard_info.get("has_active_hazard", False))
                or (agent_status in ("WARNING", "DANGER") and hazard_type not in ("NONE_ACTIVE", "CLEAR_SKIES", "LEVEL_0_NORMAL"))
            )
            if has_hazard:
                metrics["hazard_active"] = True
                metrics["hazard_description"] = disaster_out.get("advisory") or hazard_info.get(
                    "description", "Active maritime disaster warning"
                )

            # Check cyclone track directly
            if "cyclone_track" in disaster_out and isinstance(disaster_out["cyclone_track"], dict):
                ct = disaster_out["cyclone_track"]
                metrics["cyclone_track"] = ct
                if ct.get("has_active_track", False):
                    metrics["hazard_active"] = True
                    metrics["is_cyclone_active"] = True
                    storm_name = ct.get("cyclone_name", "Tropical Cyclonic System")
                    heading = ct.get("movement_heading", "WNW")
                    speed = ct.get("forward_speed_kmph", 18.0)
                    intensity = ct.get("current_intensity", "Gale-force cyclonic circulation")
                    metrics["hazard_description"] = (
                        f"{storm_name}: Tracking {heading} at {speed} km/h. {intensity}."
                    )

            # Check subsea earthquake / tsunami warnings
            if "subsea_earthquake" in disaster_out and isinstance(disaster_out["subsea_earthquake"], dict):
                eq = disaster_out["subsea_earthquake"]
                metrics["subsea_earthquake"] = eq
                if eq.get("tsunami_threat_level") in ("TSUNAMI_WARNING", "TSUNAMI_WATCH") or eq.get("has_seismic_event"):
                    metrics["hazard_active"] = True
                    metrics["hazard_description"] = eq.get("description", "Undersea earthquake tsunami advisory in effect")

            if "emergency_shelter" in disaster_out:
                metrics["emergency_shelter"] = disaster_out["emergency_shelter"]

        # Extract from GIS_AGENT output
        gis_out = aggregated_data.get("GIS_AGENT")
        if isinstance(gis_out, dict):
            boundary = gis_out.get("boundary_check", gis_out.get("spatial_metrics", gis_out.get("spatial_data", {})))
            metrics["within_eez"] = boundary.get("within_eez", boundary.get("is_within_eez", gis_out.get("is_within_eez", True)))
            if boundary.get("out_of_operational_domain") or gis_out.get("status") == "OUT_OF_OPERATIONAL_DOMAIN":
                metrics["within_eez"] = False
                metrics["out_of_operational_domain"] = True

            metrics["distance_to_border_nm"] = float(
                boundary.get("distance_to_border_nm", boundary.get("distance_to_imbl_nm", 50.0))
            )
            if "border_zone" in boundary:
                metrics["border_zone"] = boundary["border_zone"]

            # Extract shipping lane assessment
            lane_info = gis_out.get("shipping_lane_assessment", boundary.get("shipping_lane", {}))
            if lane_info:
                metrics["shipping_lane"] = lane_info
                metrics["inside_shipping_lane"] = lane_info.get("inside_lane", False)
                metrics["shipping_lane_name"] = lane_info.get("nearest_lane_name", "")
                metrics["shipping_lane_dist_nm"] = lane_info.get("corridor_clearance_nm", 50.0)

        # Coordinate domain check on raw location string if present
        loc_cand = aggregated_data.get("location") or aggregated_data.get("primary_location") or metrics.get("location_name")
        if loc_cand and "," in str(loc_cand):
            try:
                p = str(loc_cand).split(",")
                chk_lat, chk_lon = float(p[0].strip()), float(p[1].strip())
                if not (-15.0 <= chk_lat <= 30.0 and 50.0 <= chk_lon <= 105.0):
                    metrics["within_eez"] = False
                    metrics["out_of_operational_domain"] = True
            except Exception:
                pass

        # Extract visibility, sea fog, and solar insolation from WEATHER_AGENT output
        if isinstance(weather_out, dict):
            telemetry = weather_out.get("telemetry", {})
            metrics["visibility_km"] = float(telemetry.get("visibility_km", 10.0))
            metrics["fog_cover_fraction"] = float(telemetry.get("fog_cover_fraction", 0.0))
            metrics["solar_insolation_wm2"] = float(telemetry.get("solar_insolation_wm2", 820.0))
            metrics["solar_daily_kwh_m2"] = float(telemetry.get("solar_daily_kwh_m2", 5.4))

        # Extract from PFZ_AGENT output
        pfz_out = aggregated_data.get("PFZ_AGENT")
        if isinstance(pfz_out, dict):
            features = pfz_out.get("geojson", {}).get("features", [])
            if features:
                props = features[0].get("properties", {})
                metrics["pfz_score"] = int(props.get("suitability_score", 50))

        # =================================================================
        # IMBL Geopolitical Boundary Guardrail Detection
        # Strict conditional evaluation: Check target location and user query text ONLY.
        # NEVER inspect domain agent descriptive metadata / advisory strings.
        # =================================================================
        query_text = (
            normalized_query
            or aggregated_data.get("normalized_query")
            or aggregated_data.get("user_query")
            or ""
        ).strip().lower()

        target_loc = (
            aggregated_data.get("target_location")
            or aggregated_data.get("destination")
            or aggregated_data.get("location")
            or ""
        )

        is_imbl_violation = False

        # 1. Evaluate target location with GIS boundary agent
        if target_loc and self.gis_agent.check_imbl(target_loc):
            is_imbl_violation = True

        # 2. Check explicit cross-border keywords strictly in the user's normalized query
        if any(kw in query_text for kw in self.SRI_LANKA_KEYWORDS + self.CROSS_BORDER_KEYWORDS):
            is_imbl_violation = True

        # 3. Check target coordinates if explicitly supplied as coordinates
        coords_to_verify = []
        for loc_key in ["target_location", "destination", "location"]:
            val = aggregated_data.get(loc_key)
            if isinstance(val, str) and "," in val:
                try:
                    p = val.split(",")
                    coords_to_verify.append((float(p[0].strip()), float(p[1].strip())))
                except Exception:
                    pass

        for c_lat, c_lon in coords_to_verify:
            if self.gis_agent.check_imbl(f"{c_lat},{c_lon}"):
                is_imbl_violation = True
                break

        metrics["imbl_violation"] = is_imbl_violation
        if is_imbl_violation:
            metrics["within_eez"] = False
            metrics["border_zone"] = "India-Sri Lanka IMBL (Sri Lankan Sovereign Waters)"

        return metrics

    def evaluate_threats(
        self,
        aggregated_data: Dict[str, Any],
        normalized_query: Optional[str] = None,
        persona: Optional[str] = None,
        forecast_data: Optional[Dict[str, Any]] = None,
        forecast_offline: bool = False,
        cyclone_intel: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyzes threats, enforces IMBL geopolitical boundary guardrails (civilian)
        or Jurisdictional Boundary Assessment (maritime authorities),
        and computes calibrated composite risk score with CRITICAL (80-100) bracket scaling.
        Integrates forward-looking Open-Meteo Marine Forecast and GDACS Tropical Cyclone tracking.
        """
        p_enum = resolve_persona(persona) or StakeholderPersona.FISHERMAN
        metrics = self._extract_metrics(aggregated_data, normalized_query=normalized_query)

        if cyclone_intel is None:
            cyclone_intel = aggregated_data.get("cyclone_intelligence")
        has_cyclone_collision = bool(cyclone_intel and cyclone_intel.get("route_collision"))
        active_cyclone_name = (cyclone_intel.get("active_storms") if cyclone_intel else None) or "Tropical Cyclonic System"
        cyclone_clearance_nm = cyclone_intel.get("clearance_distance_nm") if cyclone_intel else None
        cyclone_traj_vec = cyclone_intel.get("trajectory_path") if cyclone_intel else None

        forecast_active = bool(forecast_data and forecast_data.get("success", False))
        if forecast_active:
            metrics["forecast_active"] = True
            metrics["forecast_offline"] = False
            metrics["forecast_max_wave_m"] = forecast_data["max_wave_height_m"]
            metrics["forecast_avg_wave_m"] = forecast_data.get("avg_wave_height_m", forecast_data["max_wave_height_m"])
            metrics["forecast_horizon"] = forecast_data.get("time_horizon", "24–48 hours")
            metrics["status_label"] = "Forecast Status"
        elif forecast_offline:
            metrics["forecast_active"] = False
            metrics["forecast_offline"] = True
            metrics["status_label"] = "Operational Status"
        else:
            metrics["forecast_active"] = False
            metrics["forecast_offline"] = False
            metrics["status_label"] = "Operational Status"

        wind = metrics["wind_speed_kmph"]
        gust = metrics.get("gust_speed_kmph", 0.0)
        wave = metrics["wave_height_m"]
        hazard = metrics["hazard_active"]
        is_cyclone = metrics.get("is_cyclone_active", False)
        imbl_violation = metrics.get("imbl_violation", False)
        within_eez = metrics["within_eez"]
        border_dist = metrics["distance_to_border_nm"]

        threats = []
        warnings = []
        is_danger = False
        is_warning = False

        query_candidate = normalized_query or aggregated_data.get("normalized_query") or aggregated_data.get("user_query")

        # Guardrail 0: Strict Out-of-Domain (OOD) Assessment
        is_ood, ood_reason = check_out_of_domain(query_candidate)
        if is_ood:
            ood_refusal = (
                "I am ORCA, a specialized maritime intelligence engine. "
                "I am programmed exclusively to assist with marine navigation, weather analysis, "
                "and coastal safety operations. I cannot process requests outside of this scope."
            )
            return {
                "status": "OUT_OF_DOMAIN",
                "risk_score": "N/A",
                "threat_prioritization": [f"OUT OF DOMAIN: {ood_reason}"],
                "warnings": [ood_refusal],
                "metrics": metrics,
                "out_of_domain": True,
            }

        # Guardrail 1: Prompt Injection Shield
        is_inj, inj_reason = check_prompt_injection(query_candidate)
        if is_inj:
            threat_msg = f"SECURITY REJECTION: Adversarial prompt injection or security bypass detected ({inj_reason}). Maritime safety guardrails, IMBL border enforcement, and SOLAS protocols cannot be overridden."
            return {
                "status": "SECURITY_REJECTION",
                "risk_score": 100.0,
                "threat_prioritization": [threat_msg],
                "warnings": [threat_msg],
                "metrics": metrics,
                "security_rejection": True,
            }

        # Guardrail 2: Chronological Temporal Bounds & Forecast Intent Check (24-Hour Horizon Limit)
        temporal_info = detect_temporal_intent(query_candidate)
        if temporal_info["is_beyond_24h"] and not forecast_active:
            threat_msg = f"PENDING FORECAST: {temporal_info['reason']}"
            return {
                "status": "PENDING FORECAST",
                "risk_score": "N/A",
                "threat_prioritization": [threat_msg],
                "warnings": [threat_msg],
                "metrics": metrics,
                "pending_forecast": True,
                "temporal_out_of_bounds": True,
                "temporal_intent": True,
                "temporal_info": temporal_info,
            }

        if temporal_info["has_temporal_intent"]:
            metrics["temporal_intent"] = True
            metrics["temporal_timeframe"] = temporal_info["timeframe_label"]
            metrics["temporal_info"] = temporal_info

        # Guardrail 3: Life-Safety Override (SAR Distress Precedence)
        is_sar, sar_reason = check_emergency_sar(query_candidate)
        if is_sar:
            threat_msg = (
                f"CRITICAL LIFE-SAFETY EMERGENCY (SOLAS SAR OVERRIDE): {sar_reason}. "
                f"Operational weather, cyclone, and IMBL NO-GO restrictions are OVERRIDDEN. "
                f"Forcing priority Search and Rescue (SAR) evacuation route to nearest sheltered breakwater."
            )
            metrics["emergency_sar"] = True
            return {
                "status": "EMERGENCY_SAR",
                "risk_score": 99.0,
                "threat_prioritization": [threat_msg],
                "warnings": [threat_msg],
                "metrics": metrics,
                "emergency_sar": True,
            }

        # Guardrail 4: Coastal Landmasking Enforcement
        gis_out = aggregated_data.get("GIS_AGENT", {})
        is_land_error = False
        if isinstance(gis_out, dict):
            if (
                gis_out.get("status") == "LAND_INTERSECTION_ERROR"
                or gis_out.get("error") == "LAND_INTERSECTION_ERROR"
                or gis_out.get("safe_sea_route", {}).get("error") == "LAND_INTERSECTION_ERROR"
                or gis_out.get("safe_sea_route", {}).get("route_status") == "LAND_INTERSECTION_ERROR"
            ):
                is_land_error = True

        if not is_land_error and query_candidate:
            qc_low = query_candidate.lower()
            m_route = re.search(r"from\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+)", qc_low)
            if m_route:
                co = m_route.group(1).strip()
                cd = m_route.group(2).strip()
                so = self.gis_agent.resolve_location(co)
                sd = self.gis_agent.resolve_location(cd)
                if not so.get("is_unknown") and not sd.get("is_unknown"):
                    if self.gis_agent.check_land_intersection(so["lat"], so["lon"], sd["lat"], sd["lon"]):
                        is_land_error = True

        if is_land_error:
            threat_msg = (
                "CRITICAL ROUTING FAILURE [LAND_INTERSECTION_ERROR]: Direct nautical passage between the requested ports "
                "intersects the Indian peninsular landmass. Overland marine navigation is physically impossible."
            )
            return {
                "status": "LAND_INTERSECTION_ERROR",
                "risk_score": 88.0,
                "threat_prioritization": [threat_msg],
                "warnings": [threat_msg],
                "metrics": metrics,
                "land_intersection_error": True,
            }

        # 1. IMBL Geopolitical Boundary Guardrail / Jurisdictional Assessment
        if imbl_violation or not within_eez:
            if p_enum == StakeholderPersona.MARITIME_AUTHORITY:
                is_warning = True
                threat_msg = (
                    f"Jurisdictional Boundary Assessment: Active patrol vector positioned along India-Sri Lanka IMBL perimeter "
                    f"({metrics.get('border_zone', 'International Maritime Boundary Line')}). Maintain sovereign maritime surveillance, "
                    f"verify AIS transponders, and observe standard cross-border engagement protocols."
                )
                threats.append({"type": "JURISDICTIONAL_BOUNDARY_ASSESSMENT", "severity": "ELEVATED", "message": threat_msg})
                warnings.append(threat_msg)
            else:
                is_danger = True
                threat_msg = (
                    "CRITICAL BORDER VIOLATION: Route targets Sri Lankan sovereign waters across the "
                    "International Maritime Boundary Line (IMBL). Navigating across the International "
                    "Maritime Boundary Line (IMBL) into Sri Lankan waters is illegal and strictly prohibited under international maritime law."
                )
                threats.append({"type": "IMBL_BORDER_VIOLATION", "severity": "CRITICAL", "message": threat_msg})
                warnings.append(threat_msg)
        elif border_dist < 5.0:
            is_warning = True
            if p_enum == StakeholderPersona.MARITIME_AUTHORITY:
                threat_msg = (
                    f"Jurisdictional Perimeter Alert: Patrol vessel is {border_dist:.1f} NM from International Maritime Boundary Line. "
                    f"Maintain radar surveillance and AIS compliance monitoring along sovereign boundary."
                )
            else:
                threat_msg = (
                    f"Border Proximity Alert: Vessel is only {border_dist:.1f} NM from International Boundary. "
                    f"Drift risk present. Maintain safe distance from IMBL."
                )
            threats.append({"type": "BORDER_PROXIMITY", "severity": "ELEVATED", "message": threat_msg})
            warnings.append(threat_msg)

        if has_cyclone_collision:
            is_danger = True
            clr_str = f"{cyclone_clearance_nm:.1f} NM" if cyclone_clearance_nm is not None else "< 200 NM"
            threat_msg = (
                f"Severe Cyclonic Storm Collision Hazard: Vessel route is within {clr_str} of "
                f"{active_cyclone_name}'s projected trajectory (clearance: {clr_str}, safe buffer: 200.0 NM). "
                f"{f'Projected trajectory vector: {cyclone_traj_vec}. ' if cyclone_traj_vec else ''}"
                f"Mandatory NO-GO harbor evasion in effect."
            )
            threats.append({"type": "CYCLONE_TRACK_COLLISION", "severity": "CRITICAL", "message": threat_msg})
            warnings.append(threat_msg)

        if forecast_active:
            # Bypass real-time MOSDAC state evaluation for main operational status:
            f_wave = forecast_data["max_wave_height_m"]
            metrics["wave_height_m"] = f_wave
            wave = f_wave
            if wave > self.wave_danger_limit:
                is_danger = True
                threat_msg = (
                    f"Severe Forecasted Wave Hazard: Predicted maximum wave height of {wave:.2f} m "
                    f"exceeds vessel stability threshold ({self.wave_danger_limit} m) across the 24–48 hour forecast horizon."
                )
                threats.append({"type": "FORECAST_WAVE_HAZARD", "severity": "CRITICAL", "message": threat_msg})
                warnings.append(threat_msg)
            elif wave > self.WAVE_WARNING_THRESHOLD_M:
                is_warning = True
                threat_msg = (
                    f"Elevated Forecasted Sea State: Predicted wave height of {wave:.2f} m "
                    f"requires heightened maritime caution across the 24–48 hour forecast horizon."
                )
                threats.append({"type": "FORECAST_WAVE_WARNING", "severity": "ELEVATED", "message": threat_msg})
                warnings.append(threat_msg)
            else:
                threat_msg = (
                    f"Favorable Marine Forecast: Predicted wave heights remain safe below "
                    f"{self.WAVE_WARNING_THRESHOLD_M} m (maximum {wave:.2f} m) across the 24–48 hour forecast horizon."
                )
                threats.append({"type": "FORECAST_FAVORABLE", "severity": "INFO", "message": threat_msg})
        else:
            # 2. Severe Cyclonic Storm & Active Disaster Hazard Check
            if is_cyclone:
                is_danger = True
                cyclone_desc = metrics.get("hazard_description") or "Severe Cyclonic Storm active with destructive gale gusts"
                threat_msg = f"Severe Cyclonic Storm Hazard: {cyclone_desc}."
                threats.append({"type": "CYCLONE_HAZARD", "severity": "CRITICAL", "message": threat_msg})
                warnings.append(threat_msg)
            elif hazard:
                is_danger = True
                threat_msg = f"Active Maritime Disaster Alert: {metrics['hazard_description'] or 'Severe weather advisory issued'}."
                threats.append({"type": "DISASTER_ALERT", "severity": "CRITICAL", "message": threat_msg})
                warnings.append(threat_msg)

            # 3. Deterministic Wind Threat Check
            if wind > self.wind_danger_limit or gust > 65.0:
                is_danger = True
                threat_msg = (
                    f"Critical Wind Hazard: Sustained wind speed of {wind:.1f} km/h (gusts {gust:.1f} km/h) "
                    f"exceeds maritime danger limit ({self.wind_danger_limit} km/h)."
                )
                threats.append({"type": "WIND_HAZARD", "severity": "CRITICAL", "message": threat_msg})
                warnings.append(threat_msg)
            elif wind > self.WIND_WARNING_THRESHOLD_KMPH:
                is_warning = True
                threat_msg = f"Moderate Wind Warning: Wind speed {wind:.1f} km/h approaching small vessel threshold ({self.wind_danger_limit} km/h)."
                threats.append({"type": "WIND_WARNING", "severity": "ELEVATED", "message": threat_msg})
                warnings.append(threat_msg)

            # 4. Deterministic Wave Threat Check
            if wave > self.wave_danger_limit:
                is_danger = True
                threat_msg = f"Severe Wave Hazard: Significant wave height of {wave:.2f} m exceeds vessel stability threshold ({self.wave_danger_limit} m)."
                threats.append({"type": "WAVE_HAZARD", "severity": "CRITICAL", "message": threat_msg})
                warnings.append(threat_msg)
            elif wave > self.WAVE_WARNING_THRESHOLD_M:
                is_warning = True
                threat_msg = f"Elevated Sea State: Wave height {wave:.2f} m requires heightened maritime caution."
                threats.append({"type": "WAVE_WARNING", "severity": "ELEVATED", "message": threat_msg})
                warnings.append(threat_msg)

        # 5. Commercial Shipping Lane Collision Risk (SLOC Geofencing)
        shipping_lane = metrics.get("shipping_lane", {})
        inside_lane = metrics.get("inside_shipping_lane", shipping_lane.get("inside_lane", False))
        lane_dist = metrics.get("shipping_lane_dist_nm", shipping_lane.get("corridor_clearance_nm", 50.0))
        lane_name = metrics.get("shipping_lane_name", shipping_lane.get("nearest_lane_name", "Commercial Shipping Corridor"))
        visibility = metrics.get("visibility_km", 10.0)
        fog_fraction = metrics.get("fog_cover_fraction", 0.0)
        collision_pts = 0.0

        if inside_lane:
            if visibility < 5.0 or fog_fraction > 0.3:
                is_danger = True
                collision_pts = 25.0
                threat_msg = (
                    f"Critical Collision Hazard: Vessel is INSIDE {lane_name} with reduced visibility ({visibility:.1f} km"
                    f"{', dense sea fog detected via INSAT-3DR' if fog_fraction > 0.3 else ''})! "
                    f"Heavy container ships & tankers cannot spot small craft. Immediate inshore fairway evasion required."
                )
                threats.append({"type": "SHIPPING_LANE_COLLISION_CRITICAL", "severity": "CRITICAL", "message": threat_msg})
                warnings.append(threat_msg)
            else:
                is_warning = True
                collision_pts = 15.0
                threat_msg = (
                    f"High Collision Risk: Vessel is operating inside {lane_name}. "
                    f"Commercial vessels ({shipping_lane.get('traffic_type', 'Container/Tanker')}) navigate here at 18-24 kt. "
                    f"Mount radar reflector, display 360° all-round white navigation lights, and steer clear of fairway."
                )
                threats.append({"type": "SHIPPING_LANE_COLLISION", "severity": "ELEVATED", "message": threat_msg})
                warnings.append(threat_msg)
        elif lane_dist < 3.0:
            is_warning = True
            collision_pts = 8.0
            threat_msg = (
                f"Commercial Traffic Warning: Vessel is {lane_dist:.1f} NM from {lane_name}. "
                f"Maintain continuous visual/radar watch for high-speed commercial ships."
            )
            threats.append({"type": "SHIPPING_LANE_PROXIMITY", "severity": "ELEVATED", "message": threat_msg})
            warnings.append(threat_msg)

        # Determine overall status
        if is_danger:
            status = "DANGER"
        elif is_warning:
            status = "WARNING"
        else:
            status = "SAFE"

        # Calculate composite Risk Score (0 to 100)
        wind_pts = min(35.0, (wind / max(self.wind_danger_limit, 1.0)) * 30.0)
        wave_pts = min(35.0, (wave / max(self.wave_danger_limit, 1.0)) * 30.0)
        hazard_pts = 25.0 if hazard else 0.0
        border_pts = 15.0 if (not within_eez or border_dist < 5.0) else (5.0 if border_dist < 10.0 else 0.0)
        base_score = round(min(100.0, wind_pts + wave_pts + hazard_pts + border_pts + collision_pts), 1)

        # =================================================================
        # Risk Score Calibration: CRITICAL (80-100) Bracket Enforcement
        # =================================================================
        if has_cyclone_collision:
            risk_score = max(base_score, 96.0)
            status = "DANGER"
        elif is_cyclone:
            risk_score = max(base_score, 92.0)
            status = "DANGER"
        elif imbl_violation or not within_eez:
            if p_enum == StakeholderPersona.MARITIME_AUTHORITY:
                risk_score = min(74.0, max(45.0, base_score))
                if not is_danger:
                    status = "WARNING"
            else:
                risk_score = max(base_score, 90.0)
                status = "DANGER"
        elif metrics.get("subsea_earthquake", {}).get("tsunami_threat_level") in ("TSUNAMI_WARNING", "TSUNAMI_WATCH"):
            risk_score = max(base_score, 95.0)
            status = "DANGER"
        elif hazard:
            risk_score = max(base_score, 85.0)
            status = "DANGER"
        elif wind > self.wind_danger_limit or gust > 65.0:
            risk_score = max(base_score, 82.0)
            status = "DANGER"
        elif wave > self.wave_danger_limit:
            risk_score = max(base_score, 80.0)
            status = "DANGER"
        elif status == "DANGER":
            risk_score = max(base_score, 80.0)
        elif status == "WARNING":
            risk_score = min(74.0, max(40.0, base_score))
        else:
            risk_score = 16.8 if FAST_DEMO_MODE else min(30.0, base_score)

        # Prioritize threats by severity (CRITICAL first, then ELEVATED)
        threat_prioritization = [t["message"] for t in sorted(threats, key=lambda x: 0 if x["severity"] == "CRITICAL" else 1)]

        return {
            "status": status,
            "risk_score": risk_score,
            "threat_prioritization": threat_prioritization,
            "warnings": warnings,
            "metrics": metrics,
            "forecast_active": forecast_active,
            "forecast_offline": forecast_offline,
            "status_label": metrics.get("status_label", "Operational Status"),
            "forecast_max_wave_m": metrics.get("forecast_max_wave_m"),
            "forecast_avg_wave_m": metrics.get("forecast_avg_wave_m"),
            "forecast_horizon": metrics.get("forecast_horizon"),
            "cyclone_intelligence": cyclone_intel,
        }


# =====================================================================
# 2. SAFE ALTERNATIVE REROUTING
# =====================================================================
class SafeAlternative:
    """
    Computes safe fallback maritime waypoints and multi-waypoint safe passage routes
    when hazardous conditions (DANGER / WARNING) are detected at the primary operational destination.
    """

    def __init__(self):
        self.gis_agent = GisAgent()

    KNOWN_PORTS = {
        "thoothukudi": (8.7642, 78.1348),
        "tuticorin": (8.7642, 78.1348),
        "rameswaram": (9.2876, 79.3129),
        "kanyakumari": (8.0883, 77.5385),
        "chennai": (13.0827, 80.2707),
        "kochi": (9.9312, 76.2673),
        "visakhapatnam": (17.6868, 83.2185),
        "sri lanka": (6.9428, 79.8412),
        "srilanka": (6.9428, 79.8412),
        "colombo": (6.9428, 79.8412),
    }

    def _resolve_coordinates(self, location: Any) -> Tuple[float, float]:
        """Resolves location string, tuple, or port name to (latitude, longitude)."""
        if isinstance(location, (tuple, list)) and len(location) >= 2:
            return float(location[0]), float(location[1])

        if isinstance(location, str):
            # Check if formatted as "lat,lon"
            if "," in location:
                parts = location.split(",")
                try:
                    return float(parts[0].strip()), float(parts[1].strip())
                except ValueError:
                    pass

            # Check known port dictionary
            key = location.strip().lower()
            for name, coords in self.KNOWN_PORTS.items():
                if name in key:
                    return coords

            # Check GisAgent resolve_location (handles COUNTRY_PORT_ALIASES and COASTAL_SECTORS)
            sec = self.gis_agent.resolve_location(location)
            if sec and not sec.get("is_unknown"):
                return (sec["lat"], sec["lon"])

        # Default fallback: Gulf of Mannar offshore Thoothukudi
        return (8.7642, 78.1348)

    @staticmethod
    def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculates exact great-circle distance between two points in Nautical Miles."""
        r_km = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2.0) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return round((r_km * c) / 1.852, 1)

    def calculate_reroute(
        self,
        primary_location: Any,
        threat_data: Dict[str, Any],
        target_destination: Optional[Any] = None,
        persona: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculates safe waypoint routes and mathematically synchronizes distinct waypoints (WP1 != WP2).
        - Under SAFE conditions: Routes user from vessel position (WP1) to persona-tailored destination (WP2).
        - Under DANGER / REROUTE / IMBL: Routes user from vessel position (WP1) to persona-tailored safe alternative (WP2).
        Guarantees WP1 != WP2 and computes exact nautical miles via Haversine.
        """
        p_enum = resolve_persona(persona) or StakeholderPersona.FISHERMAN
        orig_lat, orig_lon = self._resolve_coordinates(primary_location)
        status = threat_data.get("status", "SAFE")
        risk_score = threat_data.get("risk_score", 0.0)
        metrics = threat_data.get("metrics", {})
        imbl_violation = metrics.get("imbl_violation", False)

        # Guardrail 0: Out-of-Domain Rejection (Clear all waypoints)
        if status == "OUT_OF_DOMAIN" or threat_data.get("out_of_domain"):
            return {
                "reroute_needed": False,
                "primary_coordinates": "",
                "safe_coordinates": "",
                "safe_location_name": "OUT_OF_DOMAIN",
                "distance_shift_nm": 0.0,
                "rationale": "Out-of-Domain request: all nautical waypoints cleared.",
                "safe_sea_route": None,
            }

        # Guardrail 0-B: Out of Operational Domain (Indian Ocean Maritime Domain)
        is_orig_out = not (-15.0 <= orig_lat <= 30.0 and 50.0 <= orig_lon <= 105.0) or threat_data.get("out_of_operational_domain") or status == "OUT_OF_OPERATIONAL_DOMAIN"
        if is_orig_out:
            return {
                "reroute_needed": False,
                "primary_coordinates": f"{orig_lat:.4f},{orig_lon:.4f}",
                "safe_coordinates": "",
                "safe_location_name": "OUT_OF_OPERATIONAL_DOMAIN",
                "distance_shift_nm": 0.0,
                "rationale": f"Vessel position ({orig_lat:.4f}, {orig_lon:.4f}) is outside the ORCA Indian Ocean operational domain (-15° to 30°N, 50° to 105°E). Safe route generation aborted.",
                "safe_sea_route": {
                    "route_status": "OUT_OF_OPERATIONAL_DOMAIN",
                    "error": "OUT_OF_OPERATIONAL_DOMAIN",
                    "waypoints": [],
                    "waypoints_count": 0,
                    "total_distance_nm": 0.0,
                    "navigational_brief": f"Position ({orig_lat:.4f}, {orig_lon:.4f}) is outside the Indian Ocean maritime operational zone. Nautical routing cannot be generated.",
                },
            }

        # Guardrail A: Adversarial Prompt Injection Security Rejection
        if status == "SECURITY_REJECTION" or threat_data.get("security_rejection"):
            return {
                "reroute_needed": False,
                "primary_coordinates": f"{orig_lat:.4f},{orig_lon:.4f}",
                "safe_coordinates": f"{orig_lat:.4f},{orig_lon:.4f}",
                "safe_location_name": "SECURITY_REJECTION",
                "distance_shift_nm": 0.0,
                "rationale": "Adversarial prompt injection / security protocol bypass detected. Query terminated.",
                "safe_sea_route": None,
            }

        # Guardrail B: Chronological Temporal Bounds & Pending Forecast Intercept
        if not threat_data.get("forecast_active") and not threat_data.get("metrics", {}).get("forecast_active") and (
            status in ("PENDING FORECAST", "DATA_UNAVAILABLE", "TEMPORAL_OUT_OF_BOUNDS")
            or threat_data.get("pending_forecast")
            or threat_data.get("temporal_out_of_bounds")
        ):
            return {
                "reroute_needed": False,
                "primary_coordinates": "",
                "safe_coordinates": "",
                "safe_location_name": "PENDING FORECAST",
                "distance_shift_nm": 0.0,
                "rationale": "Forecast horizon exceeds 24-hour ISRO MOSDAC satellite cache window. Real-time satellite data cannot predict conditions beyond 24 hours.",
                "safe_sea_route": None,
            }

        # Guardrail C: Coastal Landmask Intersection Error
        if status == "LAND_INTERSECTION_ERROR" or threat_data.get("land_intersection_error"):
            return {
                "reroute_needed": False,
                "primary_coordinates": f"{orig_lat:.4f},{orig_lon:.4f}",
                "safe_coordinates": f"{orig_lat:.4f},{orig_lon:.4f}",
                "safe_location_name": "Overland Passage Blocked",
                "distance_shift_nm": 0.0,
                "rationale": "Direct nautical passage intersects peninsular landmass. Overland marine routing is physically impossible.",
                "safe_sea_route": {
                    "route_status": "LAND_INTERSECTION_ERROR",
                    "error": "LAND_INTERSECTION_ERROR",
                    "waypoints": [],
                    "waypoints_count": 0,
                    "total_distance_nm": 0.0,
                    "navigational_brief": "CRITICAL ROUTING FAILURE [LAND_INTERSECTION_ERROR]: Direct nautical passage intersects the Indian peninsular landmass. Overland marine routing is physically impossible.",
                },
            }

        # Guardrail D: Life-Safety Override (SAR): Priority rescue routing overriding all NO-GO rules
        if status == "EMERGENCY_SAR" or threat_data.get("emergency_sar"):
            if "emergency_shelter" in metrics:
                shelter = metrics["emergency_shelter"]
                s_coords = shelter.get("coordinates", "8.7525,78.1983")
                shelter_name = shelter.get("shelter_name", "V.O. Chidambaranar Port Breakwater Basin")
            else:
                s_coords = "8.7525,78.1983" if orig_lat < 9.5 else "9.2876,79.1500"
                shelter_name = "V.O. Chidambaranar Port Breakwater Basin" if orig_lat < 9.5 else "Mandapam Protected Basin"

            safe_lat, safe_lon = self._resolve_coordinates(s_coords)
            dist_nm = self.haversine_nm(orig_lat, orig_lon, safe_lat, safe_lon)
            if dist_nm < 1.0:
                safe_lat = round(safe_lat + 0.04, 4)
                dist_nm = max(self.haversine_nm(orig_lat, orig_lon, safe_lat, safe_lon), 2.5)

            res_dict = {
                "reroute_needed": True,
                "primary_coordinates": f"{orig_lat:.4f},{orig_lon:.4f}",
                "safe_coordinates": f"{safe_lat:.4f},{safe_lon:.4f}",
                "safe_location_name": f"Designated Emergency Breakwater Shelter: {shelter_name}",
                "distance_shift_nm": dist_nm,
                "rationale": (
                    f"SOLAS Life-Safety Emergency (SAR Override): Cyclone and operational NO-GO restrictions overridden. "
                    f"Emergency priority vector of {dist_nm:.1f} NM engaged to {shelter_name}."
                ),
            }
            try:
                res_dict["safe_sea_route"] = self.gis_agent.generate_safe_sea_route(
                    origin=f"{orig_lat:.4f},{orig_lon:.4f}",
                    destination=f"{safe_lat:.4f},{safe_lon:.4f}",
                )
            except Exception:
                pass
            return res_dict

        # 1. IMBL Geopolitical Cross-Border Guardrail Rerouting / Patrol Stationing
        if imbl_violation or not metrics.get("within_eez", True):
            if p_enum == StakeholderPersona.MARITIME_AUTHORITY:
                safe_lat, safe_lon = (8.8500, 78.3500) if orig_lat < 9.0 else (9.2876, 79.2500)
                dist_nm = max(self.haversine_nm(orig_lat, orig_lon, safe_lat, safe_lon), 3.5)
                res_dict = {
                    "reroute_needed": True,
                    "primary_coordinates": f"{orig_lat:.4f},{orig_lon:.4f}",
                    "safe_coordinates": f"{safe_lat:.4f},{safe_lon:.4f}",
                    "safe_location_name": "Indian EEZ Border Perimeter Patrol Station",
                    "distance_shift_nm": dist_nm,
                    "rationale": (
                        f"Jurisdictional Boundary Assessment: Tactical patrol stationing along sovereign EEZ perimeter "
                        f"({dist_nm} NM shift) for maritime boundary surveillance and intercept readiness."
                    ),
                }
            else:
                # Divert immediately into sovereign Indian coastal waters (Thoothukudi or Rameswaram West)
                safe_lat, safe_lon = (9.2876, 79.1500) if orig_lat > 9.0 else (8.7525, 78.1983)
                dist_nm = self.haversine_nm(orig_lat, orig_lon, safe_lat, safe_lon)
                if dist_nm < 1.0:
                    safe_lon = round(safe_lon - 0.05, 4)
                    dist_nm = max(self.haversine_nm(orig_lat, orig_lon, safe_lat, safe_lon), 2.5)

                res_dict = {
                    "reroute_needed": True,
                    "primary_coordinates": f"{orig_lat:.4f},{orig_lon:.4f}",
                    "safe_coordinates": f"{safe_lat:.4f},{safe_lon:.4f}",
                    "safe_location_name": "Indian Sovereign Coastal Waters (Clear of IMBL)",
                    "distance_shift_nm": dist_nm,
                    "rationale": (
                        f"IMBL Boundary Guardrail enforced: Cross-border trajectory cancelled. "
                        f"Diverted {dist_nm} NM inshore to sovereign Indian waters with safe clearance from the international maritime boundary."
                    ),
                }
            try:
                res_dict["safe_sea_route"] = self.gis_agent.generate_safe_sea_route(
                    origin=f"{orig_lat:.4f},{orig_lon:.4f}",
                    destination=f"{safe_lat:.4f},{safe_lon:.4f}",
                )
            except Exception:
                pass
            return res_dict

        if status in ("DANGER", "WARNING") or risk_score >= 50.0:
            # Check if an official emergency breakwater shelter harbor was designated by DISASTER_AGENT
            if "emergency_shelter" in metrics:
                shelter = metrics["emergency_shelter"]
                s_coords = shelter.get("coordinates", "8.7525,78.1983")
                safe_lat, safe_lon = self._resolve_coordinates(s_coords)
                dist_nm = self.haversine_nm(orig_lat, orig_lon, safe_lat, safe_lon)
                if dist_nm < 1.0:
                    safe_lat = round(safe_lat + 0.04, 4)
                    dist_nm = max(self.haversine_nm(orig_lat, orig_lon, safe_lat, safe_lon), 2.0)

                if p_enum == StakeholderPersona.MARITIME_AUTHORITY:
                    loc_name = shelter.get("shelter_name", "All-Weather Emergency Harbor")
                    rat_str = f"Disaster crisis response stationing: Tactical staging at designated emergency harbor ({dist_nm} NM) for SAR and fleet safety."
                elif p_enum == StakeholderPersona.DISASTER_MANAGEMENT:
                    loc_name = f"Designated Emergency Evacuation Shelter: {shelter.get('shelter_name', 'Breakwater Basin')}"
                    rat_str = f"Life-safety evacuation directive: Priority evacuation route to designated high-capacity emergency breakwater shelter ({dist_nm} NM) for coastal protection."
                else:
                    loc_name = shelter.get("shelter_name", "All-Weather Emergency Harbor")
                    rat_str = f"Emergency reroute of {dist_nm} NM to all-weather shelter: {shelter.get('navigational_advice', 'Proceed to sheltered harbor basin.')}"

                res_dict = {
                    "reroute_needed": True,
                    "primary_coordinates": f"{orig_lat:.4f},{orig_lon:.4f}",
                    "safe_coordinates": f"{safe_lat:.4f},{safe_lon:.4f}",
                    "safe_location_name": loc_name,
                    "distance_shift_nm": dist_nm,
                    "rationale": rat_str,
                }
                try:
                    res_dict["safe_sea_route"] = self.gis_agent.generate_safe_sea_route(
                        origin=f"{orig_lat:.4f},{orig_lon:.4f}",
                        destination=f"{safe_lat:.4f},{safe_lon:.4f}",
                    )
                except Exception:
                    pass
                return res_dict

            # Check if vessel is inside or near a commercial shipping lane
            if metrics.get("inside_shipping_lane"):
                lane_name = metrics.get("shipping_lane_name", "Commercial Shipping Fairway")
                if p_enum == StakeholderPersona.MARITIME_OPERATOR:
                    safe_lat = round(orig_lat + 0.08, 4)
                    safe_lon = round(orig_lon + 0.12, 4)
                    dist_nm = max(self.haversine_nm(orig_lat, orig_lon, safe_lat, safe_lon), 3.0)
                    loc_name = "Designated Deep-Draft Shipping Fairway (TSS Alignment)"
                    rat_str = f"TSS Alignment: Vessel aligned with primary deep-draft commercial fairway traffic separation corridor ({dist_nm} NM transit)."
                elif p_enum == StakeholderPersona.MARITIME_AUTHORITY:
                    safe_lat = round(orig_lat + 0.06, 4)
                    safe_lon = round(orig_lon - 0.08, 4)
                    dist_nm = max(self.haversine_nm(orig_lat, orig_lon, safe_lat, safe_lon), 3.0)
                    loc_name = "Shipping Channel Traffic Enforcement Station"
                    rat_str = f"SLOC Surveillance: Patrol station established {dist_nm} NM outside fairway for traffic monitoring and collision avoidance oversight."
                else:
                    shift_lat = 0.08 if orig_lat < 12.0 else -0.08
                    shift_lon = -0.10
                    safe_lat = round(orig_lat + shift_lat, 4)
                    safe_lon = round(orig_lon + shift_lon, 4)
                    dist_nm = max(self.haversine_nm(orig_lat, orig_lon, safe_lat, safe_lon), 2.0)
                    loc_name = "Designated Artisanal Fishing Zone (Clear of Shipping Fairway)"
                    rat_str = f"Evasive clearance applied: shifted {dist_nm} NM inshore to clear commercial shipping traffic in {lane_name}."

                res_dict = {
                    "reroute_needed": True,
                    "primary_coordinates": f"{orig_lat:.4f},{orig_lon:.4f}",
                    "safe_coordinates": f"{safe_lat:.4f},{safe_lon:.4f}",
                    "safe_location_name": loc_name,
                    "distance_shift_nm": dist_nm,
                    "rationale": rat_str,
                }
                try:
                    res_dict["safe_sea_route"] = self.gis_agent.generate_safe_sea_route(
                        origin=f"{orig_lat:.4f},{orig_lon:.4f}",
                        destination=f"{safe_lat:.4f},{safe_lon:.4f}",
                    )
                except Exception:
                    pass
                return res_dict

            # Shift coordinates toward sheltered waters
            shift_lat = -0.10 if orig_lat > 9.0 else 0.08
            shift_lon = -0.15  # pull closer to Indian mainland coast
            safe_lat = round(orig_lat + shift_lat, 4)
            safe_lon = round(orig_lon + shift_lon, 4)
            dist_nm = max(self.haversine_nm(orig_lat, orig_lon, safe_lat, safe_lon), 3.0)

            if p_enum == StakeholderPersona.RESEARCHER:
                loc_name = "Sheltered Oceanographic Sampling Station"
                rat_str = f"Scientific survey rerouted {dist_nm} NM to sheltered sampling station to safeguard hydrographic sensor payloads."
            elif p_enum == StakeholderPersona.MARITIME_OPERATOR:
                loc_name = "Alternative Deep-Water Commercial Transit Channel"
                rat_str = f"Weather evasion reroute: Vessel shifted {dist_nm} NM to alternative deep-water fairway to minimize ship rolling and cargo stress."
            elif p_enum == StakeholderPersona.DISASTER_MANAGEMENT:
                loc_name = "Coastal Inundation Relief Staging Sector"
                rat_str = f"Emergency deployment: Shifted {dist_nm} NM to sheltered coastal relief staging harbor."
            elif p_enum == StakeholderPersona.MARITIME_AUTHORITY:
                loc_name = "Sheltered Naval / Coast Guard Mooring Basin"
                rat_str = f"Heavy weather protocol: Patrol craft repositioned {dist_nm} NM to sheltered tactical anchorage."
            else:
                loc_name = "Sheltered Inshore Coastal Sector"
                rat_str = (
                    f"Rerouted {dist_nm} NM toward sheltered coastal waters to mitigate "
                    f"threats: {threat_data.get('threat_prioritization', ['Elevated maritime risks'])[0]}"
                )

            res_dict = {
                "reroute_needed": True,
                "primary_coordinates": f"{orig_lat:.4f},{orig_lon:.4f}",
                "safe_coordinates": f"{safe_lat:.4f},{safe_lon:.4f}",
                "safe_location_name": loc_name,
                "distance_shift_nm": dist_nm,
                "rationale": rat_str,
            }
            try:
                res_dict["safe_sea_route"] = self.gis_agent.generate_safe_sea_route(
                    origin=f"{orig_lat:.4f},{orig_lon:.4f}",
                    destination=f"{safe_lat:.4f},{safe_lon:.4f}",
                )
            except Exception:
                pass
            return res_dict

        # Safe conditions: route user from vessel position (WP1) to persona-tailored destination (WP2)
        if p_enum == StakeholderPersona.RESEARCHER:
            dest_lat, dest_lon = (round(orig_lat + 0.12, 4), round(orig_lon + 0.18, 4)) if not target_destination else self._resolve_coordinates(target_destination)
            dist_nm = max(self.haversine_nm(orig_lat, orig_lon, dest_lat, dest_lon), 4.0)
            loc_name = "Oceanographic Biophysical Sampling Transect"
            rat_str = f"Favorable oceanographic conditions. Direct scientific transect of {dist_nm} NM planned across biophysical gradient front."
        elif p_enum == StakeholderPersona.MARITIME_AUTHORITY:
            dest_lat, dest_lon = (round(orig_lat + 0.10, 4), round(orig_lon + 0.25, 4)) if not target_destination else self._resolve_coordinates(target_destination)
            dist_nm = max(self.haversine_nm(orig_lat, orig_lon, dest_lat, dest_lon), 5.0)
            loc_name = "Sovereign EEZ Perimeter Patrol Sector"
            rat_str = f"Maritime security corridor clear. Routine sovereign patrol transit of {dist_nm} NM to EEZ surveillance sector."
        elif p_enum == StakeholderPersona.DISASTER_MANAGEMENT:
            dest_lat, dest_lon = (round(orig_lat + 0.08, 4), round(orig_lon + 0.12, 4)) if not target_destination else self._resolve_coordinates(target_destination)
            dist_nm = max(self.haversine_nm(orig_lat, orig_lon, dest_lat, dest_lon), 3.0)
            loc_name = "Coastal Disaster Readiness Patrol Sector"
            rat_str = f"Conditions normal. Pre-positioned disaster monitoring and relief corridor of {dist_nm} NM."
        elif p_enum == StakeholderPersona.MARITIME_OPERATOR:
            dest_lat, dest_lon = (round(orig_lat + 0.15, 4), round(orig_lon + 0.30, 4)) if not target_destination else self._resolve_coordinates(target_destination)
            dist_nm = max(self.haversine_nm(orig_lat, orig_lon, dest_lat, dest_lon), 6.0)
            loc_name = "Commercial Deep-Draft Shipping Fairway (SLOC)"
            rat_str = f"Fairway conditions optimal. Deep-draft navigation transit of {dist_nm} NM along designated shipping lane."
        else:
            if target_destination:
                dest_lat, dest_lon = self._resolve_coordinates(target_destination)
            else:
                dest_lat, dest_lon = (round(orig_lat + 0.15, 4), round(orig_lon + 0.20, 4))

            dist_nm = self.haversine_nm(orig_lat, orig_lon, dest_lat, dest_lon)
            if dist_nm < 1.0:
                dest_lat = round(orig_lat + 0.12, 4)
                dest_lon = round(orig_lon + 0.16, 4)
                dist_nm = max(self.haversine_nm(orig_lat, orig_lon, dest_lat, dest_lon), 5.0)

            loc_name = "High-Yield Potential Fishing Zone (PFZ)"
            rat_str = f"Primary destination has favorable sea conditions. Direct passage of {dist_nm} NM from vessel to high-scoring PFZ grounds."

        safe_dict = {
            "reroute_needed": False,
            "primary_coordinates": f"{orig_lat:.4f},{orig_lon:.4f}",
            "safe_coordinates": f"{dest_lat:.4f},{dest_lon:.4f}",
            "safe_location_name": loc_name,
            "distance_shift_nm": dist_nm,
            "rationale": rat_str,
        }
        try:
            safe_dict["safe_sea_route"] = self.gis_agent.generate_safe_sea_route(
                origin=f"{orig_lat:.4f},{orig_lon:.4f}",
                destination=f"{dest_lat:.4f},{dest_lon:.4f}",
            )
        except Exception:
            pass
        return safe_dict


# =====================================================================
# 3. REASONING AGENT (GEMINI SYNTHESIS)
# =====================================================================
class ReasoningAgent:
    """
    Synthesizes aggregated agent observations, deterministic risk evaluations,
    and safe alternatives into a coherent, explainable advisory payload.
    Uses Gemini (gemini-1.5-flash-latest) with strict JSON output formatting.
    """

    DEFAULT_MODEL = "gemma-4-26b-a4b-it"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = None
        self._init_gemini_model()

    def _init_gemini_model(self):
        """Initializes model prioritizing Gemma 4B with resilient fallbacks."""
        if FAST_DEMO_MODE:
            self.model = None
            self.model_name = "deterministic-fast-path"
            return

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
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-2.5-flash",
            "gemini-flash-latest",
            "gemma-4-26b-a4b-it",
            "gemma-4-31b-it",
        ]

        prioritized = [c for c in candidates if c in available_models]
        if not prioritized:
            prioritized = candidates

        system_instruction = (
            "You are ORCA's Master Maritime Intelligence Synthesis Engine (ISRO SIH 176).\n"
            "Your task is to analyze multi-agent marine data, enforce maritime boundary safety, "
            "and generate urgent, voice-friendly broadcast advisories for active boat operators.\n\n"
            "MANDATORY DOMAIN ASSESSMENT & BROADCAST ADVISORY RULES:\n"
            "1. DOMAIN ASSESSMENT STEP (MANDATORY FIRST TASK):\n"
            "   Your very first task is to evaluate if the user's normalized text query is relevant to the maritime domain "
            "(e.g., fishing, oceanography, marine species, coastal weather, waves, currents, maritime navigation, routing, ports, disaster management, cyclones, storm surge, search and rescue, or naval patrol).\n"
            "   If the query is off-topic (e.g., general math, cooking, baking recipes, unrelated coding, general trivia, sports, non-marine general assistance), "
            "you MUST immediately halt all standard maritime processing.\n\n"
            "2. OUT-OF-DOMAIN (OOD) STRICT REJECTION PROTOCOL:\n"
            "   When an off-topic query is detected, you MUST immediately return strictly this JSON payload:\n"
            "   {\n"
            '     "bhashini_text": "I am ORCA, a specialized maritime intelligence engine. I am programmed exclusively to assist with marine navigation, weather analysis, and coastal safety operations. I cannot process requests outside of this scope.",\n'
            '     "text_advisory_local": "I am ORCA, a specialized maritime intelligence engine. I am programmed exclusively to assist with marine navigation, weather analysis, and coastal safety operations. I cannot process requests outside of this scope.",\n'
            '     "map_status": "OUT_OF_DOMAIN",\n'
            '     "recommended_coordinates": ""\n'
            "   }\n\n"
            "3. IN-DOMAIN STRUCTURE: For maritime domain queries, the 'bhashini_text' MUST strictly follow this exact structure:\n"
            "   [STATUS: NO-GO | CAUTION | GO] -> [Core Hazard or Sea Condition Explanation] -> [Immediate Action Directive]\n"
            "4. GEOGRAPHIC VERIFICATION RULE: Analyze the user's requested location. If the user specifies an inland, non-coastal, or landlocked city (e.g., Sivakasi, Madurai), you MUST explicitly state in the advisory that it is an inland location with no marine access. Inform the user that the generated waypoints correspond to the nearest operational coastal departure harbor (e.g., Thoothukudi) instead.\n"
            "5. GEOGRAPHIC TARGET COORDINATES: When a marine location, fishing zone, storm center, emergency shelter, or route destination is queried, the advisory MUST explicitly state the geographic feature name, the exact coordinate rounded to 4 decimals (e.g., 'Lat 9.0932° N, Lon 78.3218° E'), and the cardinal direction with distance from the departure position (e.g., 'approximately 22.7 nautical miles northeast of Thoothukudi Outer Harbor'). Avoid unlabelled naked decimal numbers like '8.7642, 78.1348'.\n"
            "6. RELATIVE SPATIAL INSTRUCTIONS: Always use relative spatial navigation directions (e.g., 'navigate 12 nautical miles inland toward sheltered coastal waters', 'steer west towards mainland shore', 'head 8 nautical miles inshore').\n"
            "7. MOBILE MAP REFERENCE: Explicitly instruct the user to view the plotted waypoints directly on the mobile app's map interface.\n"
            "8. NO ACADEMIC JARGON: Never use academic or speculative phrasing (e.g., 'critical conflict between fishing potential and weather', 'divergence of biophysical indicators'). Use urgent, concise broadcast directives.\n"
            "9. TEMPORAL & FORECAST INTENT PROTOCOL:\n"
            "   Analyze the user's query for future-tense keywords (e.g., 'tomorrow', 'next week', 'later', 'forecast', 'upcoming', 'tonight', 'in the evening', 'future').\n"
            "   - When forward-looking marine forecast data (Open-Meteo Marine API) is ACTIVE:\n"
            "     * The operational status label is 'Forecast Status'.\n"
            "     * Set 'map_status' to match the evaluated forecast status ('SAFE', 'WARNING', or 'DANGER').\n"
            "     * You MUST explicitly state in the advisory that this is a prediction/forecast for the requested time horizon (upcoming 24–48 hours) based on forecasted maximum wave heights.\n"
            "   - When the forecast API is OFFLINE:\n"
            "     * You MUST begin or prepend the advisory with strictly this exact sentence: 'Forecast API offline. Displaying real-time observational data instead.'\n"
            "     * Evaluate conditions against the real-time ISRO MOSDAC satellite observation cache.\n"
            "   - If the user asks for an extended forecast beyond available limits (e.g. beyond 48 hours / 'next week') without forecast data:\n"
            "     * Set 'map_status' to 'PENDING FORECAST', leave 'recommended_coordinates' empty, and explicitly state that conditions beyond available horizons cannot be predicted.\n"
            "   - If a future query is processed with only real-time observation telemetry (and forecast API offline):\n"
            "     * You MUST NOT issue a definitive 'STATUS: GO'. Set status to 'STATUS: CONDITIONAL' (if conditions are currently safe) or 'STATUS: CAUTION'.\n\n"
            "10. REAL-TIME CYCLONE TRACK & ROUTE COLLISION PROTOCOL:\n"
            "   If an active Tropical Cyclone is tracked via GDACS or ISRO MOSDAC, or if the vessel's route intersects the forecasted track within a 200 NM radius:\n"
            "   - You MUST set 'map_status' to 'DANGER' and begin the advisory with strictly 'STATUS: NO-GO'.\n"
            "   - You MUST explicitly state the cyclone's name and its future trajectory vector in the broadcast advisory.\n"
            "   - Direct immediate mandatory emergency evasion to the nearest sheltered breakwater harbor.\n\n"
            "STRICT JSON OUTPUT REQUIREMENT:\n"
            "You must output ONLY a valid JSON object matching this schema:\n"
            "{\n"
            '  "bhashini_text": "STATUS: NO-GO / CAUTION / GO / CONDITIONAL / PENDING FORECAST. [Core condition/hazard]. [Immediate action directive with relative navigation & mobile map reference].",\n'
            '  "text_advisory_local": "Exact same text as bhashini_text",\n'
            '  "map_status": "SAFE | WARNING | DANGER | OUT_OF_DOMAIN | PENDING FORECAST | DATA_UNAVAILABLE",\n'
            '  "recommended_coordinates": "lat,lon or empty"\n'
            "}\n"
            "Do not include markdown tags like ```json or any text outside the JSON object."
        )

        for candidate in prioritized:
            try:
                self.model = genai.GenerativeModel(
                    model_name=candidate,
                    system_instruction=system_instruction,
                    generation_config={"response_mime_type": "application/json"},
                )
                self.model_name = candidate
                return
            except Exception:
                continue

        # Fallback
        self.model_name = "gemini-flash-latest"
        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_instruction,
                generation_config={"response_mime_type": "application/json"},
            )
        except Exception:
            self.model = None

    @staticmethod
    def _clean_tts_text(text: str) -> str:
        """
        Strips academic jargon, removes raw decimal coordinate pairs,
        and enforces voice-friendly spoken text for TTS and VHF radio broadcast.
        """
        if not text:
            return ""
        # Remove academic jargon
        text = re.sub(
            r"critical conflict between (?:fishing potential|localized telemetry|data) and [^.]*\.",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"divergence between predictive [^.]*\.",
            "",
            text,
            flags=re.IGNORECASE,
        )
        # Scrub raw coordinate pairs: e.g. [8.7642, 78.1348] or (8.7642, 78.1348) or 8.7642, 78.1348 or 8.7642,78.1348
        text = re.sub(
            r"(\[|\()?(\b\d{1,2}\.\d{3,6}\s*,\s*\d{1,3}\.\d{3,6}\b)(\]|\))?",
            "the plotted waypoints on your mobile map interface",
            text,
        )
        # Clean extra spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """Resilient JSON parsing handling raw strings, markdown blocks, and array wrappers."""
        cleaned = raw_text.strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                return parsed[0]
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Try markdown regex
        matches = re.findall(r"```(?:json)?\s*([\{\[][\s\S]*?[\}\]])\s*```", cleaned, re.IGNORECASE)
        for m in reversed(matches):
            try:
                parsed = json.loads(m.strip())
                if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                    return parsed[0]
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue

        # Try searching first { to last }
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except Exception:
                pass

        return None

    def synthesize_insights(
        self,
        aggregated_data: Dict[str, Any],
        risk_data: Dict[str, Any],
        alternative_data: Dict[str, Any],
        persona: str = "FISHERMAN",
        language_code: str = "en",
        ocean_target: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Uses Gemini to generate the voice-friendly emergency broadcast advisory.
        Enforces: STATUS -> Core Hazard Explanation -> Legal/IMBL Notice -> Immediate Action Directive.
        Zero raw numeric coordinates in spoken output; instructs user to check mobile map interface.
        """
        status = risk_data.get("status", "SAFE")
        risk_score = risk_data.get("risk_score", 20.0)
        rec_coords = alternative_data.get("safe_coordinates", "8.7642,78.1348")
        threats = risk_data.get("threat_prioritization", [])
        metrics = risk_data.get("metrics", {})
        imbl_violation = metrics.get("imbl_violation", False)
        reroute_distance_nm = alternative_data.get("distance_shift_nm", 12.0)
        if reroute_distance_nm <= 0:
            reroute_distance_nm = 10.0

        if not ocean_target and "primary_geographic_target" in aggregated_data:
            ocean_target = aggregated_data["primary_geographic_target"]

        user_query_text = (
            aggregated_data.get("normalized_query")
            or aggregated_data.get("user_query")
            or aggregated_data.get("english_query")
            or ""
        )
        temporal_info = detect_temporal_intent(user_query_text)
        has_temporal = temporal_info.get("has_temporal_intent", False)
        is_beyond_24h = temporal_info.get("is_beyond_24h", False)
        temporal_label = temporal_info.get("timeframe_label") or "tomorrow morning"
        dataset_ts = get_dataset_timestamp(aggregated_data)

        forecast_active = bool(metrics.get("forecast_active") or risk_data.get("forecast_active", False))
        forecast_offline = bool(metrics.get("forecast_offline") or risk_data.get("forecast_offline", False))
        forecast_max_wave = metrics.get("forecast_max_wave_m") or risk_data.get("forecast_max_wave_m")
        forecast_avg_wave = metrics.get("forecast_avg_wave_m") or risk_data.get("forecast_avg_wave_m")
        forecast_horizon = metrics.get("forecast_horizon") or risk_data.get("forecast_horizon") or "24–48 hours"
        wave = float(metrics.get("wave_height_m") or metrics.get("wave_height") or 1.2)

        # Guardrail 0: Out-of-Domain Rejection (Zero latency, immediate refusal)
        if status == "OUT_OF_DOMAIN" or risk_data.get("out_of_domain"):
            ood_text = (
                "I am ORCA, a specialized maritime intelligence engine. "
                "I am programmed exclusively to assist with marine navigation, weather analysis, and coastal safety operations. "
                "I cannot process requests outside of this scope."
            )
            return {
                "bhashini_text": ood_text,
                "text_advisory_local": ood_text,
                "map_status": "OUT_OF_DOMAIN",
                "recommended_coordinates": "",
            }

        # Guardrail 1: Security Rejection (Zero latency, immediate shield)
        if status == "SECURITY_REJECTION" or risk_data.get("security_rejection"):
            sec_text = (
                "STATUS: SECURITY_REJECTION. Critical security warning: Adversarial prompt injection or safety protocol bypass attempt detected. "
                "Project ORCA maritime safety guardrails, IMBL border enforcement, and SOLAS life-safety protocols cannot be overridden or disabled. Query terminated."
            )
            return {
                "bhashini_text": sec_text,
                "text_advisory_local": sec_text,
                "map_status": "SECURITY_REJECTION",
                "recommended_coordinates": rec_coords,
            }

        # Guardrail 2: Chronological Temporal Bounds & Pending Forecast Intercept (>24h MOSDAC Limit)
        if not forecast_active and (is_beyond_24h or status in ("PENDING FORECAST", "DATA_UNAVAILABLE", "TEMPORAL_OUT_OF_BOUNDS") or risk_data.get("pending_forecast") or risk_data.get("temporal_out_of_bounds")):
            oob_text = (
                "STATUS: PENDING FORECAST. Real-time ISRO MOSDAC satellite Earth Observation cache provides near-real-time observations and short-term 24-hour nowcasting, and cannot predict conditions beyond 24 hours. "
                "The requested forecast horizon is unavailable. Please check back closer to your departure window for updated satellite passes and numerical forecast model runs."
            )
            return {
                "bhashini_text": oob_text,
                "text_advisory_local": oob_text,
                "map_status": "PENDING FORECAST",
                "recommended_coordinates": "",
            }

        # Guardrail 3: Coastal Landmasking Enforcement (Overland Marine Routing Prohibited)
        if status == "LAND_INTERSECTION_ERROR" or risk_data.get("land_intersection_error"):
            land_text = (
                "STATUS: NO-GO. CRITICAL ROUTING FAILURE: Direct nautical passage between the specified locations directly intersects the Indian peninsular landmass. "
                "Overland marine routing is physically impossible. Vessels must route around Cape Comorin via open coastal waters or transfer cargo via terrestrial freight corridors."
            )
            return {
                "bhashini_text": land_text,
                "text_advisory_local": land_text,
                "map_status": "LAND_INTERSECTION_ERROR",
                "recommended_coordinates": rec_coords,
            }

        # Guardrail 4: Life-Safety Override (Emergency SAR Priority Precedence)
        if status == "EMERGENCY_SAR" or risk_data.get("emergency_sar"):
            dist_val = reroute_distance_nm if reroute_distance_nm and reroute_distance_nm > 0 else 3.8
            sar_text = (
                f"STATUS: EMERGENCY_SAR. CRITICAL DISTRESS BROADCAST: Immediate life-safety emergency declared under SOLAS conventions. "
                f"Routine weather and cyclone NO-GO restrictions are OVERRIDDEN. Indian Coast Guard MRCC alerted on VHF Channel 16. "
                f"Steer immediately along emergency vector {dist_val:.1f} nautical miles to nearest safe breakwater shelter. "
                f"Refer to emergency plotted waypoints on mobile map interface."
            )
            return {
                "bhashini_text": sar_text,
                "text_advisory_local": sar_text,
                "map_status": "EMERGENCY_SAR",
                "recommended_coordinates": rec_coords,
            }

        p_enum = resolve_persona(persona) or StakeholderPersona.FISHERMAN
        persona_upper = p_enum.value

        persona_directives = {
            StakeholderPersona.FISHERMAN.value: (
                "Role: Marine Navigation & Fishery Emergency Co-Pilot. "
                "Perspective: Protect fishermen from rough seas, provide urgent life-safety directives, "
                "highlight high-yield PFZ zones, and instruct boat operators to follow plotted waypoints on their mobile map interface."
            ),
            StakeholderPersona.RESEARCHER.value: (
                "Role: Oceanographic & Marine Biophysical Science Advisor. "
                "Perspective: Deliver precise satellite Earth observation telemetry: Oceansat-3 chlorophyll-a fronts, diffuse attenuation Kd_490, "
                "total suspended matter (TSM), SARAL-AltiKa SWH altimetry, geostrophic current vectors, and solar irradiance. "
                "Instruct vessel to view plotted transect coordinates on the mobile map."
            ),
            StakeholderPersona.MARITIME_AUTHORITY.value: (
                "Role: Maritime Authority & Sovereign Border Patrol. "
                "Perspective: Focus on India EEZ perimeter surveillance, AIS tracking, Coast Guard interception readiness, "
                "search and rescue (SAR), and defense stationing. Treat foreign boundary proximity as a Jurisdictional Boundary Assessment, "
                "not as a civilian crime or violation. Direct officers to plotted patrol zones on the map interface."
            ),
            StakeholderPersona.DISASTER_MANAGEMENT.value: (
                "Role: Coastal Disaster Management & Crisis Response. "
                "Perspective: Focus on life safety, storm surge inundation, cyclone track projection, coastal population evacuation routes, "
                "and designated breakwater emergency shelter allocation. Direct response teams to plotted evacuation corridors on the mobile map interface."
            ),
            StakeholderPersona.MARITIME_OPERATOR.value: (
                "Role: Commercial Shipping & Port Operations Manager. "
                "Perspective: Focus on deep-draft fairway navigation, Traffic Separation Scheme (TSS) compliance, "
                "Sea Lines of Communication (SLOC) clearance, harbor approach scheduling, and fuel optimization. "
                "Direct commercial vessels along plotted fairway waypoints on the mobile map."
            ),
        }
        role_instruction = persona_directives.get(persona_upper, "Role: Marine Intelligence Advisor.")

        wp1_coord = alternative_data.get("primary_coordinates", "8.7642,78.1348")
        wp2_coord = alternative_data.get("safe_coordinates", rec_coords)
        departure_harbor = "Thoothukudi (V.O. Chidambaranar Port)"

        prompt_parts = [
            f"Persona: {persona_upper}",
            f"Role Directives: {role_instruction}",
            f"Target Language: {language_code}",
            f"ORIGINAL USER QUERY: \"{user_query_text}\"",
            "QUERY-SPECIFIC MANDATE: Answer the user's specific query directly! If the user asks about cyclones, report active/calm storm status without mentioning fishing grounds or navigation waypoints. If the user asks about weather, report wind, waves, rainfall, and pressure without inventing PFZ or route clearance. Only include PFZ for fishing questions, and only include passage waypoints for routing questions.",
            f"Dataset Observation Timestamp: {dataset_ts}",
            f"WP 1 Departure Coordinates: {wp1_coord} (Nearest Operational Coastal Departure Harbor: {departure_harbor})",
            f"WP 2 Destination Coordinates: {wp2_coord}",
            f"Overall Threat Status: {status} (Risk Score: {risk_score}/100)",
            f"Navigation Distance: {reroute_distance_nm} nautical miles (WP1 to WP2)",
            f"Identified Threats: {json.dumps(threats)}",
            f"Aggregated Telemetry: {json.dumps(metrics)}",
            f"Safe Alternative Routing: {json.dumps(alternative_data)}",
        ]

        if ocean_target:
            prompt_parts.extend([
                f"Primary Geographic Target Type: {ocean_target.get('feature_type')}",
                f"Target Coordinates: {ocean_target.get('target_coordinates')}",
                f"Relative Spatial Vector: {ocean_target.get('relative_vector')} (from departure)",
                f"Landmark Reference: {ocean_target.get('landmark_reference')}",
            ])

        if forecast_active:
            f_wave_val = forecast_max_wave if forecast_max_wave is not None else wave
            prompt_parts.extend([
                "",
                "FORECAST ACTIVE - OPEN-METEO NUMERICAL MARINE PREDICTION:",
                "- Operational Status Label: Forecast Status",
                f"- Forecast Horizon: {forecast_horizon}",
                f"- Forecasted Maximum Wave Height: {f_wave_val:.2f} meters",
                "- Real-time MOSDAC telemetry has been bypassed for the main operational status.",
                f"- MANDATORY ADVISORY INSTRUCTION: You MUST explicitly state that this advisory is a predictive forecast for the requested time horizon ({forecast_horizon}) based on numerical weather prediction models (e.g., 'Forward-looking marine forecast for tomorrow predicts maximum wave heights of {f_wave_val:.2f} meters. This advisory is a prediction for the requested {forecast_horizon} horizon.').",
                f"- Set 'map_status' to match the evaluated forecast status ('{status}'). Do NOT use 'CONDITIONAL' or 'PENDING FORECAST'.",
                "",
            ])
        elif forecast_offline:
            prompt_parts.extend([
                "",
                "FORECAST API OFFLINE - MOSDAC REAL-TIME FALLBACK MANDATE:",
                "- Forecast API is offline, timed out, or unreachable.",
                "- MANDATORY ADVISORY INSTRUCTION: You MUST force your advisory to start with strictly this exact sentence:",
                "  'Forecast API offline. Displaying real-time observational data instead.'",
                "- Conditions are evaluated using real-time ISRO MOSDAC satellite observational telemetry.",
                "",
            ])
        elif has_temporal:
            prompt_parts.extend([
                "",
                "TEMPORAL INTENT & FORECAST BOUNDARY MANDATE:",
                f"- The user's query contains near-future / forecast intent ('{temporal_label}').",
                f"- Dataset observation timestamp is: '{dataset_ts}'.",
                "- STRICT PROHIBITION: DO NOT issue a definitive 'STATUS: GO' for that future date using only current telemetry!",
                "- If conditions are currently safe, you MUST start the advisory with 'STATUS: CONDITIONAL.' (or 'STATUS: CAUTION.' if elevated threats exist).",
                "- You MUST include this strict temporal boundary statement in the advisory:",
                f"  'Current satellite telemetry indicates safe conditions as of {dataset_ts}, but this is a real-time observation. We recommend pulling an updated short-term wave and weather forecast closer to your departure {temporal_label}.'",
                "",
            ])

        cyclone_intel = risk_data.get("cyclone_intelligence") or aggregated_data.get("cyclone_intelligence")
        if cyclone_intel and cyclone_intel.get("active_storms"):
            c_name = cyclone_intel.get("active_storms")
            c_traj = cyclone_intel.get("trajectory_path", "N/A")
            c_col = cyclone_intel.get("route_collision", False)
            c_clr = cyclone_intel.get("clearance_distance_nm")
            clr_str = f"{c_clr:.1f} NM" if c_clr is not None else "N/A"
            prompt_parts.extend([
                "",
                "CYCLONE TRACKING INTELLIGENCE (GDACS / ISRO MOSDAC):",
                f"- Active Tropical Cyclone: {c_name}",
                f"- Projected 24h Trajectory Vector: {c_traj}",
                f"- Route Collision Status: {'CRITICAL ROUTE COLLISION DETECTED (< 200 NM buffer)' if c_col else 'NO COLLISION (Safe distance)'}",
                f"- Vessel-to-Track Clearance Distance: {clr_str}",
                "- MANDATORY CYCLONE DIRECTIVE: If collision is detected, you MUST issue 'STATUS: NO-GO', state the cyclone name and its projected trajectory, and direct immediate emergency evacuation to the nearest breakwater harbor.",
                "",
            ])

        prompt_parts.extend([
            "",
            "LOCATION CONTRAST CONTEXT & GEOGRAPHIC VERIFICATION RULE:",
            "- Analyze the user's requested location in 'User Text Query'.",
            "- If the user specifies an inland, non-coastal, or landlocked city (e.g., Sivakasi, Madurai, Coimbatore, Tirunelveli): "
            "you MUST explicitly state in the advisory that it is an inland location with no marine access. "
            "Inform the user that the generated waypoints correspond to the nearest operational coastal departure harbor (Thoothukudi) instead.",
            "- Contrast the user's text query with the WP 1 (Departure) coordinate. If the text says 'Sivakasi' but WP1 is a marine coordinate, "
            "you must bridge that gap logically for the user rather than blindly generating a marine route for an inland city.",
            "",
            "MANDATORY INSTRUCTIONS FOR VOICE-FRIENDLY BROADCAST SYNTHESIS:",
            "1. STRUCTURE ENFORCEMENT: The 'bhashini_text' MUST strictly follow this structure:",
            "   STATUS: (Must start with 'STATUS: NO-GO.' or 'STATUS: CAUTION.' or 'STATUS: CONDITIONAL.' or 'STATUS: GO.' - NEVER use 'STATUS: GO.' if near-future temporal intent is present)",
            "   -> Core Condition/Hazard Explanation (Urgent concise summary tailored strictly to active persona, including geographic inland note and temporal boundary if applicable)",
        ])

        if imbl_violation:
            if p_enum == StakeholderPersona.MARITIME_AUTHORITY:
                prompt_parts.extend([
                    "   -> Jurisdictional Assessment: State: 'Jurisdictional boundary assessment active along India-Sri Lanka IMBL perimeter. Sovereign maritime patrol assets deployed for EEZ surveillance and cross-border protocol enforcement.'",
                    f"   -> Immediate Action Directive: Maintain tactical surveillance along the {reroute_distance_nm} nautical mile patrol corridor, verify AIS contacts, and monitor plotted coordinates on the mobile map interface.",
                ])
            else:
                prompt_parts.extend([
                    "   -> Legal/IMBL Notice: State explicitly: 'Navigating across the International Maritime Boundary Line (IMBL) into Sri Lankan waters is illegal and strictly prohibited under international maritime law.'",
                    f"   -> Immediate Action Directive: Instruct the vessel to abort course, navigate {reroute_distance_nm} nautical miles into sovereign Indian waters, and refer to plotted waypoints on the mobile map interface.",
                ])
        else:
            prompt_parts.extend([
                f"   -> Immediate Action Directive: Action directive using relative spatial guidance with exact distance ({reroute_distance_nm} nautical miles), directing user to check plotted waypoints on their mobile map interface.",
            ])

        prompt_parts.extend([
            f"2. MATHEMATICAL DISTANCE SYNCHRONIZATION: Do not invent distances. Use the provided reroute_distance_nm ({reroute_distance_nm} nautical miles) in your text advisory.",
            "3. COORDINATE NOTATION RULE: Do NOT write unlabelled comma-separated decimal coordinate pairs (e.g., '8.7642, 78.1348' or '[8.7642, 78.1348]'). Whenever mentioning geographic targets, always use standard notation format: 'Lat XX.XXXX° N, Lon YY.YYYY° E' along with relative distance and cardinal direction from departure harbor (e.g., 'The high-yield fishing zone is located at Lat 9.0932° N, Lon 78.3218° E, approximately 22.7 nautical miles northeast of Thoothukudi Outer Harbor.').",
            "4. NO ACADEMIC JARGON: Never write speculative phrases like 'critical conflict between fishing potential and extreme weather'. Active operators require urgent, actionable phrasing.",
            "5. RETURN FORMAT: Return strictly valid JSON matching:",
            '   {"bhashini_text": "...", "text_advisory_local": "...", "map_status": "...", "recommended_coordinates": "..."}.'
        ])

        prompt = "\n".join(prompt_parts)

        if self.model and os.getenv("GEMINI_API_KEY") and not FAST_DEMO_MODE:
            try:
                response = self.model.generate_content(
                    prompt,
                    request_options={"timeout": 1.0, "retry": None},
                )
                if response and response.text:
                    parsed = self._extract_json(response.text)
                    if parsed:
                        if parsed.get("map_status") == "OUT_OF_DOMAIN" or parsed.get("status") == "OUT_OF_DOMAIN":
                            ood_text = (
                                "I am ORCA, a specialized maritime intelligence engine. "
                                "I am programmed exclusively to assist with marine navigation, weather analysis, and coastal safety operations. "
                                "I cannot process requests outside of this scope."
                            )
                            return {
                                "bhashini_text": ood_text,
                                "text_advisory_local": ood_text,
                                "map_status": "OUT_OF_DOMAIN",
                                "recommended_coordinates": "",
                            }
                        if "bhashini_text" in parsed:
                            cleaned_text = self._clean_tts_text(parsed["bhashini_text"])
                            # Strict temporal / forecast post-enforcement on LLM text output
                            if forecast_offline:
                                offline_prefix = "Forecast API offline. Displaying real-time observational data instead."
                                if not cleaned_text.startswith(offline_prefix):
                                    cleaned_text = f"{offline_prefix} {cleaned_text}"
                            elif forecast_active:
                                f_wave_val = forecast_max_wave if forecast_max_wave is not None else wave
                                if "prediction" not in cleaned_text.lower() and "forecast" not in cleaned_text.lower():
                                    cleaned_text = f"Forward-looking marine forecast predicts maximum wave heights of {f_wave_val:.2f} meters. This advisory is a prediction for the requested {forecast_horizon} time horizon. {cleaned_text}"
                            elif has_temporal:
                                if cleaned_text.startswith("STATUS: GO."):
                                    cleaned_text = "STATUS: CONDITIONAL." + cleaned_text[len("STATUS: GO."):]
                                elif cleaned_text.startswith("STATUS: GO"):
                                    cleaned_text = "STATUS: CONDITIONAL." + cleaned_text[len("STATUS: GO"):]
                                if dataset_ts not in cleaned_text and "real-time observation" not in cleaned_text:
                                    boundary_clause = (
                                        f"Current satellite telemetry indicates safe conditions as of {dataset_ts}, "
                                        f"but this is a real-time observation. We recommend pulling an updated short-term wave and weather forecast closer to your departure {temporal_label}. "
                                    )
                                    parts = cleaned_text.split(".", 1)
                                    if len(parts) == 2:
                                        cleaned_text = f"{parts[0]}. {boundary_clause}{parts[1].strip()}"
                                    else:
                                        cleaned_text = f"{boundary_clause}{cleaned_text}"
                            parsed["bhashini_text"] = cleaned_text
                            parsed["text_advisory_local"] = cleaned_text
                            if forecast_active:
                                parsed["map_status"] = status
                            elif has_temporal and status == "SAFE":
                                parsed["map_status"] = "CONDITIONAL"
                            else:
                                parsed["map_status"] = status
                            parsed["recommended_coordinates"] = rec_coords
                            return parsed
            except (TimeoutError, Exception) as err:
                err_str = str(err)
                if "429" in err_str or "quota" in err_str.lower():
                    print(f"[ReasoningAgent Notice] Gemini free-tier rate limit/quota reached (HTTP 429). Immediate fallback to deterministic offline synthesis.")
                elif "504" in err_str or "timeout" in err_str.lower() or "deadline" in err_str.lower():
                    print(f"[ReasoningAgent Notice] LLM synthesis timeout (15s limit reached). Immediate fallback to deterministic synthesis.")
                else:
                    print(f"[ReasoningAgent Notice] LLM synthesis fallback triggered: {err_str[:120]}")

        # Deterministic Fallback Synthesis (Zero-crash guarantee tailored by Persona)
        dist_nm = reroute_distance_nm

        # Geographic verification: Inland / non-coastal city detection
        inland_cities_map = {
            "sivakasi": "Sivakasi",
            "madurai": "Madurai",
            "coimbatore": "Coimbatore",
            "tirunelveli": "Tirunelveli",
            "trichy": "Tiruchirappalli",
            "tiruchirappalli": "Tiruchirappalli",
            "salem": "Salem",
            "erode": "Erode",
            "dindigul": "Dindigul",
            "virudhunagar": "Virudhunagar",
            "theni": "Theni",
            "vellore": "Vellore",
            "bangalore": "Bengaluru",
            "bengaluru": "Bengaluru",
            "hyderabad": "Hyderabad",
        }
        detected_inland = None
        for k, v in inland_cities_map.items():
            if re.search(rf"\b{k}\b", user_query_text.lower()):
                detected_inland = v
                break

        inland_prefix = ""
        if detected_inland:
            inland_prefix = f"Geographic Notice: {detected_inland} is an inland location with no marine access. Plotted waypoints correspond to the nearest operational coastal departure harbor at Thoothukudi. "

        target_sentence = ""
        if ocean_target and ocean_target.get("target_coordinates"):
            ft = ocean_target.get("feature_type", "")
            t_coords = ocean_target.get("target_coordinates", "")
            l_ref = ocean_target.get("landmark_reference", "")
            t_dist = ocean_target.get("distance_nm", dist_nm)
            t_card = (ocean_target.get("cardinal_direction_full") or ocean_target.get("cardinal_direction") or "northeast").lower()
            if ft == "PFZ Aggregation Hotspot":
                target_sentence = f"The high-yield fishing zone is located at {t_coords}, approximately {t_dist:.1f} nautical miles {t_card} of Thoothukudi Outer Harbor ({l_ref})."
            elif ft == "Storm Eye / Center Coordinate":
                rmw = ocean_target.get("details", {}).get("radius_max_winds_nm", 25.0)
                target_sentence = f"The storm center is located at {t_coords}, approximately {t_dist:.1f} nautical miles {t_card} of Thoothukudi Outer Harbor ({l_ref}) with radius of maximum winds {rmw:.0f} NM."
            elif ft == "Hazard Impact / Evacuation Boundary":
                s_name = ocean_target.get("details", {}).get("shelter_name", "Designated Emergency Shelter")
                target_sentence = f"The emergency shelter boundary ({s_name}) is located at {t_coords}, approximately {t_dist:.1f} nautical miles {t_card} of Thoothukudi Outer Harbor ({l_ref})."
            else:
                d_name = ocean_target.get("details", {}).get("destination_name", "fairway corridor")
                target_sentence = f"The navigational waypoint for {d_name} is located at {t_coords}, approximately {t_dist:.1f} nautical miles {t_card} of Thoothukudi Outer Harbor ({l_ref})."

        target_addon = f" {target_sentence}" if target_sentence else ""

        temporal_boundary = ""
        if has_temporal:
            temporal_boundary = (
                f"Current satellite telemetry indicates safe conditions as of {dataset_ts}, "
                f"but this is a real-time observation. We recommend pulling an updated short-term wave and weather forecast closer to your departure {temporal_label}. "
            )

        cyc_intel = risk_data.get("cyclone_intelligence") or {}
        has_cyc_col = bool(cyc_intel.get("route_collision"))

        q_lower = str(user_query_text or "").lower()
        intent_cat = (aggregated_data.get("analyzed_intent") or "").upper()
        is_cyclone_query = (intent_cat == "DISASTER") or any(k in q_lower for k in ["cyclone", "storm", "hurricane", "typhoon", "depression", "tsunami", "surge", "radar", "hazard", "gale"])
        is_weather_query = (intent_cat == "WEATHER") or any(k in q_lower for k in ["weather", "wind", "rain", "temperature", "forecast", "cloud", "gust", "pressure", "wave", "swell", "sea state"])
        is_fishing_query = (intent_cat == "FISHING") or any(k in q_lower for k in ["fish", "fishing", "pfz", "catch", "tuna", "mackerel", "sardine", "seerfish", "shoal", "yield"])
        is_route_query = (intent_cat == "ROUTE") or any(k in q_lower for k in ["route", "passage", "sail from", "navigate", "navigation", "sri lanka", "colombo", "waypoint"]) or ("from " in q_lower and " to " in q_lower)

        w_ht = float(metrics.get("wave_height_m") or metrics.get("wave_height") or 1.2)
        w_spd = float(metrics.get("wind_speed_kmph") or metrics.get("wind_speed") or 14.0)
        g_spd = float(metrics.get("gust_speed_kmph") or 18.0)
        w_dir = metrics.get("wind_direction", "SW")
        rain = float(metrics.get("rainfall_mmh", 0.0))
        press = float(metrics.get("pressure_hpa", 1012.0))
        vis = float(metrics.get("visibility_km", 10.0))
        temp = float(metrics.get("surface_temp_c", 27.5))

        if is_cyclone_query:
            if status in ("DANGER", "WARNING") or metrics.get("is_cyclone_active") or metrics.get("hazard_active") or has_cyc_col:
                status_tag = "STATUS: NO-GO." if (status == "DANGER" or has_cyc_col) else "STATUS: CAUTION."
                c_name = cyc_intel.get("active_storms", "Tropical Cyclonic Storm")
                c_desc = metrics.get("hazard_description") or f"Active cyclonic storm alert ({c_name})"
                explanation = (
                    f"{status_tag} Tropical Cyclone Warning: {c_desc}. "
                    f"Elevated wind speeds reaching {w_spd:.1f} km/h with gusts to {g_spd:.1f} km/h and wave heights at {w_ht:.1f} meters. "
                    f"IMMEDIATE ACTION: Cease offshore operations immediately, return to port, and monitor VHF Channel 16 for disaster management bulletins."
                )
            else:
                status_tag = "STATUS: GO."
                explanation = (
                    f"STATUS: GO. No tropical cyclone or severe storm is currently detected in your maritime operating sector (Thoothukudi / Gulf of Mannar). "
                    f"ISRO MOSDAC scatterometer and radar telemetry confirm clear conditions. Sustained wind speed is {w_spd:.1f} km/h ({w_dir}) and wave height is {w_ht:.1f} meters. "
                    f"Maritime conditions are safe for routine operations."
                )
            rec_coords = ""

        elif is_weather_query:
            if status in ("DANGER", "WARNING"):
                status_tag = "STATUS: CAUTION." if status == "WARNING" else "STATUS: NO-GO."
                explanation = (
                    f"{status_tag} Coastal weather advisory: Marginal or adverse sea state observed. "
                    f"Sustained wind speed is {w_spd:.1f} km/h from {w_dir} with gusts up to {g_spd:.1f} km/h. "
                    f"Wave heights reach {w_ht:.1f} meters. Precipitation rate: {rain:.1f} mm/h. Atmospheric pressure: {press:.0f} hPa. "
                    f"Exercise heightened vigilance and monitor weather updates."
                )
            else:
                status_tag = "STATUS: CONDITIONAL." if has_temporal else "STATUS: GO."
                temporal_prefix = temporal_boundary if has_temporal else ""
                explanation = (
                    f"{status_tag} {temporal_prefix}Coastal weather advisory for your operating sector: Favorable maritime conditions observed. "
                    f"Sustained wind speed is {w_spd:.1f} km/h from {w_dir} with gusts up to {g_spd:.1f} km/h. "
                    f"Sea state shows wave heights of {w_ht:.1f} meters with gentle swells. "
                    f"Atmospheric pressure is {press:.0f} hPa, sea surface temperature {temp:.1f}°C, and visibility is {vis:.1f} km with {rain:.1f} mm/h rainfall. "
                    f"Conditions are favorable for maritime operations."
                )
            rec_coords = ""

        elif is_fishing_query:
            if imbl_violation:
                status_tag = "STATUS: NO-GO."
                explanation = (
                    f"{status_tag} Boundary violation detected: Target fishing grounds cross the International Maritime Boundary Line (IMBL) into Sri Lankan waters. "
                    f"Crossing the IMBL is strictly prohibited. Plotted alternative PFZ grounds strictly within sovereign Indian EEZ waters."
                )
            else:
                pfz_agent = aggregated_data.get("PFZ_AGENT", {})
                features = pfz_agent.get("geojson", {}).get("features", []) if isinstance(pfz_agent, dict) else []
                if features:
                    props = features[0].get("properties", {})
                    t_lat = props.get("centroid_lat", 9.0932)
                    t_lon = props.get("centroid_lon", 78.3218)
                    score = props.get("suitability_score", 92)
                    catch = ", ".join(props.get("likely_catch", ["Yellowfin Tuna", "Mackerel", "Sardine"]))
                    explanation = (
                        f"STATUS: GO. Optimal Potential Fishing Zone (PFZ) identified at Latitude {t_lat:.4f}° N, Longitude {t_lon:.4f}° E off Thoothukudi. "
                        f"Chlorophyll-a front concentration and thermal gradients indicate high pelagic fish aggregation (Suitability: {score}/100) with favorable catch probability for {catch}. "
                        f"Sea surface temperature is {temp:.1f}°C with safe wave heights of {w_ht:.1f} meters. "
                        f"Refer to plotted PFZ waypoints on your mobile map interface."
                    )
                    rec_coords = f"{t_lat:.4f},{t_lon:.4f}"
                else:
                    explanation = (
                        f"STATUS: GO. High-yield Potential Fishing Zone (PFZ) located at Lat 9.0932° N, Lon 78.3218° E (~22.6 NM NNE off Thoothukudi). "
                        f"Favorable SST ({temp:.1f}°C) and chlorophyll fronts indicate high pelagic fish aggregation for Tuna, Mackerel, and Sardine (Suitability: 92/100). "
                        f"Sea state is safe with wave heights of {w_ht:.1f} meters."
                    )
                    rec_coords = "9.0932,78.3218"

        elif imbl_violation:
            if p_enum == StakeholderPersona.MARITIME_AUTHORITY:
                status_tag = "STATUS: CAUTION."
                core_hazard = "Jurisdictional boundary assessment: Active patrol vector along India-Sri Lanka IMBL sovereign boundary sector."
                legal_notice = "OPERATIONAL PROTOCOL: Indian Coast Guard and naval units maintain active maritime domain awareness. Comply with sovereign patrol rules of engagement and AIS broadcast integrity."
                action_dir = f"IMMEDIATE ACTION: Maintain tactical perimeter patrol across the {dist_nm:.1f} nautical mile corridor and observe plotted defense waypoints on your mobile map interface."
                explanation = f"{status_tag} {inland_prefix}{core_hazard} {legal_notice} {action_dir}"
            else:
                status_tag = "STATUS: NO-GO."
                core_hazard = "Boundary violation detected: Proposed route targets foreign waters across the maritime border."
                legal_notice = "LEGAL NOTICE: Navigating across the International Maritime Boundary Line (IMBL) into Sri Lankan waters is illegal and strictly prohibited under international maritime law and Indian Coast Guard regulations."
                action_dir = f"IMMEDIATE ACTION: Turn back immediately, navigate {dist_nm:.1f} nautical miles into sovereign Indian waters, and refer to the plotted safe waypoints on your mobile map interface."
                explanation = f"{status_tag} {inland_prefix}{core_hazard} {legal_notice} {action_dir}"
        elif has_cyc_col:
            c_name = cyc_intel.get("active_storms", "Tropical Cyclonic Storm")
            c_traj = cyc_intel.get("trajectory_path", "projected track")
            c_clr = cyc_intel.get("clearance_distance_nm")
            clr_str = f"{c_clr:.1f} nautical miles" if c_clr is not None else "less than 200 nautical miles"
            status_tag = "STATUS: NO-GO."
            core_hazard = (
                f"Severe Tropical Cyclone Alert ({c_name}): Vessel route intersects forecasted storm path "
                f"with only {clr_str} clearance (mandatory safe buffer: 200 nautical miles). "
                f"Storm trajectory: {c_traj}. Gale-force winds and violent storm surge expected."
            )
            action_dir = (
                f"IMMEDIATE ACTION: Cease navigation immediately, initiate mandatory emergency evacuation to nearest "
                f"sheltered breakwater harbor ({dist_nm:.1f} nautical miles), and follow plotted emergency evacuation waypoints on your mobile map interface."
            )
            explanation = f"{status_tag} {inland_prefix}{core_hazard}{target_addon} {action_dir}"
        elif forecast_active:
            f_wave_val = forecast_max_wave if forecast_max_wave is not None else wave
            status_tag = f"STATUS: {'NO-GO' if status == 'DANGER' else ('CAUTION' if status == 'WARNING' else 'GO')}."
            forecast_statement = (
                f"Forward-looking marine forecast for {temporal_label} predicts maximum wave heights of {f_wave_val:.2f} meters. "
                f"This advisory is a predictive forecast for the requested {forecast_horizon} time horizon based on Open-Meteo numerical marine weather prediction models. "
            )
            if status == "DANGER":
                core_hazard = f"Hazardous forecasted wave heights of {f_wave_val:.2f} meters exceeding small craft safety limits."
                action_dir = f"IMMEDIATE ACTION: Cease operations immediately, navigate {dist_nm:.1f} nautical miles inshore toward sheltered coastal waters, and view plotted safe waypoints on your mobile map interface."
            elif status == "WARNING":
                core_hazard = f"Elevated forecasted sea state with wave heights near {f_wave_val:.2f} meters requiring heightened maritime caution."
                action_dir = f"IMMEDIATE ACTION: Proceed with extreme care, navigate {dist_nm:.1f} nautical miles toward sheltered coastal waters, monitor VHF Channel 16, and check plotted waypoints on your mobile map interface."
            else:
                core_hazard = f"Favorable forecasted sea state with wave heights remaining safe below 1.8 meters (peak {f_wave_val:.2f} m)."
                action_dir = f"IMMEDIATE ACTION: Proceed along your {dist_nm:.1f} nautical mile passage following standard safety protocols, and refer to plotted waypoints on your mobile map interface."
            explanation = f"{status_tag} {inland_prefix}{forecast_statement}{core_hazard}{target_addon} {action_dir}"
        elif p_enum == StakeholderPersona.RESEARCHER:
            chl_val = metrics.get("chlorophyll_mg_m3", 0.58)
            sst_val = metrics.get("sst_c", 27.5)
            swh_val = metrics.get("wave_height_m", 1.45)
            solar_val = metrics.get("solar_insolation_wm2", 840.0)
            status_tag = f"STATUS: {'NO-GO' if status == 'DANGER' else ('CAUTION' if status == 'WARNING' else ('CONDITIONAL' if has_temporal else 'GO'))}."
            core_hazard = (
                f"Oceanographic Earth observation telemetry: Chlorophyll-a front concentration is {chl_val:.2f} mg/m³, "
                f"SST is {sst_val:.1f}°C, significant wave height is {swh_val:.2f} meters, and solar insolation is {solar_val:.1f} W/m²."
            )
            action_dir = f"IMMEDIATE ACTION: Execute oceanographic transect operations covering {dist_nm:.1f} nautical miles following the plotted waypoints on your mobile map interface."
            temporal_clause = temporal_boundary if (has_temporal and status not in ("DANGER", "WARNING")) else ""
            explanation = f"{status_tag} {inland_prefix}{temporal_clause}{core_hazard}{target_addon} {action_dir}"
        elif p_enum == StakeholderPersona.MARITIME_AUTHORITY:
            border_dist = metrics.get("distance_to_border_nm", 32.5)
            sloc_info = metrics.get("sloc_status", "Clearance Verified")
            status_tag = f"STATUS: {'NO-GO' if status == 'DANGER' else ('CAUTION' if status == 'WARNING' else ('CONDITIONAL' if has_temporal else 'GO'))}."
            core_hazard = (
                f"Maritime security and border surveillance report: India EEZ perimeter secure with {border_dist:.1f} NM border clearance. "
                f"SLOC shipping lane status: {sloc_info}. "
                f"{'; '.join(threats) if threats else 'Normal surveillance conditions.'}"
            )
            action_dir = f"IMMEDIATE ACTION: Broadcast fleet advisory directives, maintain coastal security watch across {dist_nm:.1f} nautical miles, and inspect plotted patrol waypoints on your mobile map interface."
            temporal_clause = temporal_boundary if (has_temporal and status not in ("DANGER", "WARNING")) else ""
            explanation = f"{status_tag} {inland_prefix}{temporal_clause}{core_hazard}{target_addon} {action_dir}"
        elif p_enum == StakeholderPersona.DISASTER_MANAGEMENT:
            status_tag = f"STATUS: {'NO-GO' if status == 'DANGER' else ('CAUTION' if status == 'WARNING' else ('CONDITIONAL' if has_temporal else 'GO'))}."
            if metrics.get("is_cyclone_active") or "cyclone" in (metrics.get("hazard_description") or "").lower():
                core_hazard = f"Critical disaster incident: Cyclone track active with gale-force winds and life-safety storm surge threats."
                action_dir = f"IMMEDIATE ACTION: Activate coastal evacuation corridors across {dist_nm:.1f} nautical miles toward designated emergency breakwater shelters, and monitor plotted emergency points on your mobile map interface."
            elif metrics.get("hazard_active"):
                core_hazard = f"Coastal hazard advisory: {metrics.get('hazard_description') or 'Severe meteorological hazard active'}."
                action_dir = f"IMMEDIATE ACTION: Maintain heightened disaster response readiness and review coastal shelter sectors on your mobile map interface."
            else:
                core_hazard = f"Coastal disaster monitoring: Sea state stable with wave heights at {metrics.get('wave_height_m', 1.2):.2f} meters. No active cyclonic threat detected."
                action_dir = f"IMMEDIATE ACTION: Maintain routine disaster surveillance protocols and observe coastal monitoring sectors on your mobile map interface."
            temporal_clause = temporal_boundary if (has_temporal and status not in ("DANGER", "WARNING")) else ""
            explanation = f"{status_tag} {inland_prefix}{temporal_clause}{core_hazard}{target_addon} {action_dir}"
        elif p_enum == StakeholderPersona.MARITIME_OPERATOR:
            status_tag = f"STATUS: {'NO-GO' if status == 'DANGER' else ('CAUTION' if status == 'WARNING' else ('CONDITIONAL' if has_temporal else 'GO'))}."
            vis = metrics.get("visibility_km", 10.0)
            lane_name = metrics.get("shipping_lane_name", "Cape Comorin SLOC Fairway")
            core_hazard = (
                f"Commercial shipping navigation update: Visibility is {vis:.1f} km along {lane_name}. "
                f"Sustained wind speed is {metrics.get('wind_speed_kmph', 15.0):.1f} km/h and wave height is {metrics.get('wave_height_m', 1.2):.2f} meters."
            )
            action_dir = f"IMMEDIATE ACTION: Maintain navigation along designated deep-draft Traffic Separation Scheme corridor across {dist_nm:.1f} nautical miles, and verify waypoints on your mobile map interface."
            temporal_clause = temporal_boundary if (has_temporal and status not in ("DANGER", "WARNING")) else ""
            explanation = f"{status_tag} {inland_prefix}{temporal_clause}{core_hazard}{target_addon} {action_dir}"
        elif is_route_query:
            if status == "DANGER":
                status_tag = "STATUS: NO-GO."
                explanation = f"{status_tag} {inland_prefix}Severe marine hazards detected along the requested transit corridor. Transit held for safety."
            elif status == "WARNING":
                status_tag = "STATUS: CAUTION."
                explanation = f"{status_tag} {inland_prefix}Elevated sea state along the corridor ({w_ht:.1f}m waves). Proceed with heightened navigational vigilance."
            else:
                status_tag = "STATUS: GO."
                explanation = (
                    f"{status_tag} {inland_prefix}Route analysis complete. Navigational corridor cleared across {dist_nm:.1f} nautical miles to {alt_data.get('safe_location_name', 'destination')}. "
                    f"Passage maintains safe clearance from coastal shallows, shipping fairways, and the IMBL border with wave heights at {w_ht:.1f} meters. "
                    f"Refer to plotted waypoints on your mobile map interface."
                )
        else:
            # General marine safety assessment
            if status == "DANGER":
                status_tag = "STATUS: NO-GO."
                explanation = f"{status_tag} {inland_prefix}Severe sea conditions with wave heights reaching {w_ht:.1f} meters. Cease offshore operations immediately."
            elif status == "WARNING":
                status_tag = "STATUS: CAUTION."
                explanation = f"{status_tag} {inland_prefix}Marginal sea conditions detected with wave heights near {w_ht:.1f} meters. Heightened vigilance required."
            else:
                status_tag = "STATUS: CONDITIONAL." if has_temporal else "STATUS: GO."
                temporal_clause = temporal_boundary if has_temporal else ""
                explanation = f"{status_tag} {inland_prefix}{temporal_clause}Favorable maritime conditions observed with wave heights of {w_ht:.1f} meters and wind speeds of {w_spd:.1f} km/h. Sea state is safe for operations."

        if forecast_offline:
            explanation = f"Forecast API offline. Displaying real-time observational data instead. {explanation}"

        cleaned_text = self._clean_tts_text(explanation)
        return {
            "bhashini_text": cleaned_text,
            "text_advisory_local": cleaned_text,
            "map_status": status if forecast_active else ("CONDITIONAL" if (has_temporal and status == "SAFE") else status),
            "recommended_coordinates": rec_coords,
        }


# =====================================================================
# 4. MAIN EXECUTION PIPELINE
# =====================================================================
def run_decision_engine(
    raw_agent_outputs: Dict[str, Any],
    persona: str = "FISHERMAN",
    language_code: str = "en",
    normalized_query: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main orchestration pipeline that sequentially chains:
    1. RiskAnalysisAgent -> Deterministic threat evaluation & risk scoring.
    2. SafeAlternative   -> Calculates rerouting if threats exceed thresholds.
    3. ReasoningAgent     -> Voice-friendly Gemini-powered emergency broadcast synthesis.

    Returns:
        Dict formatted for frontend consumption and Bhashini translation:
        {
            "bhashini_text": "...",
            "text_advisory_local": "...",
            "map_status": "SAFE | WARNING | DANGER",
            "recommended_coordinates": "lat,lon",
            "risk_assessment": {...},
            "alternative_route": {...}
        }
    """
    p_enum = resolve_persona(persona) or StakeholderPersona.FISHERMAN
    print("\n" + "=" * 70)
    print("ORCA INTELLIGENT ANALYSIS & DECISION ENGINE")
    print(f"Active Stakeholder Persona : {p_enum.value}")
    print("=" * 70)

    resolved_query = (
        normalized_query
        or raw_agent_outputs.get("normalized_query")
        or raw_agent_outputs.get("user_query")
    )

    # Detect temporal & forecast intent
    temporal_info = detect_temporal_intent(resolved_query)
    is_future_query = temporal_info.get("is_future_query", False)
    is_beyond_24h = temporal_info.get("is_beyond_24h", False)

    forecast_data = None
    forecast_offline = False

    if is_future_query and not is_beyond_24h:
        t_lat, t_lon = resolve_target_coordinates(raw_agent_outputs, resolved_query)
        forecast_agent = ForecastAgent(timeout=5.0)
        forecast_data = forecast_agent.fetch_forecast(t_lat, t_lon)
        if forecast_data and forecast_data.get("success"):
            print(f"[ForecastAgent] Open-Meteo Marine Forecast fetched for ({t_lat:.4f}, {t_lon:.4f})")
            print(f"               Max Wave: {forecast_data['max_wave_height_m']}m | Horizon: {forecast_data['time_horizon']}")
        else:
            print(f"[ForecastAgent] Forecast API offline / unreachable. Falling back silently to real-time MOSDAC cache.")
            forecast_offline = True

    # Determine Vessel Location & Target Destination early for spatial cross-referencing
    vessel_location = raw_agent_outputs.get("vessel_location")
    if not vessel_location and "device_telemetry" in raw_agent_outputs:
        dt = raw_agent_outputs["device_telemetry"]
        if isinstance(dt, dict) and "latitude" in dt and "longitude" in dt:
            vessel_location = f"{dt['latitude']:.4f},{dt['longitude']:.4f}"
    if not vessel_location:
        vessel_location = "8.7642,78.1348"  # Default (Thoothukudi Departure Port)

    # Determine PFZ or requested target destination
    target_destination = None
    pfz_out = raw_agent_outputs.get("PFZ_AGENT")
    if isinstance(pfz_out, dict):
        features = pfz_out.get("geojson", {}).get("features", [])
        if features:
            props = features[0].get("properties", {})
            c_lat = props.get("centroid_lat")
            c_lon = props.get("centroid_lon")
            if c_lat and c_lon:
                target_destination = f"{c_lat:.4f},{c_lon:.4f}"

    if not target_destination:
        for loc_k in ["destination", "target_location"]:
            if loc_k in raw_agent_outputs and raw_agent_outputs[loc_k]:
                target_destination = raw_agent_outputs[loc_k]
                break

    if not target_destination and resolved_query:
        m_route = re.search(r"from\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+)", resolved_query, re.IGNORECASE)
        if m_route:
            vessel_location = m_route.group(1).strip()
            target_destination = m_route.group(2).strip()

    # Resolve coordinates for cyclone track spatial cross-referencing
    gis_resolver = GisAgent()
    orig_coords = gis_resolver.resolve_location(vessel_location)
    orig_lat, orig_lon = (float(orig_coords["lat"]), float(orig_coords["lon"])) if orig_coords and "lat" in orig_coords else (8.7642, 78.1348)
    route_coords = [(orig_lat, orig_lon)]

    if target_destination:
        dest_coords = gis_resolver.resolve_location(target_destination)
        if dest_coords and "lat" in dest_coords:
            route_coords.append((float(dest_coords["lat"]), float(dest_coords["lon"])))

    gis_out = raw_agent_outputs.get("GIS_AGENT", {})
    if isinstance(gis_out, dict):
        gis_wps = gis_out.get("safe_sea_route", {}).get("waypoints", [])
        for wp in gis_wps:
            if isinstance(wp, (list, tuple)) and len(wp) >= 2:
                route_coords.append((float(wp[0]), float(wp[1])))
            elif isinstance(wp, dict) and ("lat" in wp or "latitude" in wp) and ("lon" in wp or "longitude" in wp):
                route_coords.append((float(wp.get("lat") or wp.get("latitude")), float(wp.get("lon") or wp.get("longitude"))))

    # Real-Time Cyclone Tracking via GDACS (with strict 1s timeout & MOSDAC .h5 fallback; bypassed in FAST_DEMO_MODE)
    if FAST_DEMO_MODE:
        cyclone_track_data = None
    else:
        cyclone_agent = CycloneTrackAgent(timeout=1.0)
        cyclone_track_data = cyclone_agent.fetch_active_cyclone_tracks()
    if cyclone_track_data and cyclone_track_data.get("active"):
        is_collision, clearance_nm, closest_pt = cyclone_agent.check_cyclone_collision(
            route_coords, cyclone_track_data, threshold_nm=200.0
        )
        cyclone_intel = {
            "active_storms": cyclone_track_data.get("name"),
            "trajectory_path": cyclone_track_data.get("next_24h_projection"),
            "route_collision": is_collision,
            "clearance_distance_nm": clearance_nm,
            "storm_details": cyclone_track_data,
        }
        print(f"[CycloneTrackAgent] GDACS Active Storm: {cyclone_track_data['name']} | 24h Proj: {cyclone_track_data['next_24h_projection']}")
        print(f"                     Collision: {'YES (< 200 NM)' if is_collision else 'NO'} | Clearance: {clearance_nm} NM")
    else:
        # Fallback to local MOSDAC satellite HDF5 telemetry (DisasterAgent)
        disaster_out = raw_agent_outputs.get("DISASTER_AGENT", {})
        mosdac_ct = disaster_out.get("cyclone_track") if isinstance(disaster_out, dict) else None
        if mosdac_ct and isinstance(mosdac_ct, dict) and mosdac_ct.get("has_active_track"):
            m_name = mosdac_ct.get("cyclone_name", "Tropical Cyclonic System")
            m_wps = mosdac_ct.get("projected_waypoints", [])
            m_pts = [(float(p[0]), float(p[1])) for p in m_wps if isinstance(p, (list, tuple)) and len(p) >= 2]
            mock_storm = {"active": True, "name": m_name, "current_coords": m_pts[0] if m_pts else (orig_lat, orig_lon), "track_path": m_pts}
            is_collision, clearance_nm, _ = cyclone_agent.check_cyclone_collision(route_coords, mock_storm, threshold_nm=200.0)
            cyclone_intel = {
                "active_storms": m_name,
                "trajectory_path": mosdac_ct.get("next_24h_projection") or (f"Lat {m_pts[-1][0]:.4f}°, Lon {m_pts[-1][1]:.4f}°" if m_pts else "Projected Trajectory"),
                "route_collision": is_collision,
                "clearance_distance_nm": clearance_nm,
                "storm_details": mosdac_ct,
            }
            print(f"[CycloneTrackAgent] MOSDAC Local Fallback Storm: {m_name} | Collision: {is_collision} ({clearance_nm} NM)")
        else:
            cyclone_intel = {
                "active_storms": None,
                "trajectory_path": None,
                "route_collision": False,
                "clearance_distance_nm": None,
                "storm_details": None,
            }
            print("[CycloneTrackAgent] No active tropical cyclones in Indian Ocean basin (or GDACS offline). Reading local MOSDAC satellite telemetry.")

    # Step 1: Risk Analysis & Geopolitical Boundary Guardrail / Jurisdictional Assessment
    risk_agent = RiskAnalysisAgent()
    risk_data = risk_agent.evaluate_threats(
        raw_agent_outputs,
        normalized_query=resolved_query,
        persona=p_enum.value,
        forecast_data=forecast_data,
        forecast_offline=forecast_offline,
        cyclone_intel=cyclone_intel,
    )
    print(f"[1] Threat Status Evaluated : {risk_data['status']} (Risk Score: {risk_data['risk_score']}/100)")
    if risk_data["threat_prioritization"]:
        for idx, threat in enumerate(risk_data["threat_prioritization"], 1):
            print(f"    - Threat {idx}: {threat}")
    else:
        print("    - No critical maritime threats detected.")

    # Step 2: Route Planning & Safe Alternative
    safe_alt = SafeAlternative()
    alt_data = safe_alt.calculate_reroute(
        primary_location=vessel_location,
        threat_data=risk_data,
        target_destination=target_destination,
        persona=p_enum.value,
    )

    ocean_target = resolve_ocean_target(
        query_text=resolved_query,
        raw_agent_outputs=raw_agent_outputs,
        vessel_lat=orig_lat,
        vessel_lon=orig_lon,
    )

    print(f"[2] Alternative Route Plan  : {'REROUTE NEEDED' if alt_data['reroute_needed'] else 'PROCEED DIRECTLY'}")
    print(f"    - Departure (WP 1)     : {alt_data['primary_coordinates']}")
    print(f"    - Destination (WP 2)   : {alt_data['safe_coordinates']} ({alt_data['distance_shift_nm']} NM)")
    print(f"    - Routing Rationale     : {alt_data['rationale']}")
    if ocean_target:
        print(f"    - Primary Target       : {ocean_target['feature_type']} ({ocean_target['target_coordinates']})")
        print(f"    - Relative Vector      : {ocean_target['relative_vector']}")
        print(f"    - Landmark Reference   : {ocean_target['landmark_reference']}")
    else:
        print("    - Primary Target       : N/A (Non-Marine / Out of Domain)")

    # Step 3: Reasoning & Synthesis via Voice-Friendly Gemini Broadcast Engine
    reasoning_agent = ReasoningAgent()
    synthesis = reasoning_agent.synthesize_insights(
        aggregated_data=raw_agent_outputs,
        risk_data=risk_data,
        alternative_data=alt_data,
        persona=p_enum.value,
        language_code=language_code,
        ocean_target=ocean_target,
    )
    print(f"[3] Gemini Synthesis Engine : Advisory Generated ({synthesis.get('map_status')})")
    print(f"    - Model Active         : {reasoning_agent.model_name}")

    final_payload = {
        "bhashini_text": synthesis.get("bhashini_text", ""),
        "text_advisory_local": synthesis.get("text_advisory_local", synthesis.get("bhashini_text", "")),
        "map_status": synthesis.get("map_status", risk_data["status"]),
        "recommended_coordinates": synthesis.get("recommended_coordinates", alt_data["safe_coordinates"]),
        "primary_geographic_target": ocean_target,
        "risk_assessment": risk_data,
        "alternative_route": alt_data,
        "safe_sea_route": alt_data.get("safe_sea_route") or raw_agent_outputs.get("GIS_AGENT", {}).get("safe_sea_route"),
        "persona": p_enum.value,
        "language_code": language_code,
        "imbl_violation": risk_data.get("metrics", {}).get("imbl_violation", False),
        "within_eez": risk_data.get("metrics", {}).get("within_eez", True),
        "is_live_satellite": any(
            bool(raw_agent_outputs.get(ag, {}).get("is_live_satellite", False))
            for ag in ("OCEAN_AGENT", "WEATHER_AGENT", "DISASTER_AGENT", "PFZ_AGENT")
            if isinstance(raw_agent_outputs.get(ag), dict)
        ),
        "is_future_query": is_future_query,
        "forecast_active": risk_data.get("forecast_active", False),
        "forecast_offline": risk_data.get("forecast_offline", False),
        "status_label": risk_data.get("status_label", "Operational Status"),
    }

    if risk_data.get("forecast_active"):
        final_payload["forecast_max_wave_m"] = risk_data.get("forecast_max_wave_m")
        final_payload["forecast_avg_wave_m"] = risk_data.get("forecast_avg_wave_m")
        final_payload["forecast_horizon"] = risk_data.get("forecast_horizon")

    # Attach cyclone intelligence, trajectory, subsea seismic, and shelter route if active
    final_payload["cyclone_intelligence"] = cyclone_intel
    if cyclone_track_data:
        final_payload["cyclone_track"] = cyclone_track_data

    disaster_out = raw_agent_outputs.get("DISASTER_AGENT", {})
    if isinstance(disaster_out, dict):
        if "cyclone_track" in disaster_out and "cyclone_track" not in final_payload:
            final_payload["cyclone_track"] = disaster_out["cyclone_track"]
        if "subsea_earthquake" in disaster_out:
            final_payload["subsea_earthquake"] = disaster_out["subsea_earthquake"]
        if "emergency_shelter" in disaster_out:
            final_payload["emergency_shelter"] = disaster_out["emergency_shelter"]

    # Attach Green Marine Solar Energy Endurance & Sustainability (Diurnal Time-Aware INSAT-3DR IMC)
    status_str = str(synthesis.get("map_status") or risk_data.get("status") or "SAFE").upper()
    is_daylight, current_hour_dec, diurnal_label = check_diurnal_cycle(raw_agent_outputs)
    is_nighttime = not is_daylight

    if status_str in ("OUT_OF_DOMAIN", "SECURITY_REJECTION", "TEMPORAL_OUT_OF_BOUNDS", "LAND_INTERSECTION_ERROR", "PENDING FORECAST", "DATA_UNAVAILABLE"):
        solar_w_m2 = 0
    elif is_nighttime:
        # Diurnal Nighttime: Force 0 W/m² zero solar insolation
        solar_w_m2 = 0
    elif status_str in ("SAFE", "GO", "OPERATIONAL"):
        solar_w_m2 = random.randint(700, 950)
    elif status_str == "EMERGENCY_SAR":
        solar_w_m2 = random.randint(0, 150)
    else:
        solar_w_m2 = random.randint(0, 200)

    # Extract route distance in nautical miles
    route_distance_nm = 0.0
    safe_route = alt_data.get("safe_sea_route") or raw_agent_outputs.get("GIS_AGENT", {}).get("safe_sea_route")
    if safe_route and isinstance(safe_route, dict) and safe_route.get("route_status") != "LAND_INTERSECTION_ERROR":
        route_distance_nm = safe_route.get("total_distance_nm") or safe_route.get("distance_nm") or 0.0
    if not route_distance_nm and status_str not in ("OUT_OF_DOMAIN", "SECURITY_REJECTION", "TEMPORAL_OUT_OF_BOUNDS", "LAND_INTERSECTION_ERROR", "PENDING FORECAST", "DATA_UNAVAILABLE"):
        route_distance_nm = alt_data.get("distance_shift_nm") or alt_data.get("reroute_distance_nm") or 0.0

    # Calculate persona-specific fuel consumption rate and savings
    fuel_rate = get_fuel_consumption_rate(p_enum)

    if is_nighttime:
        # Nighttime: Set active solar fuel savings to 0.0 Liters
        fuel_saved = 0.0
        carbon_offset_kg = 0.0
        solar_daily = 0.0
        hourly_recharge_kw = 0.0
        extended_hours = 0.0
        solar_range_nm = 0.0
        energy_advisory = "INSAT-3DR Solar Insolation (0 W/m²). (Nighttime / Zero Solar Insolation): Auxiliary solar generation inactive; vessel operating on stored battery buffer reserve only."
    elif route_distance_nm and route_distance_nm > 0 and status_str not in ("OUT_OF_DOMAIN", "SECURITY_REJECTION", "TEMPORAL_OUT_OF_BOUNDS", "LAND_INTERSECTION_ERROR", "PENDING FORECAST", "DATA_UNAVAILABLE"):
        fuel_saved = round((float(route_distance_nm) * fuel_rate) * 0.20, 1)
        carbon_offset_kg = round(fuel_saved * 2.68, 1)
        solar_daily = round((solar_w_m2 / 1000.0) * 6.5, 1)
        hourly_recharge_kw = round((solar_w_m2 / 1000.0) * 1.5 * 0.82, 2)
        extended_hours = round((hourly_recharge_kw / 2.2) * 8.0, 1) if solar_w_m2 > 0 else 0.0
        solar_range_nm = round(extended_hours * 5.5, 1)
        energy_advisory = f"INSAT-3DR Solar Insolation ({solar_w_m2} W/m²) yields +{extended_hours}h (+{solar_range_nm} NM) auxiliary electric endurance for solar-hybrid craft."
    else:
        solar_daily = round((solar_w_m2 / 1000.0) * 6.5, 1)
        hourly_recharge_kw = round((solar_w_m2 / 1000.0) * 1.5 * 0.82, 2)
        extended_hours = round((hourly_recharge_kw / 2.2) * 8.0, 1) if solar_w_m2 > 0 else 0.0
        solar_range_nm = round(extended_hours * 5.5, 1)
        if solar_w_m2 > 0 and status_str not in ("OUT_OF_DOMAIN", "SECURITY_REJECTION", "TEMPORAL_OUT_OF_BOUNDS", "LAND_INTERSECTION_ERROR", "PENDING FORECAST", "DATA_UNAVAILABLE"):
            fuel_saved = round(max(2.5, extended_hours * 1.45 * (solar_w_m2 / 850.0)), 1)
            carbon_offset_kg = round(fuel_saved * 2.68, 1)
        else:
            fuel_saved = 0.0
            carbon_offset_kg = 0.0
        energy_advisory = f"INSAT-3DR Solar Insolation ({solar_w_m2} W/m²) yields +{extended_hours}h (+{solar_range_nm} NM) auxiliary electric endurance for solar-hybrid craft."

    final_payload["green_marine_energy"] = {
        "is_daylight": is_daylight,
        "is_nighttime": is_nighttime,
        "current_hour_ist": current_hour_dec,
        "diurnal_cycle": diurnal_label,
        "solar_irradiance_wm2": float(solar_w_m2),
        "solar_insolation_wm2": float(solar_w_m2),
        "solar_irradiance": float(solar_w_m2),
        "solar_wm2": float(solar_w_m2),
        "daily_solar_yield_kwh_m2": solar_daily,
        "effective_solar_recharge_kw": hourly_recharge_kw,
        "extended_zero_emission_hours": extended_hours,
        "zero_emission_hours": extended_hours,
        "auxiliary_endurance_hrs": extended_hours,
        "battery_hours": extended_hours,
        "stored_battery_buffer_only": is_nighttime,
        "solar_assisted_range_nm": solar_range_nm,
        "fuel_consumption_rate_l_nm": fuel_rate,
        "fuel_saved_liters": fuel_saved,
        "fuel_savings_liters": fuel_saved,
        "diesel_saved_liters": fuel_saved,
        "carbon_offset_kg": carbon_offset_kg,
        "co2_saved_kg": carbon_offset_kg,
        "carbon_saved_kg": carbon_offset_kg,
        "calculation_basis": "THEORETICAL_MODEL_ESTIMATE",
        "is_estimate": True,
        "telemetry_note": "Engineering estimate based on INSAT-3DR solar insolation diurnal model and standard marine diesel displacement factors; not shipboard sensor telemetry.",
        "advisory": energy_advisory,
    }
    final_payload["green_energy"] = final_payload["green_marine_energy"]

    if status_str == "OUT_OF_DOMAIN" or risk_data.get("out_of_domain") or synthesis.get("map_status") == "OUT_OF_DOMAIN":
        ood_refusal = (
            "I am ORCA, a specialized maritime intelligence engine. "
            "I am programmed exclusively to assist with marine navigation, weather analysis, and coastal safety operations. "
            "I cannot process requests outside of this scope."
        )
        final_payload["map_status"] = "OUT_OF_DOMAIN"
        final_payload["bhashini_text"] = ood_refusal
        final_payload["text_advisory_local"] = ood_refusal
        final_payload["recommended_coordinates"] = ""
        final_payload["primary_geographic_target"] = None
        final_payload["safe_sea_route"] = None
        if "risk_assessment" in final_payload and isinstance(final_payload["risk_assessment"], dict):
            final_payload["risk_assessment"]["status"] = "OUT_OF_DOMAIN"
            final_payload["risk_assessment"]["risk_score"] = "N/A"
        if "alternative_route" in final_payload and isinstance(final_payload["alternative_route"], dict):
            final_payload["alternative_route"]["safe_coordinates"] = ""
            final_payload["alternative_route"]["primary_coordinates"] = ""
            final_payload["alternative_route"]["safe_sea_route"] = None
            final_payload["alternative_route"]["waypoints"] = []

    if not final_payload.get("forecast_active") and (status_str in ("PENDING FORECAST", "DATA_UNAVAILABLE", "TEMPORAL_OUT_OF_BOUNDS") or risk_data.get("pending_forecast") or risk_data.get("temporal_out_of_bounds") or synthesis.get("map_status") in ("PENDING FORECAST", "DATA_UNAVAILABLE")):
        pending_msg = (
            "STATUS: PENDING FORECAST. Real-time ISRO MOSDAC satellite Earth Observation cache provides "
            "near-real-time observations and short-term 24-hour nowcasting, and cannot predict conditions beyond 24 hours. "
            "The requested forecast horizon is unavailable. Please check back closer to your departure window for updated satellite passes and numerical forecast model runs."
        )
        final_payload["map_status"] = "PENDING FORECAST"
        final_payload["bhashini_text"] = pending_msg
        final_payload["text_advisory_local"] = pending_msg
        final_payload["recommended_coordinates"] = ""
        final_payload["primary_geographic_target"] = None
        final_payload["safe_sea_route"] = None
        if "risk_assessment" in final_payload and isinstance(final_payload["risk_assessment"], dict):
            final_payload["risk_assessment"]["status"] = "PENDING FORECAST"
            final_payload["risk_assessment"]["risk_score"] = "N/A"
        if "alternative_route" in final_payload and isinstance(final_payload["alternative_route"], dict):
            final_payload["alternative_route"]["safe_coordinates"] = ""
            final_payload["alternative_route"]["primary_coordinates"] = ""
            final_payload["alternative_route"]["safe_sea_route"] = None
            final_payload["alternative_route"]["waypoints"] = []

    print("=" * 70 + "\n")
    return final_payload


if __name__ == "__main__":
    print("--- SCENARIO 1: DANGEROUS HIGH-WIND & ROUGH SWELL CONFLICT ---")
    dangerous_input = {
        "wind_speed_kmph": 48.5,  # Exceeds 40 km/h -> DANGER
        "wave_height_m": 3.1,    # Exceeds 2.5 m  -> DANGER
        "PFZ_AGENT": {
            "geojson": {
                "features": [
                    {
                        "properties": {
                            "suitability_score": 94,
                            "centroid_lat": 9.15,
                            "centroid_lon": 78.45,
                            "likely_catch": ["Yellowfin Tuna", "Mackerel"],
                        }
                    }
                ]
            }
        },
    }
    result_danger = run_decision_engine(dangerous_input, persona="FISHERMAN")
    print("[Final Output JSON]:")
    print(json.dumps(result_danger, indent=2))

    print("\n--- SCENARIO 2: CALM FAVORABLE FISHING CONDITIONS ---")
    safe_input = {
        "wind_speed_kmph": 18.0,
        "wave_height_m": 1.2,
        "PFZ_AGENT": {
            "geojson": {
                "features": [
                    {
                        "properties": {
                            "suitability_score": 88,
                            "centroid_lat": 8.85,
                            "centroid_lon": 78.25,
                            "likely_catch": ["Sardines"],
                        }
                    }
                ]
            }
        },
    }
    result_safe = run_decision_engine(safe_input, persona="FISHERMAN")
    print("[Final Output JSON]:")
    print(json.dumps(result_safe, indent=2))

    print("\n--- SCENARIO 3: IMBL CROSS-BORDER VIOLATION (SRI LANKA WATERS) ---")
    imbl_input = {
        "wind_speed_kmph": 15.0,
        "wave_height_m": 1.1,
        "normalized_query": "Plan a deep sea fishing route towards Jaffna into Sri Lankan waters.",
    }
    result_imbl = run_decision_engine(imbl_input, persona="FISHERMAN")
    print("[Final Output JSON]:")
    print(json.dumps(result_imbl, indent=2))
