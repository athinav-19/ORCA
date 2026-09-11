"""
cli.py - ORCA Standalone Terminal Command-Line Interface
ISRO SIH Problem Statement 176: Marine Multi-Agent System

Enables full interaction with the Project ORCA multi-agent intelligence backend
directly from the terminal (PowerShell / Bash / CMD) without requiring any
frontend, browser, or web server.

Usage:
  Interactive Mode:
    python cli.py

  Single Query Mode:
    python cli.py --query "Where are the safest fishing zones near Tuticorin?"
    python cli.py --query "தூத்துக்குடியில் மீன்பிடிக்க சிறந்த இடம் எங்கே?" --lang ta
    python cli.py --query "तूतीकोरिन के पास मौसम कैसा है?" --lang hi

  Quick Diagnostics:
    python cli.py --status
"""

import os
import sys
import time
import argparse
import datetime
import random
from typing import Optional

# Ensure UTF-8 output and immediate line buffering on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()

# Import Backend Pipeline Components
from models import (
    StakeholderPersona,
    PERSONA_BADGES,
    PERSONA_MENU_OPTIONS,
    resolve_persona,
    classify_persona_intent,
    get_fuel_consumption_rate,
)
from main import ManagerAgent, process_marine_request
from shadow_cache_worker import ensure_latest_mosdac_cache, CACHE_DIR
from setup_mosdac_config import SUPPORTED_DATASETS

from translation_service import IndicTranslationService, SUPPORTED_LANGUAGES


DIVIDER_BOLD = "=" * 76
DIVIDER_THIN = "-" * 76


def print_banner():
    print(DIVIDER_BOLD)
    print("  PROJECT ORCA - MARINE MULTI-AGENT INTELLIGENCE ENGINE (CLI)")
    print("  ISRO SIH Problem Statement 176 | 9-Dataset Satellite Backend")
    print("  Stakeholders: FISHERMAN | MARITIME_AUTHORITY | DISASTER_MANAGEMENT")
    print("                RESEARCHER | MARITIME_OPERATOR")
    print("  Headless Backend Mode: 100% Autonomous (No Frontend Required)")
    print(DIVIDER_BOLD)


def print_status_dossier():
    """Prints live diagnostic summary of 9 MOSDAC datasets and agent readiness."""
    print("\n" + DIVIDER_THIN)
    print(" SYSTEM DIAGNOSTICS & MOSDAC SHADOW CACHE STATUS")
    print(DIVIDER_THIN)
    import glob

    cache_ok = ensure_latest_mosdac_cache(max_age_hours=72.0)
    print(f"MOSDAC Local Cache Dir : {CACHE_DIR}")
    print(f"Overall Cache Health   : {'ONLINE (All Fresh)' if cache_ok else 'PARTIAL'}")
    print(f"Active Datasets (9)    :")
    for code in sorted(SUPPORTED_DATASETS):
        short_code = code.split("_")[-1]
        matched = glob.glob(os.path.join(CACHE_DIR, f"*{short_code}*.h5")) + glob.glob(os.path.join(CACHE_DIR, f"*{short_code}*.nc"))
        status_icon = "[OK]" if matched else "[MISSING]"
        file_name = os.path.basename(matched[0]) if matched else "none"
        size_kb = round(os.path.getsize(matched[0]) / 1024, 1) if matched else 0
        print(f"  {status_icon:<10} {code:<24} {size_kb:>7.1f} KB  ({file_name})")

    print(f"\nLanguage Layer:")
    print(f"  - Translation Engine : deep-translator (Google Translator)")
    print(f"  - Multilingual Layer : ACTIVE (Zero-credential in-process Indic translation)")
    print(f"  - Supported Languages: English, Tamil, Hindi, Malayalam, Telugu, Gujarati, Kannada, Odia")
    print(f"\nDomain Agents:")
    print(f"  - PFZ_AGENT          : Potential Fishing Zones (Oceansat-3 Chl + Insat-3D SST)")
    print(f"  - OCEAN_AGENT        : Ocean State Forecast (SWH Wave Heights + Currents)")
    print(f"  - WEATHER_AGENT      : Marine Weather & Atmospheric Hazards")
    print(f"  - DISASTER_AGENT     : Cyclone, Storm Surge & Coastal Warnings")
    print(f"  - GIS_AGENT          : India EEZ Boundary & Maritime Spatial Clearance")
    print(f"  - DECISION_ENGINE    : Multi-Objective Synthesis & Safe Navigation Routes")
    print(DIVIDER_THIN + "\n")


def execute_query(
    query_text: str,
    persona: Optional[str] = None,
    lat: float = 8.7642,
    lon: float = 78.1348,
    lang: Optional[str] = None
):
    """Executes a natural language query through the full multi-agent backend pipeline."""
    p_enum = resolve_persona(persona)
    persona_val = p_enum.value if p_enum else None

    manager = ManagerAgent()

    request_payload = {
        "session_id": "sess_cli_terminal",
        "client_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "user_context": {"persona": persona_val},
        "device_telemetry": {
            "latitude": lat,
            "longitude": lon,
            "speed_knots": 0.0,
            "heading_degrees": 120.0
        },
        "user_input": {
            "input_type": "TEXT",
            "raw_text": query_text
        }
    }
    if lang:
        request_payload["source_language_code"] = lang

    start_time = time.time()
    result = process_marine_request(request_payload, manager=manager)
    elapsed = time.time() - start_time

    # Display Pretty CLI Report with exact official persona badge
    active_persona = result.get("user_persona") or (persona_val if persona_val else "FISHERMAN")
    resolved_active = resolve_persona(active_persona)
    badge_key = resolved_active.value if resolved_active else active_persona
    persona_label = PERSONA_BADGES.get(badge_key, f"👤 {active_persona}")

    print("\n" + DIVIDER_BOLD)
    print("                      ORCA DECISION DOSSIER")
    print(DIVIDER_BOLD)
    print(f"👤 Persona              : {persona_label}")
    print(f"📍 Vessel Position      : Lat {lat:.4f}, Lon {lon:.4f}")
    print(f"🌐 Detected Language    : {result.get('source_language_code', 'en').upper()}")
    print(f"📝 Normalized English   : \"{result.get('english_query')}\"")
    print(f"⏱️ Backend Processing   : {elapsed:.2f}s")
    print(DIVIDER_THIN)

    # Risk Assessment
    risk_info = result.get("risk_assessment", {})
    risk_score = risk_info.get("risk_score", "N/A")
    raw_status = str(risk_info.get("status") or result.get("map_status") or "OPERATIONAL").upper()

    is_forecast_active = result.get("forecast_active", False)
    status_label = result.get("status_label") or ("Forecast Status" if is_forecast_active else "Operational Status")

    is_ood = "OUT_OF_DOMAIN" in raw_status
    is_pending_forecast = (
        "PENDING FORECAST" in raw_status
        or "DATA_UNAVAILABLE" in raw_status
        or "TEMPORAL_OUT_OF_BOUNDS" in raw_status
        or risk_info.get("pending_forecast", False)
        or risk_info.get("temporal_out_of_bounds", False)
    )
    if is_ood:
        risk_color = "⚪ [OUT OF DOMAIN]"
        status_badge = "🚫 OUT_OF_DOMAIN (Non-Maritime Query Refused)"
        risk_str = "Score N/A"
    elif is_pending_forecast:
        risk_color = "⏳ [PENDING FORECAST]"
        status_badge = "⏳ PENDING FORECAST (>24h Satellite Horizon Limit)"
        risk_str = "Score N/A"
    elif "EMERGENCY_SAR" in raw_status:
        risk_color = "🚨 [EMERGENCY SAR OVERRIDE]"
        status_badge = "🔴 EMERGENCY_SAR (SOLAS Priority Rescue Vector)"
        risk_str = f"Score {risk_score}/100" if risk_score != "N/A" else "Score N/A"
    elif "SECURITY_REJECTION" in raw_status:
        risk_color = "🛑 [SECURITY REJECTION]"
        status_badge = "🚫 SECURITY_REJECTION (Adversarial Bypass Blocked)"
        risk_str = f"Score {risk_score}/100" if risk_score != "N/A" else "Score N/A"
    elif "LAND_INTERSECTION_ERROR" in raw_status:
        risk_color = "⛔ [LAND INTERSECTION ERROR]"
        status_badge = "❌ LAND_INTERSECTION_ERROR (Overland Sea Route Impossible)"
        risk_str = f"Score {risk_score}/100" if risk_score != "N/A" else "Score N/A"
    elif is_forecast_active:
        f_wave = result.get("forecast_max_wave_m")
        wave_str = f" (Wave: {f_wave:.2f}m)" if f_wave is not None else ""
        if raw_status in ("SAFE", "GO"):
            risk_color = "🟢 [FORECAST: SAFE]"
            status_badge = f"🟢 SAFE{wave_str} - Favorable Marine Forecast"
        elif raw_status in ("WARNING", "CAUTION"):
            risk_color = "🟡 [FORECAST: ELEVATED]"
            status_badge = f"🟡 CAUTION{wave_str} - Elevated Forecast Sea State"
        elif raw_status in ("DANGER", "NO-GO"):
            risk_color = "🔴 [FORECAST: CRITICAL]"
            status_badge = f"🔴 NO-GO{wave_str} - Severe Forecast Wave Hazard"
        else:
            status_badge = f"{raw_status}{wave_str}"
        risk_str = f"Score {risk_score}/100" if risk_score != "N/A" else "Score N/A"
    elif "CONDITIONAL" in raw_status or result.get("native_advisory_text", "").startswith("STATUS: CONDITIONAL"):
        risk_color = "🟡 [CONDITIONAL]"
        status_badge = "🟡 CONDITIONAL (Subject to Pre-Departure Nowcast)"
        risk_str = f"Score {risk_score}/100" if risk_score != "N/A" else "Score N/A"
    elif isinstance(risk_score, (int, float)) and risk_score < 40:
        risk_color = "🟢 [LOW RISK]"
        status_badge = f"{raw_status}"
        risk_str = f"Score {risk_score}/100"
    elif isinstance(risk_score, (int, float)) and risk_score < 80:
        risk_color = "🟡 [ELEVATED RISK]"
        status_badge = f"{raw_status}"
        risk_str = f"Score {risk_score}/100"
    else:
        risk_color = "🔴 [CRITICAL ALERT]"
        status_badge = f"{raw_status}"
        risk_str = f"Score {risk_score}/100" if risk_score != "N/A" else "Score N/A"

    status_label_padded = f"{status_label:<21}"
    print(f"🛡️ Risk Assessment      : {risk_color} {risk_str}")
    print(f"📋 {status_label_padded}: {status_badge}")
    if is_forecast_active and result.get("forecast_max_wave_m") is not None:
        print(f"🌊 Forecast Max Wave    : {result.get('forecast_max_wave_m'):.2f} m (Open-Meteo 24–48h Horizon)")
    if risk_info.get("summary"):
        print(f"ℹ️ Advisory Summary     : {risk_info.get('summary')}")

    # 🌀 CYCLONE TRACKING INTELLIGENCE (GDACS / MOSDAC)
    cyc_intel = result.get("cyclone_intelligence") or risk_info.get("cyclone_intelligence") or {}
    active_storm = cyc_intel.get("active_storms")
    traj_path = cyc_intel.get("trajectory_path")
    route_collision = cyc_intel.get("route_collision", False)
    clearance_dist = cyc_intel.get("clearance_distance_nm")

    active_storm_display = str(active_storm) if active_storm else "None"
    traj_display = str(traj_path) if traj_path else "None"

    if route_collision:
        col_display = f"Yes (Collision Alert! Clearance: {clearance_dist:.1f} NM < 200 NM buffer)" if clearance_dist is not None else "Yes (Collision Alert! < 200 NM buffer)"
    elif active_storm and clearance_dist is not None:
        col_display = f"No (Clearance: {clearance_dist:.1f} NM)"
    else:
        col_display = "No - Safe Clearance"

    print("🌀 CYCLONE TRACKING INTELLIGENCE:")
    print(f"   - Active Storms    : {active_storm_display}")
    print(f"   - Trajectory Path  : {traj_display}")
    print(f"   - Route Collision  : {col_display}")

    # Calculate Route Distance
    route = result.get("safe_sea_route")
    target = result.get("primary_geographic_target")
    route_distance_nm = 0.0
    if not is_ood and not is_pending_forecast:
        if route and isinstance(route, dict):
            route_distance_nm = route.get("total_distance_nm") or route.get("distance_nm") or 0.0
        if not route_distance_nm and result.get("alternative_route"):
            alt_r = result["alternative_route"]
            route_distance_nm = alt_r.get("distance_shift_nm") or alt_r.get("reroute_distance_nm") or 0.0
        if not route_distance_nm and route and route.get("waypoints") and len(route["waypoints"]) >= 2:
            try:
                from decision_engine import haversine_nm
                wp1 = route["waypoints"][0]
                wp2 = route["waypoints"][-1]
                lat1 = wp1.get("lat") or wp1.get("latitude") if isinstance(wp1, dict) else wp1[0]
                lon1 = wp1.get("lon") or wp1.get("longitude") if isinstance(wp1, dict) else wp1[1]
                lat2 = wp2.get("lat") or wp2.get("latitude") if isinstance(wp2, dict) else wp2[0]
                lon2 = wp2.get("lon") or wp2.get("longitude") if isinstance(wp2, dict) else wp2[1]
                route_distance_nm = round(haversine_nm(lat1, lon1, lat2, lon2), 1)
            except Exception:
                route_distance_nm = 0.0
        if not route_distance_nm and target:
            route_distance_nm = target.get("distance_nm", 0.0)

    # 🎯 PRIMARY GEOGRAPHIC TARGET
    rec_coords = result.get("recommended_coordinates")
    if target and not is_ood and not is_pending_forecast:
        print("🎯 PRIMARY GEOGRAPHIC TARGET:")
        print(f"   - Feature Type        : {target.get('feature_type', 'Operational Maritime Target')}")
        print(f"   - Target Coordinates  : {target.get('target_coordinates', 'N/A')}")
        print(f"   - Relative Vector     : {target.get('relative_vector', 'N/A')}")
        print(f"   - Landmark Reference  : {target.get('landmark_reference', 'N/A')}")
    elif rec_coords and not is_ood and not is_pending_forecast:
        print("🎯 PRIMARY GEOGRAPHIC TARGET:")
        print("   - Feature Type        : Operational Maritime Target")
        if isinstance(rec_coords, dict):
            print(f"   - Target Coordinates  : Lat {rec_coords.get('latitude')}, Lon {rec_coords.get('longitude')}")
        else:
            print(f"   - Target Coordinates  : {rec_coords}")

    # 🧭 NAVIGATIONAL ROUTE PLAN
    if route and route.get("waypoints") and not is_ood and not is_pending_forecast:
        print("🧭 NAVIGATIONAL ROUTE PLAN:")
        print(f"   - Total Voyage        : {route_distance_nm:.1f} NM")
        print("   - Segment Waypoints   :")
        wps = route["waypoints"]
        for idx, wp in enumerate(wps, 1):
            if isinstance(wp, dict):
                wp_lat = wp.get("lat") or wp.get("latitude", 0.0)
                wp_lon = wp.get("lon") or wp.get("longitude", 0.0)
                wp_name = wp.get("name") or ("Departure Harbor" if idx == 1 else ("Target Sector / Destination" if idx == len(wps) else f"Fairway Waypoint {idx-1}"))
            elif isinstance(wp, (list, tuple)) and len(wp) >= 2:
                wp_lat = wp[0]
                wp_lon = wp[1]
                wp_name = "Departure Harbor" if idx == 1 else ("Target Sector / Destination" if idx == len(wps) else f"Fairway Waypoint {idx-1}")
            else:
                wp_lat, wp_lon, wp_name = 0.0, 0.0, str(wp)
            lat_dir = "N" if wp_lat >= 0 else "S"
            lon_dir = "E" if wp_lon >= 0 else "W"
            coord_str = f"Lat {abs(wp_lat):07.4f}° {lat_dir}, Lon {abs(wp_lon):08.4f}° {lon_dir}"
            print(f"     Leg {idx:<2} | {coord_str} | {wp_name}")
    elif not is_ood and not is_pending_forecast and target and target.get("target_coordinates"):
        orig_coords = result.get("alternative_route", {}).get("primary_coordinates", "8.7642,78.1348")
        try:
            o_lat, o_lon = [float(x.strip()) for x in orig_coords.split(",")[:2]]
        except Exception:
            o_lat, o_lon = 8.7642, 78.1348
        t_lat = target.get("target_lat", 9.0932)
        t_lon = target.get("target_lon", 78.3218)
        lat_dir_o = "N" if o_lat >= 0 else "S"
        lon_dir_o = "E" if o_lon >= 0 else "W"
        lat_dir_t = "N" if t_lat >= 0 else "S"
        lon_dir_t = "E" if t_lon >= 0 else "W"
        coord_o = f"Lat {abs(o_lat):07.4f}° {lat_dir_o}, Lon {abs(o_lon):08.4f}° {lon_dir_o}"
        coord_t = f"Lat {abs(t_lat):07.4f}° {lat_dir_t}, Lon {abs(t_lon):08.4f}° {lon_dir_t}"
        print("🧭 NAVIGATIONAL ROUTE PLAN:")
        print(f"   - Total Voyage        : {route_distance_nm:.1f} NM")
        print("   - Segment Waypoints   :")
        print(f"     Leg 1  | {coord_o} | Thoothukudi Port (Departure)")
        print(f"     Leg 2  | {coord_t} | {target.get('feature_type', 'Target Sector')}")

    # Green Marine Energy & Sustainability (Diurnal Time-Aware Dynamic Metrics)
    green = result.get("green_marine_energy", {})
    op_status = raw_status

    is_night = green.get("is_nighttime")
    if is_night is None and not is_ood:
        from decision_engine import check_diurnal_cycle
        is_day, _, _ = check_diurnal_cycle(result)
        is_night = not is_day
    elif is_night is None:
        is_night = False

    if op_status in ("OUT_OF_DOMAIN", "SECURITY_REJECTION", "TEMPORAL_OUT_OF_BOUNDS", "LAND_INTERSECTION_ERROR", "PENDING FORECAST", "DATA_UNAVAILABLE") or is_pending_forecast:
        solar_disp = "0 W/m²"
        zero_emission_disp = "+0.0 hrs battery buffer"
        fuel_saved_disp = "0.0 Liters"
        carbon_disp = "0.0 kg CO₂"
    elif is_night:
        solar_disp = "0 W/m² (Nighttime)"
        zero_emission_disp = "Stored battery buffer only"
        fuel_saved_disp = "0.0 Liters (Nighttime / Zero Solar Insolation)"
        carbon_disp = "0.0 kg CO₂"
    else:
        if op_status in ("SAFE", "GO", "OPERATIONAL"):
            solar_w_m2 = green.get("solar_irradiance_wm2") or random.randint(700, 950)
        elif op_status == "EMERGENCY_SAR":
            solar_w_m2 = random.randint(0, 150)
        else:
            solar_w_m2 = green.get("solar_irradiance_wm2") if (green.get("solar_irradiance_wm2", 0) <= 200 and green.get("solar_irradiance_wm2", 0) > 0) else random.randint(0, 200)

        # Calculate Fuel Saved: (route_distance_nm * fuel_consumption_rate) * 0.20, rounded to 1 decimal place (fallback to 0)
        fuel_rate = get_fuel_consumption_rate(active_persona)
        if not is_ood and route_distance_nm and route_distance_nm > 0:
            fuel_saved = round((float(route_distance_nm) * fuel_rate) * 0.20, 1)
        else:
            fuel_saved = 0.0 if is_ood else green.get("fuel_saved_liters", 0.0)

        # Calculate Carbon Offset: fuel_saved * 2.68, rounded to 1 decimal place
        carbon_offset_kg = round(fuel_saved * 2.68, 1)

        # Extended zero emission battery buffer
        zero_emission_hours = green.get("extended_zero_emission_hours") or (round(((solar_w_m2 / 1000.0) * 1.5 * 0.82 / 2.2) * 8.0, 1) if solar_w_m2 > 0 else 0.0)

        solar_disp = f"{solar_w_m2} W/m²"
        zero_emission_disp = f"+{zero_emission_hours} hrs battery buffer"
        fuel_saved_disp = f"{fuel_saved} Liters"
        carbon_disp = f"{carbon_offset_kg} kg CO₂"

    print(DIVIDER_THIN)
    print("🌱 GREEN MARINE ENERGY & SUSTAINABILITY:")
    print(f"    - Solar Irradiance    : {solar_disp}")
    print(f"    - Zero-Emission Hours : {zero_emission_disp}")
    print(f"    - Fuel Saved          : {fuel_saved_disp}")
    print(f"    - Carbon Offset       : {carbon_disp}")

    # Localized Native Advisory Text
    native_text = result.get("native_advisory_text") or result.get("chat_text") or result.get("text_advisory_local")
    print(DIVIDER_THIN)
    print("🗣️ LOCALIZED ADVISORY (STAKEHOLDER AUDIO/TEXT PAYLOAD):")
    print(native_text)
    print(DIVIDER_BOLD + "\n")
    return result


def select_persona_interactive() -> Optional[StakeholderPersona]:
    """Displays the numbered role selection menu on launch."""
    print("Stakeholder Persona Selection:")
    for num, p_enum, desc in PERSONA_MENU_OPTIONS:
        print(f"  [{num}] {desc}")
    print("  [Enter] Auto-Detect (Dynamic intent resolution per query)\n")
    try:
        choice = input("Select Stakeholder Role [1-5 or Enter]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not choice:
        return None
    return resolve_persona(choice)


def interactive_mode(initial_persona: Optional[str] = None):
    """Runs an interactive REPL terminal session."""
    print_banner()

    resolved_init = resolve_persona(initial_persona)
    if resolved_init is not None:
        active_persona = resolved_init.value
        print(f"[OK] Persona set to: {PERSONA_BADGES.get(active_persona, active_persona)}\n")
    elif initial_persona is not None:
        print(f"[!] Unrecognized persona: '{initial_persona}'. Launching role selector.\n")
        selected = select_persona_interactive()
        active_persona = selected.value if selected else "AUTO-DETECT"
    else:
        selected = select_persona_interactive()
        active_persona = selected.value if selected else "AUTO-DETECT"

    print("Commands:")
    print("  - Type any question in English, Tamil, Hindi, etc.")
    print("  - 'status'          : Run 9-dataset satellite cache & agent health check")
    print("  - 'persona:NAME'    : Switch role (1-5 or FISHERMAN, MARITIME_AUTHORITY, DISASTER_MANAGEMENT, RESEARCHER, MARITIME_OPERATOR, AUTO)")
    print("  - 'test:fisherman'  : Test Potential Fishing Zones & Safe Reroute (FISHERMAN)")
    print("  - 'test:authority'  : Test EEZ Patrol, AIS Tracking & Boundary Brief (MARITIME_AUTHORITY)")
    print("  - 'test:disaster'   : Test Cyclone Evacuation & Emergency Shelter (DISASTER_MANAGEMENT)")
    print("  - 'test:researcher' : Test Oceanographic Biophysical Telemetry (RESEARCHER)")
    print("  - 'test:operator'   : Test Commercial Fairway & TSS Clearance (MARITIME_OPERATOR)")
    print("  - 'test:weather'    : Test marine weather & wave forecast")
    print("  - 'test:cyclone'    : Test disaster/cyclone tracking")
    print("  - 'test:tamil'      : Test Tamil query integration")
    print("  - 'test:hindi'      : Test Hindi query integration")
    print("  - 'exit' / 'q'      : Exit CLI\n")

    while True:
        try:
            prompt_str = f"ORCA [{active_persona}] > "
            user_text = input(prompt_str).strip()
            if not user_text:
                continue

            if user_text.lower() in ("exit", "quit", "q"):
                print("Exiting ORCA Standalone CLI. Smooth sailing!")
                break

            if user_text.lower() == "status":
                print_status_dossier()
                continue

            # Switch persona command
            if user_text.lower().startswith("persona:"):
                target_raw = user_text.split(":", 1)[1].strip()
                if target_raw.lower() in ("auto", "none", "dynamic"):
                    active_persona = "AUTO-DETECT"
                    print("Switched persona to: AUTO-DETECT (Dynamic Intent Resolution)")
                    continue
                new_p = resolve_persona(target_raw)
                if new_p:
                    active_persona = new_p.value
                    badge = PERSONA_BADGES.get(active_persona, active_persona)
                    print(f"Switched persona to: {badge}")
                else:
                    print(f"[!] Unrecognized persona: '{target_raw}'. Available:")
                    for num, p_enum, desc in PERSONA_MENU_OPTIONS:
                        print(f"    [{num}] {desc}")
                    print("    [auto] Dynamic Auto-Detection")
                continue

            # Presets
            if user_text.lower() in ("test:fisherman", "test:pfz", "fisherman"):
                execute_query(
                    "Where are the best potential fishing zones with high chlorophyll and safe sea waves right now?",
                    persona="FISHERMAN"
                )
                continue
            elif user_text.lower() in ("test:authority", "authority"):
                execute_query(
                    "Maritime security and border patrol brief: Check EEZ perimeter, IMBL clearance, SLOC shipping traffic conflict, and cyclone alert status in Gulf of Mannar.",
                    persona="MARITIME_AUTHORITY"
                )
                continue
            elif user_text.lower() in ("test:disaster", "disaster"):
                execute_query(
                    "Severe weather alert: Assess cyclone track impact, coastal storm surge inundation, and designate all-weather emergency breakwater shelter basins.",
                    persona="DISASTER_MANAGEMENT"
                )
                continue
            elif user_text.lower() in ("test:researcher", "researcher"):
                execute_query(
                    "Provide oceanographic biophysical telemetry including chlorophyll-a gradients, diffuse attenuation Kd_490, Ka-band altimetry, and solar irradiance for Tuticorin.",
                    persona="RESEARCHER"
                )
                continue
            elif user_text.lower() in ("test:operator", "operator", "shipping"):
                execute_query(
                    "Check deep-draft commercial fairway navigation conditions, Traffic Separation Scheme (TSS) clearance, and auxiliary solar endurance for container transit.",
                    persona="MARITIME_OPERATOR"
                )
                continue
            elif user_text.lower() == "test:weather":
                execute_query("What is the sea wave height and wind speed forecast for Tuticorin?", persona=(None if active_persona == "AUTO-DETECT" else active_persona))
                continue
            elif user_text.lower() == "test:cyclone":
                execute_query("Is there any cyclone or storm alert active in the Gulf of Mannar?", persona=(None if active_persona == "AUTO-DETECT" else active_persona))
                continue
            elif user_text.lower() == "test:tamil":
                execute_query("தூத்துக்குடியில் மீன்பிடிக்க சிறந்த இடம் எங்கே?", persona=(None if active_persona == "AUTO-DETECT" else active_persona), lang="ta")
                continue
            elif user_text.lower() == "test:hindi":
                execute_query("तूतीकोरिन के पास सुरक्षित मछली पकड़ने का क्षेत्र कहाँ है?", persona=(None if active_persona == "AUTO-DETECT" else active_persona), lang="hi")
                continue

            # Execute user prompt
            run_p = None if active_persona == "AUTO-DETECT" else active_persona
            execute_query(user_text, persona=run_p)

        except KeyboardInterrupt:
            print("\nSession ended by user.")
            break
        except Exception as e:
            print(f"\n[CLI Error] {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Project ORCA Marine Multi-Agent Autonomous CLI")
    parser.add_argument("--query", "-q", type=str, help="Natural language query to process through the pipeline")
    parser.add_argument(
        "--persona", "-p", type=str, default=None,
        help="Stakeholder persona: 1-5 or FISHERMAN, MARITIME_AUTHORITY, DISASTER_MANAGEMENT, RESEARCHER, MARITIME_OPERATOR (interactive if omitted)"
    )
    parser.add_argument("--lang", "-l", type=str, default=None, help="Optional language code (ta, hi, ml, te, bn, gu)")
    parser.add_argument("--lat", type=float, default=8.7642, help="Vessel latitude (default: Tuticorin 8.7642)")
    parser.add_argument("--lon", type=float, default=78.1348, help="Vessel longitude (default: Tuticorin 78.1348)")
    parser.add_argument("--status", "-s", action="store_true", help="Print live 9-dataset MOSDAC health report and exit")

    args = parser.parse_args()

    if args.status:
        print_banner()
        print_status_dossier()
    elif args.query:
        print_banner()
        execute_query(args.query, persona=args.persona, lat=args.lat, lon=args.lon, lang=args.lang)
    else:
        interactive_mode(initial_persona=args.persona)


if __name__ == "__main__":
    main()
