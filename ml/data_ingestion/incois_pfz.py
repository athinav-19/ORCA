"""
incois_pfz.py - INCOIS PFZ Advisory Ingestion & Ground Truth Data Adapter
SIH 2026 Problem Statement SIH26176

Handles ingestion, caching, and spatial queries of official Potential Fishing Zone (PFZ)
advisories issued by the Indian National Centre for Ocean Information Services (INCOIS).

Features:
1. Live INCOIS GeoServer WFS connector (PFZ_Automation:pfzlines) fetching official daily lines.
2. Local persistent caching in data/incois_pfz/.
3. Spatial nearest-advisory search with distance and bearing from coastal landing centers.
4. Clean schema and validation adapter for multi-year historical institutional shapefiles/CSVs.
5. Strict provenance tracking: Never fabricates synthetic advisory records.
"""

import os
import ssl
import json
import math
import urllib.request
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

INCOIS_DATA_DIR = os.path.join("data", "incois_pfz")
LOCAL_ADVISORY_CATALOG = os.path.join(INCOIS_DATA_DIR, "advisories_catalog.csv")
INCOIS_WFS_URL = (
    "https://incois.gov.in/geoserver/PFZ_Automation/ows?"
    "service=WFS&version=1.1.0&request=GetFeature&"
    "typeName=PFZ_Automation:pfzlines&outputFormat=application/json"
)


def fetch_live_incois_wfs_pfzlines(save_to_cache: bool = True, timeout_sec: int = 15) -> Optional[Dict[str, Any]]:
    """
    Fetches the official live daily PFZ lines from the INCOIS GeoServer WFS endpoint.
    Caches the GeoJSON locally if successful. Returns None if network or server is unreachable.
    NEVER generates fabricated fallback data.
    """
    os.makedirs(INCOIS_DATA_DIR, exist_ok=True)
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    cache_path = os.path.join(INCOIS_DATA_DIR, f"incois_live_pfzlines_{today_str}.json")

    # Check local cache first for today's file
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                return cached_data
        except Exception:
            pass

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        req = urllib.request.Request(
            INCOIS_WFS_URL,
            headers={"User-Agent": "ORCA-Marine-Platform/1.0 (INCOIS-PFZ-Client)"},
        )
        with urllib.request.urlopen(req, timeout=timeout_sec, context=ctx) as resp:
            if resp.status == 200:
                raw_bytes = resp.read()
                data = json.loads(raw_bytes.decode("utf-8"))
                features = data.get("features", [])
                if features:
                    payload = {
                        "fetched_at": datetime.utcnow().isoformat() + "Z",
                        "source": "INCOIS GeoServer WFS (PFZ_Automation:pfzlines)",
                        "source_url": INCOIS_WFS_URL,
                        "advisory_date": today_str,
                        "feature_count": len(features),
                        "features": features,
                    }
                    if save_to_cache:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(payload, f, indent=2)
                    return payload
    except Exception as e:
        print(f"[INCOIS WFS Warning] Could not fetch live PFZ lines from INCOIS: {e}")
        return None

    return None


def get_incois_advisories_for_location(
    target_lat: float,
    target_lon: float,
    max_dist_km: float = 120.0,
) -> List[Dict[str, Any]]:
    """
    Searches live or cached official INCOIS PFZ line features within max_dist_km of a coastal coordinate.
    Returns matched advisory segments with distance, bearing, and INCOIS metadata.
    """
    wfs_data = fetch_live_incois_wfs_pfzlines(save_to_cache=True)
    if not wfs_data or not wfs_data.get("features"):
        return []

    matched = []
    features = wfs_data["features"]

    for feat in features:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])

        # Geometry can be LineString or MultiLineString
        all_pts = []
        if geom.get("type") == "LineString":
            all_pts = coords
        elif geom.get("type") == "MultiLineString":
            for line in coords:
                all_pts.extend(line)

        if not all_pts:
            continue

        # Find closest point along line segment to target
        min_d = float("inf")
        closest_pt = None
        for pt in all_pts:
            p_lon, p_lat = pt[0], pt[1]
            d_lat = (p_lat - target_lat) * 111.0
            d_lon = (p_lon - target_lon) * 111.0 * math.cos(math.radians(target_lat))
            d = math.sqrt(d_lat**2 + d_lon**2)
            if d < min_d:
                min_d = d
                closest_pt = (p_lat, p_lon)

        if min_d <= max_dist_km and closest_pt is not None:
            c_lat, c_lon = closest_pt
            d_nm = round(min_d / 1.852, 1)

            # Calculate compass bearing from target to PFZ point
            d_lon_rad = math.radians(c_lon - target_lon)
            lat1_rad = math.radians(target_lat)
            lat2_rad = math.radians(c_lat)
            y = math.sin(d_lon_rad) * math.cos(lat2_rad)
            x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(d_lon_rad)
            initial_bearing = math.degrees(math.atan2(y, x))
            compass_bearing = (initial_bearing + 360) % 360

            directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
            bearing_str = directions[int((compass_bearing + 11.25) / 22.5) % 16]

            matched.append({
                "uid": props.get("UID"),
                "state_name": props.get("State_Name"),
                "sector": props.get("SECTORBOUN"),
                "julian_day": props.get("Julian_day"),
                "year": props.get("Year"),
                "distance_km": round(min_d, 2),
                "distance_nm": d_nm,
                "bearing": f"{round(compass_bearing)}° ({bearing_str})",
                "nearest_latitude": round(c_lat, 4),
                "nearest_longitude": round(c_lon, 4),
                "line_length_km": props.get("Length"),
                "source": "INCOIS GeoServer WFS (PFZ_Automation:pfzlines)",
                "advisory_date": wfs_data.get("advisory_date"),
            })

    # Sort by closest distance
    matched.sort(key=lambda x: x["distance_km"])
    return matched


def inspect_incois_data_status() -> Dict[str, Any]:
    """
    Audits the availability of genuine INCOIS historical and live PFZ advisories.
    Returns status and schema readiness without fabricating any non-existent records.
    """
    os.makedirs(INCOIS_DATA_DIR, exist_ok=True)
    has_catalog = os.path.exists(LOCAL_ADVISORY_CATALOG)
    file_count = len(os.listdir(INCOIS_DATA_DIR)) if os.path.exists(INCOIS_DATA_DIR) else 0

    # Test live WFS connectivity
    live_wfs_available = False
    try:
        live = fetch_live_incois_wfs_pfzlines(save_to_cache=True, timeout_sec=5)
        if live and live.get("feature_count", 0) > 0:
            live_wfs_available = True
    except Exception:
        live_wfs_available = False

    return {
        "source": "INCOIS (Indian National Centre for Ocean Information Services)",
        "advisory_dir": INCOIS_DATA_DIR,
        "files_found": file_count,
        "catalog_exists": has_catalog,
        "live_wfs_endpoint": INCOIS_WFS_URL,
        "live_wfs_available": live_wfs_available,
        "data_classification": "RULE_EMULATION",
        "status": "PENDING_REAL_GROUND_TRUTH",
        "limitation_note": (
            "INCOIS GeoServer WFS serves the operational daily advisory (PFZ_Automation:pfzlines). "
            "However, multi-decade digital archives of INCOIS PFZ polygons (2005-2025) are not "
            "publicly exposed via open REST/WFS endpoints without authenticated institutional credentials. "
            "Therefore, multi-year model training maintains status PENDING_REAL_GROUND_TRUTH to prevent "
            "unscientific data fabrication."
        ),
        "schema_columns": [
            "advisory_date",
            "sector_name",
            "latitude",
            "longitude",
            "bearing_deg",
            "distance_km",
            "depth_m",
            "valid_until",
            "incois_confidence",
        ]
    }


def load_incois_ground_truth() -> Optional[pd.DataFrame]:
    """
    Loads verified INCOIS PFZ records if present on disk.
    Returns None if genuine records are not yet placed in data/incois_pfz/.
    NEVER generates fabricated records.
    """
    if os.path.exists(LOCAL_ADVISORY_CATALOG):
        try:
            df = pd.read_csv(LOCAL_ADVISORY_CATALOG)
            return df
        except Exception as e:
            print(f"[INCOIS Adapter Error] Could not parse catalog: {e}")
            return None
    return None
