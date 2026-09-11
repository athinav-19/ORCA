"""
cyclone_track_agent.py - Real-Time Cyclone Tracking Agent using GDACS Public API
ISRO SIH Problem Statement 176: Marine Multi-Agent System

Ingests real-time Tropical Cyclone (TC) advisories and forecast tracks from GDACS:
- Primary Endpoint: https://www.gdacs.org/gdacsapi/api/events/geteventdata?eventtype=TC
- Fallback Endpoint: https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH?eventlist=TC
- Strict 5.0-second network timeout with graceful None fallback
- Filters active storms for the Indian Ocean & Bay of Bengal region
- Extracts cyclone name, current coordinates, and future forecasted trajectory
- Evaluates 200 NM route collision threshold for operational safety overrides
"""

import math
import requests
from typing import Dict, Any, List, Optional, Tuple


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two coordinates in Nautical Miles."""
    r_nm = 3440.065
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    )
    return 2.0 * r_nm * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


class CycloneTrackAgent:
    """
    Agent monitoring active Tropical Cyclones (TC) in the Indian Ocean & Bay of Bengal
    via the Global Disaster Alert and Coordination System (GDACS) API.
    """

    PRIMARY_URL = "https://www.gdacs.org/gdacsapi/api/events/geteventdata?eventtype=TC"
    SEARCH_URL = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH?eventlist=TC"

    # Indian Ocean & Bay of Bengal bounding box
    LAT_MIN = -10.0
    LAT_MAX = 30.0
    LON_MIN = 45.0
    LON_MAX = 105.0

    INDIAN_OCEAN_REGIONAL_KEYWORDS = {
        "india", "sri lanka", "bangladesh", "myanmar", "maldives",
        "oman", "yemen", "somalia", "pakistan", "arabian sea",
        "bay of bengal", "indian ocean", "andaman", "nicobar"
    }

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    def _is_in_indian_ocean(self, lat: float, lon: float, country_str: str = "") -> bool:
        """Determines if a storm coordinate or country falls within the Indian Ocean / Bay of Bengal basin."""
        if self.LAT_MIN <= lat <= self.LAT_MAX and self.LON_MIN <= lon <= self.LON_MAX:
            return True
        c_low = country_str.lower()
        return any(kw in c_low for kw in self.INDIAN_OCEAN_REGIONAL_KEYWORDS)

    def _fetch_geometry_tracks(self, geom_url: str) -> List[Tuple[float, float]]:
        """Extracts forecasted LineString track points from the GDACS geometry endpoint."""
        track_points: List[Tuple[float, float]] = []
        try:
            resp = requests.get(geom_url, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                for feat in data.get("features", []):
                    geom = feat.get("geometry", {})
                    g_type = geom.get("type")
                    if g_type == "LineString":
                        coords = geom.get("coordinates", [])
                        for pt in coords:
                            # GDACS GeoJSON is [lon, lat] -> convert to (lat, lon)
                            if len(pt) >= 2:
                                p_lat, p_lon = round(float(pt[1]), 4), round(float(pt[0]), 4)
                                if (p_lat, p_lon) not in track_points:
                                    track_points.append((p_lat, p_lon))
                    elif g_type == "Point":
                        pt = geom.get("coordinates", [])
                        if len(pt) >= 2:
                            p_lat, p_lon = round(float(pt[1]), 4), round(float(pt[0]), 4)
                            if (p_lat, p_lon) not in track_points:
                                track_points.append((p_lat, p_lon))
        except Exception:
            pass
        return track_points

    def parse_cyclone_features(self, features: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Parses GeoJSON features from GDACS to identify an active Tropical Cyclone
        in the Indian Ocean / Bay of Bengal region.
        """
        for feat in features:
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})
            event_type = props.get("eventtype") or feat.get("eventtype")
            if event_type and str(event_type).upper() != "TC":
                continue

            # Check coordinates (GeoJSON [lon, lat])
            coords = geom.get("coordinates") or props.get("coordinates")
            if not coords or len(coords) < 2:
                continue

            lon = float(coords[0])
            lat = float(coords[1])
            country = props.get("country", "")

            # Check geographic relevance
            if not self._is_in_indian_ocean(lat, lon, country):
                continue

            # Check if active / current
            is_current = str(props.get("iscurrent", "")).strip().lower() == "true"
            if not is_current:
                continue

            storm_name = (
                props.get("eventname")
                or props.get("name")
                or f"TC-{props.get('eventid', 'Active')}"
            )

            # Retrieve track coordinates if available via url.geometry
            track_points: List[Tuple[float, float]] = []
            geom_url = props.get("url", {}).get("geometry")
            if geom_url:
                track_points = self._fetch_geometry_tracks(geom_url)

            # If track points are empty, synthesize forward projection from position
            current_coords = (round(lat, 4), round(lon, 4))
            if current_coords not in track_points:
                track_points.insert(0, current_coords)

            # Determine next 24-hour projection coordinate
            if len(track_points) > 1:
                proj_pt = track_points[min(3, len(track_points) - 1)]
            else:
                # Approximate 24h northwestward displacement (~18 km/h -> ~230 NM)
                proj_pt = (round(lat + 2.0, 4), round(lon - 1.5, 4))
                track_points.append(proj_pt)

            p_lat, p_lon = proj_pt
            lat_dir = "N" if p_lat >= 0 else "S"
            lon_dir = "E" if p_lon >= 0 else "W"
            proj_str = f"Lat {abs(p_lat):07.4f}° {lat_dir}, Lon {abs(p_lon):08.4f}° {lon_dir}"

            severity_data = props.get("severitydata", {})
            max_wind = severity_data.get("severity") or 65.0
            alert_level = props.get("alertlevel") or "Orange"

            return {
                "active": True,
                "name": storm_name,
                "current_coords": current_coords,
                "track_path": track_points,
                "next_24h_projection": proj_str,
                "next_24h_coords": proj_pt,
                "alert_level": alert_level,
                "max_wind_kmh": float(max_wind),
                "source": "GDACS Real-Time Tropical Cyclone Advisory",
            }

        return None

    def fetch_active_cyclone_tracks(self) -> Optional[Dict[str, Any]]:
        """
        Queries GDACS public API for currently active Tropical Cyclones in the Indian Ocean.
        Strict 5-second timeout. Returns None if API fails, times out, or no storms are active.
        """
        try:
            # 1. Attempt primary endpoint
            resp = requests.get(self.PRIMARY_URL, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                features = data.get("features", [])
                if isinstance(features, list) and features:
                    parsed = self.parse_cyclone_features(features)
                    if parsed:
                        return parsed

            # 2. Seamless fallback to search list endpoint
            resp_search = requests.get(self.SEARCH_URL, timeout=self.timeout)
            if resp_search.status_code == 200:
                data = resp_search.json()
                features = data.get("features", [])
                if isinstance(features, list) and features:
                    return self.parse_cyclone_features(features)

        except Exception:
            # Silently fail and return None on any network timeout or JSON error
            return None

        return None

    def check_cyclone_collision(
        self,
        route_points: List[Tuple[float, float]],
        cyclone_data: Optional[Dict[str, Any]],
        threshold_nm: float = 200.0,
    ) -> Tuple[bool, float, Optional[Tuple[float, float]]]:
        """
        Evaluates spatial cross-reference between route points (departure, waypoints, destination)
        and the cyclone's current coordinate + projected track path.
        Returns (collision_detected, min_distance_nm, closest_track_point).
        """
        if not cyclone_data or not cyclone_data.get("active"):
            return False, float("inf"), None

        if not route_points:
            return False, float("inf"), None

        all_cyclone_points = [cyclone_data["current_coords"]]
        for pt in cyclone_data.get("track_path", []):
            if pt not in all_cyclone_points:
                all_cyclone_points.append(pt)

        min_dist = float("inf")
        closest_point = None

        for r_lat, r_lon in route_points:
            for c_lat, c_lon in all_cyclone_points:
                dist = haversine_nm(r_lat, r_lon, c_lat, c_lon)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = (c_lat, c_lon)

        min_dist_rounded = round(min_dist, 1) if min_dist != float("inf") else float("inf")
        collision_detected = min_dist <= threshold_nm
        return collision_detected, min_dist_rounded, closest_point


def fetch_active_cyclone_tracks(timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    """Convenience module function for fetching active cyclone tracks."""
    agent = CycloneTrackAgent(timeout=timeout)
    return agent.fetch_active_cyclone_tracks()

