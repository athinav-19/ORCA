"""
bathymetry_geography.py - Indian EEZ Bathymetry & Geographic Feature Calculations
SIH 2026 Problem Statement SIH26176

Provides distance-to-coast calculation, continental shelf bathymetry estimation,
and coastal proximity features across the Indian maritime zone.
"""

import math
import numpy as np
from typing import Dict, Any, Tuple

# Key Indian coastal landmarks along East and West coasts for distance calculation
INDIAN_COASTLINE_COORDINATES = [
    # Gujarat / West Coast North
    (23.25, 68.50), (22.25, 68.90), (21.60, 69.60), (20.90, 70.35),
    (20.70, 70.90), (21.10, 72.80), (20.00, 72.70), (18.95, 72.80),
    # Maharashtra / Goa
    (18.00, 73.00), (16.90, 73.30), (15.50, 73.75), (14.80, 74.10),
    # Karnataka / Kerala / West Coast South
    (13.35, 74.70), (12.90, 74.80), (11.85, 75.35), (10.80, 75.85),
    (9.95, 76.25), (9.00, 76.50), (8.50, 76.90), (8.08, 77.55), # Kanyakumari
    # Tamil Nadu / East Coast South
    (8.75, 78.15), (9.28, 79.15), (10.30, 79.85), (11.75, 79.75),
    (13.08, 80.28), (14.00, 80.15), # Chennai / Andhra
    # Andhra Pradesh / Odisha / Bengal / East Coast North
    (15.50, 80.25), (16.90, 82.25), (17.70, 83.30), (18.30, 84.00),
    (19.80, 85.85), (20.30, 86.65), (21.50, 87.10), (21.65, 87.50),
    (21.75, 88.00), (21.80, 88.80)
]


def calculate_distance_to_coast_nm(lat: float, lon: float) -> float:
    """
    Computes minimum geodesic distance in Nautical Miles to the Indian coastline.
    """
    min_dist_km = float("inf")
    lat_r = math.radians(lat)

    for c_lat, c_lon in INDIAN_COASTLINE_COORDINATES:
        d_lat = (c_lat - lat) * 111.0
        d_lon = (c_lon - lon) * 111.0 * math.cos((lat_r + math.radians(c_lat)) / 2.0)
        dist_km = math.hypot(d_lat, d_lon)
        if dist_km < min_dist_km:
            min_dist_km = dist_km

    return round(min_dist_km / 1.852, 2)


def estimate_bathymetric_depth_m(lat: float, lon: float) -> float:
    """
    Approximates ocean depth (m) using continental shelf distance curves:
    - Inner shelf (<10 NM): 10 - 50m
    - Mid shelf (10-30 NM): 50 - 100m
    - Shelf break (30-60 NM): 100 - 200m
    - Continental slope (60-100 NM): 200 - 1500m
    - Abyssal plain (>100 NM): 1500 - 3500m
    """
    dist_nm = calculate_distance_to_coast_nm(lat, lon)
    if dist_nm <= 5.0:
        return 15.0
    elif dist_nm <= 20.0:
        return 45.0
    elif dist_nm <= 40.0:
        return 90.0
    elif dist_nm <= 60.0:
        return 180.0
    elif dist_nm <= 100.0:
        return 850.0
    else:
        return min(3200.0, 1000.0 + (dist_nm - 100.0) * 15.0)


def get_geographic_features(lat: float, lon: float) -> Dict[str, Any]:
    """Returns spatial and bathymetric descriptors for a geographic location."""
    dist_nm = calculate_distance_to_coast_nm(lat, lon)
    depth_m = estimate_bathymetric_depth_m(lat, lon)
    is_shelf = depth_m <= 200.0
    return {
        "distance_to_coast_nm": dist_nm,
        "distance_to_coast_km": round(dist_nm * 1.852, 2),
        "bathymetry_depth_m": depth_m,
        "is_continental_shelf": is_shelf,
        "is_deep_ocean": depth_m > 1000.0,
    }


if __name__ == "__main__":
    print("Mumbai offshore (18.9, 72.5):", get_geographic_features(18.9, 72.5))
    print("Deep Bay of Bengal (12.0, 88.0):", get_geographic_features(12.0, 88.0))

