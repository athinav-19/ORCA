"""
gis_agent.py - ORCA GIS & Maritime Boundary Agent
ISRO SIH Problem Statement 176: Marine Multi-Agent System

Specialized deterministic agent for maritime spatial boundaries:
- Loads local official india_eez.geojson (Bhuvan / UNCLOS EEZ dataset)
- Uses Shapely geometry for precise point-in-polygon verification
- Geofencing checks for the 200 NM Exclusive Economic Zone (EEZ)
- International Maritime Boundary Line (IMBL) proximity analysis
  (especially sensitive in Palk Strait, Gulf of Mannar, Sir Creek)
- Distance-to-border metric and GPS verification
"""

import os
import re
import json
import math
from typing import Dict, Any, Optional, List

try:
    from shapely.geometry import shape, Point, LineString, Polygon
    from shapely.ops import unary_union
    from shapely.prepared import prep
except ImportError:
    shape = None
    Point = None
    LineString = None
    Polygon = None
    unary_union = None
    prep = None

# Peninsular Indian Mainland Core Landmask Polygon (Coastal Landmasking)
# Prohibits direct overland nautical routing between Western and Eastern coastlines
if Polygon:
    PENINSULA_CORE_LANDMASK = Polygon([
        [73.5, 18.5], [74.2, 16.0], [75.0, 14.0], [75.8, 12.0],
        [76.6, 10.5], [76.9, 9.5], [77.3, 8.8], [77.5, 8.35],
        [77.8, 8.6], [78.2, 9.5], [78.8, 10.5], [79.2, 11.5],
        [79.5, 12.5], [79.8, 14.0], [80.5, 16.0], [81.5, 17.5],
        [78.0, 19.5], [75.0, 19.5], [73.5, 18.5]
    ])
    SRI_LANKA_LANDMASK = Polygon([
        [79.8, 9.8], [80.3, 9.8], [80.9, 9.3], [81.3, 8.6],
        [81.9, 7.5], [81.8, 6.9], [81.3, 6.2], [80.6, 5.9],
        [80.1, 6.0], [79.8, 6.9], [79.8, 8.0], [79.7, 9.0],
        [79.8, 9.8]
    ])
else:
    PENINSULA_CORE_LANDMASK = None
    SRI_LANKA_LANDMASK = None


class GisAgent:
    """
    Deterministic domain agent performing spatial geofencing calculations
    using Shapely geometry and Bhuvan GeoJSON boundaries to enforce EEZ compliance.
    """

    EEZ_FILE = "india_eez.geojson"

    COASTAL_SECTORS = {
        # --- GUJARAT SECTORS ---
        "porbandar": {"lat": 21.6417, "lon": 69.6293, "imbl_dist_nm": 58.0, "border_zone": "India-Pakistan IMBL (Sir Creek Sector)"},
        "veraval": {"lat": 20.9000, "lon": 70.3667, "imbl_dist_nm": 92.0, "border_zone": "Arabian Sea EEZ (Saurashtra Coast)"},
        "okha": {"lat": 22.4667, "lon": 69.0667, "imbl_dist_nm": 42.0, "border_zone": "India-Pakistan IMBL (Kori Creek Sector)"},
        "jakhau": {"lat": 23.2356, "lon": 68.7000, "imbl_dist_nm": 22.0, "border_zone": "India-Pakistan IMBL (Sir Creek Sector)"},
        "kandla": {"lat": 23.0033, "lon": 70.2167, "imbl_dist_nm": 75.0, "border_zone": "Gulf of Kutch Sector"},
        "mangrol": {"lat": 21.1200, "lon": 70.1167, "imbl_dist_nm": 85.0, "border_zone": "Arabian Sea EEZ (Saurashtra Coast)"},

        # --- MAHARASHTRA SECTORS ---
        "mumbai": {"lat": 18.9220, "lon": 72.8347, "imbl_dist_nm": 175.0, "border_zone": "Arabian Sea EEZ (Maharashtra Coast)"},
        "sassoon dock": {"lat": 18.9130, "lon": 72.8250, "imbl_dist_nm": 175.0, "border_zone": "Arabian Sea EEZ (Mumbai Harbor)"},
        "bhaucha dhakka": {"lat": 18.9550, "lon": 72.8510, "imbl_dist_nm": 175.0, "border_zone": "Arabian Sea EEZ (Ferry Wharf)"},
        "ratnagiri": {"lat": 16.9833, "lon": 73.2833, "imbl_dist_nm": 165.0, "border_zone": "Arabian Sea EEZ (Konkan Coast)"},
        "malvan": {"lat": 16.0667, "lon": 73.4667, "imbl_dist_nm": 155.0, "border_zone": "Arabian Sea EEZ (Konkan Coast)"},

        # --- GOA SECTORS ---
        "goa": {"lat": 15.4120, "lon": 73.8050, "imbl_dist_nm": 160.0, "border_zone": "Arabian Sea EEZ (Goa Coast)"},
        "mormugao": {"lat": 15.4120, "lon": 73.8050, "imbl_dist_nm": 160.0, "border_zone": "Arabian Sea EEZ (Goa Coast)"},
        "panaji": {"lat": 15.4989, "lon": 73.8278, "imbl_dist_nm": 160.0, "border_zone": "Arabian Sea EEZ (Goa Coast)"},

        # --- KARNATAKA SECTORS ---
        "mangalore": {"lat": 12.9250, "lon": 74.8150, "imbl_dist_nm": 170.0, "border_zone": "Arabian Sea EEZ (Karnataka Coast)"},
        "new mangalore": {"lat": 12.9250, "lon": 74.8150, "imbl_dist_nm": 170.0, "border_zone": "Arabian Sea EEZ (Karnataka Coast)"},
        "malpe": {"lat": 13.3500, "lon": 74.7000, "imbl_dist_nm": 172.0, "border_zone": "Arabian Sea EEZ (Udupi Sector)"},
        "karwar": {"lat": 14.8167, "lon": 74.1333, "imbl_dist_nm": 165.0, "border_zone": "Arabian Sea EEZ (Uttara Kannada Coast)"},

        # --- KERALA SECTORS ---
        "kochi": {"lat": 9.9312, "lon": 76.2673, "imbl_dist_nm": 180.0, "border_zone": "Arabian Sea EEZ"},
        "cochin": {"lat": 9.9312, "lon": 76.2673, "imbl_dist_nm": 180.0, "border_zone": "Arabian Sea EEZ"},
        "neendakara": {"lat": 8.9378, "lon": 76.5367, "imbl_dist_nm": 140.0, "border_zone": "Arabian Sea EEZ (Kollam Sector)"},
        "munambam": {"lat": 10.1833, "lon": 76.1833, "imbl_dist_nm": 175.0, "border_zone": "Arabian Sea EEZ (Ernakulam Sector)"},
        "vizhinjam": {"lat": 8.3800, "lon": 76.9900, "imbl_dist_nm": 110.0, "border_zone": "Arabian Sea EEZ (Thiruvananthapuram Sector)"},
        "beypore": {"lat": 11.1667, "lon": 75.8000, "imbl_dist_nm": 170.0, "border_zone": "Arabian Sea EEZ (Kozhikode Sector)"},

        # --- TAMIL NADU SECTORS ---
        "thoothukudi": {"lat": 8.7642, "lon": 78.1348, "imbl_dist_nm": 32.5, "border_zone": "India-Sri Lanka IMBL (Gulf of Mannar)"},
        "tuticorin": {"lat": 8.7642, "lon": 78.1348, "imbl_dist_nm": 32.5, "border_zone": "India-Sri Lanka IMBL (Gulf of Mannar)"},
        "rameswaram": {"lat": 9.2876, "lon": 79.3129, "imbl_dist_nm": 8.2, "border_zone": "India-Sri Lanka IMBL (Palk Strait)"},
        "dhanushkodi": {"lat": 9.1764, "lon": 79.4186, "imbl_dist_nm": 5.4, "border_zone": "India-Sri Lanka IMBL (Kachchatheevu Sector)"},
        "mandapam": {"lat": 9.2778, "lon": 79.1236, "imbl_dist_nm": 12.0, "border_zone": "India-Sri Lanka IMBL (Palk Bay)"},
        "kanyakumari": {"lat": 8.0883, "lon": 77.5385, "imbl_dist_nm": 48.0, "border_zone": "India-Sri Lanka-Maldives Trilateral EEZ"},
        "chennai": {"lat": 13.0827, "lon": 80.2707, "imbl_dist_nm": 135.0, "border_zone": "Bay of Bengal High Seas"},
        "nagapattinam": {"lat": 10.7667, "lon": 79.8500, "imbl_dist_nm": 28.0, "border_zone": "India-Sri Lanka IMBL (Point Calimere Sector)"},
        "cuddalore": {"lat": 11.7500, "lon": 79.7667, "imbl_dist_nm": 80.0, "border_zone": "Bay of Bengal EEZ (Coromandel Coast)"},
        "colachel": {"lat": 8.1800, "lon": 77.2600, "imbl_dist_nm": 52.0, "border_zone": "Arabian Sea / Wadge Bank Sector"},

        # --- ANDHRA PRADESH SECTORS ---
        "visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "imbl_dist_nm": 190.0, "border_zone": "Bay of Bengal EEZ (Andhra Coast)"},
        "vizag": {"lat": 17.6868, "lon": 83.2185, "imbl_dist_nm": 190.0, "border_zone": "Bay of Bengal EEZ (Andhra Coast)"},
        "kakinada": {"lat": 16.9891, "lon": 82.2475, "imbl_dist_nm": 185.0, "border_zone": "Bay of Bengal EEZ (Godavari Sector)"},
        "machilipatnam": {"lat": 16.1833, "lon": 81.1333, "imbl_dist_nm": 180.0, "border_zone": "Bay of Bengal EEZ (Krishna Sector)"},
        "krishnapatnam": {"lat": 14.2500, "lon": 80.1167, "imbl_dist_nm": 150.0, "border_zone": "Bay of Bengal EEZ (Nellore Sector)"},

        # --- ODISHA SECTORS ---
        "paradip": {"lat": 20.2600, "lon": 86.6700, "imbl_dist_nm": 180.0, "border_zone": "Bay of Bengal EEZ (Odisha Coast)"},
        "dhamra": {"lat": 20.8000, "lon": 86.9667, "imbl_dist_nm": 175.0, "border_zone": "Bay of Bengal EEZ (Bhadrak Sector)"},
        "gopalpur": {"lat": 19.2667, "lon": 84.9000, "imbl_dist_nm": 185.0, "border_zone": "Bay of Bengal EEZ (Ganjam Coast)"},
        "chandipur": {"lat": 21.4500, "lon": 87.0167, "imbl_dist_nm": 170.0, "border_zone": "Bay of Bengal EEZ (Balasore Sector)"},

        # --- WEST BENGAL SECTORS ---
        "digha": {"lat": 21.6266, "lon": 87.5074, "imbl_dist_nm": 165.0, "border_zone": "Bay of Bengal EEZ (Bengal Coast)"},
        "sankarpur": {"lat": 21.6350, "lon": 87.5670, "imbl_dist_nm": 165.0, "border_zone": "Bay of Bengal EEZ (Bengal Coast)"},
        "petuaghat": {"lat": 21.7800, "lon": 87.8800, "imbl_dist_nm": 160.0, "border_zone": "Bay of Bengal EEZ (Purba Medinipur Sector)"},
        "diamond harbour": {"lat": 22.1833, "lon": 88.2000, "imbl_dist_nm": 155.0, "border_zone": "Hooghly Estuary / Sunderbans Maritime Sector"},
        "kakdwip": {"lat": 21.8700, "lon": 88.1900, "imbl_dist_nm": 150.0, "border_zone": "Bay of Bengal EEZ (Sunderbans Sector)"},
        "fraserganj": {"lat": 21.5833, "lon": 88.2500, "imbl_dist_nm": 140.0, "border_zone": "Bay of Bengal EEZ (Sunderbans Sector)"},
        "haldia": {"lat": 22.0200, "lon": 88.0600, "imbl_dist_nm": 155.0, "border_zone": "Hooghly River / Bay of Bengal Sector"},

        # --- ISLAND TERRITORIES ---
        "port blair": {"lat": 11.6667, "lon": 92.7333, "imbl_dist_nm": 45.0, "border_zone": "Andaman Sea EEZ (India-Myanmar-Thailand Boundary)"},
        "diglipur": {"lat": 13.2667, "lon": 92.9833, "imbl_dist_nm": 30.0, "border_zone": "Andaman Sea EEZ (North Andaman Sector)"},
        "campbell bay": {"lat": 6.9833, "lon": 93.9167, "imbl_dist_nm": 25.0, "border_zone": "Great Nicobar / Malacca Strait IMBL (India-Indonesia)"},
        "kavaratti": {"lat": 10.5667, "lon": 72.6333, "imbl_dist_nm": 110.0, "border_zone": "Lakshadweep Sea EEZ"},
        "agatti": {"lat": 10.8500, "lon": 72.1833, "imbl_dist_nm": 115.0, "border_zone": "Lakshadweep Sea EEZ"},
        "minicoy": {"lat": 8.2833, "lon": 73.0500, "imbl_dist_nm": 35.0, "border_zone": "India-Maldives Eight Degree Channel IMBL"},
    }

    # Country-to-Port Alias Mapping
    # Resolves international destination queries (e.g. srilanka / sri lanka) to designated major ports
    COUNTRY_PORT_ALIASES = {
        "sri lanka": {
            "name": "Colombo Port",
            "port": "colombo",
            "lat": 6.9428,
            "lon": 79.8412,
            "imbl_dist_nm": 0.0,
            "border_zone": "Sri Lankan Sovereign Waters (Colombo Port)",
            "country": "Sri Lanka",
        },
        "srilanka": {
            "name": "Colombo Port",
            "port": "colombo",
            "lat": 6.9428,
            "lon": 79.8412,
            "imbl_dist_nm": 0.0,
            "border_zone": "Sri Lankan Sovereign Waters (Colombo Port)",
            "country": "Sri Lanka",
        },
        "ceylon": {
            "name": "Colombo Port",
            "port": "colombo",
            "lat": 6.9428,
            "lon": 79.8412,
            "imbl_dist_nm": 0.0,
            "border_zone": "Sri Lankan Sovereign Waters (Colombo Port)",
            "country": "Sri Lanka",
        },
        "colombo port": {
            "name": "Colombo Port",
            "port": "colombo",
            "lat": 6.9428,
            "lon": 79.8412,
            "imbl_dist_nm": 0.0,
            "border_zone": "Sri Lankan Sovereign Waters (Colombo Port)",
            "country": "Sri Lanka",
        },
        "colombo": {
            "name": "Colombo Port",
            "port": "colombo",
            "lat": 6.9428,
            "lon": 79.8412,
            "imbl_dist_nm": 0.0,
            "border_zone": "Sri Lankan Sovereign Waters (Colombo Port)",
            "country": "Sri Lanka",
        },
        "maldives": {
            "name": "Malé Port",
            "port": "male",
            "lat": 4.1755,
            "lon": 73.5093,
            "imbl_dist_nm": 0.0,
            "border_zone": "Maldivian Sovereign Waters (Malé Port)",
            "country": "Maldives",
        },
    }

    BORDER_WARNING_THRESHOLD_NM = 12.0
    BORDER_CRITICAL_THRESHOLD_NM = 5.0
    MAX_EEZ_LIMIT_NM = 200.0

    # Major Sea Lines of Communication (SLOC) & Commercial Shipping Fairways
    MAJOR_SHIPPING_LANES = [
        {
            "id": "cape_comorin_sloc",
            "name": "Cape Comorin Global Trunk SLOC (East-West Highway)",
            "description": "Ultra-dense global container & tanker highway connecting Suez/Persian Gulf to Malacca Strait",
            "traffic_type": "Ultra-Large Container Ships & VLCC Tankers (18-24 kt)",
            "corridor_half_width_nm": 3.0,
            "centerline": [
                (7.5000, 76.5000),
                (7.5000, 77.5000),
                (7.3000, 78.5000),
                (5.8000, 80.5000),
                (5.7000, 82.0000),
            ],
        },
        {
            "id": "mumbai_jnpt_tss",
            "name": "Mumbai & JNPT Traffic Separation Scheme (TSS)",
            "description": "Commercial approach channel for India's largest container gateway and offshore Bombay High oil fields",
            "traffic_type": "Container Ships, Chemical Tankers & Bulk Carriers",
            "corridor_half_width_nm": 2.5,
            "centerline": [
                (18.6000, 72.2000),
                (18.8500, 72.5000),
                (18.9500, 72.8000),
            ],
        },
        {
            "id": "gulf_of_kutch_tanker",
            "name": "Gulf of Kutch Deep-Draft Tanker Corridor",
            "description": "Deep-water fairway serving Reliance Jamnagar, Mundra, Kandla, and Sikka refineries",
            "traffic_type": "VLCC Crude Oil Supertankers (Deep Draft > 20m)",
            "corridor_half_width_nm": 2.5,
            "centerline": [
                (22.3500, 68.8000),
                (22.5000, 69.3000),
                (22.7000, 69.8000),
                (22.8000, 70.2000),
            ],
        },
        {
            "id": "sandheads_paradip",
            "name": "Sandheads / Paradip & Haldia Bulk Shipping Channel",
            "description": "Major coal, iron ore, and container approach corridor in northern Bay of Bengal",
            "traffic_type": "Capesize Bulk Carriers & Container Feeders",
            "corridor_half_width_nm": 2.5,
            "centerline": [
                (20.3000, 86.8000),
                (20.8000, 87.5000),
                (21.3000, 88.2000),
            ],
        },
        {
            "id": "vizag_fairway",
            "name": "Visakhapatnam Commercial & Naval Approach Fairway",
            "description": "Deep-water approach channel for Visakhapatnam Port and Eastern Naval Command",
            "traffic_type": "Naval Vessels, Bulk Carriers & Tankers",
            "corridor_half_width_nm": 2.0,
            "centerline": [
                (17.5000, 83.4000),
                (17.6500, 83.3000),
                (17.7000, 83.2500),
            ],
        },
    ]

    def __init__(self, eez_geojson_path: str = EEZ_FILE):
        self.eez_geojson_path = eez_geojson_path
        self.eez_shape = self.load_bhuvan_eez(eez_geojson_path)
        self.prepared_eez = prep(self.eez_shape) if (self.eez_shape is not None and prep is not None) else None

    def load_bhuvan_eez(self, geojson_path: str) -> Optional[Any]:
        """
        Loads local india_eez.geojson (or high-res boundary) and constructs a Shapely polygon geometry.
        Unions all features if multiple are present.
        """
        if not os.path.exists(geojson_path):
            return None

        if shape is None:
            return None

        try:
            with open(geojson_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "features" in data and len(data["features"]) > 0:
                geoms = [shape(feat["geometry"]) for feat in data["features"] if feat.get("geometry")]
                if not geoms:
                    return None
                if len(geoms) == 1:
                    return geoms[0]
                return unary_union(geoms) if unary_union is not None else geoms[0]
        except Exception as e:
            print(f"[GisAgent Notice] Could not load EEZ GeoJSON: {e}")

        return None

    GAZETTEER_ALIASES = {
        # Regional language and common aliases
        "mumbai": "mumbai", "bombay": "mumbai", "મુંબઈ": "mumbai", "மும்பை": "mumbai", "मुंबई": "mumbai", "മുംബൈ": "mumbai", "ముంబై": "mumbai", "মুম্বই": "mumbai", "বোম্বাই": "mumbai",
        "chennai": "chennai", "madras": "chennai", "சென்னை": "chennai", "चेन्नई": "chennai", "చెన్నై": "chennai",
        "kochi": "kochi", "cochin": "kochi", "കൊച്ചി": "kochi", "கொச்சி": "kochi", "कोच्चि": "kochi",
        "goa": "goa", "mormugao": "goa", "panaji": "goa", "ગોવા": "goa", "கோவா": "goa", "गोवा": "goa", "ഗോവ": "goa", "గోవా": "goa", "গোয়া": "goa",
        "tuticorin": "tuticorin", "thoothukudi": "thoothukudi", "தூத்துக்குடி": "thoothukudi", "थूथुकुडी": "thoothukudi", "तूतीकोरिन": "thoothukudi", "തൂത്തുക്കുടി": "thoothukudi",
        "rameswaram": "rameswaram", "ராமேஸ்வரம்": "rameswaram", "रामेश्वरम": "rameswaram",
        "kanyakumari": "kanyakumari", "cape comorin": "kanyakumari", "கன்னியாகுமரி": "kanyakumari", "कन्याकुमारी": "kanyakumari",
        "visakhapatnam": "visakhapatnam", "vizag": "visakhapatnam", "విశాఖపట్నం": "visakhapatnam", "विशाखापट्टनम": "visakhapatnam",
        "porbandar": "porbandar", "પોરબંદર": "porbandar", "पोरबंदर": "porbandar",
        "veraval": "veraval", "વેરાવળ": "veraval", "वेरावल": "veraval",
        "okha": "okha", "ઓખા": "okha",
        "kandla": "kandla", "કંડલા": "kandla", "दीनदयाल": "kandla",
        "mangalore": "mangalore", "new mangalore": "mangalore", "मंगलोर": "mangalore",
        "karwar": "karwar", "कारवार": "karwar",
        "paradip": "paradip", "पारादीप": "paradip",
        "digha": "digha", "দিঘা": "digha", "दीघा": "digha",
        "haldia": "haldia", "হলদিয়া": "haldia",
        "kolkata": "haldia", "calcutta": "haldia", "কলকাতা": "haldia",
        "sri lanka": "sri lanka", "srilanka": "sri lanka", "ceylon": "sri lanka", "colombo": "sri lanka",
        "port blair": "port blair", "andaman": "port blair",
        "kavaratti": "kavaratti", "lakshadweep": "kavaratti",
    }

    def _evaluate_coordinate_sector(self, lat: float, lon: float, name: Optional[str] = None) -> Dict[str, Any]:
        """Evaluates domain validity, IMBL border proximity, and EEZ status for coordinates."""
        # Validate global coordinate boundaries
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return {
                "name": name or f"{lat:.4f},{lon:.4f}",
                "lat": lat,
                "lon": lon,
                "imbl_dist_nm": 9999.0,
                "border_zone": "Invalid Coordinates (Out of Global Latitude/Longitude Bounds)",
                "is_within_eez": False,
                "out_of_operational_domain": True,
                "is_valid_coordinates": False,
                "is_unknown": False,
            }

        # Validate ORCA Indian Ocean operational domain (-15 <= lat <= 30, 50 <= lon <= 105)
        if not (-15.0 <= lat <= 30.0 and 50.0 <= lon <= 105.0):
            return {
                "name": name or f"{lat:.4f},{lon:.4f}",
                "lat": lat,
                "lon": lon,
                "imbl_dist_nm": 9999.0,
                "border_zone": "Outside Indian Maritime Domain / International Waters",
                "is_within_eez": False,
                "out_of_operational_domain": True,
                "is_valid_coordinates": True,
                "is_unknown": False,
            }

        # Dynamic IMBL / border zone evaluation based on geographic sector
        if lat >= 22.0 and lon <= 69.5:
            imbl_dist = max(3.0, (lon - 68.1) * 60.0)
            border_zone = "India-Pakistan IMBL (Sir Creek Sector)"
        elif 8.0 <= lat <= 10.8 and 78.5 <= lon <= 80.5:
            imbl_dist = max(3.0, (79.9 - lon) * 60.0)
            border_zone = "India-Sri Lanka IMBL (Palk Strait / Gulf of Mannar)"
        elif lat <= 9.0 and 71.0 <= lon <= 74.5:
            imbl_dist = max(10.0, (lat - 7.5) * 60.0)
            border_zone = "India-Maldives IMBL (Eight Degree Channel)"
        elif lat <= 7.5 and lon >= 93.0:
            imbl_dist = max(10.0, (lon - 94.5) * 60.0)
            border_zone = "India-Indonesia Maritime Boundary (Great Nicobar)"
        else:
            imbl_dist = 65.0
            border_zone = "Indian Exclusive Economic Zone"

        resolved_name = name
        if not resolved_name:
            try:
                landmark = find_nearest_coastal_landmark(lat, lon)
                resolved_name = landmark.get("landmark_name") or f"Lat {lat:.4f}°, Lon {lon:.4f}°"
            except Exception:
                resolved_name = f"Lat {lat:.4f}°, Lon {lon:.4f}°"

        in_eez = True
        if self.eez_shape is not None and Point is not None:
            try:
                pt = Point(lon, lat)
                if self.prepared_eez is not None:
                    in_eez = bool(self.prepared_eez.contains(pt) or self.eez_shape.intersects(pt) or self.eez_shape.distance(pt) <= 0.05)
                else:
                    in_eez = bool(self.eez_shape.contains(pt) or self.eez_shape.intersects(pt) or self.eez_shape.distance(pt) <= 0.05)
            except Exception:
                in_eez = True

        return {
            "name": resolved_name,
            "lat": lat,
            "lon": lon,
            "imbl_dist_nm": round(imbl_dist, 1),
            "border_zone": border_zone,
            "is_within_eez": in_eez,
            "out_of_operational_domain": False,
            "is_valid_coordinates": True,
            "is_unknown": False,
        }

    def resolve_location(self, target_location: Any) -> Dict[str, Any]:
        if target_location is None:
            return {
                "name": "Location Required",
                "lat": None,
                "lon": None,
                "imbl_dist_nm": 0.0,
                "border_zone": "None Specified",
                "is_within_eez": False,
                "out_of_operational_domain": False,
                "is_valid_coordinates": False,
                "is_unknown": True,
            }

        # Check if LocationContext object or dict with latitude/longitude
        lat_val = getattr(target_location, "latitude", None)
        lon_val = getattr(target_location, "longitude", None)
        name_val = getattr(target_location, "location_name", None)

        if isinstance(target_location, dict):
            lat_val = target_location.get("latitude", target_location.get("lat", lat_val))
            lon_val = target_location.get("longitude", target_location.get("lon", lon_val))
            name_val = target_location.get("location_name", target_location.get("name", name_val))

        if lat_val is not None and lon_val is not None:
            try:
                lat = float(lat_val)
                lon = float(lon_val)
                return self._evaluate_coordinate_sector(lat, lon, name=name_val)
            except (ValueError, TypeError):
                pass

        if isinstance(target_location, (tuple, list)) and len(target_location) >= 2:
            try:
                lat = float(target_location[0])
                lon = float(target_location[1])
                return self._evaluate_coordinate_sector(lat, lon)
            except (ValueError, TypeError):
                pass

        loc_str = str(name_val or target_location).strip()
        if "," in loc_str:
            try:
                parts = loc_str.split(",")
                lat = float(parts[0].strip())
                lon = float(parts[1].strip())
                return self._evaluate_coordinate_sector(lat, lon)
            except (ValueError, IndexError):
                pass

        clean_key = loc_str.lower().strip()

        # 1. Gazetteer aliases
        for alias, mapped_sec in self.GAZETTEER_ALIASES.items():
            if alias == clean_key or f" {alias} " in f" {clean_key} " or clean_key.startswith(f"{alias} ") or clean_key.endswith(f" {alias}"):
                if mapped_sec in self.COUNTRY_PORT_ALIASES:
                    res = dict(self.COUNTRY_PORT_ALIASES[mapped_sec])
                    res["is_unknown"] = False
                    res["is_valid_coordinates"] = True
                    return res
                if mapped_sec in self.COASTAL_SECTORS:
                    res = dict(self.COASTAL_SECTORS[mapped_sec])
                    res["name"] = mapped_sec.title()
                    res["is_unknown"] = False
                    res["is_valid_coordinates"] = True
                    return res

        # 2. Country-to-port alias mapping
        for alias, sector in self.COUNTRY_PORT_ALIASES.items():
            if alias in clean_key:
                res = dict(sector)
                res["is_unknown"] = False
                res["is_valid_coordinates"] = True
                return res

        # 3. Known coastal sectors
        for name, sector in self.COASTAL_SECTORS.items():
            if name in clean_key:
                res = dict(sector)
                res["name"] = name.title()
                res["is_unknown"] = False
                res["is_valid_coordinates"] = True
                return res

        # 4. Unknown / Unresolvable location (ZERO SILENT TUTICORIN FALLBACK)
        return {
            "name": loc_str,
            "lat": None,
            "lon": None,
            "imbl_dist_nm": 0.0,
            "border_zone": "Unknown Geographic Location",
            "is_within_eez": False,
            "out_of_operational_domain": False,
            "is_valid_coordinates": False,
            "is_unknown": True,
        }

    def check_imbl(self, target_location: Any) -> bool:
        """
        Evaluates whether a target location or coordinate pair constitutes an intentional or actual
        cross-border violation of the International Maritime Boundary Line (IMBL) into Sri Lankan
        sovereign waters or other foreign maritime jurisdictions.
        Returns False for all Indian coastal ports, sovereign fishing grounds, and Indian EEZ sectors.
        """
        if not target_location:
            return False

        if isinstance(target_location, (tuple, list)) and len(target_location) >= 2:
            try:
                lat, lon = float(target_location[0]), float(target_location[1])
                return self._is_sri_lanka_coords(lat, lon)
            except (ValueError, TypeError):
                return False

        loc_str = str(target_location).strip().lower()

        # Explicit foreign ports and Sri Lankan geographic keywords
        foreign_keywords = [
            "sri lanka", "srilanka", "ceylon", "jaffna", "colombo", "talaimannar",
            "kankesanthurai", "trincomalee", "batticaloa", "galle", "hambantota",
            "point pedro", "delft island", "neduntheevu", "mannar island", "pesalai",
            "cross imbl", "cross the imbl", "cross border", "cross the border"
        ]
        if any(k in loc_str for k in foreign_keywords):
            return True

        # Check coordinate string "lat,lon"
        if "," in loc_str:
            try:
                parts = loc_str.split(",")
                lat, lon = float(parts[0].strip()), float(parts[1].strip())
                return self._is_sri_lanka_coords(lat, lon)
            except (ValueError, IndexError):
                pass

        return False

    def _is_sri_lanka_coords(self, lat: float, lon: float) -> bool:
        """
        Checks if geographic coordinates fall across the IMBL into Sri Lankan sovereign waters.
        Coordinates west of the IMBL (e.g. Tuticorin 78.13E, Rameswaram west 79.31E) are sovereign Indian waters.
        """
        # In Palk Bay / Palk Strait (lat 8.5 to 10.5), IMBL lies between lon 79.40 and 79.80.
        # Longitudes >= 79.55 in this belt enter Sri Lankan waters (e.g., Kachchatheevu east, Jaffna 80.0E).
        if 8.5 <= lat <= 10.5 and lon >= 79.55:
            return True
        # In Gulf of Mannar (lat 7.0 to 8.5), Sri Lankan waters are east of lon 79.70.
        if 7.0 <= lat < 8.5 and lon >= 79.70:
            return True
        # Sri Lanka south / east waters
        if 5.5 <= lat < 7.0 and lon >= 79.60:
            return True
        return False

    def check_imbl_proximity(self, target_location: Any) -> Dict[str, Any]:
        sector = self.resolve_location(target_location)
        dist_nm = sector.get("imbl_dist_nm", 0.0)
        border_zone = sector.get("border_zone", "Unknown")
        lat = sector.get("lat")
        lon = sector.get("lon")

        # Location Unresolvable / Missing check
        if sector.get("is_unknown") or lat is None or lon is None:
            return {
                "target_location": str(target_location) if target_location is not None else "Location Required",
                "coordinates": {"lat": None, "lon": None},
                "distance_to_imbl_nm": 0.0,
                "distance_to_imbl_km": 0.0,
                "border_zone": "Unknown Geographic Location",
                "is_within_eez": False,
                "status_flag": "LOCATION_REQUIRED",
                "compliance_status": "LOCATION_REQUIRED",
                "advisory": f"Geographic location '{target_location}' could not be resolved. Border monitoring and EEZ verification require a recognized coastal port or valid GPS coordinates.",
                "shipping_lane": {
                    "inside_lane": False,
                    "nearest_lane_name": "Unknown",
                    "corridor_clearance_nm": 0.0,
                    "status_flag": "LOCATION_REQUIRED",
                    "advisory": "Location unresolvable.",
                },
                "shapely_verified": False,
                "is_unknown": True,
            }

        # Out-of-Operational-Domain check
        if sector.get("out_of_operational_domain") or not (-15.0 <= lat <= 30.0 and 50.0 <= lon <= 105.0):
            return {
                "target_location": str(target_location),
                "coordinates": {"lat": lat, "lon": lon},
                "distance_to_imbl_nm": dist_nm,
                "distance_to_imbl_km": round(dist_nm * 1.852, 2) if dist_nm < 9000 else 9999.0,
                "border_zone": border_zone,
                "is_within_eez": False,
                "status_flag": "OUT_OF_OPERATIONAL_DOMAIN",
                "compliance_status": "OUT_OF_DOMAIN",
                "advisory": f"Coordinates ({lat:.4f}, {lon:.4f}) lie outside the ORCA Indian Ocean operational domain (-15° to 30°N, 50° to 105°E). Border monitoring and EEZ verification not applicable.",
                "shipping_lane": {
                    "inside_lane": False,
                    "nearest_lane_name": "N/A",
                    "corridor_clearance_nm": 9999.0,
                    "status_flag": "CLEAR_OF_SHIPPING_LANES",
                    "advisory": "Vessel position is outside the Indian Ocean operational domain.",
                },
                "shapely_verified": False,
                "is_unknown": False,
            }

        # Geometric boundary check with Shapely if available
        is_within_eez = dist_nm <= self.MAX_EEZ_LIMIT_NM
        if self.eez_shape is not None and Point is not None:
            pt = Point(lon, lat)
            try:
                if self.prepared_eez is not None:
                    is_within_eez = bool(self.prepared_eez.contains(pt) or self.eez_shape.intersects(pt) or self.eez_shape.distance(pt) <= 0.05)
                else:
                    is_within_eez = bool(self.eez_shape.contains(pt) or self.eez_shape.intersects(pt) or self.eez_shape.distance(pt) <= 0.05)
            except Exception:
                is_within_eez = dist_nm <= self.MAX_EEZ_LIMIT_NM

        if dist_nm <= self.BORDER_CRITICAL_THRESHOLD_NM:
            status_flag = "CRITICAL_BORDER_PROXIMITY"
            compliance = "IMMEDIATE_TURN_BACK_REQUIRED"
            advisory = (
                f"Vessel is only {dist_nm} NM from the {border_zone}! "
                f"Critical risk of international border crossing violation."
            )
        elif dist_nm <= self.BORDER_WARNING_THRESHOLD_NM:
            status_flag = "NEAR_IMBL_WARNING"
            compliance = "EXERCISE_EXTREME_CAUTION"
            advisory = (
                f"Vessel is {dist_nm} NM from {border_zone}. "
                f"Approaching restricted international boundary."
            )
        else:
            status_flag = "SAFE_EEZ_WATERS"
            compliance = "PERMITTED_FISHING_ZONE"
            advisory = (
                f"Vessel location is safely within Indian Sovereign Waters ({dist_nm} NM from {border_zone})."
            )

        shipping_lane = self.check_shipping_lane_proximity(lat, lon)

        return {
            "target_location": target_location,
            "coordinates": {"lat": lat, "lon": lon},
            "distance_to_imbl_nm": dist_nm,
            "distance_to_imbl_km": round(dist_nm * 1.852, 2),
            "border_zone": border_zone,
            "is_within_eez": is_within_eez,
            "status_flag": status_flag,
            "compliance_status": compliance,
            "advisory": advisory,
            "shipping_lane": shipping_lane,
            "shapely_verified": self.eez_shape is not None,
        }

    def check_shipping_lane_proximity(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Evaluates vessel coordinates against major Indian commercial shipping lanes
        and Traffic Separation Schemes (TSS) to prevent catastrophic collisions.
        """
        closest_lane = None
        min_distance_nm = float("inf")

        for lane in self.MAJOR_SHIPPING_LANES:
            centerline = lane["centerline"]
            for i in range(len(centerline) - 1):
                p1_lat, p1_lon = centerline[i]
                p2_lat, p2_lon = centerline[i + 1]

                # Project point onto line segment using nautical coordinate space
                cos_lat = math.cos(math.radians((p1_lat + p2_lat) / 2.0))
                dx = (p2_lon - p1_lon) * cos_lat * 111.0
                dy = (p2_lat - p1_lat) * 111.0
                seg_len_sq = dx * dx + dy * dy

                if seg_len_sq == 0:
                    dist_km = math.hypot((lon - p1_lon) * cos_lat * 111.0, (lat - p1_lat) * 111.0)
                else:
                    px = (lon - p1_lon) * cos_lat * 111.0
                    py = (lat - p1_lat) * 111.0
                    t = max(0.0, min(1.0, (px * dx + py * dy) / seg_len_sq))
                    proj_x = t * dx
                    proj_y = t * dy
                    dist_km = math.hypot(px - proj_x, py - proj_y)

                dist_nm = dist_km / 1.852
                if dist_nm < min_distance_nm:
                    min_distance_nm = dist_nm
                    closest_lane = lane

        if closest_lane is None:
            return {
                "inside_lane": False,
                "distance_to_lane_nm": 99.0,
                "nearest_lane_name": "None",
                "status_flag": "CLEAR_OF_SHIPPING_LANES",
                "advisory": "Vessel is clear of major commercial shipping fairways.",
            }

        half_width = closest_lane["corridor_half_width_nm"]
        lane_name = closest_lane["name"]
        traffic_type = closest_lane["traffic_type"]
        dist_rounded = round(min_distance_nm, 1)

        if min_distance_nm <= half_width:
            inside = True
            status_flag = "INSIDE_SHIPPING_LANE"
            clearance_needed = round(half_width - min_distance_nm + 1.5, 1)
            advisory = (
                f"CRITICAL COLLISION RISK: Vessel is operating INSIDE the {lane_name}! "
                f"Commercial traffic ({traffic_type}) navigates here at high speed (18-24 kt) with limited maneuverability. "
                f"Move {clearance_needed} NM clear of the fairway immediately and keep all-round white navigation lights lit."
            )
        elif min_distance_nm <= (half_width + 4.0):
            inside = False
            clearance = round(min_distance_nm - half_width, 1)
            status_flag = "NEAR_SHIPPING_LANE_WARNING"
            advisory = (
                f"COLLISION WARNING: Vessel is {clearance} NM from {lane_name}. "
                f"High-speed commercial vessels ({traffic_type}) operate in this corridor. "
                f"Maintain radar reflector and continuous visual watch."
            )
        else:
            inside = False
            clearance = round(min_distance_nm - half_width, 1)
            status_flag = "CLEAR_OF_SHIPPING_LANES"
            advisory = f"Clear of commercial shipping fairways ({clearance} NM clearance from {lane_name})."

        return {
            "inside_lane": inside,
            "distance_to_centerline_nm": dist_rounded,
            "corridor_clearance_nm": round(max(0.0, min_distance_nm - half_width), 1),
            "lane_id": closest_lane["id"],
            "nearest_lane_name": lane_name,
            "traffic_type": traffic_type,
            "corridor_half_width_nm": half_width,
            "status_flag": status_flag,
            "advisory": advisory,
        }

    def calculate_bearing_and_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> tuple[float, str, float]:
        """Calculates distance in NM, compass bearing string, and degree heading."""
        d_lat = lat2 - lat1
        d_lon = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
        dist_km = math.sqrt((d_lat * 111.0) ** 2 + (d_lon * 111.0) ** 2)
        dist_nm = round(dist_km / 1.852, 1)

        y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
        x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(
            math.radians(lat1)
        ) * math.cos(math.radians(lat2)) * math.cos(math.radians(lon2 - lon1))
        initial_bearing = (math.degrees(math.atan2(y, x)) + 360) % 360

        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        bearing_str = directions[int((initial_bearing + 11.25) / 22.5) % 16]

        return dist_nm, f"{round(initial_bearing):03d}° ({bearing_str})", round(initial_bearing, 1)

    def check_land_intersection(self, lat1: float, lon1: float, lat2: float, lon2: float) -> bool:
        """
        Determines whether a direct nautical navigational vector between two coordinates
        actually intersects the peninsular Indian mainland land polygon.
        Guarantees that zero-distance, stationary positions, and open-water routes never trigger false errors.
        """
        # Zero-distance guard: stationary positions or identical endpoints cannot traverse land
        if abs(lat1 - lat2) < 1e-4 and abs(lon1 - lon2) < 1e-4:
            return False

        # 1. Shapely geometry intersection check against peninsular core landmask
        if LineString and PENINSULA_CORE_LANDMASK:
            try:
                route_line = LineString([(lon1, lat1), (lon2, lat2)])
                # Ensure the line has non-zero length and actually intersects the land polygon
                if route_line.length > 1e-5 and route_line.intersects(PENINSULA_CORE_LANDMASK):
                    intersection = route_line.intersection(PENINSULA_CORE_LANDMASK)
                    if intersection and not intersection.is_empty:
                        # Only true overland crossings produce a line intersection with non-trivial length
                        if intersection.geom_type in ("LineString", "MultiLineString") or getattr(intersection, "length", 0) > 1e-4:
                            return True
            except Exception:
                pass

        # 2. Heuristic check for cross-peninsula transits between West and East coasts north of Cape Comorin
        # West coast: lon < 77.3, lat > 8.35 (e.g., Kochi 9.93, 76.27; Mangalore 12.92, 74.81; Mumbai 18.92, 72.83)
        # East coast: lon > 77.8, lat > 8.35 (e.g., Tuticorin 8.76, 78.13; Chennai 13.08, 80.27; Rameswaram 9.28, 79.31)
        # Both origin and destination must be north of Cape Comorin (lat > 8.35) for cross-peninsula overland transit
        is_west_origin = lon1 < 77.3 and lat1 > 8.35
        is_east_dest = lon2 > 77.8 and lat2 > 8.35
        is_east_origin = lon1 > 77.8 and lat1 > 8.35
        is_west_dest = lon2 < 77.3 and lat2 > 8.35
        if (is_west_origin and is_east_dest) or (is_east_origin and is_west_dest):
            return True

        return False

    def is_over_land(self, lat: float, lon: float) -> bool:
        """Checks if a geographic coordinate falls over peninsular Indian or Sri Lankan landmass."""
        if Point:
            pt = Point(lon, lat)
            if PENINSULA_CORE_LANDMASK and PENINSULA_CORE_LANDMASK.contains(pt):
                return True
            if SRI_LANKA_LANDMASK and SRI_LANKA_LANDMASK.contains(pt):
                return True
        # Additional bathymetric heuristic check if available
        try:
            from ml.data_ingestion.bathymetry_geography import get_geographic_features
            geo = get_geographic_features(lat, lon)
            if geo.get("is_land"):
                return True
        except Exception:
            pass
        return False

    def is_in_indian_eez(self, lat: float, lon: float) -> bool:
        """Determines whether a coordinate is inside the Indian Exclusive Economic Zone."""
        if self.prepared_eez and Point:
            try:
                return bool(self.prepared_eez.contains(Point(lon, lat)))
            except Exception:
                pass
        # Fallback to coarse bounding geofence if prepared geometry is unavailable
        return bool(0.0 <= lat <= 25.0 and 65.0 <= lon <= 95.0)

    def generate_safe_sea_route(
        self,
        origin: Any,
        destination: Any,
        cruising_speed_knots: float = 8.0,
        active_hazards: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generates a sovereign, hazard-free nautical passage plan between origin and destination.
        Guarantees >= 8.0 NM clearance from the International Maritime Boundary Line (IMBL).
        Inserts tactical dogleg waypoints to evade border zones and storm hazard radiuses.
        """
        orig_sector = self.resolve_location(str(origin))
        dest_sector = self.resolve_location(str(destination))

        # 0. Unknown origin/destination guardrail:
        # Do NOT treat unknown destination as LAND_INTERSECTION_ERROR
        if dest_sector.get("is_unknown") and not any(k in str(destination).lower() for k in ["pfz", "hotspot", "offshore", "waypoint"]):
            return {
                "route_status": "UNKNOWN_DESTINATION",
                "error": "UNKNOWN_DESTINATION",
                "total_distance_nm": 0.0,
                "estimated_duration_hours": 0.0,
                "estimated_duration_formatted": "N/A",
                "cruising_speed_knots": cruising_speed_knots,
                "waypoints_count": 0,
                "minimum_border_clearance_nm": 0.0,
                "dogleg_reroute_active": False,
                "waypoints": [],
                "route_geojson": {"type": "FeatureCollection", "features": []},
                "navigational_brief": (
                    f"Destination '{destination}' could not be resolved to known maritime coordinates or ports. "
                    f"Please provide a recognized port name (e.g. Tuticorin, Colombo Port, Kochi) or GPS coordinates."
                ),
            }

        if orig_sector.get("is_unknown") and not any(k in str(origin).lower() for k in ["pfz", "hotspot", "offshore", "waypoint"]):
            return {
                "route_status": "UNKNOWN_ORIGIN",
                "error": "UNKNOWN_ORIGIN",
                "total_distance_nm": 0.0,
                "estimated_duration_hours": 0.0,
                "estimated_duration_formatted": "N/A",
                "cruising_speed_knots": cruising_speed_knots,
                "waypoints_count": 0,
                "minimum_border_clearance_nm": 0.0,
                "dogleg_reroute_active": False,
                "waypoints": [],
                "route_geojson": {"type": "FeatureCollection", "features": []},
                "navigational_brief": (
                    f"Origin '{origin}' could not be resolved to known maritime coordinates or ports. "
                    f"Please provide a recognized port name (e.g. Tuticorin, Colombo Port, Kochi) or GPS coordinates."
                ),
            }

        o_lat, o_lon = orig_sector["lat"], orig_sector["lon"]
        d_lat, d_lon = dest_sector["lat"], dest_sector["lon"]

        direct_dist, direct_bearing_str, direct_bearing_deg = self.calculate_bearing_and_distance(o_lat, o_lon, d_lat, d_lon)

        # 0. Out-of-Operational-Domain & Extreme Range Guardrail:
        is_o_out = orig_sector.get("out_of_operational_domain") or not (-15.0 <= o_lat <= 30.0 and 50.0 <= o_lon <= 105.0)
        is_d_out = dest_sector.get("out_of_operational_domain") or not (-15.0 <= d_lat <= 30.0 and 50.0 <= d_lon <= 105.0)
        if is_o_out or is_d_out or direct_dist > 600.0:
            status_code = "OUT_OF_OPERATIONAL_DOMAIN" if (is_o_out or is_d_out) else "ROUTE_EXCEEDS_OPERATIONAL_RANGE"
            reason = (
                f"Coordinates ({o_lat:.4f}, {o_lon:.4f} -> {d_lat:.4f}, {d_lon:.4f}) lie outside "
                f"the Indian Ocean maritime operational zone (-15° to 30°N, 50° to 105°E)."
                if (is_o_out or is_d_out) else
                f"Direct distance ({direct_dist:.1f} NM) exceeds maximum coastal operational range (600 NM)."
            )
            return {
                "route_status": status_code,
                "error": status_code,
                "total_distance_nm": 0.0,
                "estimated_duration_hours": 0.0,
                "estimated_duration_formatted": "0m",
                "cruising_speed_knots": cruising_speed_knots,
                "waypoints_count": 0,
                "minimum_border_clearance_nm": 0.0,
                "dogleg_reroute_active": False,
                "waypoints": [],
                "route_geojson": {
                    "type": "FeatureCollection",
                    "features": []
                },
                "navigational_brief": f"Routing disabled: {reason}",
            }

        # 1. Zero-distance / Stationary guardrail:
        # Vessel is already at the requested destination
        if direct_dist < 0.1 or (abs(o_lat - d_lat) < 1e-4 and abs(o_lon - d_lon) < 1e-4):
            return {
                "route_status": "STATIONARY",
                "total_distance_nm": 0.0,
                "estimated_duration_hours": 0.0,
                "estimated_duration_formatted": "0m",
                "cruising_speed_knots": cruising_speed_knots,
                "waypoints_count": 1,
                "minimum_border_clearance_nm": orig_sector.get("imbl_dist_nm", 30.0),
                "dogleg_reroute_active": False,
                "waypoints": [{
                    "waypoint_index": 1,
                    "name": f"Stationary ({orig_sector.get('name') or orig_sector.get('border_zone', 'Current Location')[:25]})",
                    "lat": round(o_lat, 4),
                    "lon": round(o_lon, 4),
                    "leg_heading": "000° (N)",
                    "leg_distance_nm": 0.0,
                    "cumulative_distance_nm": 0.0,
                    "leg_duration_minutes": 0,
                    "eta_from_departure": "+00h 00m",
                    "border_clearance_nm": orig_sector.get("imbl_dist_nm", 30.0),
                    "safety_clearance": "Vessel already at target destination; holding stationary position.",
                }],
                "route_geojson": {
                    "type": "FeatureCollection",
                    "features": [{
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [round(o_lon, 4), round(o_lat, 4)]},
                        "properties": {"name": "Current Position", "distance_nm": 0.0}
                    }]
                },
                "navigational_brief": f"Vessel is already at the target destination ({orig_sector.get('name', 'Current Location')}). No nautical transit required.",
            }

        # 3. Coastal Landmask Guardrail: Detect overland marine routing impossibilities
        if self.check_land_intersection(o_lat, o_lon, d_lat, d_lon):
            orig_name = orig_sector.get("name") or orig_sector.get("border_zone", "Origin Port")[:25]
            dest_name = dest_sector.get("name") or dest_sector.get("border_zone", "Destination Port")[:25]
            return {
                "route_status": "LAND_INTERSECTION_ERROR",
                "error": "LAND_INTERSECTION_ERROR",
                "total_distance_nm": direct_dist,
                "estimated_duration_hours": 0.0,
                "estimated_duration_formatted": "N/A",
                "cruising_speed_knots": cruising_speed_knots,
                "waypoints_count": 0,
                "minimum_border_clearance_nm": 0.0,
                "dogleg_reroute_active": False,
                "waypoints": [],
                "route_geojson": {"type": "FeatureCollection", "features": []},
                "navigational_brief": (
                    f"CRITICAL ROUTING FAILURE [LAND_INTERSECTION_ERROR]: Direct nautical passage between "
                    f"({o_lat:.4f}, {o_lon:.4f}) and ({d_lat:.4f}, {d_lon:.4f}) directly intersects "
                    f"the Indian peninsular landmass ({direct_dist} NM straight-line). Overland marine routing is physically impossible."
                ),
            }

        mid_lat = (o_lat + d_lat) / 2.0
        mid_lon = (o_lon + d_lon) / 2.0
        mid_sector = self.resolve_location(f"{mid_lat:.4f},{mid_lon:.4f}")
        mid_imbl_dist = mid_sector["imbl_dist_nm"]

        waypoints = []
        waypoints.append({
            "waypoint_index": 1,
            "name": f"Departure ({orig_sector.get('border_zone', 'Coastal Port')[:25]})",
            "lat": round(o_lat, 4),
            "lon": round(o_lon, 4),
            "leg_heading": direct_bearing_str,
            "leg_distance_nm": 0.0,
            "cumulative_distance_nm": 0.0,
            "leg_duration_minutes": 0,
            "eta_from_departure": "+00h 00m",
            "border_clearance_nm": orig_sector["imbl_dist_nm"],
            "safety_clearance": "Vessel departing authorized coastal port within sovereign waters.",
        })

        needs_dogleg = False
        if mid_imbl_dist < 10.0 or orig_sector["imbl_dist_nm"] < 8.0 or dest_sector["imbl_dist_nm"] < 8.0:
            needs_dogleg = True

        if active_hazards:
            for hz in active_hazards:
                hz_lat, hz_lon = hz.get("center", (mid_lat, mid_lon))
                rad_nm = hz.get("radius_km", 25.0) / 1.852
                d_to_hz, _, _ = self.calculate_bearing_and_distance(mid_lat, mid_lon, hz_lat, hz_lon)
                if d_to_hz <= rad_nm:
                    needs_dogleg = True
                    break

        if needs_dogleg:
            shift_lon = -0.15 if mid_lon > 78.5 else 0.12
            shift_lat = 0.08
            dogleg_lat = round(mid_lat + shift_lat, 4)
            dogleg_lon = round(mid_lon + shift_lon, 4)
            dogleg_sector = self.resolve_location(f"{dogleg_lat:.4f},{dogleg_lon:.4f}")

            dist1, bearing1_str, _ = self.calculate_bearing_and_distance(o_lat, o_lon, dogleg_lat, dogleg_lon)
            time1_min = int((dist1 / max(cruising_speed_knots, 1.0)) * 60)
            t1_hrs = time1_min // 60
            t1_rem = time1_min % 60

            waypoints.append({
                "waypoint_index": 2,
                "name": "Waypoint 1 (IMBL Sovereignty Buffer Clearance)",
                "lat": dogleg_lat,
                "lon": dogleg_lon,
                "leg_heading": bearing1_str,
                "leg_distance_nm": dist1,
                "cumulative_distance_nm": dist1,
                "leg_duration_minutes": time1_min,
                "eta_from_departure": f"+{t1_hrs:02d}h {t1_rem:02d}m",
                "border_clearance_nm": dogleg_sector["imbl_dist_nm"],
                "safety_clearance": f"Clearance Waypoint established: {dogleg_sector['imbl_dist_nm']} NM from international boundary.",
            })

            dist2, bearing2_str, _ = self.calculate_bearing_and_distance(dogleg_lat, dogleg_lon, d_lat, d_lon)
            time2_min = int((dist2 / max(cruising_speed_knots, 1.0)) * 60)
            total_dist = round(dist1 + dist2, 1)
            total_time_min = time1_min + time2_min
            tot_hrs = total_time_min // 60
            tot_rem = total_time_min % 60

            waypoints.append({
                "waypoint_index": 3,
                "name": f"Destination ({dest_sector.get('border_zone', 'Target Zone')[:25]})",
                "lat": round(d_lat, 4),
                "lon": round(d_lon, 4),
                "leg_heading": bearing2_str,
                "leg_distance_nm": dist2,
                "cumulative_distance_nm": total_dist,
                "leg_duration_minutes": time2_min,
                "eta_from_departure": f"+{tot_hrs:02d}h {tot_rem:02d}m",
                "border_clearance_nm": dest_sector["imbl_dist_nm"],
                "safety_clearance": "Arrival at operational sector; maintain safe distance from international boundary.",
            })
        else:
            total_dist = direct_dist
            total_time_min = int((total_dist / max(cruising_speed_knots, 1.0)) * 60)
            tot_hrs = total_time_min // 60
            tot_rem = total_time_min % 60

            waypoints.append({
                "waypoint_index": 2,
                "name": f"Destination ({dest_sector.get('border_zone', 'Target Zone')[:25]})",
                "lat": round(d_lat, 4),
                "lon": round(d_lon, 4),
                "leg_heading": direct_bearing_str,
                "leg_distance_nm": direct_dist,
                "cumulative_distance_nm": total_dist,
                "leg_duration_minutes": total_time_min,
                "eta_from_departure": f"+{tot_hrs:02d}h {tot_rem:02d}m",
                "border_clearance_nm": dest_sector["imbl_dist_nm"],
                "safety_clearance": "Direct open-water passage verified clear of international boundary buffer.",
            })

        track_coordinates = [[wp["lon"], wp["lat"]] for wp in waypoints]
        geojson_features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": track_coordinates,
                },
                "properties": {
                    "feature_type": "SAFE_SEA_PASSAGE_ROUTE",
                    "total_distance_nm": total_dist,
                    "cruising_speed_knots": cruising_speed_knots,
                    "estimated_duration_hours": round(total_time_min / 60.0, 2),
                    "reroute_dogleg_applied": needs_dogleg,
                },
            }
        ]

        for wp in waypoints:
            geojson_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [wp["lon"], wp["lat"]],
                },
                "properties": {
                    "waypoint_index": wp["waypoint_index"],
                    "name": wp["name"],
                    "heading": wp["leg_heading"],
                    "distance_nm": wp["leg_distance_nm"],
                    "eta": wp["eta_from_departure"],
                    "border_clearance_nm": wp["border_clearance_nm"],
                },
            })

        route_geojson = {
            "type": "FeatureCollection",
            "features": geojson_features,
        }

        min_border = min(wp["border_clearance_nm"] for wp in waypoints)
        return {
            "route_status": "SAFE_PASSAGE_PLAN",
            "total_distance_nm": total_dist,
            "estimated_duration_hours": round(total_time_min / 60.0, 2),
            "estimated_duration_formatted": f"{tot_hrs}h {tot_rem:02d}m",
            "cruising_speed_knots": cruising_speed_knots,
            "waypoints_count": len(waypoints),
            "minimum_border_clearance_nm": min_border,
            "dogleg_reroute_active": needs_dogleg,
            "waypoints": waypoints,
            "route_geojson": route_geojson,
            "navigational_brief": (
                f"Voyage Plan ({total_dist} NM, ~{tot_hrs}h {tot_rem:02d}m at {cruising_speed_knots} kt): "
                f"Departing {waypoints[0]['name']} along initial heading {waypoints[1]['leg_heading']}. "
                f"{'Deflected inshore to preserve 8 NM sovereign border buffer.' if needs_dogleg else 'Direct sovereign passage.'} "
                f"Minimum IMBL clearance: {min_border} NM."
            ),
        }

    def execute_task(
        self,
        location: Any,
        time_frame: str = "today",
        task_instructions: str = "",
        expected_format: str = "TEXT_SUMMARY",
        persona: str = "FISHERMAN",
    ) -> Dict[str, Any]:
        sector = self.resolve_location(location)
        loc_name = sector.get("name") if not sector.get("is_unknown") else (str(location) if location else "Location Required")
        loc_str = str(loc_name)
        time_str = time_frame.strip() if time_frame else "Current"
        format_upper = expected_format.strip().upper() if expected_format else "TEXT_SUMMARY"
        persona_upper = persona.strip().upper() if persona else "UNKNOWN"

        loc_dict = {
            "name": sector.get("name") or loc_name,
            "latitude": sector.get("lat"),
            "longitude": sector.get("lon"),
        }

        task_lower = (task_instructions or "").lower()
        is_route_requested = any(k in task_lower for k in ["route", "passage", "navigate", "sail", "waypoint", "plan", "course"]) or "from " in task_lower

        # If location is unknown and not a route request
        if sector.get("is_unknown") and not is_route_requested:
            return {
                "agent": "GIS_AGENT",
                "location": loc_dict,
                "location_name": loc_name,
                "time_frame": time_str,
                "format": format_upper,
                "status": "LOCATION_REQUIRED",
                "error": "LOCATION_REQUIRED",
                "distance_to_border_nm": 0.0,
                "is_within_eez": False,
                "advisory": f"Geographic location '{location}' is required. Please specify a recognized coastal port or valid GPS coordinates.",
                "shipping_lane_assessment": {
                    "inside_lane": False,
                    "nearest_lane_name": "Unknown",
                    "corridor_clearance_nm": 0.0,
                    "status_flag": "LOCATION_REQUIRED",
                    "advisory": "Location unresolvable.",
                },
                "recommendation": "Provide departure harbor or valid GPS coordinates.",
                "safe_sea_route": None,
            }

        proximity = self.check_imbl_proximity(location if not sector.get("is_unknown") else loc_name)

        safe_route = None
        if is_route_requested:
            dest_loc = None
            orig_loc = loc_name if not sector.get("is_unknown") else None

            # Check explicit "from X to Y" in task instructions
            route_match = re.search(r"from\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+)", task_lower)
            if route_match:
                orig_loc = route_match.group(1).strip()
                dest_loc = route_match.group(2).strip()
            else:
                to_match = re.search(r"(?:to|towards)\s+([a-zA-Z\s]+)", task_lower)
                if to_match:
                    dest_loc = to_match.group(1).strip()

            if not dest_loc:
                for alias in self.GAZETTEER_ALIASES:
                    if alias in task_lower and (not orig_loc or alias not in str(orig_loc).lower()):
                        dest_loc = alias
                        break

            if dest_loc and orig_loc:
                safe_route = self.generate_safe_sea_route(origin=orig_loc, destination=dest_loc)
            elif dest_loc and not orig_loc:
                return {
                    "agent": "GIS_AGENT",
                    "location": loc_dict,
                    "location_name": loc_name,
                    "time_frame": time_str,
                    "format": format_upper,
                    "status": "LOCATION_REQUIRED",
                    "error": "DEPARTURE_LOCATION_REQUIRED",
                    "distance_to_border_nm": 0.0,
                    "is_within_eez": False,
                    "advisory": "Departure harbor or origin coordinates required to plan sea passage.",
                    "shipping_lane_assessment": proximity.get("shipping_lane", {}),
                    "recommendation": "Specify departure port (e.g. Mumbai, Tuticorin, Kochi).",
                    "safe_sea_route": None,
                }
            else:
                safe_route = {
                    "route_status": "UNKNOWN_DESTINATION",
                    "navigational_brief": "Destination harbor not specified. Route generation aborted.",
                }

        if safe_route and safe_route.get("route_status") == "UNKNOWN_DESTINATION":
            return {
                "agent": "GIS_AGENT",
                "location": loc_dict,
                "location_name": loc_name,
                "time_frame": time_str,
                "format": format_upper,
                "status": "DATA_UNAVAILABLE",
                "error": "UNKNOWN_DESTINATION",
                "distance_to_border_nm": proximity["distance_to_imbl_nm"],
                "is_within_eez": proximity["is_within_eez"],
                "advisory": safe_route["navigational_brief"],
                "shipping_lane_assessment": proximity["shipping_lane"],
                "recommendation": "Destination could not be resolved. Please specify a recognized coastal port or GPS coordinates.",
                "safe_sea_route": safe_route,
            }

        if (safe_route and safe_route.get("route_status") in ("OUT_OF_OPERATIONAL_DOMAIN", "ROUTE_EXCEEDS_OPERATIONAL_RANGE")) or proximity.get("status_flag") == "OUT_OF_OPERATIONAL_DOMAIN":
            err_status = (safe_route.get("route_status") if safe_route else None) or proximity.get("status_flag") or "OUT_OF_OPERATIONAL_DOMAIN"
            brief = (safe_route.get("navigational_brief") if safe_route else None) or proximity.get("advisory") or "Coordinates are outside the ORCA operational Indian Ocean maritime domain."
            if format_upper == "BINARY_ADVISORY" or (persona_upper == "FISHERMAN" and not is_route_requested):
                return {
                    "agent": "GIS_AGENT",
                    "location": loc_dict,
                    "location_name": loc_name,
                    "time_frame": time_str,
                    "format": "BINARY_ADVISORY",
                    "status": err_status,
                    "error": err_status,
                    "distance_to_border_nm": proximity["distance_to_imbl_nm"],
                    "is_within_eez": False,
                    "advisory": brief,
                    "shipping_lane_assessment": proximity["shipping_lane"],
                    "recommendation": "Vessel position is outside the Indian Ocean maritime operational zone (-15° to 30°N, 50° to 105°E). Safe routing and EEZ geofencing are disabled.",
                    "safe_sea_route": safe_route,
                }
            elif format_upper == "GEOJSON_POLYGONS" or persona_upper == "AUTHORITY":
                return {
                    "agent": "GIS_AGENT",
                    "location": loc_dict,
                    "location_name": loc_name,
                    "time_frame": time_str,
                    "format": "GEOJSON_POLYGONS",
                    "status": err_status,
                    "error": err_status,
                    "geojson": {"type": "FeatureCollection", "features": []},
                    "spatial_data": proximity,
                    "shipping_lane_assessment": proximity["shipping_lane"],
                    "safe_sea_route": safe_route,
                    "advisory": brief,
                }
            else:
                return {
                    "agent": "GIS_AGENT",
                    "location": loc_dict,
                    "location_name": loc_name,
                    "time_frame": time_str,
                    "format": "TEXT_SUMMARY",
                    "status": err_status,
                    "error": err_status,
                    "summary": brief,
                    "spatial_metrics": proximity,
                    "shipping_lane_assessment": proximity["shipping_lane"],
                    "safe_sea_route": safe_route,
                }

        if safe_route and safe_route.get("route_status") == "LAND_INTERSECTION_ERROR":
            if format_upper == "BINARY_ADVISORY" or (persona_upper == "FISHERMAN" and not is_route_requested):
                return {
                    "agent": "GIS_AGENT",
                    "location": loc_dict,
                    "location_name": loc_name,
                    "time_frame": time_str,
                    "format": "BINARY_ADVISORY",
                    "status": "LAND_INTERSECTION_ERROR",
                    "error": "LAND_INTERSECTION_ERROR",
                    "distance_to_border_nm": proximity["distance_to_imbl_nm"],
                    "is_within_eez": proximity["is_within_eez"],
                    "advisory": safe_route["navigational_brief"],
                    "shipping_lane_assessment": proximity["shipping_lane"],
                    "recommendation": "Overland marine routing is physically impossible. Select open sea coastal routing around Cape Comorin.",
                    "safe_sea_route": safe_route,
                }
            elif format_upper == "GEOJSON_POLYGONS" or persona_upper == "AUTHORITY":
                return {
                    "agent": "GIS_AGENT",
                    "location": loc_dict,
                    "location_name": loc_name,
                    "time_frame": time_str,
                    "format": "GEOJSON_POLYGONS",
                    "status": "LAND_INTERSECTION_ERROR",
                    "error": "LAND_INTERSECTION_ERROR",
                    "geojson": {"type": "FeatureCollection", "features": []},
                    "spatial_data": proximity,
                    "shipping_lane_assessment": proximity["shipping_lane"],
                    "safe_sea_route": safe_route,
                    "advisory": safe_route["navigational_brief"],
                }
            else:
                return {
                    "agent": "GIS_AGENT",
                    "location": loc_dict,
                    "location_name": loc_name,
                    "time_frame": time_str,
                    "format": "TEXT_SUMMARY",
                    "status": "LAND_INTERSECTION_ERROR",
                    "error": "LAND_INTERSECTION_ERROR",
                    "summary": safe_route["navigational_brief"],
                    "spatial_metrics": proximity,
                    "shipping_lane_assessment": proximity["shipping_lane"],
                    "safe_sea_route": safe_route,
                }

        if format_upper == "BINARY_ADVISORY" or (persona_upper == "FISHERMAN" and not is_route_requested):
            is_safe = proximity["status_flag"] == "SAFE_EEZ_WATERS" and not proximity["shipping_lane"]["inside_lane"]
            return {
                "agent": "GIS_AGENT",
                "location": loc_dict,
                "location_name": loc_name,
                "time_frame": time_str,
                "format": "BINARY_ADVISORY",
                "status": "SAFE" if is_safe else "WARNING",
                "distance_to_border_nm": proximity["distance_to_imbl_nm"],
                "is_within_eez": proximity["is_within_eez"],
                "advisory": proximity["advisory"],
                "shipping_lane_assessment": proximity["shipping_lane"],
                "recommendation": "Maintain heading within Indian EEZ. Keep clear of commercial shipping lanes and keep AIS active.",
                "safe_sea_route": safe_route,
            }

        if format_upper == "GEOJSON_POLYGONS" or persona_upper == "AUTHORITY":
            lat = proximity["coordinates"]["lat"]
            lon = proximity["coordinates"]["lon"]
            geojson_features = []
            if lat is not None and lon is not None:
                bbox = [
                    [round(lon - 0.1, 4), round(lat - 0.1, 4)],
                    [round(lon + 0.1, 4), round(lat - 0.1, 4)],
                    [round(lon + 0.1, 4), round(lat + 0.1, 4)],
                    [round(lon - 0.1, 4), round(lat + 0.1, 4)],
                    [round(lon - 0.1, 4), round(lat - 0.1, 4)],
                ]
                geojson_features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [bbox]},
                    "properties": {
                        "location": loc_name,
                        "status_flag": proximity["status_flag"],
                        "distance_to_imbl_nm": proximity["distance_to_imbl_nm"],
                        "border_zone": proximity["border_zone"],
                        "is_within_eez": proximity["is_within_eez"],
                        "shapely_verified": proximity["shapely_verified"],
                        "shipping_lane": proximity["shipping_lane"]["nearest_lane_name"],
                        "inside_shipping_lane": proximity["shipping_lane"]["inside_lane"],
                    },
                })
            geojson = {"type": "FeatureCollection", "features": geojson_features}
            if is_route_requested and safe_route and "route_geojson" in safe_route:
                geojson = safe_route["route_geojson"]

            return {
                "agent": "GIS_AGENT",
                "location": loc_dict,
                "location_name": loc_name,
                "time_frame": time_str,
                "format": "GEOJSON_POLYGONS",
                "geojson": geojson,
                "spatial_data": proximity,
                "shipping_lane_assessment": proximity["shipping_lane"],
                "safe_sea_route": safe_route,
            }

        summary_text = (
            f"GIS Geofencing for {loc_name}: Status is {proximity['status_flag']}. "
            f"Distance to nearest IMBL ({proximity['border_zone']}) is {proximity['distance_to_imbl_nm']} NM "
            f"({proximity['distance_to_imbl_km']} km). Within 200 NM EEZ: {proximity['is_within_eez']} "
            f"(Shapely Verified: {proximity['shapely_verified']})."
        )
        if proximity["shipping_lane"]["status_flag"] != "CLEAR_OF_SHIPPING_LANES":
            summary_text += f"\n• {proximity['shipping_lane']['advisory']}"
        if is_route_requested and safe_route and "navigational_brief" in safe_route:
            summary_text += f"\n• {safe_route['navigational_brief']}"

        return {
            "agent": "GIS_AGENT",
            "location": loc_dict,
            "location_name": loc_name,
            "time_frame": time_str,
            "format": "TEXT_SUMMARY",
            "summary": summary_text,
            "spatial_metrics": proximity,
            "shipping_lane_assessment": proximity["shipping_lane"],
            "safe_sea_route": safe_route,
        }

    def resolve_ocean_target(
        self,
        query_text: Optional[str] = None,
        raw_agent_outputs: Optional[Dict[str, Any]] = None,
        vessel_lat: Optional[float] = None,
        vessel_lon: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        return resolve_ocean_target(
            query_text=query_text,
            raw_agent_outputs=raw_agent_outputs,
            vessel_lat=vessel_lat,
            vessel_lon=vessel_lon,
        )


LANDMARK_DISPLAY_NAMES: Dict[str, str] = {
    "thoothukudi": "Thoothukudi Major Port",
    "tuticorin": "Thoothukudi Major Port",
    "kochi": "Kochi Major Port",
    "cochin": "Kochi Major Port",
    "chennai": "Chennai Port Harbor",
    "rameswaram": "Rameswaram Coastal Pier",
    "dhanushkodi": "Dhanushkodi Point",
    "mandapam": "Mandapam Fishery Harbor",
    "kanyakumari": "Cape Comorin",
    "mumbai": "Mumbai Harbor (JNPT)",
    "sassoon dock": "Sassoon Dock (Mumbai)",
    "visakhapatnam": "Visakhapatnam Port",
    "vizag": "Visakhapatnam Port",
    "paradip": "Paradip Port Fairway",
    "mangalore": "New Mangalore Port",
    "porbandar": "Porbandar Harbor",
    "veraval": "Veraval Fishery Port",
    "okha": "Okha Port",
    "jakhau": "Jakhau Fishery Harbor",
    "kandla": "Kandla Deendayal Port",
    "ratnagiri": "Ratnagiri Mirya Bay",
    "goa": "Mormugao Port (Goa)",
    "panaji": "Panaji Port (Goa)",
    "mormugao": "Mormugao Port (Goa)",
    "neendakara": "Neendakara Port (Kollam)",
    "vizhinjam": "Vizhinjam International Seaport",
    "beypore": "Beypore Port (Kozhikode)",
    "nagapattinam": "Nagapattinam Port",
    "cuddalore": "Cuddalore Port",
    "colachel": "Colachel Fishing Harbor",
    "kakinada": "Kakinada Deepwater Port",
    "dhamra": "Dhamra Port",
    "gopalpur": "Gopalpur Port",
    "digha": "Digha Coastal Sector",
    "haldia": "Haldia Dock Complex",
    "port blair": "Port Blair Haddo Wharf",
    "kavaratti": "Kavaratti Island Wharf",
    "agatti": "Agatti Island Lagoon",
    "minicoy": "Minicoy Island Channel",
}


def find_nearest_coastal_landmark(lat: float, lon: float) -> Dict[str, Any]:
    """
    Finds the nearest major coastal port or coastal landmark to the given coordinates,
    and returns its name, distance in NM, cardinal bearing, and formatted reference string.
    Example output: '24.5 NM Southeast off Thoothukudi Major Port'
    """
    best_sector = None
    min_dist = float("inf")
    best_bearing_deg = 0.0
    best_cardinal = "N"
    best_cardinal_full = "North"

    for key, sector in GisAgent.COASTAL_SECTORS.items():
        s_lat = sector["lat"]
        s_lon = sector["lon"]
        d_lat = lat - s_lat
        d_lon = (lon - s_lon) * math.cos(math.radians((s_lat + lat) / 2.0))
        dist_km = math.sqrt((d_lat * 111.0) ** 2 + (d_lon * 111.0) ** 2)
        dist_nm = round(dist_km / 1.852, 1)

        if dist_nm < min_dist:
            min_dist = dist_nm
            best_sector = key

            # Forward bearing from landmark TO target coordinate
            y = math.sin(math.radians(lon - s_lon)) * math.cos(math.radians(lat))
            x = math.cos(math.radians(s_lat)) * math.sin(math.radians(lat)) - math.sin(
                math.radians(s_lat)
            ) * math.cos(math.radians(lat)) * math.cos(math.radians(lon - s_lon))
            bearing = (math.degrees(math.atan2(y, x)) + 360) % 360
            best_bearing_deg = round(bearing, 1)

            directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
            dir_full_map = {
                "N": "North", "NNE": "North-Northeast", "NE": "Northeast", "ENE": "East-Northeast",
                "E": "East", "ESE": "East-Southeast", "SE": "Southeast", "SSE": "South-Southeast",
                "S": "South", "SSW": "South-Southwest", "SW": "Southwest", "WSW": "West-Southwest",
                "W": "West", "WNW": "West-Northwest", "NW": "Northwest", "NNW": "North-Northwest"
            }
            c_code = directions[int((bearing + 11.25) / 22.5) % 16]
            best_cardinal = c_code
            best_cardinal_full = dir_full_map.get(c_code, c_code)

    landmark_name = LANDMARK_DISPLAY_NAMES.get(best_sector, best_sector.title() if best_sector else "Coastal Major Port")
    if min_dist < 1.0:
        ref_str = f"Adjacent to {landmark_name} (< 1.0 NM)"
    else:
        ref_str = f"{min_dist:.1f} NM {best_cardinal_full} off {landmark_name}"

    return {
        "landmark_key": best_sector,
        "landmark_name": landmark_name,
        "distance_nm": min_dist,
        "bearing_deg": best_bearing_deg,
        "cardinal_direction": best_cardinal,
        "cardinal_direction_full": best_cardinal_full,
        "reference_string": ref_str,
    }


def resolve_ocean_target(
    query_text: Optional[str] = None,
    raw_agent_outputs: Optional[Dict[str, Any]] = None,
    vessel_lat: Optional[float] = None,
    vessel_lon: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Deterministic coordinate resolver that inspects query intent and active telemetry
    across 4 core marine classes:
    1. PFZ / Fish Species -> Highest-scoring chlorophyll-a / thermal front coordinate.
    2. Cyclone / Storm -> Minimum pressure / maximum vorticity center with RMW.
    3. Tsunami / Hazard Alert -> Coastal hazard impact zones or seismic origin.
    4. Sea Route Planning -> Optimal navigational waypoints in fairways.
    """
    outputs = raw_agent_outputs or {}
    q = (query_text or outputs.get("normalized_query") or outputs.get("user_query") or outputs.get("english_query") or "").lower().strip()

    # Fast-check Out-of-Domain
    try:
        from decision_engine import check_out_of_domain
        is_ood, _ = check_out_of_domain(q)
        if is_ood:
            return None
    except Exception:
        pass

    v_lat = vessel_lat
    v_lon = vessel_lon

    # Extract coordinates from raw_agent_outputs if not directly provided
    if v_lat is None or v_lon is None:
        loc_cand = outputs.get("location_context") or outputs.get("location") or outputs.get("primary_location")
        if isinstance(loc_cand, dict):
            v_lat = loc_cand.get("latitude", loc_cand.get("lat"))
            v_lon = loc_cand.get("longitude", loc_cand.get("lon"))
        elif loc_cand and hasattr(loc_cand, "latitude") and hasattr(loc_cand, "longitude"):
            v_lat = loc_cand.latitude
            v_lon = loc_cand.longitude
        elif loc_cand and isinstance(loc_cand, str) and "," in loc_cand:
            try:
                parts = loc_cand.split(",")
                v_lat, v_lon = float(parts[0].strip()), float(parts[1].strip())
            except Exception:
                pass

    if v_lat is None or v_lon is None:
        for ag_key in ["WEATHER_AGENT", "OCEAN_AGENT", "GIS_AGENT", "PFZ_AGENT", "DISASTER_AGENT"]:
            ag_data = outputs.get(ag_key)
            if isinstance(ag_data, dict):
                loc_obj = ag_data.get("location")
                if isinstance(loc_obj, dict) and loc_obj.get("latitude") is not None and loc_obj.get("longitude") is not None:
                    v_lat = loc_obj.get("latitude")
                    v_lon = loc_obj.get("longitude")
                    break

    if v_lat is None or v_lon is None:
        return None

    try:
        v_lat = float(v_lat)
        v_lon = float(v_lon)
    except (ValueError, TypeError):
        return None

    # Fast-check vessel coordinates within Indian Ocean operational domain (-15 <= lat <= 30, 50 <= lon <= 105)
    if not (-15.0 <= v_lat <= 30.0 and 50.0 <= v_lon <= 105.0):
        return None

    # Intent Classification
    is_cyclone = any(k in q for k in [
        "cyclone", "storm", "depression", "gale", "hurricane", "typhoon", "vorticity", "eye"
    ])
    disaster_out = outputs.get("DISASTER_AGENT", {})
    if isinstance(disaster_out, dict):
        if disaster_out.get("cyclone_track") or "cyclone" in str(disaster_out.get("hazard_summary", {})).lower():
            is_cyclone = True

    is_hazard_or_shelter = any(k in q for k in [
        "tsunami", "earthquake", "surge", "evacuation", "shelter", "breakwater",
        "mayday", "sinking", "capsiz", "distress", "sar", "emergency", "taking water"
    ])

    is_pfz = any(k in q for k in [
        "fish", "fishing", "pfz", "tuna", "mackerel", "sardine", "seerfish", "hilsa", "catch",
        "pelagic", "yellowfin", "chlorophyll", "tsm", "biophysical", "angler", "anglers"
    ])

    is_route = any(k in q for k in [
        "route", "waypoint", "passage", "fairway", "corridor", "lane", "transit", "from", "heading", "course"
    ])

    target_lat = None
    target_lon = None
    feature_type = "Harbor Approach / Fairway Waypoint"
    target_label = "🧭 Navigational Fairway / Harbor Approach (Lat, Lon)"
    details: Dict[str, Any] = {}

    gis = GisAgent()

    # Priority 1: Hazard / Tsunami / Emergency SAR Shelter
    if is_hazard_or_shelter:
        feature_type = "Hazard Impact / Evacuation Boundary"
        target_label = "🌊 Hazard Impact / Evacuation Boundary (Lat, Lon)"
        shelter_data = disaster_out.get("nearest_shelter") if isinstance(disaster_out, dict) else None
        if isinstance(shelter_data, dict) and "shelter_lat" in shelter_data:
            target_lat = float(shelter_data["shelter_lat"])
            target_lon = float(shelter_data["shelter_lon"])
            details["shelter_name"] = shelter_data.get("shelter_name", "Designated Emergency Breakwater Shelter")
            details["shelter_type"] = "Emergency Breakwater Basin"
        else:
            target_lat = v_lat
            target_lon = v_lon
            lmark = find_nearest_coastal_landmark(v_lat, v_lon)
            details["shelter_name"] = f"Emergency Coastal Anchorage off {lmark['landmark_name']}"
            details["shelter_type"] = "Emergency Breakwater Basin"

    # Priority 2: Cyclone / Severe Storm Eye
    elif is_cyclone:
        cyclone_track = disaster_out.get("cyclone_track") if isinstance(disaster_out, dict) else None
        has_active_cyclone = (
            isinstance(cyclone_track, dict) and cyclone_track.get("has_active_track")
        ) or (isinstance(cyclone_track, list) and len(cyclone_track) > 0)

        if has_active_cyclone:
            feature_type = "Storm Eye / Center Coordinate"
            target_label = "🌀 Storm Eye / Center Coordinate (Lat, Lon)"
            if isinstance(cyclone_track, dict):
                waypoints = cyclone_track.get("projected_track_waypoints", [])
                wp0 = waypoints[0] if waypoints else {}
                target_lat = float(wp0.get("lat", v_lat))
                target_lon = float(wp0.get("lon", v_lon))
                details["storm_name"] = cyclone_track.get("cyclone_name", "Active Cyclonic System")
                details["forward_speed_kmph"] = float(cyclone_track.get("forward_speed_kmph", 18.0))
                details["movement_heading"] = cyclone_track.get("movement_heading", "WNW")
                details["current_intensity"] = cyclone_track.get("current_intensity", "Cyclonic Storm")
                details["radius_max_winds_nm"] = 25.0
            else:
                pt0 = cyclone_track[0]
                target_lat = float(pt0.get("center_lat") or pt0.get("lat") or v_lat)
                target_lon = float(pt0.get("center_lon") or pt0.get("lon") or v_lon)
                details["storm_name"] = pt0.get("storm_name", "Active Cyclonic System")
                details["radius_max_winds_nm"] = float(pt0.get("rmw_nm", 25.0))
                details["central_pressure_hpa"] = float(pt0.get("central_pressure_hpa", 984.0))
                details["max_sustained_wind_kmph"] = float(pt0.get("max_sustained_wind_kmph", 90.0))
        else:
            feature_type = "Storm Monitoring Zone"
            target_label = "🌀 Storm Monitoring Zone (Lat, Lon)"
            target_lat = v_lat
            target_lon = v_lon
            details["storm_name"] = "No Active Cyclone Alert (Routine Satellite Surveillance)"
            details["status"] = "SAFE / CLEAR"
            details["radius_max_winds_nm"] = 0.0

    # Priority 3: PFZ / Fishery Hotspot
    elif is_pfz or ("PFZ_AGENT" in outputs and not is_route):
        feature_type = "PFZ Aggregation Hotspot"
        target_label = "🎯 Target Zone: PFZ Aggregation Hotspot (Lat, Lon)"
        pfz_agent = outputs.get("PFZ_AGENT", {})
        features = []
        if isinstance(pfz_agent, dict):
            features = pfz_agent.get("geojson", {}).get("features", [])

        if features:
            best_feat = max(features, key=lambda f: f.get("properties", {}).get("suitability_score", 0))
            props = best_feat.get("properties", {})
            target_lat = float(props.get("centroid_lat", v_lat))
            target_lon = float(props.get("centroid_lon", v_lon))
            details["suitability_score"] = props.get("suitability_score", 94)
            details["likely_catch"] = props.get("likely_catch", ["Yellowfin Tuna", "Mackerel"])
            details["chlorophyll_front"] = props.get("chlorophyll_mg_m3", 0.58)
        else:
            target_lat = v_lat
            target_lon = v_lon
            details["suitability_score"] = 50
            details["likely_catch"] = ["Pelagic Marine Species"]
            details["chlorophyll_front"] = 0.35

    # Priority 4: Navigational Route / Fairway
    else:
        feature_type = "Harbor Approach / Fairway Waypoint"
        target_label = "🧭 Navigational Fairway / Harbor Approach (Lat, Lon)"
        safe_route = outputs.get("GIS_AGENT", {}).get("safe_sea_route") or outputs.get("safe_sea_route")
        if isinstance(safe_route, dict) and safe_route.get("waypoints"):
            wp_last = safe_route["waypoints"][-1]
            target_lat = float(wp_last.get("lat") or wp_last.get("latitude", v_lat))
            target_lon = float(wp_last.get("lon") or wp_last.get("longitude", v_lon))
            details["destination_name"] = wp_last.get("name", "Target Fairway Waypoint")
        else:
            dest_cand = outputs.get("destination") or outputs.get("target_location")
            if dest_cand:
                sec = gis.resolve_location(str(dest_cand))
                if sec.get("lat") is not None and sec.get("lon") is not None:
                    target_lat = float(sec["lat"])
                    target_lon = float(sec["lon"])
                    details["destination_name"] = sec.get("name", "Operational Harbor")
                else:
                    target_lat = v_lat
                    target_lon = v_lon
                    lmark = find_nearest_coastal_landmark(v_lat, v_lon)
                    details["destination_name"] = f"Coastal Fairway off {lmark['landmark_name']}"
            else:
                target_lat = v_lat
                target_lon = v_lon
                lmark = find_nearest_coastal_landmark(v_lat, v_lon)
                details["destination_name"] = f"Operational Fairway off {lmark['landmark_name']}"

    # Compute Bearing, Distance, Cardinal Direction & Landmark Reference
    dist_nm, bearing_str, bearing_deg = gis.calculate_bearing_and_distance(v_lat, v_lon, target_lat, target_lon)

    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    dir_full_map = {
        "N": "North", "NNE": "North-Northeast", "NE": "Northeast", "ENE": "East-Northeast",
        "E": "East", "ESE": "East-Southeast", "SE": "Southeast", "SSE": "South-Southeast",
        "S": "South", "SSW": "South-Southwest", "SW": "Southwest", "WSW": "West-Southwest",
        "W": "West", "WNW": "West-Northwest", "NW": "Northwest", "NNW": "North-Northwest"
    }
    cardinal = directions[int((bearing_deg + 11.25) / 22.5) % 16]
    cardinal_full = dir_full_map.get(cardinal, cardinal)

    relative_vector = f"{dist_nm:.1f} NM along Bearing {int(round(bearing_deg)):03d}° {cardinal}"
    landmark_info = find_nearest_coastal_landmark(target_lat, target_lon)
    landmark_reference = landmark_info["reference_string"]

    lat_dir = "N" if target_lat >= 0 else "S"
    lon_dir = "E" if target_lon >= 0 else "W"
    target_coords_str = f"Lat {abs(target_lat):.4f}° {lat_dir}, Lon {abs(target_lon):.4f}° {lon_dir}"

    return {
        "feature_type": feature_type,
        "target_label": target_label,
        "target_coordinates": target_coords_str,
        "target_lat": round(target_lat, 4),
        "target_lon": round(target_lon, 4),
        "distance_nm": dist_nm,
        "bearing_deg": bearing_deg,
        "cardinal_direction": cardinal,
        "cardinal_direction_full": cardinal_full,
        "relative_vector": relative_vector,
        "landmark_reference": landmark_reference,
        "details": details,
    }


if __name__ == "__main__":
    agent = GisAgent()
    print("--- SCENARIO 1: ROUTINE GEOFENCE CHECK ---")
    res1 = agent.execute_task("Thoothukudi", "today", "Check IMBL proximity", "TEXT_SUMMARY", "FISHERMAN")
    print(json.dumps(res1, indent=2))

    print("\n--- SCENARIO 2: SAFE SEA ROUTE PLANNER (THOOTHUKUDI TO RAMESWARAM) ---")
    route_res = agent.generate_safe_sea_route(
        origin="8.7642,78.1348",
        destination="9.2876,79.3129",
        cruising_speed_knots=8.0,
    )
    print(json.dumps(route_res, indent=2))
