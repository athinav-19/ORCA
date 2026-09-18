"""
ibtracs_cyclone.py - Official NOAA/NCEI IBTrACS Cyclone Ingestion for North Indian Ocean
SIH 2026 Problem Statement SIH26176

Downloads, caches, and parses authoritative historical tropical cyclone best-tracks
from the NOAA International Best Track Archive for Climate Stewardship (IBTrACS)
for the North Indian Ocean (Arabian Sea and Bay of Bengal), covering 2005-2025.
"""

import os
import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List

IBTRACS_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
LOCAL_CACHE_PATH = os.path.join("data", "ibtracs_NI.csv")


def ensure_ibtracs_dataset(force_download: bool = False, timeout_sec: int = 30) -> str:
    """
    Downloads NOAA IBTrACS North Indian Ocean archive if not present.
    Returns path to valid CSV file.
    """
    os.makedirs(os.path.dirname(LOCAL_CACHE_PATH), exist_ok=True)

    if not force_download and os.path.exists(LOCAL_CACHE_PATH) and os.path.getsize(LOCAL_CACHE_PATH) > 100000:
        print(f"[IBTrACS OK] Local dataset verified at {LOCAL_CACHE_PATH} ({os.path.getsize(LOCAL_CACHE_PATH) / 1024:.1f} KB)")
        return LOCAL_CACHE_PATH

    print(f"[IBTrACS Download] Fetching official NOAA IBTrACS dataset from {IBTRACS_URL}...")
    req = urllib.request.Request(IBTRACS_URL, headers={"User-Agent": "ORCA-Marine-System/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as response, open(LOCAL_CACHE_PATH, "wb") as out_file:
            data = response.read()
            out_file.write(data)
        print(f"[IBTrACS OK] Downloaded {len(data) / 1024:.1f} KB to {LOCAL_CACHE_PATH}")
        return LOCAL_CACHE_PATH
    except Exception as e:
        print(f"[IBTrACS Warning] Download failed: {e}")
        if os.path.exists(LOCAL_CACHE_PATH):
            print("[IBTrACS Fallback] Reusing existing local file.")
            return LOCAL_CACHE_PATH
        raise RuntimeError(f"Failed to obtain NOAA IBTrACS dataset: {e}")


def load_cleaned_cyclone_records(
    start_year: int = 2005,
    end_year: int = 2025,
) -> pd.DataFrame:
    """
    Loads and cleans IBTrACS records for the North Indian Ocean basin.
    Standardizes units:
    - Wind: km/h (converted from knots)
    - Pressure: hPa / mb
    - Coordinates: degrees North, degrees East
    """
    filepath = ensure_ibtracs_dataset()

    # Line 1 is headers, Line 2 is units description
    df = pd.read_csv(
        filepath,
        skiprows=[1],
        low_memory=False,
        na_values=[" ", "", "NOT_NAMED"],
    )

    # Convert ISO_TIME to datetime
    df["datetime"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    df = df.dropna(subset=["datetime", "LAT", "LON"])

    # Filter year range
    df["year"] = df["datetime"].dt.year
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

    # Cast coordinates
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")

    # Select best wind (WMO_WIND or NEWDELHI_WIND or USA_WIND in knots)
    wmo_wind = pd.to_numeric(df["WMO_WIND"], errors="coerce")
    imd_wind = pd.to_numeric(df["NEWDELHI_WIND"], errors="coerce")
    usa_wind = pd.to_numeric(df["USA_WIND"], errors="coerce")
    wind_kts = imd_wind.fillna(wmo_wind).fillna(usa_wind).fillna(30.0)
    df["wind_speed_kmh"] = wind_kts * 1.852

    # Select best pressure (WMO_PRES or NEWDELHI_PRES or USA_PRES in hPa)
    wmo_pres = pd.to_numeric(df["WMO_PRES"], errors="coerce")
    imd_pres = pd.to_numeric(df["NEWDELHI_PRES"], errors="coerce")
    usa_pres = pd.to_numeric(df["USA_PRES"], errors="coerce")
    df["surface_pressure_hpa"] = imd_pres.fillna(wmo_pres).fillna(usa_pres).fillna(1005.0)

    # Forward speed and direction
    df["storm_speed_kmh"] = pd.to_numeric(df["STORM_SPEED"], errors="coerce").fillna(15.0) * 1.852
    df["storm_dir_deg"] = pd.to_numeric(df["STORM_DIR"], errors="coerce").fillna(315.0)  # NW default

    # IMD Cyclone Classification
    # Depression: 31-49 km/h (17-27 kts)
    # Deep Depression: 50-61 km/h (28-33 kts)
    # Cyclonic Storm: 62-88 km/h (34-47 kts)
    # Severe Cyclonic Storm: 89-117 km/h (48-63 kts)
    # Very Severe Cyclonic Storm: 118-166 km/h (64-89 kts)
    # Extremely Severe: 167-221 km/h (90-119 kts)
    # Super Cyclonic Storm: >= 222 km/h (>= 120 kts)
    def classify_imd(wind_kmh: float) -> str:
        if wind_kmh >= 118:
            return "SEVERE_CYCLONE"
        elif wind_kmh >= 62:
            return "CYCLONIC_STORM"
        elif wind_kmh >= 45:
            return "DEPRESSION"
        else:
            return "LOW_PRESSURE"

    df["hazard_class"] = df["wind_speed_kmh"].apply(classify_imd)
    df["is_hazard"] = (df["wind_speed_kmh"] >= 45.0).astype(int)
    df["is_severe"] = (df["wind_speed_kmh"] >= 62.0).astype(int)

    # Sort chronologically (Crucial for NO DATA LEAKAGE)
    df = df.sort_values("datetime").reset_index(drop=True)

    print(f"[IBTrACS OK] Loaded {len(df)} storm observations across {df['NAME'].nunique()} named systems ({start_year}-{end_year}).")
    return df


if __name__ == "__main__":
    records = load_cleaned_cyclone_records()
    print(records[["datetime", "NAME", "LAT", "LON", "wind_speed_kmh", "surface_pressure_hpa", "hazard_class"]].head(10))

