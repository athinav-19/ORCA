"""
ml/data_ingestion/openmeteo_weather_real.py
Authoritative ERA5 Historical Weather Ingestion via Open-Meteo Archive API.
Classification: [REANALYSIS] — ECMWF ERA5 Global Atmospheric Reanalysis.

Coverage: 1980-01-01 00:00:00 to 2026-09-16 23:00:00 (Hourly resampled to 6-hourly).
Indian Maritime Sectors:
  1. Gulf of Mannar (8.76N, 78.25E)
  2. Konkan / Mumbai Coast (18.92N, 72.83E)
  3. Malabar / Kochi Coast (9.93N, 76.26E)
  4. Coromandel / Chennai Coast (13.08N, 80.27E)
  5. Goa Offshore (15.41N, 73.80E)
  6. North Bay of Bengal (20.31N, 86.61E)

Variables:
  - wind_speed_kmh (km/h)
  - wind_dir_deg (degrees)
  - u_wind_kmh, v_wind_kmh (meteorological components)
  - surface_pressure_hpa (hPa)
  - precipitation_mm (mm)
  - relative_humidity_pct (%)
  - temperature_2m_c (Celsius)
  - wave_height_m (Pierson-Moskowitz oceanographic derivation from wind speed, labeled DERIVED)
"""

import os
import time
import json
import math
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime

CACHE_DIR = os.path.join("data", "real_weather_era5")

ORCA_STATIONS = [
    {"name": "Gulf_of_Mannar",      "lat":  8.76, "lon": 78.25},
    {"name": "Konkan_Mumbai",        "lat": 18.92, "lon": 72.83},
    {"name": "Malabar_Kochi",        "lat":  9.93, "lon": 76.26},
    {"name": "Coromandel_Chennai",   "lat": 13.08, "lon": 80.27},
    {"name": "Goa_Offshore",         "lat": 15.41, "lon": 73.80},
    {"name": "North_Bay_of_Bengal",  "lat": 20.31, "lon": 86.61},
]

CHUNK_RANGES = [
    ("1980-01-01", "1989-12-31", "1980s"),
    ("1990-01-01", "1999-12-31", "1990s"),
    ("2000-01-01", "2009-12-31", "2000s"),
    ("2010-01-01", "2019-12-31", "2010s"),
    ("2020-01-01", "2026-09-16", "2020_2026"),
]

HOURLY_VARS = [
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "precipitation",
    "relative_humidity_2m",
    "temperature_2m",
]


def fetch_era5_chunk(lat: float, lon: float, start_date: str, end_date: str, max_retries: int = 5) -> Optional[pd.DataFrame]:
    """Fetch one chunk of ERA5 reanalysis from Open-Meteo Archive API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "wind_speed_unit": "kmh",
        "timezone": "UTC",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": "ORCA-RealData-Pipeline/2.0 (SIH2026; Ocean Meteorological Reanalysis)"}

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            
            h = data["hourly"]
            df = pd.DataFrame({
                "datetime": pd.to_datetime(h["time"]),
                "wind_speed_kmh": h["wind_speed_10m"],
                "wind_dir_deg": h["wind_direction_10m"],
                "surface_pressure_hpa": h["surface_pressure"],
                "precipitation_mm": h["precipitation"],
                "relative_humidity_pct": h["relative_humidity_2m"],
                "temperature_2m_c": h["temperature_2m"],
            })
            return df
        except Exception as e:
            wait_time = 2 * (attempt + 1)
            print(f"      [Attempt {attempt+1}/{max_retries} failed] {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

    return None


def download_all_real_era5(force_refresh: bool = False) -> pd.DataFrame:
    """Download and cache genuine ERA5 reanalysis from 1980-01-01 to 2026-09-16."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    all_station_dfs = []

    print("================================================================================")
    print("INGESTING GENUINE ERA5 REANALYSIS (1980-01-01 TO 2026-09-16) FOR 6 MARITIME NODES")
    print("================================================================================")

    for stn in ORCA_STATIONS:
        name = stn["name"]
        lat, lon = stn["lat"], stn["lon"]
        stn_cache = os.path.join(CACHE_DIR, f"era5_6h_{name}_1980_2026.parquet")

        if not force_refresh and os.path.exists(stn_cache):
            df_stn = pd.read_parquet(stn_cache)
            print(f"[CACHE HIT] {name}: Loaded {len(df_stn):,} 6-hourly fixes ({df_stn['datetime'].min()} to {df_stn['datetime'].max()})")
            all_station_dfs.append(df_stn)
            continue

        print(f"\n[DOWNLOADING ERA5] Station: {name} (Lat: {lat}, Lon: {lon})")
        chunk_dfs = []

        for start_dt, end_dt, tag in CHUNK_RANGES:
            chunk_file = os.path.join(CACHE_DIR, f"raw_hourly_{name}_{tag}.parquet")
            if not force_refresh and os.path.exists(chunk_file):
                df_chunk = pd.read_parquet(chunk_file)
                print(f"  -> Chunk {tag}: Cached ({len(df_chunk):,} hourly fixes)")
            else:
                print(f"  -> Fetching {start_dt} to {end_dt}...", end="", flush=True)
                t0 = time.time()
                df_chunk = fetch_era5_chunk(lat, lon, start_dt, end_dt)
                if df_chunk is None:
                    raise RuntimeError(f"Failed to fetch ERA5 data for {name} ({tag})")
                df_chunk.to_parquet(chunk_file, index=False)
                t1 = time.time()
                print(f" Done in {round(t1-t0, 1)}s ({len(df_chunk):,} hours)")
                time.sleep(1.0) # Polite rate limiting

            chunk_dfs.append(df_chunk)

        full_hourly = pd.concat(chunk_dfs, ignore_index=True)
        full_hourly = full_hourly.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

        # Resample to 6-hourly fixes (00:00, 06:00, 12:00, 18:00 UTC) for precise forecast horizon alignment
        full_hourly = full_hourly.set_index("datetime")
        df_6h = full_hourly.resample("6h").first().reset_index()

        # Add spatial identifiers & meteorological wind components
        df_6h["station"] = name
        df_6h["latitude"] = lat
        df_6h["longitude"] = lon
        df_6h["data_source"] = "ECMWF_ERA5_OpenMeteo"
        df_6h["data_classification"] = "REANALYSIS"

        # Meteorological U and V components (km/h)
        rad = np.radians(df_6h["wind_dir_deg"])
        df_6h["u_wind_kmh"] = - df_6h["wind_speed_kmh"] * np.sin(rad)
        df_6h["v_wind_kmh"] = - df_6h["wind_speed_kmh"] * np.cos(rad)

        # Oceanographic Pierson-Moskowitz significant wave height derivation (m)
        # H_s ≈ 0.0246 * (U_10 in m/s)^2 = 0.0246 * (wind_kmh / 3.6)^2
        w_ms = df_6h["wind_speed_kmh"] / 3.6
        df_6h["wave_height_m"] = np.clip(0.0246 * (w_ms ** 2) + 0.5, 0.4, 14.0).round(2)
        df_6h["wave_period_s"] = np.clip(3.8 + 1.1 * df_6h["wave_height_m"], 3.0, 16.0).round(1)
        df_6h["wave_classification"] = "DERIVED_PIERSON_MOSKOWITZ"

        df_6h.to_parquet(stn_cache, index=False)
        print(f"  [SAVED] {name}: {len(df_6h):,} 6-hourly observations saved to {stn_cache}")
        all_station_dfs.append(df_6h)

    combined_df = pd.concat(all_station_dfs, ignore_index=True)
    combined_df = combined_df.sort_values("datetime").reset_index(drop=True)
    return combined_df


def get_real_weather_inventory(combined_df: pd.DataFrame) -> Dict:
    """Generate comprehensive audit metadata for the real ERA5 dataset."""
    cache_files = [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR) if f.endswith(".parquet")]
    total_bytes = sum(os.path.getsize(f) for f in cache_files)

    return {
        "dataset_name": "ECMWF ERA5 Reanalysis Indian Maritime 1980-2026",
        "source": "European Centre for Medium-Range Weather Forecasts (ECMWF) via Open-Meteo Archive API",
        "source_url": "https://archive-api.open-meteo.com/v1/archive",
        "provenance_classification": "REANALYSIS",
        "actual_start_date": str(combined_df["datetime"].min()),
        "actual_end_date": str(combined_df["datetime"].max()),
        "temporal_resolution": "6-hourly (derived from 1-hourly raw reanalysis)",
        "spatial_resolution": "0.25 x 0.25 degree grid point sampling at 6 Indian EEZ maritime nodes",
        "number_of_nodes": len(ORCA_STATIONS),
        "nodes": [s["name"] for s in ORCA_STATIONS],
        "total_observations": len(combined_df),
        "number_of_files": len(cache_files),
        "total_storage_bytes": total_bytes,
        "total_storage_mb": round(total_bytes / (1024 * 1024), 2),
        "variables": [
            "wind_speed_kmh (REANALYSIS)",
            "wind_dir_deg (REANALYSIS)",
            "u_wind_kmh (REANALYSIS)",
            "v_wind_kmh (REANALYSIS)",
            "surface_pressure_hpa (REANALYSIS)",
            "precipitation_mm (REANALYSIS)",
            "relative_humidity_pct (REANALYSIS)",
            "temperature_2m_c (REANALYSIS)",
            "wave_height_m (DERIVED_PIERSON_MOSKOWITZ)",
            "wave_period_s (DERIVED_PIERSON_MOSKOWITZ)"
        ],
        "missing_data_percentage": {
            "wind_speed_kmh": round(100.0 * combined_df["wind_speed_kmh"].isna().sum() / len(combined_df), 3),
            "surface_pressure_hpa": round(100.0 * combined_df["surface_pressure_hpa"].isna().sum() / len(combined_df), 3),
            "precipitation_mm": round(100.0 * combined_df["precipitation_mm"].isna().sum() / len(combined_df), 3),
            "wave_height_m": round(100.0 * combined_df["wave_height_m"].isna().sum() / len(combined_df), 3)
        },
        "target_horizons_hours": [6, 12, 24, 48, 72]
    }


if __name__ == "__main__":
    df = download_all_real_era5()
    inv = get_real_weather_inventory(df)
    print("\n=== INVENTORY SUMMARY ===")
    print(json.dumps(inv, indent=2))
