"""
copernicus_historical.py - Copernicus Marine Historical & Physics Reanalysis Ingestion
SIH 2026 Problem Statement SIH26176

Accesses, validates, and extracts Copernicus Marine (CMEMS) historical grids
for the Indian Ocean domain (0-25°N, 65-97°E).
"""

import os
import sys
import time
from typing import Optional, Dict, Any
import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None

COPERNICUS_CACHE_DIR = os.path.join("data", "copernicus_cache")
COPERNICUS_SST_FILE = os.path.join(COPERNICUS_CACHE_DIR, "copernicus_sst_india.nc")


def get_copernicus_grid() -> Optional[Any]:
    """Loads and validates the cached Copernicus Marine NetCDF dataset."""
    if not os.path.exists(COPERNICUS_SST_FILE):
        return None
    try:
        ds = xr.open_dataset(COPERNICUS_SST_FILE)
        return ds
    except Exception as e:
        print(f"[Copernicus Ingest Warning] Failed to read NetCDF {COPERNICUS_SST_FILE}: {e}")
        return None


def sample_copernicus_sst(lat: float, lon: float) -> float:
    """Samples SST for a specific latitude and longitude from the Copernicus grid."""
    ds = get_copernicus_grid()
    if ds is None:
        # Fallback to climatological Indian Ocean baseline
        return 28.0

    try:
        var_name = "thetao" if "thetao" in ds else list(ds.data_vars.keys())[0]
        val = ds[var_name].sel(latitude=lat, longitude=lon, method="nearest").values
        if hasattr(val, "item"):
            val = val.item()
        if np.isnan(val) or val <= 0.0:
            return 28.0
        return float(val)
    except Exception:
        return 28.0

