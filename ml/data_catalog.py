"""
data_catalog.py - Comprehensive Dataset Catalog Generator for ORCA ML
SIH 2026 Problem Statement SIH26176

Inspects, documents, and catalogues all historical and operational datasets across
MOSDAC, Copernicus Marine, NOAA IBTrACS, and bathymetric grids.
"""

import os
import csv
import glob
from typing import List, Dict, Any

CATALOG_CSV_PATH = os.path.join("ml", "data_catalog.csv")

DATASET_DEFINITIONS: List[Dict[str, str]] = [
    {
        "dataset": "MOSDAC_INSAT3DR_LST",
        "source": "ISRO MOSDAC",
        "variables": "Sea Surface Temperature (SST)",
        "start_date": "2016-09-08",
        "end_date": "2026-NRT",
        "resolution": "4.0 km",
        "format": "HDF5 (.h5)",
        "coverage": "Indian Ocean (0-40N, 40-110E)",
        "download_status": "AVAILABLE_CACHE",
        "quality": "Level-2B Validated Thermal Radiance",
        "model_usage": "Disaster, Weather, PFZ",
    },
    {
        "dataset": "MOSDAC_INSAT3DR_HEM",
        "source": "ISRO MOSDAC",
        "variables": "Hydro-Estimator Rainfall Rate (mm/h)",
        "start_date": "2016-09-08",
        "end_date": "2026-NRT",
        "resolution": "4.0 km",
        "format": "HDF5 (.h5)",
        "coverage": "Indian Ocean & South Asia",
        "download_status": "AVAILABLE_CACHE",
        "quality": "Level-2B Hydro-Estimator Precipitation",
        "model_usage": "Disaster, Weather",
    },
    {
        "dataset": "MOSDAC_INSAT3DR_OLR",
        "source": "ISRO MOSDAC",
        "variables": "Outgoing Longwave Radiation (OLR, W/m2)",
        "start_date": "2016-09-08",
        "end_date": "2026-NRT",
        "resolution": "4.0 km",
        "format": "HDF5 (.h5)",
        "coverage": "Indian Ocean & South Asia",
        "download_status": "AVAILABLE_CACHE",
        "quality": "Level-2B Deep Convection Proxy",
        "model_usage": "Disaster",
    },
    {
        "dataset": "MOSDAC_INSAT3DR_CTP",
        "source": "ISRO MOSDAC",
        "variables": "Cloud Top Pressure (CTP, hPa), Cloud Top Temp",
        "start_date": "2016-09-08",
        "end_date": "2026-NRT",
        "resolution": "4.0 km",
        "format": "HDF5 (.h5)",
        "coverage": "Indian Ocean",
        "download_status": "AVAILABLE_CACHE",
        "quality": "Level-2B Atmospheric Barometric Layer",
        "model_usage": "Weather, Disaster",
    },
    {
        "dataset": "MOSDAC_OCEANSAT3_OCM",
        "source": "ISRO MOSDAC",
        "variables": "Chlorophyll-a (mg/m3), Kd490 Turbidity, TSM",
        "start_date": "2022-11-26",
        "end_date": "2026-NRT",
        "resolution": "1.0 km / 360m",
        "format": "HDF5 (.h5)",
        "coverage": "Arabian Sea & Bay of Bengal",
        "download_status": "AVAILABLE_CACHE",
        "quality": "Level-2B Ocean Color Radiometry",
        "model_usage": "PFZ",
    },
    {
        "dataset": "MOSDAC_OCEANSAT3_OSCAT",
        "source": "ISRO MOSDAC",
        "variables": "10m Ocean Surface Wind Speed & Direction",
        "start_date": "2022-11-26",
        "end_date": "2026-NRT",
        "resolution": "25 km / 50 km swath",
        "format": "HDF5 (.h5)",
        "coverage": "Indian Ocean Basin",
        "download_status": "AVAILABLE_CACHE",
        "quality": "Level-2B Scatterometer Wind Vectors",
        "model_usage": "Weather, Disaster, PFZ",
    },
    {
        "dataset": "MOSDAC_SARAL_SWH",
        "source": "ISRO / CNES MOSDAC",
        "variables": "Significant Wave Height (SWH, m), Wind Speed",
        "start_date": "2013-02-25",
        "end_date": "2026-NRT",
        "resolution": "Along-track 1 Hz (~7 km)",
        "format": "HDF5 (.h5)",
        "coverage": "Indian Ocean Basin",
        "download_status": "AVAILABLE_CACHE",
        "quality": "Level-2P Altimeter Significant Wave Height",
        "model_usage": "Weather, Disaster",
    },
    {
        "dataset": "COPERNICUS_MARINE_CMEMS_PHYSICS",
        "source": "Copernicus Marine Service",
        "variables": "Potential Temperature (thetao/SST), Currents (uo, vo), Sea Level (zos)",
        "start_date": "2005-01-01",
        "end_date": "2026-NRT",
        "resolution": "0.083 deg (~9 km)",
        "format": "NetCDF4 (.nc)",
        "coverage": "India Bounding Box (0-25N, 65-97E)",
        "download_status": "SYNCED_CACHE",
        "quality": "Level-4 Multi-Mission Reanalysis / Physics",
        "model_usage": "Weather, PFZ, Disaster",
    },
    {
        "dataset": "NOAA_NCEI_IBTRACS_NORTH_INDIAN_OCEAN",
        "source": "NOAA / NCEI / IMD",
        "variables": "Cyclone Tracks, Central Pressure, Sustained Wind Speed, Category",
        "start_date": "2005-01-01",
        "end_date": "2025-12-31",
        "resolution": "3-hourly / 6-hourly temporal tracks",
        "format": "CSV / NetCDF",
        "coverage": "North Indian Ocean (Bay of Bengal & Arabian Sea)",
        "download_status": "VERIFIED_ACCESSIBLE",
        "quality": "Authoritative Official Best Track (WMO)",
        "model_usage": "Disaster (Ground Truth)",
    },
    {
        "dataset": "GEBCO_INDIAN_BATHYMETRY",
        "source": "GEBCO / INCOIS / GIS Agent",
        "variables": "Bathymetry Depth (m), Distance to Coast (nm), Seabed Slope",
        "start_date": "2005-01-01",
        "end_date": "2026-NRT",
        "resolution": "0.05 deg (~5 km)",
        "format": "GeoTIFF / GeoJSON / NetCDF",
        "coverage": "Indian EEZ & Territorial Waters",
        "download_status": "LOCAL_BUILTIN",
        "quality": "Standard Maritime Bathymetric Model",
        "model_usage": "PFZ, Disaster",
    },
]


def generate_data_catalog(output_path: str = CATALOG_CSV_PATH) -> str:
    """Generates ml/data_catalog.csv documenting all datasets and their operational status."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Audit local cache files to reflect actual live status
    mosdac_files = glob.glob(os.path.join("data", "mosdac_cache", "*.*"))
    copernicus_files = glob.glob(os.path.join("data", "copernicus_cache", "*.*"))

    headers = [
        "dataset",
        "source",
        "variables",
        "start_date",
        "end_date",
        "resolution",
        "format",
        "coverage",
        "download_status",
        "quality",
        "model_usage",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in DATASET_DEFINITIONS:
            # Check local file presence to update status
            ds = row["dataset"]
            if "MOSDAC" in ds:
                short_code = ds.split("_")[-1]
                matched = [f for f in mosdac_files if short_code in f and not f.endswith(".part")]
                if matched:
                    row["download_status"] = f"CACHED ({len(matched)} files)"
                else:
                    row["download_status"] = "API_AVAILABLE"
            elif "COPERNICUS" in ds:
                if copernicus_files:
                    row["download_status"] = f"CACHED ({len(copernicus_files)} files)"
            writer.writerow(row)

    print(f"[OK] Generated Dataset Catalog at: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_data_catalog()

