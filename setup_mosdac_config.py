"""
setup_mosdac_config.py - MOSDAC Download API Configuration Generator
ISRO SIH Problem Statement 176: Marine Multi-Agent System

Generates the exact config.json schema required by ISRO's MOSDAC Data Download API (mdapi.py).
Prepares cache and logging directories for the shadow cache worker.
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()


# Target ISRO MOSDAC Satellite Product Suites (9 Product Suites)
SUPPORTED_DATASETS = [
    "3RIMG_L2B_LST",  # INSAT-3DR Imager: Land & Sea Surface Temperature (PFZ & Ocean)
    "3RIMG_L2B_HEM",  # INSAT-3DR Hydro-Estimator: High-Resolution Precipitation / Rainfall (Weather & Disaster)
    "3RIMG_L2B_OLR",  # INSAT-3DR Outgoing Longwave Radiation: Cyclone & Deep Convection (Disaster)
    "3RIMG_L2B_CTP",  # INSAT-3DR Cloud Top Pressure: Atmospheric Fronts & Barometric Tendency (Weather)
    "3OOCM_L2B_OCM",  # Oceansat-3 (EOS-06) Ocean Color Monitor: Chlorophyll-a, Turbidity Kd490 & TSM (PFZ)
    "3OSCAT_L2B_WND", # Oceansat-3 Scatterometer: 10m Ocean Surface Wind Speed & Direction (Weather & Ocean)
    "SARAL_L2P_SWH",  # SARAL-AltiKa Altimeter: Significant Wave Height & Geostrophic Currents (Ocean)
    "3RIMG_L2B_FOG",  # INSAT-3DR Fog Product: Sea & Coastal Fog Coverage / Navigation Visibility (Weather & GIS)
    "3RIMG_L2B_IMC",  # INSAT-3DR Insolation: Daily Solar Energy Flux & Green Vessel Range (Weather & Decision)
]


def generate_mosdac_config(
    output_path: str = "config.json",
    dataset_id: str = "3RIMG_L2B_LST",
) -> dict:
    """
    Creates config.json matching the exact ISRO MOSDAC API specification.
    Pulls credentials from .env if available, otherwise sets template placeholders.
    """
    username = os.getenv("MOSDAC_USERNAME", "YOUR_USERNAME")
    password = os.getenv("MOSDAC_PASSWORD", "YOUR_PASSWORD")

    cache_dir = "./data/mosdac_cache"
    log_dir = "./logs"

    # Ensure required target directories exist
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    config_schema = {
        "user_credentials": {
            "username/email": username,
            "username": username,
            "password": password,
        },
        "search_parameters": {
            "datasetId": dataset_id,
            "startTime": "",
            "endTime": "",
            "count": os.getenv("MOSDAC_SYNC_COUNT", "5"),
            "boundingBox": "65.0,5.0,97.0,30.0",
            "gId": "",
        },
        "supported_datasets": SUPPORTED_DATASETS,
        "download_settings": {
            "download_path": cache_dir,
            "organize_by_date": False,
            "skip_user_input": True,
            "generate_error_logs": True,
            "error_logs_dir": log_dir,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config_schema, f, indent=4)

    print(f"[OK] Generated MOSDAC configuration at: {os.path.abspath(output_path)} for dataset {dataset_id}")
    print(f"[i] Cache directory initialized at  : {os.path.abspath(cache_dir)}")
    print(f"[i] Logs directory initialized at   : {os.path.abspath(log_dir)}")
    return config_schema


if __name__ == "__main__":
    generate_mosdac_config()
