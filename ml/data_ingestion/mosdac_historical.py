"""
mosdac_historical.py - ISRO MOSDAC Historical Satellite Data Ingestion
SIH 2026 Problem Statement SIH26176

Scans, validates, and indexes available ISRO MOSDAC HDF5 satellite products
(INSAT-3DR LST, HEM, OLR, CTP; Oceansat-3 OCM, OSCAT; SARAL SWH)
for machine learning training and verification.
"""

import os
import glob
import re
import datetime
from typing import List, Dict, Any, Optional

try:
    import h5py
except ImportError:
    h5py = None

MOSDAC_CACHE_DIR = os.path.join("data", "mosdac_cache")


def is_file_valid_hdf5(filepath: str) -> bool:
    """Verifies that an HDF5 file is uncorrupted and readable."""
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 1024:
        return False
    if filepath.endswith(".part"):
        return False
    if h5py is None:
        return True
    try:
        with h5py.File(filepath, "r") as f:
            return len(f.keys()) > 0
    except Exception:
        return False


def index_mosdac_cache() -> List[Dict[str, Any]]:
    """
    Scans the local MOSDAC cache and indexes every valid satellite product.
    Extracts satellite mission, product type, acquisition date/time, and size.
    """
    records = []
    if not os.path.exists(MOSDAC_CACHE_DIR):
        return records

    pattern = os.path.join(MOSDAC_CACHE_DIR, "*.h5")
    for fpath in glob.glob(pattern):
        if not is_file_valid_hdf5(fpath):
            continue

        fname = os.path.basename(fpath)
        size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)

        # Typical filenames:
        # 3RIMG_09SEP2026_0745_L2B_CTP_V01R00.h5
        # 3OSCAT_09SEP2026_0745_L2B_WND_V01R00.h5
        # SARAL_09SEP2026_0745_L2P_SWH_V01R00.h5
        match = re.search(r"([A-Z0-9]+)_(\d{2}[A-Z]{3}\d{4})_(\d{4})_([A-Z0-9]+)_([A-Z0-9]+)", fname)
        if match:
            satellite = match.group(1)
            date_str = match.group(2)
            time_str = match.group(3)
            level = match.group(4)
            product = match.group(5)
            try:
                dt = datetime.datetime.strptime(f"{date_str}_{time_str}", "%d%b%Y_%H%M")
            except Exception:
                dt = datetime.datetime.fromtimestamp(os.path.getmtime(fpath))
        else:
            satellite = "UNKNOWN"
            product = fname.split("_")[0]
            dt = datetime.datetime.fromtimestamp(os.path.getmtime(fpath))

        records.append({
            "filename": fname,
            "filepath": fpath,
            "satellite": satellite,
            "product": product,
            "datetime": dt,
            "size_mb": size_mb,
        })

    records.sort(key=lambda x: x["datetime"])
    return records


if __name__ == "__main__":
    indexed = index_mosdac_cache()
    print(f"[MOSDAC Index] Found {len(indexed)} valid HDF5 satellite products in cache.")
    for r in indexed[:5]:
        print(f"  - {r['filename']} ({r['product']}, {r['size_mb']} MB, {r['datetime']})")

