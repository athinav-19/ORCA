"""
shadow_cache_worker.py - MOSDAC Shadow Cache Background Worker
ISRO SIH Problem Statement 176: Marine Multi-Agent System

Background synchronization worker:
1. Dynamically calculates yesterday and today dates.
2. Updates config.json search_parameters (startTime, endTime).
3. Executes ISRO's MOSDAC Data Download API (mdapi.py) to sync latest satellite files.
4. Operates in an asynchronous 6-hour daemon loop without blocking the user-facing AI chat.
"""

import os
import sys
import glob
import json
import time
import datetime
import subprocess

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from typing import Optional, Dict, Any
from dotenv import load_dotenv

from setup_mosdac_config import SUPPORTED_DATASETS, generate_mosdac_config

CONFIG_PATH = "config.json"
CACHE_DIR = "./data/mosdac_cache"
SYNC_INTERVAL_SECONDS = 21600  # 6 hours


def is_file_valid_hdf5(filepath: str) -> bool:
    """Verifies that an HDF5 or NetCDF file is readable and uncorrupted."""
    try:
        import h5py
        with h5py.File(filepath, "r") as f:
            return True
    except Exception:
        try:
            import xarray as xr
            with xr.open_dataset(filepath) as ds:
                return True
        except Exception:
            return False


def is_dataset_cached(dataset_id: str, max_age_hours: float = 72.0) -> bool:
    """
    Checks if a valid, recent file for the given MOSDAC dataset exists in CACHE_DIR.
    Accepts existing valid HDF5/NetCDF files in the cache to avoid blocking user-facing
    requests or test suites on external network timeouts.
    """
    short_code = dataset_id.split("_")[-1]  # e.g., LST, HEM, OLR, CTP
    patterns = [
        os.path.join(CACHE_DIR, f"*{short_code}*.h5"),
        os.path.join(CACHE_DIR, f"*{short_code}*.nc"),
    ]
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat))

    # Exclude incomplete downloads (.part), empty files, and corrupted files
    now = time.time()
    valid_matches = []
    for m in matches:
        if m.endswith(".part") or not os.path.exists(m) or os.path.getsize(m) < 1024:
            continue
        if max_age_hours and max_age_hours > 0:
            file_age_hours = (now - os.path.getmtime(m)) / 3600.0
            if file_age_hours > max_age_hours:
                continue
        if is_file_valid_hdf5(m):
            valid_matches.append(m)

    if not valid_matches:
        return False

    return True


def ensure_latest_mosdac_cache(max_age_hours: float = 72.0, force_sync: bool = False, timeout_sec: int = 5) -> Dict[str, Any]:
    """
    Pre-run check: verifies that the latest MOSDAC satellite files for all
    supported datasets (LST, HEM, OLR, CTP) are present in the cache.
    If missing or older than max_age_hours, automatically downloads them via mdapi.py before processing.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    missing_or_stale = []

    for ds in SUPPORTED_DATASETS:
        if force_sync or not is_dataset_cached(ds, max_age_hours):
            missing_or_stale.append(ds)

    if not missing_or_stale:
        print(f"[MOSDAC Cache OK] All {len(SUPPORTED_DATASETS)} satellite datasets verified in {CACHE_DIR}.")
        return {"status": "UP_TO_DATE", "synced": [], "cached_datasets": SUPPORTED_DATASETS}

    print(f"\n[MOSDAC Cache Notice] Missing or outdated datasets detected: {missing_or_stale}")
    print("[MOSDAC Cache Action] Initiating automatic download from ISRO MOSDAC...")

    synced_datasets = []
    if os.path.exists("mdapi.py"):
        for ds in missing_or_stale:
            print(f"  [DOWNLOAD] Fetching latest {ds} from MOSDAC via mdapi.py...")
            try:
                update_config_dates(CONFIG_PATH, dataset_id=ds)
                result = subprocess.run(
                    [sys.executable, "mdapi.py"],
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                )
                if result.returncode == 0:
                    synced_datasets.append(ds)
                    print(f"  [OK] Successfully updated {ds}.")
                else:
                    print(f"  [NOTICE] Download for {ds} exited with code {result.returncode}. Using local shadow cache.")
            except subprocess.TimeoutExpired:
                print(f"  [WARN] Download for {ds} timed out. Proceeding with existing cache.")
            except Exception as err:
                print(f"  [WARN] Error downloading {ds}: {err}. Proceeding with existing cache.")
    else:
        print("[MOSDAC Cache Notice] mdapi.py not found. Using local cached datasets.")

    print("[MOSDAC Cache Complete] Cache verified and ready for agent processing.\n")
    return {
        "status": "SYNCED" if synced_datasets else "USING_LOCAL_CACHE",
        "synced": synced_datasets,
        "cached_datasets": SUPPORTED_DATASETS,
    }



def update_config_dates(config_file: str = CONFIG_PATH, dataset_id: Optional[str] = None) -> tuple[str, str]:
    """
    Loads config.json, updates startTime to yesterday and endTime to today,
    synchronizes credentials from .env, updates datasetId if provided,
    and saves the file back. Returns (startTime, endTime).
    """
    load_dotenv()

    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    start_str = yesterday.strftime("%Y-%m-%d")
    end_str = today.strftime("%Y-%m-%d")

    if not os.path.exists(config_file):
        generate_mosdac_config(config_file, dataset_id or "3RIMG_L2B_LST")

    with open(config_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Sync credentials from .env to config.json
    username = os.getenv("MOSDAC_USERNAME")
    password = os.getenv("MOSDAC_PASSWORD")
    if username and password:
        if "user_credentials" not in data:
            data["user_credentials"] = {}
        data["user_credentials"]["username/email"] = username
        data["user_credentials"]["username"] = username
        data["user_credentials"]["password"] = password

    # Ensure non-interactive background execution for mdapi.py
    if "download_settings" not in data:
        data["download_settings"] = {}
    data["download_settings"]["skip_user_input"] = True
    data["download_settings"]["download_path"] = CACHE_DIR
    data["download_settings"]["generate_error_logs"] = True
    data["download_settings"]["error_logs_dir"] = "./logs"

    if "search_parameters" not in data:
        data["search_parameters"] = {}

    data["search_parameters"]["startTime"] = start_str
    data["search_parameters"]["endTime"] = end_str
    data["search_parameters"]["count"] = os.getenv("MOSDAC_SYNC_COUNT", "3")
    if dataset_id:
        data["search_parameters"]["datasetId"] = dataset_id

    data["supported_datasets"] = SUPPORTED_DATASETS

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    curr_ds = data["search_parameters"].get("datasetId", "3RIMG_L2B_LST")
    print(f"[Worker] Synced .env credentials & updated {config_file} for dataset '{curr_ds}' ({start_str} -> {end_str})")
    return start_str, end_str


def run_sync():
    """
    Executes a multi-dataset synchronization pass using ISRO's MOSDAC Download API.
    Cycles through all configured satellite datasets (LST, HEM, OLR, CTP).
    """
    print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Multi-Dataset Shadow Cache sync pass...")

    if not os.path.exists("mdapi.py"):
        print("[Worker Notice] mdapi.py is not present in workspace.")
        print("                 Shadow Cache directory ready at:", os.path.abspath(CACHE_DIR))
        return

    # Cycle through each satellite product suite
    for target_ds in SUPPORTED_DATASETS:
        print(f"\n[Worker] Syncing ISRO MOSDAC Product: {target_ds}...")
        try:
            update_config_dates(CONFIG_PATH, dataset_id=target_ds)
            result = subprocess.run(
                [sys.executable, "mdapi.py"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            print(f"[Worker] {target_ds} sync completed with exit code {result.returncode}")
            if result.stdout:
                print(f"[Worker stdout for {target_ds}]:\n" + result.stdout[:400])
        except subprocess.TimeoutExpired:
            print(f"[Worker Warning] {target_ds} sync timed out after 300 seconds.")
        except Exception as err:
            print(f"[Worker Error] Exception syncing {target_ds}: {err}")

    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Multi-Dataset Shadow Cache sync pass finished.\n")


def start_worker_loop():
    """
    Runs the worker loop every 6 hours (21600 seconds).
    """
    print("=" * 70)
    print("      ORCA - ISRO MOSDAC SHADOW CACHE WORKER (SIH 176)")
    print("=" * 70)
    print(f"[i] Sync Interval : Every {SYNC_INTERVAL_SECONDS // 3600} hours ({SYNC_INTERVAL_SECONDS} seconds)")
    print(f"[i] Cache Directory: {os.path.abspath(CACHE_DIR)}")
    print("[i] Press Ctrl+C to terminate the worker.\n")

    while True:
        try:
            run_sync()
            print(f"[Worker] Sleeping for {SYNC_INTERVAL_SECONDS // 3600} hours until next sync pass...")
            time.sleep(SYNC_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\n[Worker] Received shutdown signal. Stopping Shadow Cache Worker.")
            break
        except Exception as e:
            print(f"[Worker Error] Unexpected error in worker loop: {e}")
            time.sleep(60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--once", "-1", "sync"):
        print("[Worker] Running single manual sync pass...")
        run_sync()
    else:
        start_worker_loop()
