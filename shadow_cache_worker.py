"""
shadow_cache_worker.py - Freshness-Aware MOSDAC Shadow Cache Background Worker
ISRO SIH Problem Statement 176: Marine Multi-Agent System

Background synchronization worker:
1. Dynamically calculates yesterday and today dates.
2. Updates config.json search_parameters (startTime, endTime).
3. Executes ISRO's MOSDAC Data Download API (mdapi.py) to sync latest satellite files.
4. Validates downloaded and cached HDF5/NetCDF files for corruption and structural integrity.
5. Freshness-aware lifecycle management:
   - Successful download -> validate file, record retrieval & dataset timestamp, data age, mark FRESH.
   - Timeout/failure -> inspect cached file age against max_age_hours:
     * age <= max_age_hours -> STALE_CACHE (fallback_reason="MOSDAC_DOWNLOAD_FAILED"), allow cache.
     * age > max_age_hours -> DATA_UNAVAILABLE (fallback_reason="CACHE_EXPIRED"), reject expired cache.
6. Configurable timeouts, exponential backoff retries, and independent parallel sync per dataset.
7. Zero synthetic data fabrication: never describe stale data as "current".
"""

import os
import sys
import glob
import json
import time
import re
import datetime
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv

load_dotenv()

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

from setup_mosdac_config import SUPPORTED_DATASETS, generate_mosdac_config

CONFIG_PATH = os.getenv("MOSDAC_CONFIG_FILE", "config.json")
CACHE_DIR = os.getenv("MOSDAC_CACHE_DIR", "./data/mosdac_cache")
LOGS_DIR = os.getenv("MOSDAC_LOGS_DIR", "./logs")
SYNC_INTERVAL_SECONDS = 21600  # 6 hours

# Configurable defaults via environment
DEFAULT_MAX_AGE_HOURS = float(os.getenv("MOSDAC_MAX_AGE_HOURS", "72.0"))
DEFAULT_TIMEOUT_SEC = float(os.getenv("MOSDAC_DOWNLOAD_TIMEOUT_SEC", os.getenv("MOSDAC_TIMEOUT_SEC", "5.0")))
DEFAULT_MAX_RETRIES = int(os.getenv("MOSDAC_MAX_RETRIES", "2"))
DEFAULT_BACKOFF_FACTOR = float(os.getenv("MOSDAC_BACKOFF_FACTOR", "1.5"))
DEFAULT_BASE_DELAY_SEC = float(os.getenv("MOSDAC_BASE_DELAY_SEC", "0.5"))

# Global in-memory cache registry for active decisions
_CACHE_REGISTRY: Dict[str, Dict[str, Any]] = {}
_LAST_SYNC_PASS_TIME: float = 0.0


def get_configured_timeout(config_path: str = CONFIG_PATH) -> float:
    """Retrieves download timeout from env, config.json, or default."""
    env_timeout = os.getenv("MOSDAC_DOWNLOAD_TIMEOUT_SEC") or os.getenv("MOSDAC_TIMEOUT_SEC")
    if env_timeout:
        try:
            return float(env_timeout)
        except ValueError:
            pass

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                ds = cfg.get("download_settings", {})
                if "timeout_sec" in ds:
                    return float(ds["timeout_sec"])
                if "download_timeout_seconds" in ds:
                    return float(ds["download_timeout_seconds"])
        except Exception:
            pass

    return DEFAULT_TIMEOUT_SEC


def get_configured_max_age(config_path: str = CONFIG_PATH) -> float:
    """Retrieves max cache age hours from env, config.json, or default."""
    env_max_age = os.getenv("MOSDAC_MAX_AGE_HOURS") or os.getenv("CACHE_TTL_HOURS")
    if env_max_age:
        try:
            return float(env_max_age)
        except ValueError:
            pass

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                ds = cfg.get("download_settings", {})
                if "max_age_hours" in ds:
                    return float(ds["max_age_hours"])
        except Exception:
            pass

    return DEFAULT_MAX_AGE_HOURS


def is_file_valid_hdf5(filepath: str) -> bool:
    """
    Verifies that an HDF5 or NetCDF file is readable, non-empty, and uncorrupted.
    Rejects incomplete downloads (.part) and empty/corrupted headers.
    """
    if not filepath or not os.path.exists(filepath):
        return False
    if filepath.endswith(".part"):
        return False
    try:
        if os.path.getsize(filepath) < 1024:
            return False
    except OSError:
        return False

    # Attempt HDF5 open
    try:
        import h5py
        with h5py.File(filepath, "r") as f:
            _ = list(f.keys())
            return True
    except Exception:
        pass

    # Attempt NetCDF / xarray open
    try:
        import xarray as xr
        with xr.open_dataset(filepath) as ds:
            _ = list(ds.data_vars)
            return True
    except Exception:
        return False


def extract_dataset_timestamp(filepath: str) -> Optional[datetime.datetime]:
    """
    Extracts the acquisition/dataset observation timestamp from HDF5 attributes,
    NetCDF metadata, or the standard ISRO filename pattern (e.g., 09SEP2026_0745).
    Returns a timezone-aware UTC datetime.
    """
    if not filepath or not os.path.exists(filepath):
        return None

    # 1. Try reading HDF5 attributes
    try:
        import h5py
        with h5py.File(filepath, "r") as f:
            for attr_name in ("Acquisition_End_Time", "Acquisition_Start_Time", "Product_Creation_Time"):
                raw_val = f.attrs.get(attr_name)
                if raw_val is not None:
                    if isinstance(raw_val, bytes):
                        val_str = raw_val.decode("utf-8", errors="ignore").strip()
                    else:
                        val_str = str(raw_val).strip()
                    if val_str:
                        # Common format: "09-SEP-2026T08:12:38" or "2026-09-09T08:12:38"
                        for fmt in ("%d-%b-%YT%H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                            try:
                                dt = datetime.datetime.strptime(val_str, fmt)
                                return dt.replace(tzinfo=datetime.timezone.utc)
                            except ValueError:
                                continue

            # Try Acquisition_Date + Acquisition_Time_in_GMT
            acq_d = f.attrs.get("Acquisition_Date")
            acq_t = f.attrs.get("Acquisition_Time_in_GMT")
            if acq_d and acq_t:
                d_str = (acq_d.decode("utf-8", "ignore") if isinstance(acq_d, bytes) else str(acq_d)).strip()
                t_str = (acq_t.decode("utf-8", "ignore") if isinstance(acq_t, bytes) else str(acq_t)).strip()
                try:
                    dt = datetime.datetime.strptime(f"{d_str}{t_str}", "%d%b%Y%H%M")
                    return dt.replace(tzinfo=datetime.timezone.utc)
                except ValueError:
                    pass
    except Exception:
        pass

    # 2. Extract from standard ISRO filename pattern (e.g. 3RIMG_09SEP2026_0745_L2B_LST_V01R00.h5)
    base_name = os.path.basename(filepath)
    match = re.search(r"(\d{2}[A-Za-z]{3}\d{4})_(\d{4})", base_name)
    if match:
        date_str, time_str = match.groups()
        try:
            dt = datetime.datetime.strptime(f"{date_str}_{time_str}", "%d%b%Y_%H%M")
            return dt.replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            pass

    # 3. Fallback to filesystem mtime
    try:
        mtime = os.path.getmtime(filepath)
        return datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc)
    except Exception:
        return None


def calculate_data_age_hours(
    dt: Optional[datetime.datetime],
    reference_time: Optional[datetime.datetime] = None
) -> float:
    """
    Calculates age in hours between dataset timestamp and reference_time (default: now UTC).
    """
    if dt is None:
        return float("inf")
    now_utc = reference_time or datetime.datetime.now(datetime.timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=datetime.timezone.utc)
    age_seconds = (now_utc - dt).total_seconds()
    return round(max(0.0, age_seconds / 3600.0), 1)


def inspect_cached_dataset(
    dataset_id: str,
    max_age_hours: Optional[float] = None,
    cache_dir: str = CACHE_DIR,
    reference_time: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
    """
    Inspects existing cached files for a dataset.
    Validates file integrity. If valid, calculates age and compares against max_age_hours.
    """
    if max_age_hours is None:
        max_age_hours = get_configured_max_age()

    short_code = dataset_id.split("_")[-1]  # e.g., LST, HEM, OLR, CTP, OCM, WND, SWH, FOG, IMC
    patterns = [
        os.path.join(cache_dir, f"*{short_code}*.h5"),
        os.path.join(cache_dir, f"*{short_code}*.nc"),
    ]
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat))

    valid_candidates: List[Tuple[str, datetime.datetime, float]] = []
    for m in matches:
        if not is_file_valid_hdf5(m):
            continue
        ds_time = extract_dataset_timestamp(m)
        age = calculate_data_age_hours(ds_time, reference_time=reference_time)
        valid_candidates.append((m, ds_time, age))

    if not valid_candidates:
        return {
            "dataset_id": dataset_id,
            "status": "DATA_UNAVAILABLE",
            "file_path": None,
            "age_hours": None,
            "max_age_hours": max_age_hours,
            "action": "REJECTING_EXPIRED_CACHE",
            "fallback_reason": "NO_VALID_CACHE",
            "retrieval_timestamp": None,
            "dataset_timestamp": None,
        }

    # Sort candidates by age (lowest age = most recent)
    valid_candidates.sort(key=lambda item: item[2])
    best_file, best_time, best_age = valid_candidates[0]

    if best_age <= max_age_hours:
        return {
            "dataset_id": dataset_id,
            "status": "STALE_CACHE",
            "file_path": best_file,
            "age_hours": best_age,
            "max_age_hours": max_age_hours,
            "action": "USING_EXISTING_CACHE",
            "fallback_reason": "MOSDAC_DOWNLOAD_FAILED",
            "dataset_timestamp": best_time.isoformat() if best_time else None,
            "retrieval_timestamp": datetime.datetime.fromtimestamp(
                os.path.getmtime(best_file), tz=datetime.timezone.utc
            ).isoformat(),
        }
    else:
        return {
            "dataset_id": dataset_id,
            "status": "DATA_UNAVAILABLE",
            "file_path": None,
            "age_hours": best_age,
            "max_age_hours": max_age_hours,
            "action": "REJECTING_EXPIRED_CACHE",
            "fallback_reason": "CACHE_EXPIRED",
            "dataset_timestamp": best_time.isoformat() if best_time else None,
            "retrieval_timestamp": datetime.datetime.fromtimestamp(
                os.path.getmtime(best_file), tz=datetime.timezone.utc
            ).isoformat(),
        }


def is_dataset_cached(dataset_id: str, max_age_hours: Optional[float] = None, cache_dir: str = CACHE_DIR) -> bool:
    """
    Checks if a valid, non-expired cached file exists for the given dataset.
    Returns True only if file is present, uncorrupted, and age <= max_age_hours.
    """
    inspection = inspect_cached_dataset(dataset_id, max_age_hours=max_age_hours, cache_dir=cache_dir)
    return inspection.get("status") in ("FRESH", "STALE_CACHE") and inspection.get("file_path") is not None


def get_cached_file(dataset_id: str, max_age_hours: Optional[float] = None, cache_dir: str = CACHE_DIR) -> Optional[str]:
    """
    Returns the valid cached file path if within max_age_hours, or None if expired/unavailable.
    Guarantees no expired satellite data is silently loaded.
    """
    # Check in-memory registry first
    if dataset_id in _CACHE_REGISTRY:
        reg = _CACHE_REGISTRY[dataset_id]
        if reg.get("status") == "DATA_UNAVAILABLE":
            return None
        if reg.get("file_path") and os.path.exists(reg["file_path"]):
            # If max_age_hours requested, verify freshness
            if max_age_hours is not None and reg.get("age_hours") is not None:
                if reg["age_hours"] > max_age_hours:
                    return None
            return reg["file_path"]

    inspection = inspect_cached_dataset(dataset_id, max_age_hours=max_age_hours, cache_dir=cache_dir)
    _CACHE_REGISTRY[dataset_id] = inspection
    return inspection.get("file_path")


def get_dataset_cache_status(dataset_id: str, max_age_hours: Optional[float] = None, cache_dir: str = CACHE_DIR) -> Dict[str, Any]:
    """
    Returns structured cache metadata (status, age_hours, fallback_reason, timestamps) for a dataset.
    """
    if dataset_id in _CACHE_REGISTRY:
        reg = _CACHE_REGISTRY[dataset_id]
        if max_age_hours is None or reg.get("max_age_hours") == max_age_hours:
            return reg

    inspection = inspect_cached_dataset(dataset_id, max_age_hours=max_age_hours, cache_dir=cache_dir)
    _CACHE_REGISTRY[dataset_id] = inspection
    return inspection


def get_all_cache_statuses(max_age_hours: Optional[float] = None, cache_dir: str = CACHE_DIR) -> Dict[str, Dict[str, Any]]:
    """Returns structured status for all supported MOSDAC datasets."""
    statuses = {}
    for ds in SUPPORTED_DATASETS:
        statuses[ds] = get_dataset_cache_status(ds, max_age_hours=max_age_hours, cache_dir=cache_dir)
    return statuses


def save_cache_metadata(cache_dir: str = CACHE_DIR) -> None:
    """Saves current cache statuses to cache_metadata.json."""
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, "cache_metadata.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(_CACHE_REGISTRY, f, indent=2)
    except Exception:
        pass


def load_cache_metadata(cache_dir: str = CACHE_DIR) -> Dict[str, Any]:
    """Loads cache statuses from cache_metadata.json if present."""
    meta_path = os.path.join(cache_dir, "cache_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                _CACHE_REGISTRY.update(data)
                return data
        except Exception:
            pass
    return {}


def download_dataset_attempt(
    dataset_id: str,
    timeout_sec: float,
    config_file: str = CONFIG_PATH,
    cache_dir: str = CACHE_DIR,
) -> Dict[str, Any]:
    """
    Executes a single non-interactive MOSDAC download attempt for dataset_id using mdapi.py.
    Uses an isolated configuration file per thread to prevent collision.
    """
    if not os.path.exists("mdapi.py"):
        return {"success": False, "reason": "MDAPI_NOT_FOUND"}

    # Prepare thread-safe isolated config for dataset_id
    temp_cfg = f"config_{dataset_id}_{int(time.time() * 1000)}.json"
    try:
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)
        start_str = yesterday.strftime("%Y-%m-%d")
        end_str = today.strftime("%Y-%m-%d")

        base_cfg = {}
        if os.path.exists(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    base_cfg = json.load(f)
            except Exception:
                base_cfg = {}

        if not base_cfg:
            base_cfg = generate_mosdac_config(temp_cfg, dataset_id=dataset_id)

        # Update search parameters and credentials
        if "user_credentials" not in base_cfg:
            base_cfg["user_credentials"] = {}
        username = os.getenv("MOSDAC_USERNAME")
        password = os.getenv("MOSDAC_PASSWORD")
        if username:
            base_cfg["user_credentials"]["username/email"] = username
            base_cfg["user_credentials"]["username"] = username
        if password:
            base_cfg["user_credentials"]["password"] = password

        if "download_settings" not in base_cfg:
            base_cfg["download_settings"] = {}
        base_cfg["download_settings"]["skip_user_input"] = True
        base_cfg["download_settings"]["download_path"] = cache_dir
        base_cfg["download_settings"]["generate_error_logs"] = True
        base_cfg["download_settings"]["error_logs_dir"] = LOGS_DIR

        if "search_parameters" not in base_cfg:
            base_cfg["search_parameters"] = {}
        base_cfg["search_parameters"]["startTime"] = start_str
        base_cfg["search_parameters"]["endTime"] = end_str
        base_cfg["search_parameters"]["datasetId"] = dataset_id
        base_cfg["search_parameters"]["count"] = os.getenv("MOSDAC_SYNC_COUNT", "3")

        with open(temp_cfg, "w", encoding="utf-8") as f:
            json.dump(base_cfg, f, indent=2)

        # Snapshot cache directory before download
        before_files = set(glob.glob(os.path.join(cache_dir, "*")))

        env = os.environ.copy()
        env["MOSDAC_CONFIG_FILE"] = temp_cfg

        result = subprocess.run(
            [sys.executable, "mdapi.py"],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
        )

        if result.returncode == 0:
            after_files = set(glob.glob(os.path.join(cache_dir, "*")))
            new_files = list(after_files - before_files)
            # Find the new file matching dataset short code
            short_code = dataset_id.split("_")[-1]
            matching_new = [f for f in new_files if short_code in f and not f.endswith(".part")]
            if matching_new:
                target_file = matching_new[0]
            else:
                # Check for newest valid file
                cand = glob.glob(os.path.join(cache_dir, f"*{short_code}*.h5")) + glob.glob(os.path.join(cache_dir, f"*{short_code}*.nc"))
                target_file = max(cand, key=os.path.getmtime) if cand else None

            if target_file and is_file_valid_hdf5(target_file):
                return {"success": True, "file_path": target_file}
            else:
                return {"success": False, "reason": "CORRUPTED_DOWNLOAD"}
        else:
            return {"success": False, "reason": f"EXIT_CODE_{result.returncode}"}

    except subprocess.TimeoutExpired:
        return {"success": False, "reason": "TIMEOUT"}
    except Exception as err:
        return {"success": False, "reason": f"ERROR_{err}"}
    finally:
        if os.path.exists(temp_cfg):
            try:
                os.remove(temp_cfg)
            except OSError:
                pass


def sync_single_dataset(
    dataset_id: str,
    max_age_hours: Optional[float] = None,
    timeout_sec: Optional[float] = None,
    max_retries: Optional[int] = None,
    backoff_factor: Optional[float] = None,
    base_delay_sec: Optional[float] = None,
    cache_dir: str = CACHE_DIR,
    force_sync: bool = False,
    config_file: str = CONFIG_PATH,
    reference_time: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
    """
    Downloads a single MOSDAC dataset with exponential backoff retry.
    Applies strict freshness-aware lifecycle:
    1. Download succeeds -> validate file, record retrieval & dataset timestamp, age, mark FRESH.
    2. Download fails/times out -> inspect existing cache:
       - age <= max_age_hours -> STALE_CACHE, USING_EXISTING_CACHE, fallback_reason="MOSDAC_DOWNLOAD_FAILED".
       - age > max_age_hours -> DATA_UNAVAILABLE, REJECTING_EXPIRED_CACHE.
    3. Outputs exact required operational logs.
    """
    if max_age_hours is None:
        max_age_hours = get_configured_max_age(config_file)
    if timeout_sec is None:
        timeout_sec = get_configured_timeout(config_file)
    if max_retries is None:
        max_retries = DEFAULT_MAX_RETRIES
    if backoff_factor is None:
        backoff_factor = DEFAULT_BACKOFF_FACTOR
    if base_delay_sec is None:
        base_delay_sec = DEFAULT_BASE_DELAY_SEC

    # Fast path: if not force_sync and already verified fresh (<1 hour old), return FRESH
    if not force_sync:
        inspection = inspect_cached_dataset(dataset_id, max_age_hours=max_age_hours, cache_dir=cache_dir, reference_time=reference_time)
        if inspection["status"] == "STALE_CACHE" and inspection.get("age_hours") is not None and inspection["age_hours"] < 1.0:
            inspection["status"] = "FRESH"
            inspection["action"] = "CACHE_UP_TO_DATE"
            inspection["fallback_reason"] = None
            _CACHE_REGISTRY[dataset_id] = inspection
            return inspection

    last_failure_reason = "UNKNOWN"

    for attempt in range(1, max_retries + 1):
        res = download_dataset_attempt(dataset_id, timeout_sec=timeout_sec, config_file=config_file, cache_dir=cache_dir)
        if res.get("success"):
            # Requirement 1: Validate, record retrieval & dataset timestamp, calculate data age, mark status = FRESH
            file_path = res["file_path"]
            retrieval_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
            ds_time = extract_dataset_timestamp(file_path)
            data_age = calculate_data_age_hours(ds_time, reference_time=reference_time)
            
            decision = {
                "dataset_id": dataset_id,
                "status": "FRESH",
                "action": "DOWNLOAD_SUCCESS",
                "file_path": file_path,
                "retrieval_timestamp": retrieval_ts,
                "dataset_timestamp": ds_time.isoformat() if ds_time else None,
                "age_hours": data_age,
                "max_age_hours": max_age_hours,
                "fallback_reason": None,
            }
            _CACHE_REGISTRY[dataset_id] = decision
            print(f"[DOWNLOAD SUCCESS]\ndataset={dataset_id}\nstatus=FRESH\nage_hours={data_age}\nfile={os.path.basename(file_path)}")
            return decision

        last_failure_reason = res.get("reason", "DOWNLOAD_ERROR")
        if attempt < max_retries:
            delay = base_delay_sec * (backoff_factor ** (attempt - 1))
            time.sleep(delay)

    # All download attempts failed / timed out
    # Format max_age_hours string representation cleanly (e.g., 72 instead of 72.0 if integer)
    max_age_display = int(max_age_hours) if float(max_age_hours).is_integer() else max_age_hours

    # Requirement 6: Log exact download failure header
    print(f"\n[DOWNLOAD FAILED]\ndataset={dataset_id}\nreason={last_failure_reason}")

    # Inspect existing cache
    inspection = inspect_cached_dataset(dataset_id, max_age_hours=max_age_hours, cache_dir=cache_dir, reference_time=reference_time)
    age_hours = inspection.get("age_hours")

    if age_hours is not None:
        print(f"\n[CACHE CHECK]\nage_hours={age_hours}\nmax_age_hours={max_age_display}")
    else:
        print(f"\n[CACHE CHECK]\nage_hours=None\nmax_age_hours={max_age_display}")

    # Requirement 3 & 4: Make cache decision
    if inspection["status"] == "STALE_CACHE" and inspection.get("file_path"):
        print(f"\n[CACHE DECISION]\nstatus=STALE_CACHE\naction=USING_EXISTING_CACHE\n")
        inspection["fallback_reason"] = "MOSDAC_DOWNLOAD_FAILED"
        inspection["action"] = "USING_EXISTING_CACHE"
        _CACHE_REGISTRY[dataset_id] = inspection
        return inspection
    else:
        print(f"\n[CACHE DECISION]\nstatus=DATA_UNAVAILABLE\naction=REJECTING_EXPIRED_CACHE\n")
        inspection["status"] = "DATA_UNAVAILABLE"
        inspection["file_path"] = None  # Crucial: do NOT use cached dataset
        inspection["action"] = "REJECTING_EXPIRED_CACHE"
        if not inspection.get("fallback_reason") or inspection["fallback_reason"] == "NO_VALID_CACHE":
            inspection["fallback_reason"] = "MOSDAC_DOWNLOAD_FAILED" if inspection.get("fallback_reason") == "NO_VALID_CACHE" else "CACHE_EXPIRED"
        _CACHE_REGISTRY[dataset_id] = inspection
        return inspection


def sync_datasets_independently(
    datasets: Optional[List[str]] = None,
    max_age_hours: Optional[float] = None,
    timeout_sec: Optional[float] = None,
    max_retries: Optional[int] = None,
    backoff_factor: Optional[float] = None,
    base_delay_sec: Optional[float] = None,
    max_workers: int = 4,
    cache_dir: str = CACHE_DIR,
    force_sync: bool = False,
    reference_time: Optional[datetime.datetime] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Executes independent concurrent downloads across all requested datasets.
    Prevents a single slow dataset from blocking other datasets.
    """
    target_datasets = datasets or SUPPORTED_DATASETS
    results: Dict[str, Dict[str, Any]] = {}

    worker_count = min(max_workers, len(target_datasets)) if target_datasets else 1

    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="mosdac-sync") as executor:
        future_map = {
            executor.submit(
                sync_single_dataset,
                ds,
                max_age_hours=max_age_hours,
                timeout_sec=timeout_sec,
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                base_delay_sec=base_delay_sec,
                cache_dir=cache_dir,
                force_sync=force_sync,
                reference_time=reference_time,
            ): ds
            for ds in target_datasets
        }

        for future in as_completed(future_map):
            ds = future_map[future]
            try:
                dec = future.result()
                results[ds] = dec
            except Exception as e:
                results[ds] = {
                    "dataset_id": ds,
                    "status": "DATA_UNAVAILABLE",
                    "file_path": None,
                    "age_hours": None,
                    "max_age_hours": max_age_hours or DEFAULT_MAX_AGE_HOURS,
                    "action": "REJECTING_EXPIRED_CACHE",
                    "fallback_reason": f"EXCEPTION_{e}",
                }

    save_cache_metadata(cache_dir=cache_dir)
    return results


def ensure_latest_mosdac_cache(
    max_age_hours: Optional[float] = None,
    force_sync: bool = False,
    timeout_sec: Optional[float] = None,
    max_workers: int = 4,
    cache_dir: str = CACHE_DIR,
) -> Dict[str, Any]:
    """
    Pre-run check: verifies that the latest MOSDAC satellite files for all
    supported datasets are present in the cache.
    Freshness-aware:
    - If valid cached file is <= max_age_hours, permits STALE_CACHE on download failure.
    - If valid cached file is > max_age_hours, marks DATA_UNAVAILABLE and rejects expired cache.
    - Independent concurrent execution prevents slow downloads from stalling the application.
    """
    global _LAST_SYNC_PASS_TIME
    os.makedirs(cache_dir, exist_ok=True)
    if max_age_hours is None:
        max_age_hours = get_configured_max_age()
    if timeout_sec is None:
        timeout_sec = get_configured_timeout()

    # Short-term cooldown: if sync pass ran within last 60 seconds and not forced, return cached registry
    now_mono = time.time()
    if not force_sync and (now_mono - _LAST_SYNC_PASS_TIME) < 60.0 and len(_CACHE_REGISTRY) >= len(SUPPORTED_DATASETS):
        fresh_count = sum(1 for d in _CACHE_REGISTRY.values() if d.get("status") == "FRESH")
        stale_count = sum(1 for d in _CACHE_REGISTRY.values() if d.get("status") == "STALE_CACHE")
        unavail_count = sum(1 for d in _CACHE_REGISTRY.values() if d.get("status") == "DATA_UNAVAILABLE")
        return {
            "status": "UP_TO_DATE" if fresh_count == len(SUPPORTED_DATASETS) else ("STALE_CACHE" if stale_count > 0 else "DATA_UNAVAILABLE"),
            "fresh": fresh_count,
            "stale": stale_count,
            "unavailable": unavail_count,
            "datasets": _CACHE_REGISTRY,
        }

    # Fast path for FAST_DEMO_MODE: inspect cache freshness without blocking on network
    is_fast_demo = os.getenv("FAST_DEMO_MODE", "false").lower() in ("true", "1", "yes")
    if is_fast_demo and not force_sync:
        for ds in SUPPORTED_DATASETS:
            get_dataset_cache_status(ds, max_age_hours=max_age_hours, cache_dir=cache_dir)
        fresh_count = sum(1 for d in _CACHE_REGISTRY.values() if d.get("status") == "FRESH")
        stale_count = sum(1 for d in _CACHE_REGISTRY.values() if d.get("status") == "STALE_CACHE")
        unavail_count = sum(1 for d in _CACHE_REGISTRY.values() if d.get("status") == "DATA_UNAVAILABLE")
        return {
            "status": "UP_TO_DATE" if fresh_count == len(SUPPORTED_DATASETS) else ("STALE_CACHE" if stale_count > 0 else "DATA_UNAVAILABLE"),
            "fresh": fresh_count,
            "stale": stale_count,
            "unavailable": unavail_count,
            "datasets": _CACHE_REGISTRY,
        }

    # Evaluate missing or outdated datasets
    to_sync = []
    for ds in SUPPORTED_DATASETS:
        inspection = inspect_cached_dataset(ds, max_age_hours=max_age_hours, cache_dir=cache_dir)
        if force_sync or inspection["status"] != "FRESH":
            to_sync.append(ds)

    if not to_sync:
        print(f"[MOSDAC Cache OK] All {len(SUPPORTED_DATASETS)} satellite datasets verified FRESH in {cache_dir}.")
        return {"status": "UP_TO_DATE", "synced": [], "cached_datasets": SUPPORTED_DATASETS}

    print(f"\n[MOSDAC Cache Notice] Missing or outdated datasets detected: {to_sync}")
    print("[MOSDAC Cache Action] Initiating independent parallel download from ISRO MOSDAC...")

    results = sync_datasets_independently(
        datasets=to_sync,
        max_age_hours=max_age_hours,
        timeout_sec=timeout_sec,
        max_workers=max_workers,
        cache_dir=cache_dir,
        force_sync=force_sync,
    )

    synced = [ds for ds, res in results.items() if res.get("status") == "FRESH"]
    stale = [ds for ds, res in results.items() if res.get("status") == "STALE_CACHE"]
    unavailable = [ds for ds, res in results.items() if res.get("status") == "DATA_UNAVAILABLE"]

    overall_status = "SYNCED" if synced and not unavailable else ("STALE_CACHE" if stale and not unavailable else "DATA_UNAVAILABLE")

    print(f"[MOSDAC Cache Complete] Sync pass finished: {len(synced)} FRESH, {len(stale)} STALE_CACHE, {len(unavailable)} DATA_UNAVAILABLE.\n")

    _LAST_SYNC_PASS_TIME = time.time()
    return {
        "status": overall_status,
        "synced": synced,
        "stale": stale,
        "unavailable": unavailable,
        "datasets": results,
    }


def update_config_dates(config_file: str = CONFIG_PATH, dataset_id: Optional[str] = None) -> tuple[str, str]:
    """
    Loads config.json, updates startTime to yesterday and endTime to today,
    synchronizes credentials from .env, updates datasetId if provided,
    and saves the file back. Returns (startTime, endTime).
    Maintained for full backward compatibility with test suites.
    """
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
    data["download_settings"]["error_logs_dir"] = LOGS_DIR

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
    Runs independent concurrent downloads across all supported datasets.
    """
    print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Multi-Dataset Shadow Cache sync pass...")
    timeout = get_configured_timeout()
    max_age = get_configured_max_age()
    ensure_latest_mosdac_cache(max_age_hours=max_age, force_sync=True, timeout_sec=timeout)
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Multi-Dataset Shadow Cache sync pass finished.\n")


def start_worker_loop():
    """Runs the background worker loop periodically."""
    print("=" * 70)
    print("      ORCA - ISRO MOSDAC SHADOW CACHE WORKER (SIH 176)")
    print("=" * 70)
    print(f"[i] Sync Interval : Every {SYNC_INTERVAL_SECONDS // 3600} hours ({SYNC_INTERVAL_SECONDS} seconds)")
    print(f"[i] Cache Directory: {os.path.abspath(CACHE_DIR)}")
    print(f"[i] Max Cache Age : {get_configured_max_age()} hours")
    print(f"[i] Timeout Sec   : {get_configured_timeout()}s")
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
