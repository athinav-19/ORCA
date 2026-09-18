"""
ml/feature_engineering/weather_features_real.py
Leakage-Free Feature Engineering on Genuine ECMWF ERA5 Reanalysis (1980-2026).
SIH 2026 Problem Statement SIH26176.

Constructs multi-horizon (6h, 12h, 24h, 48h, 72h) autoregressive features
from genuine historical meteorological observations:
- Inputs at time t: ONLY observations strictly available at or before t (t, t-6h, t-12h, t-24h).
- Targets: Genuine future wind speed at t+6h, t+12h, t+24h, t+48h, t+72h.
- Persistence Baseline: Predicts current observation y_t as future observation y_{t+h}.
- Zero Leakage: Strict chronological time barriers with zero future data contamination.
"""

import os
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from ml.data_ingestion.openmeteo_weather_real import download_all_real_era5

FORECAST_HORIZONS = [6, 12, 24, 48, 72]

REAL_WEATHER_FEATURE_COLS = [
    "latitude",
    "longitude",
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
    "wind_speed_kmh",
    "wind_dir_deg",
    "u_wind_kmh",
    "v_wind_kmh",
    "surface_pressure_hpa",
    "precipitation_mm",
    "relative_humidity_pct",
    "temperature_2m_c",
    "wave_height_m",
    "wave_period_s",
    "wind_lag_6h",
    "wind_lag_12h",
    "wind_lag_24h",
    "pres_diff_6h",
    "wave_lag_6h",
]


def build_real_weather_dataset(force_download: bool = False) -> pd.DataFrame:
    """
    Builds multi-horizon dataset from genuine ERA5 reanalysis across 6 maritime stations.
    Cadence: 6-hourly fixes (00:00, 06:00, 12:00, 18:00 UTC).
    """
    raw_df = download_all_real_era5(force_refresh=force_download)

    processed_stations = []

    for station_name, stn_df in raw_df.groupby("station"):
        df_s = stn_df.sort_values("datetime").copy()

        # 1. Cyclical calendar encodings
        m = df_s["datetime"].dt.month
        h = df_s["datetime"].dt.hour
        df_s["month_sin"] = np.sin(2 * np.pi * m / 12.0)
        df_s["month_cos"] = np.cos(2 * np.pi * m / 12.0)
        df_s["hour_sin"] = np.sin(2 * np.pi * h / 24.0)
        df_s["hour_cos"] = np.cos(2 * np.pi * h / 24.0)

        # 2. Strict backward lag features (ONLY at or before t)
        # With 6-hourly fixes:
        # t-6h is shift(1)
        # t-12h is shift(2)
        # t-24h is shift(4)
        df_s["wind_lag_6h"] = df_s["wind_speed_kmh"].shift(1).bfill()
        df_s["wind_lag_12h"] = df_s["wind_speed_kmh"].shift(2).bfill()
        df_s["wind_lag_24h"] = df_s["wind_speed_kmh"].shift(4).bfill()
        df_s["pres_diff_6h"] = (df_s["surface_pressure_hpa"] - df_s["surface_pressure_hpa"].shift(1).bfill()).round(2)
        df_s["wave_lag_6h"] = df_s["wave_height_m"].shift(1).bfill()

        # 3. Independent future targets at t+6h, t+12h, t+24h, t+48h, t+72h
        # shift(-1) = t+6h
        # shift(-2) = t+12h
        # shift(-4) = t+24h
        # shift(-8) = t+48h
        # shift(-12) = t+72h
        df_s["target_wind_6h"] = df_s["wind_speed_kmh"].shift(-1)
        df_s["target_wind_12h"] = df_s["wind_speed_kmh"].shift(-2)
        df_s["target_wind_24h"] = df_s["wind_speed_kmh"].shift(-4)
        df_s["target_wind_48h"] = df_s["wind_speed_kmh"].shift(-8)
        df_s["target_wind_72h"] = df_s["wind_speed_kmh"].shift(-12)

        # Drop rows where future target is undefined (at the very end of time-series)
        df_s = df_s.dropna(subset=["target_wind_72h"]).reset_index(drop=True)
        processed_stations.append(df_s)

    final_df = pd.concat(processed_stations, ignore_index=True)
    final_df = final_df.sort_values("datetime").reset_index(drop=True)
    return final_df


def get_real_weather_splits(
    df: pd.DataFrame,
    train_end_year: int = 2022,
    val_end_year: int = 2024,
    horizon_buffer_hours: int = 72,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strict chronological split with horizon boundary buffers:
    - Train: 1980-01-01 to (2022-12-31 - 72h) = 2022-12-28 18:00:00
      Guarantees all targets (up to t+72h) fall strictly within 2022.
    - Validation: 2023-01-01 to (2024-12-31 - 72h) = 2024-12-28 18:00:00
      Guarantees all targets (up to t+72h) fall strictly within 2024.
    - Test: 2025-01-01 to 2026-09-13 (genuine unseen future)
    """
    train_cutoff = pd.to_datetime(f"{train_end_year}-12-31 23:59:59") - pd.Timedelta(hours=horizon_buffer_hours)
    val_start = pd.to_datetime(f"{train_end_year + 1}-01-01 00:00:00")
    val_cutoff = pd.to_datetime(f"{val_end_year}-12-31 23:59:59") - pd.Timedelta(hours=horizon_buffer_hours)
    test_start = pd.to_datetime(f"{val_end_year + 1}-01-01 00:00:00")

    train_df = df[df["datetime"] <= train_cutoff].copy()
    val_df = df[(df["datetime"] >= val_start) & (df["datetime"] <= val_cutoff)].copy()
    test_df = df[df["datetime"] >= test_start].copy()

    # Leakage assertions
    max_train = train_df["datetime"].max()
    min_val = val_df["datetime"].min()
    max_val = val_df["datetime"].max()
    min_test = test_df["datetime"].min()

    if min_val <= max_train:
        raise ValueError(f"[DATA LEAKAGE] Validation date {min_val} overlaps with Train date {max_train}!")
    if min_test <= max_val:
        raise ValueError(f"[DATA LEAKAGE] Test date {min_test} overlaps with Validation date {max_val}!")

    # Verify no target horizon crossing
    max_train_target = max_train + pd.Timedelta(hours=horizon_buffer_hours)
    if max_train_target >= min_val:
        raise ValueError(f"[HORIZON LEAKAGE] Train t+72h target ({max_train_target}) crosses into Validation ({min_val})!")

    max_val_target = max_val + pd.Timedelta(hours=horizon_buffer_hours)
    if max_val_target >= min_test:
        raise ValueError(f"[HORIZON LEAKAGE] Val t+72h target ({max_val_target}) crosses into Test ({min_test})!")

    print(f"[Real Weather Splits OK] Chronological split with {horizon_buffer_hours}h horizon buffer:")
    print(f"  Train      : {len(train_df):,} samples ({train_df['datetime'].min()} to {train_df['datetime'].max()})")
    print(f"  Validation : {len(val_df):,} samples ({val_df['datetime'].min()} to {val_df['datetime'].max()})")
    print(f"  Test       : {len(test_df):,} samples ({test_df['datetime'].min()} to {test_df['datetime'].max()})")

    return train_df, val_df, test_df


def assert_no_real_weather_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Rigorous audit against all forms of data leakage."""
    for col in REAL_WEATHER_FEATURE_COLS:
        if "target" in col.lower():
            raise ValueError(f"[LEAKAGE] Target column '{col}' found in feature list!")

    # Check temporal barrier
    assert test_df["datetime"].min() > train_df["datetime"].max(), "Temporal barrier violated!"

    # Verify target horizon barrier
    max_train_target = train_df["datetime"].max() + pd.Timedelta(hours=72)
    assert max_train_target < test_df["datetime"].min(), "Train target horizon crosses into test set!"

    # Verify no duplicate timestamps per station in test
    for stn, grp in test_df.groupby("station"):
        dups = grp["datetime"].duplicated().sum()
        if dups > 0:
            raise ValueError(f"[LEAKAGE] {dups} duplicate timestamps found in test set for {stn}!")

    print("[Real Weather Leakage Audit PASSED] Zero target derivation, 72h horizon buffer enforced, zero duplicates.")

