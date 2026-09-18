"""
trainer.py - Multi-Horizon Weather & Marine Forecasting Model Training & Benchmarking
SIH 2026 Problem Statement SIH26176

Trains and benchmarks multi-horizon forecasting models (6h, 12h, 24h, 48h, 72h)
against persistence baselines using LightGBM and Random Forest.
Includes strict temporal leakage assertions and realistic metrics.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple, List

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

from ml.checkpoint_manager import get_logger, CheckpointManager
from ml.feature_engineering.weather_features_real import (
    REAL_WEATHER_FEATURE_COLS as WEATHER_FEATURE_COLS,
    FORECAST_HORIZONS,
    build_real_weather_dataset,
    get_real_weather_splits,
    assert_no_real_weather_leakage,
)

logger = get_logger("WeatherTrainer", "weather_training.log")
MODEL_DIR = os.path.join("ml", "models", "weather")
os.makedirs(MODEL_DIR, exist_ok=True)


def train_weather_models(force_retrain: bool = False) -> Dict[str, Any]:
    """
    Executes end-to-end training and benchmarking for Model 2 (Weather Multi-Horizon)
    using genuine ECMWF ERA5 reanalysis data from 1980 to September 2026.
    """
    ckpt = CheckpointManager()
    if not force_retrain and ckpt.is_step_completed("weather_model_trained"):
        logger.info("[Weather Model] Step already completed. Checkpoint active. Skipping.")
        metrics_file = os.path.join(MODEL_DIR, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r", encoding="utf-8") as f:
                return json.load(f)

    logger.info("==================================================")
    logger.info("PHASE: GENUINE ERA5 REANALYSIS WEATHER MODEL TRAINING (1980-2026)")
    logger.info("==================================================")
    ckpt.set_phase("TRAINING_WEATHER_MODEL")

    logger.info("Loading genuine ECMWF ERA5 multi-decade reanalysis dataset (1980-2026)...")
    df = build_real_weather_dataset()
    train_df, val_df, test_df = get_real_weather_splits(df, train_end_year=2022, val_end_year=2024)

    assert_no_real_weather_leakage(train_df, test_df)

    X_train = train_df[WEATHER_FEATURE_COLS].values
    X_val = val_df[WEATHER_FEATURE_COLS].values
    X_test = test_df[WEATHER_FEATURE_COLS].values

    logger.info(f"ERA5 Weather dataset shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    models_by_horizon = {}
    benchmarks_summary = {}

    for h in FORECAST_HORIZONS:
        target_col = f"target_wind_{h}h"
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values

        # 1. Persistence Baseline: Predict current wind as future wind
        current_wind_test = test_df["wind_speed_kmh"].values
        persist_mae = mean_absolute_error(y_test, current_wind_test)
        persist_rmse = np.sqrt(mean_squared_error(y_test, current_wind_test))
        persist_r2 = r2_score(y_test, current_wind_test)

        # 2. LightGBM Regressor
        lgb_reg = lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=25,
            random_state=42,
            verbose=-1,
        )
        lgb_reg.fit(X_train, y_train)
        lgb_pred = lgb_reg.predict(X_test)

        lgb_mae = mean_absolute_error(y_test, lgb_pred)
        lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_pred))
        lgb_r2 = r2_score(y_test, lgb_pred)

        # 3. Random Forest Regressor
        rf_reg = RandomForestRegressor(
            n_estimators=80,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )
        rf_reg.fit(X_train, y_train)
        rf_pred = rf_reg.predict(X_test)

        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        rf_r2 = r2_score(y_test, rf_pred)

        # Calculate improvement percentage over persistence
        improvement_pct = round(((persist_rmse - lgb_rmse) / persist_rmse) * 100.0, 2)

        benchmarks_summary[f"{h}h"] = {
            "persistence": {
                "mae": round(float(persist_mae), 2),
                "rmse": round(float(persist_rmse), 2),
                "r2": round(float(persist_r2), 3),
            },
            "lightgbm": {
                "mae": round(float(lgb_mae), 2),
                "rmse": round(float(lgb_rmse), 2),
                "r2": round(float(lgb_r2), 3),
                "improvement_over_baseline_pct": improvement_pct,
            },
            "random_forest": {
                "mae": round(float(rf_mae), 2),
                "rmse": round(float(rf_rmse), 2),
                "r2": round(float(rf_r2), 3),
            },
        }

        models_by_horizon[f"{h}h"] = lgb_reg
        logger.info(
            f"Horizon {h:2d}h | Baseline RMSE: {persist_rmse:.2f} -> LightGBM RMSE: {lgb_rmse:.2f} ({improvement_pct:+5.1f}% vs baseline, R2: {lgb_r2:.3f})"
        )

    # Save Models dictionary
    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    joblib.dump(models_by_horizon, model_path)
    logger.info(f"Saved multi-horizon weather models to: {model_path}")

    with open(os.path.join(MODEL_DIR, "feature_list.json"), "w", encoding="utf-8") as f:
        json.dump({"features": WEATHER_FEATURE_COLS, "horizons": FORECAST_HORIZONS}, f, indent=2)

    metadata = {
        "model_name": "WeatherMarineForecastModel",
        "model_type": "MultiHorizon_LightGBM_Regressor",
        "version": "v4.0-ERA5-RealData-1980-2026",
        "horizons_hours": FORECAST_HORIZONS,
        "trained_at": datetime.now().isoformat(),
        "data_source": "ECMWF ERA5 Reanalysis via Open-Meteo Archive API",
        "provenance_classification": "REANALYSIS",
        "training_period": f"{str(train_df['datetime'].min())[:10]} to {str(train_df['datetime'].max())[:10]}",
        "validation_period": f"{str(val_df['datetime'].min())[:10]} to {str(val_df['datetime'].max())[:10]}",
        "test_period": f"{str(test_df['datetime'].min())[:10]} to {str(test_df['datetime'].max())[:10]}",
        "train_samples": len(train_df),
        "validation_samples": len(val_df),
        "test_samples": len(test_df),
        "primary_predictand": "target_wind_{h}h",
        "persistence_comparison": "LightGBM benchmarked against genuine persistence baseline across all evaluation horizons",
        "leakage_safeguards": "Strict temporal separation, targets measured at future t+h, input lags measured strictly at or before t, zero forward contamination",
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    metrics_payload = {
        "best_model_architecture": "LightGBM_MultiHorizon",
        "horizons_evaluated": benchmarks_summary,
        "average_24h_rmse": benchmarks_summary["24h"]["lightgbm"]["rmse"],
        "baseline_24h_rmse": benchmarks_summary["24h"]["persistence"]["rmse"],
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    ckpt.mark_step_completed("weather_model_trained", details=metrics_payload)
    ckpt.update_model_status("weather", "TRAINED", metrics=metrics_payload)

    return metrics_payload


if __name__ == "__main__":
    train_weather_models(force_retrain=True)
