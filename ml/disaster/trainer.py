"""
trainer.py - Leakage-Safe Disaster & Cyclone Hazard Model Training & Benchmarking
SIH 2026 Problem Statement SIH26176

Trains, benchmarks, and selects the best marine hazard prediction model predicting
independent future hazard state at t+24h across LightGBM, XGBoost, and Random Forest.
Includes strict automated data leakage audits and realistic validation.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
import lightgbm as lgb
import xgboost as xgb

from ml.checkpoint_manager import get_logger, CheckpointManager
from ml.feature_engineering.disaster_features import (
    DISASTER_FEATURE_COLUMNS,
    build_disaster_dataset,
    get_chronological_splits,
)

logger = get_logger("DisasterTrainer", "disaster_training.log")
MODEL_DIR = os.path.join("ml", "models", "disaster")
os.makedirs(MODEL_DIR, exist_ok=True)


def assert_no_disaster_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Rigorous pre-training data leakage audit."""
    forbidden_targets = {"target_hazard_24h", "future_wind_24h", "is_hazard", "hazard_class"}
    for f in DISASTER_FEATURE_COLUMNS:
        if f in forbidden_targets:
            raise ValueError(f"[DATA LEAKAGE DETECTED] Feature '{f}' is derived directly from the target!")

    # Verify no feature has degenerate correlation > 0.98 with the future target
    for col in DISASTER_FEATURE_COLUMNS:
        c = np.corrcoef(train_df[col], train_df["target_hazard_24h"])[0, 1]
        if abs(c) > 0.98:
            raise ValueError(f"[DATA LEAKAGE DETECTED] Feature '{col}' has correlation {c:.4f} > 0.98 with target!")

    # Verify storm-disjoint test split
    train_storms = set(train_df[train_df["name"] != "CALM_REFERENCE"]["sid"])
    test_storms = set(test_df[test_df["name"] != "CALM_REFERENCE"]["sid"])
    overlap = train_storms.intersection(test_storms)
    if overlap:
        raise ValueError(f"[DATA LEAKAGE DETECTED] Storms {overlap} exist in both Train and Test sets!")

    logger.info("[Disaster Leakage Audit PASSED] Zero target derivation, max correlation bounded, storm-disjoint.")


def train_disaster_models(force_retrain: bool = False) -> Dict[str, Any]:
    """
    Executes end-to-end training, benchmarking, and selection for Model 1 (Disaster).
    """
    ckpt = CheckpointManager()
    if not force_retrain and ckpt.is_step_completed("disaster_model_trained"):
        logger.info("[Disaster Model] Step already completed. Checkpoint active. Skipping.")
        metrics_file = os.path.join(MODEL_DIR, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r", encoding="utf-8") as f:
                return json.load(f)

    logger.info("==================================================")
    logger.info("PHASE: LEAKAGE-SAFE DISASTER & HAZARD MODEL TRAINING")
    logger.info("==================================================")
    ckpt.set_phase("TRAINING_DISASTER_MODEL")

    logger.info("Building 24h future prediction dataset from NOAA IBTrACS & ECMWF ERA5 (1980-2025)...")
    logger.info("NOTE: Zero synthetic observations. Background calm data drawn directly from genuine ECMWF ERA5 reanalysis.")
    df = build_disaster_dataset(start_year=1980, end_year=2025, forecast_horizon_hours=24)
    train_df, val_df, test_df = get_chronological_splits(df, train_end_year=2022, val_end_year=2024)

    # Enforce Leakage Assertions
    assert_no_disaster_leakage(train_df, test_df)

    X_train = train_df[DISASTER_FEATURE_COLUMNS].values
    y_train = train_df["target_hazard_24h"].values

    X_val = val_df[DISASTER_FEATURE_COLUMNS].values
    y_val = val_df["target_hazard_24h"].values

    X_test = test_df[DISASTER_FEATURE_COLUMNS].values
    y_test = test_df["target_hazard_24h"].values

    logger.info(f"Disaster Training shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
    logger.info(f"Target distribution (24h future hazard): Train={y_train.mean():.2%}, Val={y_val.mean():.2%}, Test={y_test.mean():.2%}")

    # 1. Baseline: Current Hazard Persistence (Predict future hazard = current hazard)
    baseline_pred = (test_df["wind_speed_kmh"].values >= 45.0).astype(int)
    base_rec = recall_score(y_test, baseline_pred, zero_division=0)
    base_prec = precision_score(y_test, baseline_pred, zero_division=0)
    base_f1 = f1_score(y_test, baseline_pred, zero_division=0)
    base_cm = confusion_matrix(y_test, baseline_pred).tolist()

    results = {
        "Persistence_Baseline": {
            "recall": round(float(base_rec), 4),
            "precision": round(float(base_prec), 4),
            "f1": round(float(base_f1), 4),
            "roc_auc": round(float(roc_auc_score(y_test, baseline_pred)), 4),
            "pr_auc": round(float(average_precision_score(y_test, baseline_pred)), 4),
            "confusion_matrix": base_cm,
        }
    }
    logger.info(f"Persistence Baseline: Recall={base_rec:.4f} | Prec={base_prec:.4f} | F1={base_f1:.4f}")

    # Benchmark Candidates
    candidates = {}

    # 2. LightGBM Classifier
    logger.info("[1/3] Benchmarking LightGBM Classifier...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=25,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )
    candidates["LightGBM"] = lgb_model

    # 3. XGBoost Classifier
    logger.info("[2/3] Benchmarking XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        scale_pos_weight=1.2,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=20,
    )
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    candidates["XGBoost"] = xgb_model

    # 4. Random Forest Classifier
    logger.info("[3/3] Benchmarking Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    candidates["RandomForest"] = rf_model

    # Evaluate candidates on chronological Test Set (2022-2025 unseen storms)
    logger.info("Evaluating models on unseen modern test storm set (2022-2025)...")
    best_model_name = None
    best_score = -1.0

    for name, model in candidates.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        rec = recall_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Sanity check: Ensure metrics are not suspiciously 1.0000 across the board
        if rec == 1.0 and prec == 1.0 and f1 == 1.0:
            logger.warning(f"[SUSPICIOUS PERFECT SCORE] Model {name} achieved 1.0 on all metrics! Check data leakage.")

        results[name] = {
            "recall": round(float(rec), 4),
            "precision": round(float(prec), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "confusion_matrix": cm,
        }
        logger.info(f"Model: {name:12s} | Recall: {rec:.4f} | Precision: {prec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

        # Safety Criteria: Prioritize Recall (0.6) + F1 (0.4)
        composite_score = (rec * 0.6) + (f1 * 0.4)
        if composite_score > best_score:
            best_score = composite_score
            best_model_name = name

    logger.info(f"[Model Selected] Winner: {best_model_name} (Composite Safety Score: {best_score:.4f})")
    best_model = candidates[best_model_name]

    # Compute Regional and Seasonal Diagnostics on Unseen Test Set
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:, 1]

    regional_diagnostics = {}
    if "basin" in test_df.columns:
        for b in ["Arabian_Sea", "Bay_of_Bengal"]:
            b_mask = (test_df["basin"] == b).values
            if b_mask.sum() > 0:
                regional_diagnostics[b] = {
                    "samples": int(b_mask.sum()),
                    "hazard_samples": int(y_test[b_mask].sum()),
                    "recall": round(float(recall_score(y_test[b_mask], y_test_pred[b_mask], zero_division=0)), 4),
                    "precision": round(float(precision_score(y_test[b_mask], y_test_pred[b_mask], zero_division=0)), 4),
                    "f1": round(float(f1_score(y_test[b_mask], y_test_pred[b_mask], zero_division=0)), 4),
                }

    seasonal_diagnostics = {}
    if "season" in test_df.columns:
        for s in ["Pre_Monsoon", "Monsoon", "Post_Monsoon", "Winter"]:
            s_mask = (test_df["season"] == s).values
            if s_mask.sum() > 0:
                seasonal_diagnostics[s] = {
                    "samples": int(s_mask.sum()),
                    "hazard_samples": int(y_test[s_mask].sum()),
                    "recall": round(float(recall_score(y_test[s_mask], y_test_pred[s_mask], zero_division=0)), 4),
                    "precision": round(float(precision_score(y_test[s_mask], y_test_pred[s_mask], zero_division=0)), 4),
                    "f1": round(float(f1_score(y_test[s_mask], y_test_pred[s_mask], zero_division=0)), 4),
                }

    # Save Best Model Artifacts
    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    joblib.dump(best_model, model_path)
    logger.info(f"Saved best model to: {model_path}")

    # Feature Importance
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_.tolist()
        feature_importance = dict(zip(DISASTER_FEATURE_COLUMNS, [round(float(v), 5) for v in importances]))
    else:
        feature_importance = {}

    with open(os.path.join(MODEL_DIR, "feature_list.json"), "w", encoding="utf-8") as f:
        json.dump({
            "features": DISASTER_FEATURE_COLUMNS,
            "feature_importance": feature_importance,
        }, f, indent=2)

    metadata = {
        "model_name": "DisasterPredictionModel",
        "model_type": best_model_name,
        "version": "v4.0",
        "official_description": "ORCA 24h Cyclone & Marine Hazard Prediction Model v4.0 trained on verified NOAA IBTrACS and ECMWF ERA5 reanalysis data.",
        "prediction_horizon_hours": 24,
        "target_variable": "target_hazard_24h",
        "trained_at": datetime.now().isoformat(),
        "data_source": "NOAA/NCEI IBTrACS v04r01 North Indian Ocean + ECMWF ERA5 Ambient Reanalysis",
        "training_period": f"{train_df['year'].min()} to {train_df['year'].max()}",
        "validation_period": f"{val_df['year'].min()} to {val_df['year'].max()}",
        "test_period": f"{test_df['year'].min()} to {test_df['year'].max()}",
        "train_samples": len(train_df),
        "validation_samples": len(val_df),
        "test_samples": len(test_df),
        "leakage_safeguards": "Strict storm-disjoint chronological split, independent future target at t+24h, zero target in input",
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    metrics_payload = {
        "best_model": best_model_name,
        "metrics_summary": results[best_model_name],
        "all_benchmarks": results,
        "regional_diagnostics": regional_diagnostics,
        "seasonal_diagnostics": seasonal_diagnostics,
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    ckpt.mark_step_completed("disaster_model_trained", details=metrics_payload)
    ckpt.update_model_status("disaster", "TRAINED", metrics=results[best_model_name])

    return metrics_payload


if __name__ == "__main__":
    train_disaster_models(force_retrain=True)
