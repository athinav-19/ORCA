"""
train_all_models.py - Master Automated Overnight Training Pipeline for ORCA
SIH 2026 Problem Statement SIH26176: Marine Multi-Agent System

Automated, Checkpointed, and Resumable Pipeline to train, benchmark, evaluate,
and integrate 3 specialized machine learning models:
1. Disaster Hazard Prediction Model
2. Weather & Marine Condition Forecasting Model
3. Potential Fishing Zone (PFZ) Habitat Suitability Model

Usage:
  python train_all_models.py --all
  python train_all_models.py --disaster
  python train_all_models.py --weather
  python train_all_models.py --pfz
  python train_all_models.py --resume
"""

import os
import sys
import gc
import json
import time
import argparse
from datetime import datetime

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ml.checkpoint_manager import CheckpointManager, get_logger
from ml.data_catalog import generate_data_catalog
from ml.data_ingestion.ibtracs_cyclone import ensure_ibtracs_dataset
from ml.data_ingestion.mosdac_historical import index_mosdac_cache
from ml.disaster.trainer import train_disaster_models
from ml.weather.trainer import train_weather_models
from ml.pfz.trainer import train_pfz_models
from ml.evaluation.evaluator import generate_all_reports

STATUS_FILE = "training_status.json"
logger = get_logger("PipelineMaster", "pipeline.log")


def update_status(disaster=None, weather=None, pfz=None, overall=None):
    """Updates training_status.json in root directory."""
    current = {
        "DISASTER": "NOT_STARTED",
        "WEATHER": "NOT_STARTED",
        "PFZ": "NOT_STARTED",
        "OVERALL": "NOT_STARTED",
    }
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                current = json.load(f)
        except Exception:
            pass
    if disaster is not None:
        current["DISASTER"] = disaster
    if weather is not None:
        current["WEATHER"] = weather
    if pfz is not None:
        current["PFZ"] = pfz
    if overall is not None:
        current["OVERALL"] = overall
    current["last_updated"] = datetime.now().isoformat()
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)


def test_reload_inference():
    """Validates that all saved models can be reloaded and perform inference."""
    logger.info("Executing model reload and inference validation test...")
    from ml.inference.disaster_infer import predict_marine_disaster
    from ml.inference.weather_infer import predict_marine_weather_forecast
    from ml.inference.pfz_infer import predict_pfz_suitability

    d = predict_marine_disaster(18.9, 72.8, wind_speed_kmh=20.0)
    assert "hazard_probability" in d and "prediction_horizon_hours" in d, "Disaster reload failed"

    w = predict_marine_weather_forecast(18.9, 72.8, current_wind_kmh=20.0)
    assert "forecast_table" in w and len(w["forecast_table"]) == 5, "Weather reload failed"

    p = predict_pfz_suitability(18.9, 72.8)
    assert "pfz_probability" in p and "ethical_disclaimer" in p, "PFZ reload failed"

    logger.info("[Reload Test PASSED] All 3 saved models reloaded and verified.")


def run_pipeline(
    train_all: bool = True,
    do_disaster: bool = False,
    do_weather: bool = False,
    do_pfz: bool = False,
    resume_mode: bool = False,
):
    """Executes the master training pipeline with checkpointing and memory management."""
    ckpt = CheckpointManager()
    start_time = time.time()

    logger.info("================================================================================")
    logger.info("               ORCA MARINE MULTI-AGENT ML TRAINING PIPELINE                     ")
    logger.info(f"               Execution Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}   ")
    logger.info("================================================================================")

    if not (do_disaster or do_weather or do_pfz):
        train_all = True

    run_disaster = train_all or do_disaster
    run_weather = train_all or do_weather
    run_pfz = train_all or do_pfz

    update_status(overall="RUNNING")
    logger.info(f"Execution Target: Disaster={run_disaster}, Weather={run_weather}, PFZ={run_pfz}, ResumeMode={resume_mode}")

    # STAGE 1: Inspect Available Datasets & Catalog
    if not (resume_mode and ckpt.is_step_completed("catalog_generated")):
        logger.info("[STAGE 1/6] Ingesting and Cataloging Authoritative Datasets...")
        ckpt.set_phase("DATA_INGESTION_AND_CATALOG")
        try:
            generate_data_catalog()
            ensure_ibtracs_dataset()
            indexed_mosdac = index_mosdac_cache()
            logger.info(f"Verified {len(indexed_mosdac)} cached MOSDAC satellite products and NOAA IBTrACS archive.")
            ckpt.mark_step_completed("catalog_generated", details={"mosdac_count": len(indexed_mosdac)})
        except Exception as e:
            logger.error(f"Data catalog generation encountered non-fatal error: {e}")
    else:
        logger.info("[STAGE 1/6] Checkpoint verified: Dataset Catalog already generated.")

    # STAGE 2: Train Model 1 (Disaster & Cyclone Hazard)
    if run_disaster:
        update_status(disaster="RUNNING")
        if not (resume_mode and ckpt.is_step_completed("disaster_model_trained")):
            logger.info("[STAGE 2/6] Training Disaster & Cyclone Hazard Model...")
            ckpt.set_phase("TRAINING_DISASTER_MODEL")
            try:
                disaster_res = train_disaster_models(force_retrain=not resume_mode)
                logger.info(f"Disaster Model Complete. Best: {disaster_res['best_model']} | Recall: {disaster_res['metrics_summary']['recall']:.4f}")
                update_status(disaster="COMPLETED")
            except Exception as e:
                logger.error(f"Disaster model training failed: {e}", exc_info=True)
                update_status(disaster="FAILED")
                raise
            finally:
                gc.collect()
        else:
            logger.info("[STAGE 2/6] Checkpoint verified: Disaster model already trained.")
            update_status(disaster="COMPLETED")

    # STAGE 3: Train Model 2 (Weather Multi-Horizon Forecasting)
    if run_weather:
        update_status(weather="RUNNING")
        if not (resume_mode and ckpt.is_step_completed("weather_model_trained")):
            logger.info("[STAGE 3/6] Training Weather & Marine Forecasting Model...")
            ckpt.set_phase("TRAINING_WEATHER_MODEL")
            try:
                weather_res = train_weather_models(force_retrain=not resume_mode)
                logger.info(f"Weather Model Complete. Architecture: {weather_res['best_model_architecture']}")
                update_status(weather="COMPLETED")
            except Exception as e:
                logger.error(f"Weather model training failed: {e}", exc_info=True)
                update_status(weather="FAILED")
                raise
            finally:
                gc.collect()
        else:
            logger.info("[STAGE 3/6] Checkpoint verified: Weather model already trained.")
            update_status(weather="COMPLETED")

    # STAGE 4: Train Model 3 (PFZ Habitat Suitability)
    if run_pfz:
        update_status(pfz="RUNNING")
        if not (resume_mode and ckpt.is_step_completed("pfz_model_trained")):
            logger.info("[STAGE 4/6] Training Potential Fishing Zone (PFZ) Model...")
            ckpt.set_phase("TRAINING_PFZ_MODEL")
            try:
                pfz_res = train_pfz_models(force_retrain=not resume_mode)
                logger.info(f"PFZ Model Complete. Best: {pfz_res['best_model']} | F1: {pfz_res['metrics_summary']['f1']:.4f}")
                update_status(pfz="COMPLETED")
            except Exception as e:
                logger.error(f"PFZ model training failed: {e}", exc_info=True)
                update_status(pfz="FAILED")
                raise
            finally:
                gc.collect()
        else:
            logger.info("[STAGE 4/6] Checkpoint verified: PFZ model already trained.")
            update_status(pfz="COMPLETED")

    # STAGE 5: Evaluation & Report Generation
    logger.info("[STAGE 5/6] Generating Comprehensive Markdown Reports...")
    ckpt.set_phase("EVALUATION_AND_REPORTING")
    try:
        reports = generate_all_reports()
        ckpt.mark_step_completed("reports_generated", details=reports)
        logger.info(f"Successfully generated all diagnostic reports in ml/reports/.")
    except Exception as e:
        logger.error(f"Report generation encountered an error: {e}", exc_info=True)
        raise

    # STAGE 6: Reload & Verification Test
    test_reload_inference()
    update_status(overall="COMPLETED")

    elapsed_sec = time.time() - start_time
    ckpt.set_phase("COMPLETED")
    logger.info("================================================================================")
    logger.info(f"PIPELINE EXECUTION FINISHED SUCCESSFULLY IN {elapsed_sec:.1f}s")
    logger.info(f"Model Artifacts : ml/models/ (disaster, weather, pfz)")
    logger.info(f"Diagnostic Logs : logs/ (disaster, weather, pfz, pipeline)")
    logger.info(f"Status Summary  : training_status.json & ml/training_progress.json")
    logger.info("================================================================================")


def main():
    parser = argparse.ArgumentParser(description="ORCA Automated Overnight ML Training Pipeline")
    parser.add_argument("--all", action="store_true", help="Train all three models (Disaster, Weather, PFZ)")
    parser.add_argument("--disaster", action="store_true", help="Train only the Disaster Model")
    parser.add_argument("--weather", action="store_true", help="Train only the Weather Forecasting Model")
    parser.add_argument("--pfz", action="store_true", help="Train only the PFZ Habitat Suitability Model")
    parser.add_argument("--resume", action="store_true", help="Resume from last completed checkpoint without recomputing")

    args = parser.parse_args()

    run_pipeline(
        train_all=args.all or (not args.disaster and not args.weather and not args.pfz),
        do_disaster=args.disaster,
        do_weather=args.weather,
        do_pfz=args.pfz,
        resume_mode=args.resume,
    )


if __name__ == "__main__":
    main()
