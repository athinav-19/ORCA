# ORCA Machine Learning Subsystem
SIH 2026 Problem Statement SIH26176: Marine EcOsystem Reasoning with Collaborative Agents

## Overview
ORCA integrates 3 specialized, production-ready machine learning models into its multi-agent maritime reasoning architecture:
1. **Disaster Prediction Model** (`ml/models/disaster/`): Tropical cyclone risk, extreme wind/wave/rain hazard probability, and severity classification.
2. **Weather & Marine Forecasting Model** (`ml/models/weather/`): Multi-horizon forecasting (6h, 12h, 24h, 48h, 72h) for marine parameters.
3. **PFZ Habitat Suitability Model** (`ml/models/pfz/`): INCOIS-criteria biophysical front suitability.

## Training & Resumption
To run the automated overnight training pipeline:
```powershell
python train_all_models.py --all
```
To retrain a specific model:
```powershell
python train_all_models.py --disaster
python train_all_models.py --weather
python train_all_models.py --pfz
```
To resume an interrupted run:
```powershell
python train_all_models.py --resume
```

## Logs & Progress
- `logs/disaster_training.log`
- `logs/weather_training.log`
- `logs/pfz_training.log`
- `logs/pipeline.log`
- `ml/training_progress.json`
- `ml/status.json`
- `ml/data_catalog.csv`
