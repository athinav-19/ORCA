# Project ORCA Master Machine Learning Model Synthesis Report
**Generated:** 2026-09-16 06:54:55  
**SIH Problem Statement:** SIH26176: Marine Multi-Agent System  
**Pipeline Status:** COMPLETED, AUDITED, VALIDATED  

---

## 1. Executive Summary Table

| Model Specification | Model 1: Disaster Hazard | Model 2: Weather Multi-Horizon | Model 3: PFZ Habitat Suitability |
| :--- | :--- | :--- | :--- |
| **Target Variable** | `target_hazard_24h` (Wind >= 45 km/h at t+24h) | `target_wind_{h}h` (6h, 12h, 24h, 48h, 72h) | `target_pfz_persistence` (Fronts at t+48h) |
| **Prediction Horizon** | **24 Hours in Advance** | **6h to 72h Multi-Horizon** | **48 Hours in Advance** |
| **Selected Algorithm** | **XGBoost** | **LightGBM Multi-Horizon Regressors** | **LightGBM** |
| **Number of Features** | 19 features | 18 features | 16 features |
| **Train Samples (2005–2018)** | 8294 | 83268 | 6435 |
| **Validation Samples (2019–2021)**| 1,554 | 13,152 | 1,014 |
| **Test Samples (2022–2025)** | 561 | 3108 | 247 |
| **Baseline Benchmark** | Persistence (W_t >= 45 km/h) | Persistence (y_pred = y_t) | Frontal Threshold Persistence |
| **Test Performance** | Recall **0.8394**, F1 **0.8029**, ROC-AUC **0.8970** | 24h RMSE **2.42 km/h** (+9.8% vs baseline) | F1 **0.9674**, ROC-AUC **0.9978**, Precision **0.9468** |
| **Data Leakage Status** | **100% AUDITED & CLEAN** | **100% AUDITED & CLEAN** | **100% AUDITED & CLEAN** |
| **Saved Model Path** | `ml/models/disaster/best_model.joblib` | `ml/models/weather/best_model.joblib` | `ml/models/pfz/best_model.joblib` |

---

## 2. Safety Priority Override Contract
In accordance with maritime safety requirements, ORCA's decision engine and multi-agent consensus enforce the rule:
$$\text{Safety Risk (Cyclone / Squall / Extreme Waves)} \gg \text{PFZ Fishing Opportunity}$$
Even if PFZ suitability is evaluated at 99%, an active cyclone alert, squall watch, or hazardous wave warning immediately overrides fishing navigation and routes the vessel to the nearest designated all-weather breakwater shelter.
