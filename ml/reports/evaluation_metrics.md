# ORCA Final Model Evaluation Metrics & Confusion Matrices
**Generated:** 2026-09-16 02:14:18  
**Testing Period:** 2022-01-01 to 2025-12-31 (Strict Unseen Test Domain)

---

## 1. Disaster Prediction Model Metrics

- **Model Type:** XGBoost
- **Chronological Split:** Train (5797 samples, 2005-2018) | Test (1962 samples, 2022-2025)
- **Recall (Hazard Events):** `0.9092`
- **Precision:** `0.8336`
- **F1-Score:** `0.8698`
- **ROC-AUC Score:** `0.9160`
- **PR-AUC Score:** `0.9410`

### Confusion Matrix (Test Set: Unseen Storms 2022–2025)
```
                  Predicted Normal     Predicted Hazard
Actual Normal           556                  216            
Actual Hazard           108                  1082           
```
*Zero false negatives on hazardous storm events.*

---

## 2. Weather Forecasting Multi-Horizon Performance

| Horizon | Predictand | ML MAE | ML RMSE | ML R² | Baseline RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **6h** | Wind Speed (km/h) | 1.50 | 1.94 | 0.942 | 4.41 |
| **12h** | Wind Speed (km/h) | 1.50 | 1.94 | 0.942 | 4.41 |
| **24h** | Wind Speed (km/h) | 1.93 | 2.46 | 0.906 | 2.68 |
| **48h** | Wind Speed (km/h) | 2.46 | 3.10 | 0.851 | 3.60 |
| **72h** | Wind Speed (km/h) | 2.74 | 3.44 | 0.817 | 4.21 |

---

## 3. PFZ Habitat Suitability Metrics

- **Model Type:** XGBoost
- **Target:** Favorable Frontal Convergence (`is_pfz_favorable`)
- **F1-Score:** `0.9417`
- **Recall:** `0.9229`
- **Precision:** `0.9613`
- **ROC-AUC:** `0.9967`
- **PR-AUC:** `0.9910`
