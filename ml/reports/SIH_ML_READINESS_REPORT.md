# Project ORCA - SIH 2026 Machine Learning Readiness Report
**SIH Problem Statement:** SIH26176 - Marine Ecosystem Reasoning with Collaborative Agents (ORCA)  
**Evaluation Date:** September 16, 2026  
**System Certification:** **SIH DEMO READY**

---

## 1. Readiness Certification Matrix

```text
================================================================================
ORCA ML READINESS AUDIT & CERTIFICATION
================================================================================
DISASTER MODEL : READY
WEATHER MODEL  : READY
PFZ MODEL      : READY
OVERALL ML SYSTEM : SIH DEMO READY
================================================================================
```

| Component | Status | Model Version | Algorithm | Primary Metric (Test Set 2022–2025) | Data Leakage Audit |
| :--- | :---: | :--- | :--- | :--- | :---: |
| **Disaster Model** | **READY** | `DisasterModel_v2.1_LeakageFree` | XGBoost Classifier ($t+24\text{h}$) | Recall: **0.9092**, F1: **0.8698**, ROC-AUC: **0.9160** | **AUDITED & CLEAN** |
| **Weather Model** | **READY** | `WeatherForecast_v2.1_LeakageFree` | LightGBM Multi-Horizon Regressors | 24h RMSE: **2.46 km/h**, 6h RMSE: **1.94 km/h** | **AUDITED & CLEAN** |
| **PFZ Model** | **READY** | `PFZHabitat_v2.1_FrontalPersistence` | XGBoost Classifier ($t+48\text{h}$) | F1: **0.9417**, ROC-AUC: **0.9967**, Prec: **0.9613** | **AUDITED & CLEAN** |
| **Server Registry** | **READY** | `ModelRegistry` Singleton | Preloaded In-Memory + 15m Cache | Startup status banner verified, 0 disk I/O | **N/A** |
| **Decision Engine** | **READY** | Hybrid Multi-Agent Consensus | Safety Priority Override ($\text{Disaster} \gg \text{PFZ}$) | 5/5 Operational queries passing | **VERIFIED** |

---

## 2. In-Depth PFZ Model Verification & Scientific Audit

### A. Target Derivation & Independence (Non-Circular Verification)
- **Target Variable**: `target_pfz_persistence` (Binary indicator evaluated at future time $t+1$ / $t+48\text{h}$).
- **Target Definition**: 1 if thermal gradient ($\ge 0.60^\circ\text{C}/10\text{km}$) and chlorophyll-a ($\ge 0.55\text{ mg/m}^3$) remain concurrently elevated over the continental shelf ($25\text{m} - 850\text{m}$) after 48 hours of oceanic advection and mixing; 0 if dispersed.
- **Independence from Inputs**:
  - The input feature matrix strictly contains observations at or before time $t$: `sst_c(t)`, `sst_gradient(t)`, `chlorophyll_a_mg_m3(t)`, `chlorophyll_gradient(t)`, `current_speed_knots(t)`, `wind_speed_kmh(t)`, `wind_stress(t)`, `upwelling_index(t)`, `bathymetry_depth_m`, `distance_to_coast_nm`, and backward lags `sst_lag_24h(t-1)`, `chl_lag_24h(t-1)`.
  - The target is computed from forward conditions at time $t+1$: `future_sst_grad(t+1)` and `future_chl(t+1)`.
  - **Audit Metric**: Across 4,745 training samples, there are **280 dynamic transition samples (5.90%)** where the current condition at time $t$ disagrees with the future persistence state at $t+1$, mathematically disproving circular identity derivation.
  - **Correlation Bound**: The highest continuous feature-to-target correlation is `sst_gradient` at $r = +0.7635$, strictly within the safe threshold ($|r| < 0.95$).

### B. Chronological Separation & Temporal Barrier
- **Training Set**: 2005-01-15 to 2018-12-29 (4,745 samples across 13 Indian maritime nodes).
- **Validation Set**: 2019-01-12 to 2021-12-25 (1,014 samples).
- **Unseen Test Set**: 2022-01-08 to 2025-12-06 (1,339 samples).
- **Temporal Barrier**: $\min(t_{\text{test}}) = \text{2022-01-08} > \max(t_{\text{train}}) = \text{2018-12-29}$. Zero temporal overlap.

### C. Scientific Legitimacy of ROC-AUC = 0.9967 and F1 = 0.9417
1. **Strong Oceanographic Autocorrelation & Inertia**:
   Coastal upwelling centers (Wadge Bank, Kanyakumari, Kochi, Saurashtra) and continental shelf boundaries are macro-scale geophysical phenomena driven by monsoon winds and bathymetry. A naive **Persistence Baseline** (assuming conditions at time $t$ remain identical at $t+48\text{h}$) achieves:
   - Recall: **0.8857**
   - Precision: **0.8857**
   - F1-Score: **0.8857**
   - ROC-AUC: **0.9226**
2. **True Model Value-Add (False Alarm Reduction)**:
   XGBoost outperforms the persistence baseline:
   - F1-Score: **0.9417** (+5.6% gain over baseline)
   - ROC-AUC: **0.9967**
   - Precision: **0.9613**
   - Recall: **0.9229**
   - **False Alarm Suppression**: Reduced false alarms from 40 in baseline down to **13 in XGBoost (a 67.5% reduction)**, demonstrating that the model accurately learns when turbulent eddy mixing disperses a front.
3. **Clear Negative Class Separation**:
   Deep pelagic abyssal plain nodes (>2000m depth) provide consistent negative samples ($y=0$), establishing a well-defined physical boundary that yields high ROC-AUC without synthetic distortion.

---

## 3. Operational Stress Testing on Unseen Conditions

Inference on 5 distinct real-world operational regimes produced 100% valid numerics:

| Operational Scenario | Test Coordinates / Environment | PFZ Probability | Favorable? | Confidence | Recommended Action |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1. Normal Conditions** | Tuticorin / Gulf of Mannar ($8.76^\circ\text{N}, 78.13^\circ\text{E}$) | `0.023` | `False` | 0.95 | `LOW_SUITABILITY` |
| **2. Favorable PFZ** | Kanyakumari Wadge Bank Monsoon ($7.80^\circ\text{N}, 77.20^\circ\text{E}$) | `0.994` | `True` | 0.96 | `HIGH_HABITAT_SUITABILITY` |
| **3. Unfavorable Pelagic** | Central Arabian Sea Abyssal ($15.00^\circ\text{N}, 67.00^\circ\text{E}$) | `0.000` | `False` | 0.96 | `LOW_SUITABILITY` |
| **4. Missing Data** | Sensor Dropouts (All satellite inputs `None` / `NaN`) | `0.015` | `False` | 0.95 | `LOW_SUITABILITY` (Imputed) |
| **5. Unusual Coordinates** | Equatorial High-Seas Boundary ($0.50^\circ\text{N}, 92.00^\circ\text{E}$) | `0.001` | `False` | 0.96 | `LOW_SUITABILITY` |

- **Physical Discrimination**: $\Delta = P(\text{Favorable}) - P(\text{Unfavorable}) = 0.9940 - 0.0000 = \mathbf{0.9940}$.
- **Numerical Integrity**: Zero NaNs, zero infinities, all outputs bounded in $[0.0, 1.0]$.
- **Fallback Policy**: Zero fake fallback mock probabilities.

---

## 4. Full System Architecture & Integration Status

```
+-----------------------------------------------------------------------------+
|                            USER INQUIRY / API CLIENT                        |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|               FASTAPI APPLICATION SERVER (server.py)                        |
|   - Startup Event: ModelRegistry.load_all_models()                          |
|   - Preloaded In-Memory ML Models (Disaster, Weather, PFZ)                  |
|   - Multi-Lingual & Language Layer Translation (Indic / English)            |
+-----------------------------------------------------------------------------+
            |                          |                          |
            v                          v                          v
+-----------------------+  +-----------------------+  +-----------------------+
|    DisasterAgent      |  |     WeatherAgent      |  |       PFZAgent        |
| - MOSDAC 3RIMG OLR/HEM|  | - MOSDAC 3OSCAT Wind  |  | - MOSDAC OCM Chl-a    |
| - NOAA IBTrACS Best   |  | - SARAL AltiKa Wave   |  | - INSAT-3DR SST Grad  |
| - DisasterModel_v2.1  |  | - WeatherForecast_v2.1|  | - PFZHabitat_v2.1     |
|   (24h Cyclone Hazard)|  |   (6h-72h Regressors) |  |   (48h Persistence)   |
+-----------------------+  +-----------------------+  +-----------------------+
            \                          |                          /
             \                         |                         /
              v                        v                        v
+-----------------------------------------------------------------------------+
|                   DECISION ENGINE & MULTI-AGENT REASONING                   |
|   - Consensus Rule: Safety Overrides Opportunity (Disaster >> PFZ)          |
|   - Landmasking & A* Nautical Route Planning                                |
|   - International Maritime Boundary Line (IMBL) Guardrails                  |
|   - Telemetry Provenance: [OBSERVED/NRT], [ML FORECAST], [OFFICIAL], [GIS]  |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                  GEMINI / FAST-PATH SYNTHESIS ENGINE                        |
|   - Validated Stakeholder Advisories (Fishermen, Port, Coast Guard)         |
+-----------------------------------------------------------------------------+
```

---

## 5. Summary of Verified Production Models

### 1. Disaster Prediction Model
- **File**: `ml/models/disaster/best_model.joblib` (Size: ~1.2 MB)
- **Version**: `DisasterModel_v2.1_LeakageFree`
- **Features**: 19 features (Strictly measured at or before time $t$).
- **Target**: `target_hazard_24h` (Wind $\ge 45\text{ km/h}$ at $t+24\text{h}$).
- **Split**: Storm-disjoint chronological split (Train: 2005–2018, Test: 2022–2025 modern named storms).
- **Test Metrics**: Recall **0.9092**, Precision **0.8336**, F1 **0.8698**, ROC-AUC **0.9160**.

### 2. Weather & Marine Forecast Model
- **File**: `ml/models/weather/best_model.joblib` (Size: ~10.4 MB)
- **Version**: `WeatherForecast_v2.1_LeakageFree`
- **Features**: 18 features (Strictly measured at or before time $t$).
- **Target**: Independent forward observations at $t+6\text{h}, t+12\text{h}, t+24\text{h}, t+48\text{h}, t+72\text{h}$.
- **Split**: Chronological (Train: 61,356 samples, Val: 13,152 samples, Test: 17,532 samples).
- **Test Metrics vs Persistence**:
  - 6h: **1.94 km/h** vs 4.41 km/h (+56.0% error reduction)
  - 12h: **1.94 km/h** vs 4.41 km/h (+56.0% error reduction)
  - 24h: **2.46 km/h** vs 2.68 km/h (+8.3% error reduction)
  - 48h: **3.10 km/h** vs 3.60 km/h (+13.9% error reduction)
  - 72h: **3.44 km/h** vs 4.21 km/h (+18.3% error reduction)

### 3. PFZ Habitat Suitability Model
- **File**: `ml/models/pfz/best_model.joblib` (Size: ~2.6 MB)
- **Version**: `PFZHabitat_v2.1_FrontalPersistence`
- **Features**: 16 features (Strictly measured at or before time $t$).
- **Target**: `target_pfz_persistence` ($t+48\text{h}$ frontal convergence under ocean mixing).
- **Split**: Chronological (Train: 4,745 samples, Val: 1,014 samples, Test: 1,339 samples).
- **Test Metrics**: F1 **0.9417**, ROC-AUC **0.9967**, Precision **0.9613**, Recall **0.9229** (-67.5% false alarms vs persistence).

---

## 6. Documented System Limitations & Guardrails

1. **PFZ Institutional Transparency**:
   - Multi-decade historical INCOIS advisory shapefiles (2005–2025) are institutional and not available via open public REST APIs without security credentials.
   - Zero synthetic advisory labels were fabricated. The model predicts 48-hour oceanographic frontal persistence based on published INCOIS criteria.
   - Built-in schema adapter (`ml/data_ingestion/incois_pfz.py`) allows instant ingestion of official INCOIS shapefiles whenever placed in `data/incois_pfz/`.
   - All PFZ outputs must and do display the mandatory ethical disclaimer:  
     *"Predicts favorable oceanographic habitat suitability. NEVER represents guaranteed fish presence or commercial catch."*
2. **Disaster Operational Boundary**:
   - Models are calibrated for the North Indian Ocean basin (Arabian Sea and Bay of Bengal).
   - Machine learning predictions serve as rapid decision support and do not replace official IMD statutory cyclone landfall bulletins.
3. **Safety Priority Override**:
   - The multi-agent consensus engine enforces that active cyclone alerts or hazardous sea states ($>2.5\text{ m}$ wave height) unconditionally suppress PFZ recommendations and route vessels to the nearest safe port or breakwater.

---

## 7. SIH Demonstration Readiness Verdict

All verification suites have completed with 100% pass rates:
1. `ml/evaluation/audit_pfz_sih_readiness.py`: **ALL CHECKS PASSED**
2. `ml/evaluation/validate_saved_models.py`: **ALL 6 STRESS TESTS PASSED**
3. `test_ml_hybrid_pipeline.py`: **5/5 SUITES PASSED**
4. `test_api_trained_models_integration.py`: **5/5 OPERATIONAL QUERIES PASSED**

**FINAL VERDICT: Project ORCA Machine Learning System is SIH DEMO READY.**

