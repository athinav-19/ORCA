# ORCA Model 3: Potential Fishing Zone (PFZ) Habitat Suitability Report
**Generated:** 2026-09-27 23:11:55  
**Model Name:** PFZHabitatSuitabilityModel  
**Prediction Horizon:** 48 Hours Frontal Persistence (t+48h)  
**Selected Algorithm:** **LightGBM**  

---

## 1. Specification & Data Provenance
- **Target Variable:** `target_pfz_persistence` (Binary: 1 if thermal-chlorophyll front and shelf break conditions persist at t+48h; 0 if dissipated by ocean mixing).
- **Physical Datasets Ingested:**
  - MOSDAC Oceansat-3 OCM (Chlorophyll-a, Kd490 Turbidity)
  - MOSDAC INSAT-3DR LST (Sea Surface Temperature thermal gradients)
  - GEBCO Indian Ocean Bathymetric Depth Model (30m - 800m shelf break)
  - Copernicus Marine CMEMS (Current velocity and Ekman divergence)
- **Extensible Ground Truth Adapter:** `ml/data_ingestion/incois_pfz.py` (Ready to ingest external INCOIS shapefiles/CSVs).

---

## 2. Institutional Limitation & Scientific Transparency Note
> [!NOTE]
> Multi-decade archives of official INCOIS advisory shapefiles (2005–2025) are not exposed via open REST endpoints without formal institutional credentials.
> To prevent data fabrication while maintaining rigorous scientific standards:
> 1. Zero synthetic advisory labels were generated.
> 2. The model predicts **48-hour oceanographic frontal persistence** across 13 Indian shelf-break nodes.
> 3. `ml/data_ingestion/incois_pfz.py` provides the exact schema required for seamless ingestion when institutional files are placed in `data/incois_pfz/`.

---

## 3. Multi-Algorithm Benchmarks (Unseen Test Set 2022–2025)

| Algorithm / Architecture | F1-Score | Recall | Precision | ROC-AUC | Selection Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Persistence Baseline** | 0.9412 | 0.8889 | 1.0000 | 0.9444 | Baseline Reference |
| **LightGBM Classifier** | 0.9674 | 0.9889 | 0.9468 | 0.9978 | Benchmarked |
| **XGBoost Classifier** | 0.9605 | 0.9444 | 0.9770 | 0.9971 | **Benchmarked** |
| **Random Forest Classifier** | 0.9462 | 0.9778 | 0.9167 | 0.9965 | Benchmarked |

---

## 4. Confusion Matrix (Unseen Test Set 2022–2025)
```
                          Predicted Non-PFZ     Predicted 48h PFZ
Actual Non-PFZ                   152                  5              
Actual 48h PFZ Persistence       1                    89             
```

---

## 5. Mandatory Ethical Guardrail
All PFZ model predictions output the following binding disclaimer:
> **Ethical & Safety Notice:** Predicts favorable oceanographic habitat suitability based on thermal-chlorophyll front alignment. In accordance with INCOIS guidelines, PFZ suitability **NEVER represents a guarantee of fish presence or commercial catch**.
