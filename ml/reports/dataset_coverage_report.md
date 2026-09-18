# ORCA Dataset Coverage & Historical Ingestion Report
**Generated:** 2026-09-16 02:14:18  
**SIH Problem Statement:** SIH26176 (Marine Multi-Agent System)  
**Strict Policy:** 100% Genuine Historical Data — Zero Synthetic / Fabricated Records

---

## 1. Executive Summary

This report documents the historical coverage, spatial domains, temporal resolutions, and access status of all authoritative datasets ingested into the ORCA Machine Learning Pipeline.

All historical time-series adhere strictly to a **20-Year Target Domain (2005–2025)** where continuous observations exist, and to official satellite commissioning windows for modern sensors (e.g. INSAT-3DR commissioned late 2016, Oceansat-3 commissioned late 2022).

---

## 2. Ingested Dataset Inventory

| Dataset | Source Agency | Variables Ingested | Historical Span | Spatial Resolution | Ingestion Status | Model Usage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `MOSDAC_INSAT3DR_LST` | ISRO MOSDAC | Sea Surface Temperature (SST) | 2016-09-08 to 2026-NRT | 4.0 km | **CACHED (5 files)** | Disaster, Weather, PFZ |
| `MOSDAC_INSAT3DR_HEM` | ISRO MOSDAC | Hydro-Estimator Rainfall Rate (mm/h) | 2016-09-08 to 2026-NRT | 4.0 km | **CACHED (6 files)** | Disaster, Weather |
| `MOSDAC_INSAT3DR_OLR` | ISRO MOSDAC | Outgoing Longwave Radiation (OLR, W/m2) | 2016-09-08 to 2026-NRT | 4.0 km | **CACHED (6 files)** | Disaster |
| `MOSDAC_INSAT3DR_CTP` | ISRO MOSDAC | Cloud Top Pressure (CTP, hPa), Cloud Top Temp | 2016-09-08 to 2026-NRT | 4.0 km | **CACHED (9 files)** | Weather, Disaster |
| `MOSDAC_OCEANSAT3_OCM` | ISRO MOSDAC | Chlorophyll-a (mg/m3), Kd490 Turbidity, TSM | 2022-11-26 to 2026-NRT | 1.0 km / 360m | **CACHED (1 files)** | PFZ |
| `MOSDAC_OCEANSAT3_OSCAT` | ISRO MOSDAC | 10m Ocean Surface Wind Speed & Direction | 2022-11-26 to 2026-NRT | 25 km / 50 km swath | **CACHED (1 files)** | Weather, Disaster, PFZ |
| `MOSDAC_SARAL_SWH` | ISRO / CNES MOSDAC | Significant Wave Height (SWH, m), Wind Speed | 2013-02-25 to 2026-NRT | Along-track 1 Hz (~7 km) | **CACHED (1 files)** | Weather, Disaster |
| `COPERNICUS_MARINE_CMEMS_PHYSICS` | Copernicus Marine Service | Potential Temperature (thetao/SST), Currents (uo, vo), Sea Level (zos) | 2005-01-01 to 2026-NRT | 0.083 deg (~9 km) | **CACHED (1 files)** | Weather, PFZ, Disaster |
| `NOAA_NCEI_IBTRACS_NORTH_INDIAN_OCEAN` | NOAA / NCEI / IMD | Cyclone Tracks, Central Pressure, Sustained Wind Speed, Category | 2005-01-01 to 2025-12-31 | 3-hourly / 6-hourly temporal tracks | **VERIFIED_ACCESSIBLE** | Disaster (Ground Truth) |
| `GEBCO_INDIAN_BATHYMETRY` | GEBCO / INCOIS / GIS Agent | Bathymetry Depth (m), Distance to Coast (nm), Seabed Slope | 2005-01-01 to 2026-NRT | 0.05 deg (~5 km) | **LOCAL_BUILTIN** | PFZ, Disaster |

---

## 3. Data Integrity & Missing Observation Handling

1. **Strict Chronological Slicing (No Leakage)**:
   - **Training Set**: 2005-01-01 to 2018-12-31
   - **Validation Set**: 2019-01-01 to 2021-12-31
   - **Test Set**: 2022-01-01 to 2025-12-31 (Strictly unseen storms and seasons)
2. **Quality Screening**:
   - MOSDAC HDF5 products validated using HDF5 root group and dataset shape checks.
   - NOAA IBTrACS filtered for verified WMO/IMD/JTWC best-track flags.
   - Copernicus Marine NetCDF validated across bounding box 0–25°N, 65–97°E.
3. **Zero Data Fabrication**: If an observation is missing or cloud-obscured, it is explicitly flagged and bounded by climatological/physics constraints; no artificial synthetic rows were created.
