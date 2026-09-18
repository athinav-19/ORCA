# ORCA ML Feature Engineering & Physical Covariates Report
**Generated:** 2026-09-16 02:14:18  
**Target:** 3 Specialized Models (Disaster, Weather, PFZ)

---

## 1. Disaster & Cyclone Hazard Model Features

The Disaster Prediction Model incorporates 19 biophysical, atmospheric, and storm-tracking covariates:

| Feature Name | Category | Physical Rationale & Impact |
| :--- | :--- | :--- |
| `latitude`, `longitude` | Spatial | Spatial distribution of cyclogenesis in Bay of Bengal vs Arabian Sea |
| `month_sin`, `month_cos` | Temporal Cyclical | Represents smooth annual seasonal progression without discontinuity |
| `is_cyclone_season` | Temporal Binary | Flags pre-monsoon (Apr-May) and post-monsoon (Oct-Dec) high-risk windows |
| `surface_pressure_hpa` | Atmospheric | Central barometric depression (<1000 hPa indicates severe storm) |
| `pressure_tendency_6h` | Atmospheric Trend | 6-hour barometric plunge (dP/dt) indicating rapid storm intensification |
| `wind_speed_kmh` | Aerodynamic | 10m surface sustained wind velocity (IMD cyclone classification threshold) |
| `u_wind`, `v_wind` | Vector Dynamics | Zonal and meridional wind vectors resolving cyclonic circulation |
| `storm_speed_kmh` | Kinematic | Forward translation speed of storm center |
| `storm_dir_deg` | Kinematic | Bearing angle of storm movement |
| `distance_to_coast_nm` | Coastal Proximity | Distance to landfall and coastal shoaling effects |
| `bathymetry_depth_m` | Bathymetry | Continental shelf depth (<200m triggers wave shoaling and surge) |
| `sst_c`, `sst_anomaly_c` | Thermodynamic | Sea Surface Temperature (>28°C provides latent heat for cyclogenesis) |
| `olr_proxy_wm2` | Radiative Transfer | Outgoing Longwave Radiation (<180 W/m² marks deep convective cloud tops) |
| `rainfall_rate_mmh` | Hydrological | Precipitation rate indicating intense spiral cloud rainbands |
| `wave_height_m` | Hydrodynamic | Significant wave height driven by wind stress fetch |

---

## 2. Weather & Marine Condition Forecasting Features

The Weather Forecasting Model predicts multi-horizon values (6h, 12h, 24h, 48h, 72h) utilizing:
- **Autoregressive Lags**: t-6h, t-12h wind speeds, wave height lag (t-6h)
- **Pressure Tendency**: Barometric change over preceding 6 hours (Delta P_6h)
- **Diurnal & Seasonal Cycles**: 24h diurnal hour (sin/cos) and annual month (sin/cos)
- **Geographic Anchoring**: Coastal and offshore latitude/longitude coordinates

---

## 3. Potential Fishing Zone (PFZ) Biophysical Features

The PFZ Model utilizes INCOIS-validated oceanographic criteria:
- **Thermal Gradients**: Identifies temperature boundary fronts where upwelling meets ambient surface waters (Delta T >= 0.65 °C / 10km).
- **Chlorophyll-a & Gradient**: Indicates phytoplankton blooms and primary productivity (0.5 - 2.5 mg/m³).
- **Shelf Break Bathymetry**: Target continental shelf break depths (30m - 800m) where upwelling currents force nutrient upwelling.
- **Wind Stress & Upwelling Index**: Ekman transport proxy driving coastal divergence.
