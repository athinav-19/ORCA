"""
background_worker.py - Proactive Disaster & Cyclone Background Monitoring Worker
Project ORCA: Marine Multi-Agent System (ISRO SIH Problem Statement 176)

Performs proactive, autonomous monitoring of extreme maritime hazards (cyclones, squalls, tsunamis)
using the existing Disaster Agent and ISRO MOSDAC satellite datasets.

Key Features:
- Runs in an asynchronous background loop without blocking FastAPI request handling.
- Singleton guard: prevents duplicate monitoring loops during FastAPI reload.
- Severity threshold filtering: ignores harmless weather (clear skies, normal conditions).
- Hazard tracking state machine:
    NEW HAZARD       -> Dispatches proactive alert
    UPDATED HAZARD   -> Dispatches updated alert (escalated severity, >30km position shift, >30% radius change)
    ALREADY ALERTED  -> Suppresses redundant dispatch to avoid spamming users
- Safe integration with alert_service.dispatch_alert()
- Graceful startup and shutdown support
"""

import os
import asyncio
import math
import datetime
from typing import Dict, Any, List, Optional

from disaster_agent import DisasterAgent
from alert_service import alert_service, haversine_km
from supabase_client import is_supabase_configured, get_supabase_client

# Monitored Primary Maritime Sectors
DEFAULT_MONITORED_SECTORS = [
    {"name": "thoothukudi", "lat": 8.7642, "lon": 78.1348, "region": "Gulf of Mannar"},
    {"name": "chennai", "lat": 13.0827, "lon": 80.2707, "region": "North Tamil Nadu Coast"},
    {"name": "visakhapatnam", "lat": 17.6868, "lon": 83.2185, "region": "Andhra Coast"},
    {"name": "paradip", "lat": 20.3165, "lon": 86.6115, "region": "Odisha Coast"},
    {"name": "kochi", "lat": 9.9312, "lon": 76.2673, "region": "Kerala Malabar Coast"},
]

SEVERITY_WEIGHTS = {
    "NORMAL": 0,
    "LEVEL_0_NORMAL": 0,
    "CLEAR_SKIES": 0,
    "INFO": 1,
    "CAUTION": 1,
    "LEVEL_1_CAUTION": 1,
    "WARNING": 2,
    "LEVEL_2_WARNING": 2,
    "DANGER": 3,
    "LEVEL_3_DANGER": 3,
    "CRITICAL": 4,
    "LEVEL_4_CRITICAL": 4,
}


class CycloneMonitoringWorker:
    """
    Autonomous proactive background monitor running the existing DisasterAgent.
    """
    def __init__(self, interval_seconds: int = 60):
        self.interval_seconds = int(os.getenv("MONITORING_INTERVAL_SECONDS", str(interval_seconds)))
        self.disaster_agent = DisasterAgent()
        self._is_running = False
        self._task: Optional[asyncio.Task] = None
        self._hazard_state_tracker: Dict[str, Dict[str, Any]] = {}
        self._last_run_time: Optional[str] = None
        self._total_cycles = 0

    def is_running(self) -> bool:
        return self._is_running

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_running": self._is_running,
            "interval_seconds": self.interval_seconds,
            "total_cycles_completed": self._total_cycles,
            "last_run_time": self._last_run_time,
            "tracked_hazards_count": len(self._hazard_state_tracker),
        }

    async def start(self):
        """Starts the background monitoring loop if not already running."""
        if self._is_running:
            print("[Cyclone Monitor] Background worker is already running (singleton guard active).")
            return

        self._is_running = True
        self._task = asyncio.create_task(self._monitoring_loop())
        print(f"[Cyclone Monitor] Started proactive cyclone background worker (interval: {self.interval_seconds}s).")

    async def stop(self):
        """Stops the background monitoring loop cleanly."""
        if not self._is_running:
            return

        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        print("[Cyclone Monitor] Proactive background worker stopped gracefully.")

    async def _monitoring_loop(self):
        """Main periodic monitoring loop."""
        # Initial wait to let server finish startup
        await asyncio.sleep(5)

        while self._is_running:
            try:
                await self.run_monitoring_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Cyclone Monitor Error] Unexpected error in monitoring cycle: {e}")

            try:
                await asyncio.sleep(self.interval_seconds)
            except asyncio.CancelledError:
                break

    async def run_monitoring_cycle(self) -> Dict[str, Any]:
        """
        Executes one full proactive monitoring cycle across monitored maritime sectors:
        1. Query satellite telemetry via DisasterAgent.
        2. Evaluate hazard detection and severity threshold.
        3. Deduplicate / evaluate state changes (NEW, UPDATED, ALREADY ALERTED).
        4. Trigger dispatch_alert() if threshold reached.
        """
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._last_run_time = now_iso
        self._total_cycles += 1

        dispatched_alerts = []

        if not is_supabase_configured():
            return {"status": "SKIPPED", "reason": "Supabase not configured"}

        for sector in DEFAULT_MONITORED_SECTORS:
            sector_name = sector["name"]
            lat = sector["lat"]
            lon = sector["lon"]

            try:
                # Load real MOSDAC satellite observation using existing Disaster Agent
                hazard_data = self.disaster_agent.load_mosdac_hazard_data(lat, lon, sector_name)
                if not hazard_data:
                    hazard_data = self.disaster_agent.monitor_hazards(sector_name, "Today")

                has_active = hazard_data.get("has_active_hazard", False)
                severity_str = str(hazard_data.get("severity", "NORMAL")).upper()
                hazard_type = str(hazard_data.get("hazard_type", "CLEAR_SKIES")).upper()
                weight = SEVERITY_WEIGHTS.get(severity_str, 0)

                # Severity Threshold: Ignore fair weather (weight < 2, i.e. NORMAL, CAUTION)
                if not has_active or weight < 2:
                    # Harmless or normal weather; no emergency alert needed
                    continue

                # Severe Hazard Detected (Cyclone, Squall, Tsunami)
                coords = hazard_data.get("center_coordinates", [lon, lat])
                hazard_lon = coords[0] if len(coords) > 0 else lon
                hazard_lat = coords[1] if len(coords) > 1 else lat
                radius_km = float(hazard_data.get("radius_km", 100.0))
                condition_msg = hazard_data.get("condition") or hazard_data.get("description") or f"Active {hazard_type} warning."

                shelter = self.disaster_agent.find_nearest_shelter_harbor(hazard_lat, hazard_lon)
                shelter_advice = shelter.get("navigational_advice", "")
                full_message = f"{condition_msg} {shelter_advice}".strip()

                # Normalize alert severity for Supabase
                normalized_severity = "CRITICAL" if weight >= 4 else ("DANGER" if weight == 3 else "WARNING")
                clean_alert_type = "CYCLONE" if "CYCLON" in hazard_type else ("SQUALL" if "SQUALL" in hazard_type else ("TSUNAMI" if "TSUNAMI" in hazard_type else "MARINE_HAZARD"))
                title = f"{clean_alert_type} Warning: {sector_name.title()} Sector"

                # Check Hazard State Machine: NEW vs UPDATED vs ALREADY ALERTED
                hazard_key = f"{clean_alert_type}_{sector_name}"
                state_decision = self._evaluate_hazard_state(
                    hazard_key=hazard_key,
                    current_severity_weight=weight,
                    current_lat=hazard_lat,
                    current_lon=hazard_lon,
                    current_radius=radius_km,
                )

                if state_decision["action"] == "SUPPRESS":
                    print(f"[Cyclone Monitor] ALREADY ALERTED: {hazard_key} unchanged. Suppressing redundant dispatch.")
                    continue

                is_update = (state_decision["action"] == "UPDATE")
                action_label = "UPDATED HAZARD" if is_update else "NEW HAZARD"
                print(f"[Cyclone Monitor] {action_label} DETECTED: {title} | Severity={normalized_severity} | Reason={state_decision['reason']}")

                # Dispatch Alert Proactively
                dispatch_res = alert_service.dispatch_alert(
                    latitude=hazard_lat,
                    longitude=hazard_lon,
                    radius_km=radius_km,
                    severity=normalized_severity,
                    title=title,
                    message=full_message,
                    alert_type=clean_alert_type,
                    is_test=False,
                )

                # Update State Tracker
                self._hazard_state_tracker[hazard_key] = {
                    "alert_id": dispatch_res.get("alert_id"),
                    "severity_weight": weight,
                    "latitude": hazard_lat,
                    "longitude": hazard_lon,
                    "radius_km": radius_km,
                    "last_dispatched_at": now_iso,
                }

                dispatched_alerts.append({
                    "sector": sector_name,
                    "action": state_decision["action"],
                    "result": dispatch_res
                })

            except Exception as sector_err:
                print(f"[Cyclone Monitor Warning] Error evaluating sector {sector_name}: {sector_err}")

        return {
            "status": "COMPLETED",
            "timestamp": now_iso,
            "dispatched_count": len(dispatched_alerts),
            "dispatched": dispatched_alerts
        }

    def _evaluate_hazard_state(
        self,
        hazard_key: str,
        current_severity_weight: int,
        current_lat: float,
        current_lon: float,
        current_radius: float,
    ) -> Dict[str, Any]:
        """
        Evaluates whether a hazard is NEW, UPDATED, or ALREADY ALERTED:
        - NEW: Not in tracker or last alert expired (> 4 hours ago).
        - UPDATE: Severity increased, or epicenter shifted > 30 km, or radius increased > 30%.
        - SUPPRESS: Minor or negligible change (do not spam users).
        """
        prev = self._hazard_state_tracker.get(hazard_key)
        if not prev:
            return {"action": "NEW", "reason": "First detection of hazard"}

        # Check if previous alert is older than 4 hours
        try:
            last_dt = datetime.datetime.fromisoformat(prev["last_dispatched_at"])
            age_hours = (datetime.datetime.now(datetime.timezone.utc) - last_dt).total_seconds() / 3600.0
            if age_hours >= 4.0:
                return {"action": "NEW", "reason": f"Previous alert expired ({age_hours:.1f}h ago)"}
        except Exception:
            pass

        # Check 1: Severity Escalation
        if current_severity_weight > prev.get("severity_weight", 0):
            return {"action": "UPDATE", "reason": f"Severity escalated from weight {prev.get('severity_weight')} to {current_severity_weight}"}

        # Check 2: Spatial Epicenter Shift > 30 km
        prev_lat = prev.get("latitude", current_lat)
        prev_lon = prev.get("longitude", current_lon)
        shift_km = haversine_km(prev_lat, prev_lon, current_lat, current_lon)
        if shift_km >= 30.0:
            return {"action": "UPDATE", "reason": f"Hazard epicenter moved {shift_km:.1f} km"}

        # Check 3: Significant Radius Expansion (> 30%)
        prev_rad = prev.get("radius_km", current_radius)
        if prev_rad > 0 and (current_radius - prev_rad) / prev_rad >= 0.30:
            return {"action": "UPDATE", "reason": f"Affected radius expanded from {prev_rad}km to {current_radius}km"}

        # Otherwise: ALREADY ALERTED
        return {"action": "SUPPRESS", "reason": "No significant change in severity, position, or radius"}

    def trigger_controlled_hazard(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 100.0,
        severity: str = "WARNING",
        hazard_type: str = "CYCLONE",
        message: str = "CONTROLLED TEST CYCLONE HAZARD"
    ) -> Dict[str, Any]:
        """
        Controlled hazard injection method for automated testing (Part 15 Test C).
        Does NOT rely on natural weather conditions.
        """
        title = f"TEST ALERT: {hazard_type} Warning"
        return alert_service.dispatch_alert(
            latitude=latitude,
            longitude=longitude,
            radius_km=radius_km,
            severity=severity,
            title=title,
            message=message,
            alert_type=hazard_type,
            is_test=False,
        )


# Global Singleton Worker Instance
cyclone_worker = CycloneMonitoringWorker()

