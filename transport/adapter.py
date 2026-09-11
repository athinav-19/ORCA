"""
transport/adapter.py - Request & Response Adapters for Offshore Satellite Communication
ISRO SIH Problem Statement 176: Marine Multi-Agent System

Handles bidirectional translation between:
- Inbound: Satellite protocol envelope -> Internal ORCA multi-agent parameters
- Outbound: Rich ORCA multi-agent response -> Bandwidth-constrained compact satellite response
"""

import re
import time
from typing import Dict, Any, Optional
from transport.models import SatelliteMessage, SatelliteCompactResponse


class OrcaRequestAdapter:
    """
    Translates an incoming validated SatelliteMessage into standard ORCA execution arguments.
    """

    @staticmethod
    def to_core_params(satellite_msg: SatelliteMessage) -> Dict[str, Any]:
        """
        Extracts parameters required for `execute_orca_core`.
        """
        payload = satellite_msg.payload
        session_id = f"sat_{satellite_msg.message_id}"

        return {
            "query": payload.query,
            "lat": float(payload.latitude),
            "lon": float(payload.longitude),
            "persona": payload.persona or "FISHERMAN",
            "language": (payload.language or "en").lower().strip(),
            "speed_knots": float(payload.speed_knots or 0.0),
            "heading_degrees": float(payload.heading_degrees or 120.0),
            "gps_accuracy_meters": float(payload.gps_accuracy_meters or 4.5),
            "session_id": session_id,
            "input_type": "TEXT",
            "raw_audio": None,
        }


class OrcaResponseAdapter:
    """
    Translates the rich, multi-modal ORCA response dictionary into a lightweight,
    bandwidth-efficient SatelliteCompactResponse strictly tailored for satellite transmission.
    """

    @staticmethod
    def extract_wave_height(orca_result: Dict[str, Any]) -> Optional[float]:
        """Extracts wave height in meters from ORCA agent outputs."""
        # 1. Try advisory dictionary
        advisory = orca_result.get("advisory")
        if isinstance(advisory, dict):
            wh = advisory.get("wave_height")
            if wh:
                m = re.search(r"(\d+(?:\.\d+)?)", str(wh))
                if m:
                    try:
                        return float(m.group(1))
                    except ValueError:
                        pass

        # 2. Try risk_assessment metrics
        risk = orca_result.get("risk_assessment")
        if isinstance(risk, dict):
            metrics = risk.get("metrics")
            if isinstance(metrics, dict) and "wave_height_m" in metrics:
                try:
                    return float(metrics["wave_height_m"])
                except (ValueError, TypeError):
                    pass
            if "wave_height_m" in risk:
                try:
                    return float(risk["wave_height_m"])
                except (ValueError, TypeError):
                    pass

        # 3. Try domain agents
        weather = orca_result.get("WEATHER_AGENT")
        if isinstance(weather, dict) and "wave_height_m" in weather:
            try:
                return float(weather["wave_height_m"])
            except (ValueError, TypeError):
                pass

        ocean = orca_result.get("OCEAN_AGENT")
        if isinstance(ocean, dict):
            wh_list = ocean.get("wave_height_m")
            if isinstance(wh_list, list) and wh_list:
                try:
                    return float(wh_list[0])
                except (ValueError, TypeError):
                    pass

        return None

    @staticmethod
    def extract_wind_speed(orca_result: Dict[str, Any]) -> Optional[float]:
        """Extracts wind speed in km/h from ORCA agent outputs."""
        advisory = orca_result.get("advisory")
        if isinstance(advisory, dict):
            ws = advisory.get("wind_speed")
            if ws:
                m = re.search(r"(\d+(?:\.\d+)?)", str(ws))
                if m:
                    try:
                        return float(m.group(1))
                    except ValueError:
                        pass

        risk = orca_result.get("risk_assessment")
        if isinstance(risk, dict):
            metrics = risk.get("metrics")
            if isinstance(metrics, dict) and "wind_speed_kmph" in metrics:
                try:
                    return float(metrics["wind_speed_kmph"])
                except (ValueError, TypeError):
                    pass

        weather = orca_result.get("WEATHER_AGENT")
        if isinstance(weather, dict):
            telemetry = weather.get("telemetry")
            if isinstance(telemetry, dict) and "wind_speed_kmh" in telemetry:
                try:
                    return float(telemetry["wind_speed_kmh"])
                except (ValueError, TypeError):
                    pass

        return None

    @classmethod
    def to_compact_response(
        cls,
        satellite_msg: SatelliteMessage,
        orca_result: Dict[str, Any],
        cached: bool = False
    ) -> SatelliteCompactResponse:
        """
        Builds a compact, bandwidth-minimized response preserving only critical maritime decision fields.
        """
        # Threat / Risk status extraction
        risk_data = orca_result.get("risk_assessment") or {}
        threat_status = (
            orca_result.get("threat_status")
            or risk_data.get("status")
            or risk_data.get("threat_level")
            or "SAFE"
        ).upper()

        # Numeric risk score extraction
        risk_score_raw = orca_result.get("risk_score") or risk_data.get("risk_score")
        risk_score = None
        if risk_score_raw is not None and risk_score_raw != "N/A":
            try:
                risk_score = round(float(risk_score_raw), 1)
            except (ValueError, TypeError):
                risk_score = None

        # Determine overall compact status
        if orca_result.get("success") is False:
            overall_status = "ERROR"
        elif threat_status in ("DANGER", "NO-GO", "CRITICAL"):
            overall_status = "DANGER"
        elif threat_status in ("WARNING", "CAUTION"):
            overall_status = "CAUTION"
        else:
            overall_status = "SUCCESS"

        # Advisory text extraction (concise summary suitable for low-bandwidth satellite packet)
        advisory_text = (
            orca_result.get("chat_text")
            or orca_result.get("native_advisory_text")
            or orca_result.get("reply")
            or orca_result.get("response")
            or "Advisory unavailable."
        )

        # Truncate extremely long paragraphs while keeping the critical navigational facts
        advisory_clean = str(advisory_text).strip()
        if len(advisory_clean) > 600:
            # Preserve first 2 sentences or 550 chars
            sentences = advisory_clean.split(". ")
            if len(sentences) >= 2:
                advisory_clean = ". ".join(sentences[:2]) + "."
            if len(advisory_clean) > 600:
                advisory_clean = advisory_clean[:597] + "..."

        wave_m = cls.extract_wave_height(orca_result)
        wind_kmh = cls.extract_wind_speed(orca_result)

        return SatelliteCompactResponse(
            protocol_version=satellite_msg.protocol_version,
            message_id=satellite_msg.message_id,
            status=overall_status,
            risk=threat_status,
            risk_score=risk_score,
            advisory=advisory_clean,
            latitude=round(float(satellite_msg.payload.latitude), 4),
            longitude=round(float(satellite_msg.payload.longitude), 4),
            timestamp=int(time.time()),
            data_age_minutes=15,
            wave_height_m=wave_m,
            wind_speed_kmh=wind_kmh,
            cached=cached,
        )

