"""
alert_service.py - Project ORCA Alert Management & Proactive SMS Dispatch Service
ISRO SIH Problem Statement 176: Marine Multi-Agent System

Coordinates:
- Geodesic user discovery using true spherical Haversine formula (Earth radius = 6371.0 km)
- Storing alert events into Supabase `public.alerts` table
- Tracking delivery lifecycle in `public.alert_deliveries` (PENDING -> SENT / DELIVERED / FAILED)
- Idempotent delivery creation (prevents duplicate deliveries for same alert/user)
- Dispatching cellular SMS notifications via sms_provider abstraction
- Automatic integration with DisasterAgent and Risk Analysis Agent
- Duplicate active hazard alert protection
- Structured logging with masked phone numbers (zero PII exposure)
"""

import os
import re
import math
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

from supabase_client import get_supabase_client, is_supabase_configured
from sms_provider import (
    send_sms,
    mask_phone_number,
    normalize_and_validate_phone,
    normalize_phone_e164,
)


def format_cyclone_sms_message(
    title: str,
    message: str,
    severity: str,
    latitude: float,
    longitude: float,
    radius_km: float,
    location_name: Optional[str] = None
) -> str:
    """
    Constructs a concise, high-priority marine cyclone SMS warning adhering to maritime standards.
    Structure:
    ORCA MARINE ALERT
    CYCLONE WARNING

    Location: <area / coordinates> (Radius: <km> km)
    Severity: <severity>

    <short alert message>

    Stay away from hazardous sea conditions and follow official maritime/disaster-management instructions.
    """
    loc_str = location_name or f"{latitude:.2f}°N, {longitude:.2f}°E"
    clean_msg = (message or "").strip()
    return (
        f"ORCA MARINE ALERT\n"
        f"CYCLONE WARNING\n\n"
        f"Location: {loc_str} (Radius: {radius_km:.0f} km)\n"
        f"Severity: {severity.upper()}\n\n"
        f"{clean_msg}\n\n"
        f"Stay away from hazardous sea conditions and follow official maritime/disaster-management instructions."
    )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates great-circle distance between two GPS coordinates using the Haversine formula.
    Ensures precise spherical distance calculation (no simple planar degree subtraction).
    Radius of Earth = 6371.0 km.
    """
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


class AlertService:
    """
    Core service coordinating alert generation, affected stakeholder discovery,
    delivery tracking, and notification dispatch.
    """
    last_error: Optional[Dict[str, Any]] = None

    @staticmethod
    def find_users_in_radius(
        latitude: float,
        longitude: float,
        radius_km: float,
        sms_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Queries users whose CURRENT location is within radius_km of (latitude, longitude).
        Uses spherical Haversine geodesic math on users.latitude and users.longitude.
        Filters for notification_enabled = true AND sms_alerts_enabled = true when sms_only=True.
        Requires a valid phone number (at least 10 digits).
        """
        if not is_supabase_configured():
            print("[Alert Service Warning] Supabase not configured. Cannot query users in radius.")
            return []

        try:
            client = get_supabase_client()

            # Query active users and calculate exact Haversine distance
            query = client.table("users").select("*")
            resp = query.execute()
            all_users = resp.data or []

            affected_users = []
            for user in all_users:
                u_lat = user.get("latitude")
                u_lon = user.get("longitude")
                phone = user.get("phone_number")

                if u_lat is None or u_lon is None:
                    continue

                # Validate phone number
                if not phone:
                    continue
                clean_phone = re.sub(r"[^\d]", "", str(phone))
                if len(clean_phone) < 10:
                    continue

                # Check notification settings
                if sms_only:
                    sms_on = user.get("sms_alerts_enabled")
                    if sms_on is None:
                        sms_on = user.get("sms_notifications_enabled", True)
                    notif_on = user.get("notification_enabled")
                    if notif_on is None:
                        notif_on = user.get("app_notifications_enabled", True)

                    if not (bool(sms_on) and bool(notif_on)):
                        continue

                try:
                    dist = haversine_km(latitude, longitude, float(u_lat), float(u_lon))
                    if dist <= radius_km:
                        user_copy = dict(user)
                        user_copy["distance_km"] = round(dist, 2)
                        user_copy["latitude"] = float(u_lat)
                        user_copy["longitude"] = float(u_lon)
                        affected_users.append(user_copy)
                except (ValueError, TypeError):
                    continue

            print(f"[Alert System] AFFECTED_USERS_FOUND: {len(affected_users)} users within {radius_km} km of ({latitude:.4f}, {longitude:.4f})")
            return affected_users

        except Exception as e:
            print(f"[Alert System Error] Failed to find users in radius: {e}")
            return []

    @classmethod
    def create_alert(
        cls,
        alert_type: str,
        title: str,
        message: str,
        latitude: float,
        longitude: float,
        radius_km: float,
        severity: str,
        expires_at: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Inserts a new alert record into Supabase `public.alerts` table using existing schema:
        ['id', 'title', 'message', 'radius_km', 'severity', 'alert_type', 'latitude', 'longitude', 'created_at', 'expires_at']
        """
        if not is_supabase_configured():
            print("[Alert Service Warning] Supabase not configured. Cannot save alert to database.")
            return None

        try:
            client = get_supabase_client()
            now = datetime.now(timezone.utc)
            expiry = expires_at or (now + timedelta(hours=24)).isoformat()

            payload = {
                "title": title.strip(),
                "message": message.strip(),
                "radius_km": round(float(radius_km), 1),
                "severity": severity.upper(),
                "alert_type": alert_type.upper(),
                "latitude": round(float(latitude), 4),
                "longitude": round(float(longitude), 4),
                "created_at": now.isoformat(),
                "expires_at": expiry,
            }

            resp = client.table("alerts").insert(payload).execute()
            if resp.data and len(resp.data) > 0:
                created = resp.data[0]
                alert_id = created.get("id")
                cls.last_error = None
                print(f"[Alert System] ALERT CREATED: ID={alert_id} | Type={alert_type} | Severity={severity}")
                return created

            return None

        except Exception as e:
            err_msg = getattr(e, "message", None) or str(e)
            err_code = getattr(e, "code", None)
            err_details = getattr(e, "details", None)
            err_hint = getattr(e, "hint", None)
            cls.last_error = {
                "table": "alerts",
                "operation": "INSERT",
                "columns": ["title", "message", "radius_km", "severity", "alert_type", "latitude", "longitude", "created_at", "expires_at"],
                "message": err_msg,
                "code": err_code,
                "details": err_details,
                "hint": err_hint,
            }
            print(f"""
==================================================
[SUPABASE ERROR REPORT: alerts]
- Table:           alerts
- Operation:       INSERT
- Error Message:   {err_msg}
- Error Code:      {err_code}
- Details:         {err_details}
- Hint:            {err_hint}
==================================================""")
            return None

    @classmethod
    def create_deliveries_and_dispatch(
        cls,
        alert_id: str,
        affected_users: List[Dict[str, Any]],
        message: str,
        formatted_sms: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Creates `alert_deliveries` records in public.alert_deliveries with initial PENDING status,
        dispatches SMS notifications, and updates delivery status (SENT / DELIVERED / FAILED).
        Matches schema: ['id', 'alert_id', 'user_id', 'channel', 'status', 'sent_at', 'delivered_at', 'error_message', 'provider_message_id', 'created_at']
        """
        if not is_supabase_configured():
            return {
                "affected_users_count": 0,
                "sent_count": 0,
                "failed_count": 0,
                "pending_count": 0,
                "skipped_duplicates_count": 0,
                "notice": "Supabase not configured."
            }

        client = get_supabase_client()
        sms_text = formatted_sms or message

        sent_count = 0
        failed_count = 0
        pending_count = 0
        skipped_count = 0

        for user in affected_users:
            user_id = user.get("id")
            phone = user.get("phone_number")
            if not user_id or not phone:
                continue

            # Idempotency Check: (alert_id + user_id + channel="SMS")
            try:
                existing = client.table("alert_deliveries").select("id, status, channel").eq(
                    "alert_id", alert_id
                ).eq("user_id", user_id).eq("channel", "SMS").execute()

                if existing.data and len(existing.data) > 0:
                    ex_status = existing.data[0].get("status", "UNKNOWN")
                    if ex_status in ("SENT", "DELIVERED", "PENDING"):
                        print(f"[Alert System] DUPLICATE_SMS_SKIPPED: Delivery already exists with status '{ex_status}' for user {user_id} on alert {alert_id}")
                        skipped_count += 1
                        continue
            except Exception as check_err:
                print(f"[Alert System Warning] Delivery idempotency check note: {check_err}")

            # Step 1: Insert initial PENDING delivery record
            now_iso = datetime.now(timezone.utc).isoformat()
            delivery_id = str(uuid.uuid4())
            delivery_record = {
                "id": delivery_id,
                "alert_id": alert_id,
                "user_id": user_id,
                "channel": "SMS",
                "status": "PENDING",
                "created_at": now_iso,
            }

            try:
                client.table("alert_deliveries").insert(delivery_record).execute()
                print(f"[Alert System] ALERT_DELIVERY_INITIALIZED: id={delivery_id} | user={user_id} | channel=SMS | status=PENDING")
            except Exception as ins_err:
                d_msg = getattr(ins_err, "message", None) or str(ins_err)
                d_code = getattr(ins_err, "code", None)
                print(f"[Alert System Error] Failed to initialize alert_delivery for user {user_id}: {d_msg} ({d_code})")
                failed_count += 1
                continue

            # Step 2: Validate phone number before making network calls
            is_valid_phone, normalized_phone, phone_err = normalize_and_validate_phone(phone)
            if not is_valid_phone:
                fail_payload = {
                    "status": "FAILED",
                    "error_message": phone_err or "Invalid phone number format."
                }
                try:
                    client.table("alert_deliveries").update(fail_payload).eq("id", delivery_id).execute()
                except Exception as up_err:
                    print(f"[Alert System Error] Failed to update invalid phone failure: {up_err}")
                print(f"[SMS Provider] SMS submission failed | user={mask_phone_number(phone)} | reason={phone_err}")
                failed_count += 1
                continue

            # Step 3: Dispatch SMS via provider abstraction (with transient retry engine)
            dispatch_res = send_sms(normalized_phone, sms_text)
            sms_status = dispatch_res.get("status", "FAILED")
            provider_msg_id = dispatch_res.get("provider_message_id")
            sms_err = dispatch_res.get("error")

            # Step 4: Update delivery status
            update_payload: Dict[str, Any] = {}
            if sms_status == "DELIVERED":
                sent_count += 1
                update_payload = {
                    "status": "DELIVERED",
                    "sent_at": now_iso,
                    "delivered_at": now_iso,
                    "provider_message_id": provider_msg_id,
                    "error_message": None,
                }
                print(f"[Alert System] SMS_DELIVERED: user={user_id} | phone={mask_phone_number(normalized_phone)} | id={provider_msg_id}")
            elif sms_status == "SENT":
                sent_count += 1
                update_payload = {
                    "status": "SENT",
                    "sent_at": now_iso,
                    "provider_message_id": provider_msg_id,
                    "error_message": None,
                }
                print(f"[Alert System] SMS_SENT: user={user_id} | phone={mask_phone_number(normalized_phone)} | id={provider_msg_id}")
            else:
                failed_count += 1
                update_payload = {
                    "status": "FAILED",
                    "provider_message_id": provider_msg_id,
                    "error_message": sms_err or "SMS transmission failed"
                }
                print(f"[SMS Provider] SMS submission failed | user={mask_phone_number(normalized_phone)} | reason={sms_err}")

            try:
                client.table("alert_deliveries").update(update_payload).eq("id", delivery_id).execute()
            except Exception as upd_err:
                print(f"[Alert System Error] Failed to update delivery status for {delivery_id}: {upd_err}")

        return {
            "affected_users_count": len(affected_users),
            "sent_count": sent_count,
            "failed_count": failed_count,
            "pending_count": pending_count,
            "skipped_duplicates_count": skipped_count
        }

    @classmethod
    def dispatch_alert(
        cls,
        latitude: float,
        longitude: float,
        radius_km: float,
        severity: str,
        title: str,
        message: str,
        alert_type: str = "CYCLONE",
        expires_at: Optional[str] = None,
        is_test: bool = False,
    ) -> Dict[str, Any]:
        """
        Central reusable alert dispatch engine:
        1. Duplicate protection: checks if recent active alert of same type exists within 50 km.
        2. Creates alert row in public.alerts.
        3. Finds affected users from Supabase using true Haversine distance.
        4. Creates alert_deliveries records with PENDING status.
        5. Sends SMS notifications to each affected user.
        6. Updates delivery statuses.
        7. Returns sanitized summary without exposing private phone numbers.
        """
        if not is_supabase_configured():
            return {
                "success": False,
                "error": "Supabase not configured in server environment.",
                "affected_users": 0,
                "sms_sent": 0,
                "sms_failed": 0,
            }

        client = get_supabase_client()

        # Step 1: Duplicate alert protection (checks within 50 km and 4-hour window)
        if not is_test:
            try:
                four_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()
                recent_alerts = client.table("alerts").select("id, alert_type, latitude, longitude, created_at").eq(
                    "alert_type", alert_type.upper()
                ).gte("created_at", four_hours_ago).execute()

                for ra in (recent_alerts.data or []):
                    ra_lat = ra.get("latitude")
                    ra_lon = ra.get("longitude")
                    if ra_lat is not None and ra_lon is not None:
                        d_km = haversine_km(latitude, longitude, float(ra_lat), float(ra_lon))
                        if d_km <= 50.0:
                            print(f"[Alert System] DUPLICATE_ALERT_SUPPRESSED: Active {alert_type} alert ({ra['id']}) already exists within {d_km:.1f} km.")
                            return {
                                "success": True,
                                "is_duplicate": True,
                                "alert_id": ra["id"],
                                "title": title,
                                "severity": severity,
                                "alert_type": alert_type,
                                "center": {"latitude": latitude, "longitude": longitude},
                                "radius_km": radius_km,
                                "affected_users": 0,
                                "sms_sent": 0,
                                "sms_failed": 0,
                                "sms_pending": 0,
                                "skipped_duplicates": 0,
                                "notice": f"Duplicate active alert suppressed. Existing alert ({ra['id']}) active within {d_km:.1f} km."
                            }
            except Exception as dup_err:
                print(f"[Alert System Warning] Duplicate check note: {dup_err}")

        # Step 2: Create alert in public.alerts
        created_alert = cls.create_alert(
            alert_type=alert_type,
            title=title,
            message=message,
            latitude=latitude,
            longitude=longitude,
            radius_km=radius_km,
            severity=severity,
            expires_at=expires_at,
        )

        if not created_alert:
            err_info = cls.last_error or {}
            return {
                "success": False,
                "error": "Failed to create alert in Supabase",
                "supabase_error": err_info.get("message"),
                "supabase_code": err_info.get("code"),
                "affected_users": 0,
                "sms_sent": 0,
                "sms_failed": 0,
            }

        alert_id = created_alert.get("id")

        # Step 3: Find affected users within radius
        affected_users = cls.find_users_in_radius(latitude, longitude, radius_km, sms_only=True)

        # Step 4 & 5: Format concise cyclone SMS alert and dispatch
        formatted_sms = format_cyclone_sms_message(
            title=title,
            message=message,
            severity=severity,
            latitude=latitude,
            longitude=longitude,
            radius_km=radius_km,
        )

        delivery_summary = cls.create_deliveries_and_dispatch(
            alert_id=alert_id,
            affected_users=affected_users,
            message=message,
            formatted_sms=formatted_sms,
        )

        # Step 6: Log operational metrics
        print(f"""
[ALERT DISPATCH SUMMARY]
- Alert ID:          {alert_id}
- Alert Type:        {alert_type}
- Severity:          {severity}
- Affected Users:    {delivery_summary.get('affected_users_count', 0)}
- SMS Submitted:     {delivery_summary.get('sent_count', 0)}
- SMS Failures:      {delivery_summary.get('failed_count', 0)}
- Duplicate Skips:   {delivery_summary.get('skipped_duplicates_count', 0)}
""")

        # Step 7: Return safe summary (zero phone numbers)
        return {
            "success": True,
            "is_test": is_test,
            "is_duplicate": False,
            "alert_id": alert_id,
            "title": title,
            "severity": severity,
            "alert_type": alert_type,
            "center": {
                "latitude": latitude,
                "longitude": longitude,
            },
            "radius_km": radius_km,
            "affected_users": delivery_summary.get("affected_users_count", 0),
            "sms_sent": delivery_summary.get("sent_count", 0),
            "sms_failed": delivery_summary.get("failed_count", 0),
            "sms_pending": delivery_summary.get("pending_count", 0),
            "skipped_duplicates": delivery_summary.get("skipped_duplicates_count", 0),
        }

    @classmethod
    def get_alerts(cls, limit: int = 20, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves recent alerts with delivery summary statistics.
        Never exposes raw phone numbers.
        """
        if not is_supabase_configured():
            return []

        try:
            client = get_supabase_client()
            q = client.table("alerts").select("*").order("created_at", desc=True).limit(limit)
            if severity:
                q = q.eq("severity", severity.upper())
            resp = q.execute()
            alerts = resp.data or []

            for alert in alerts:
                aid = alert.get("id")
                if aid:
                    try:
                        deliv_resp = client.table("alert_deliveries").select("status").eq("alert_id", aid).execute()
                        delivs = deliv_resp.data or []
                        sent = sum(1 for d in delivs if d.get("status") in ("SENT", "DELIVERED"))
                        failed = sum(1 for d in delivs if d.get("status") == "FAILED")
                        pending = sum(1 for d in delivs if d.get("status") == "PENDING")
                        alert["delivery_summary"] = {
                            "affected_users_count": len(delivs),
                            "sent_count": sent,
                            "failed_count": failed,
                            "pending_count": pending,
                        }
                    except Exception:
                        alert["delivery_summary"] = {
                            "affected_users_count": 0,
                            "sent_count": 0,
                            "failed_count": 0,
                            "pending_count": 0,
                        }
            return alerts
        except Exception as e:
            print(f"[Alert System Error] Failed to query alerts: {e}")
            return []

    @classmethod
    def get_alert_by_id(cls, alert_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves detailed alert record with delivery breakdown.
        Never exposes raw phone numbers.
        """
        if not is_supabase_configured():
            return None

        try:
            client = get_supabase_client()
            resp = client.table("alerts").select("*").eq("id", alert_id).execute()
            if not resp.data or len(resp.data) == 0:
                return None

            alert = resp.data[0]
            deliv_resp = client.table("alert_deliveries").select(
                "id, alert_id, user_id, status, sent_at, delivered_at, error_message, created_at"
            ).eq("alert_id", alert_id).execute()
            deliveries = deliv_resp.data or []

            sent = sum(1 for d in deliveries if d.get("status") in ("SENT", "DELIVERED"))
            failed = sum(1 for d in deliveries if d.get("status") == "FAILED")
            pending = sum(1 for d in deliveries if d.get("status") == "PENDING")

            alert["affected_user_count"] = len(deliveries)
            alert["sent_count"] = sent
            alert["failed_count"] = failed
            alert["pending_count"] = pending
            alert["deliveries"] = deliveries

            return alert
        except Exception as e:
            print(f"[Alert System Error] Failed to get alert {alert_id}: {e}")
            return None

    @classmethod
    def process_disaster_agent_output(
        cls,
        disaster_output: Dict[str, Any],
        lat: float = 8.7642,
        lon: float = 78.1348
    ) -> Optional[Dict[str, Any]]:
        """
        Connects Disaster Agent & Risk Analysis findings to proactive alert dispatch:
        - If active cyclonic convection, storm surge, squall cluster, or tsunami is detected:
          triggers dispatch_alert() with automatic duplicate alert suppression.
        - If sea state is normal/fair, safely takes no action.
        """
        if not isinstance(disaster_output, dict):
            return None

        hazard_summary = disaster_output.get("hazard_summary") or {}
        has_active = hazard_summary.get("has_active_hazard", False)
        status = disaster_output.get("status", "SAFE").upper()

        if not has_active and status not in ("DANGER", "WARNING"):
            # Normal fair weather conditions - no emergency alert needed
            return None

        hazard_type = hazard_summary.get("hazard_type", "SEVERE_MARINE_HAZARD")
        severity = hazard_summary.get("severity", "WARNING")
        radius_km = float(hazard_summary.get("radius_km", 50.0))
        condition = hazard_summary.get("condition") or hazard_summary.get("description") or "Severe marine hazard detected by ISRO MOSDAC satellite telemetry."

        cyclone_details = disaster_output.get("cyclone_track") or {}
        storm_name = cyclone_details.get("cyclone_name") or "Tropical Cyclone"

        if "CYCLON" in hazard_type.upper():
            title = f"Cyclone Warning: {storm_name}"
            alert_type = "CYCLONE"
        elif "SQUALL" in hazard_type.upper():
            title = "Severe Marine Squall Warning"
            alert_type = "SQUALL"
        elif "TSUNAMI" in hazard_type.upper():
            title = "Subsea Seismic Tsunami Advisory"
            alert_type = "TSUNAMI"
        else:
            title = f"Marine Alert: {hazard_type}"
            alert_type = "MARINE_HAZARD"

        shelter = disaster_output.get("nearest_shelter") or {}
        shelter_advice = shelter.get("navigational_advice", "")
        full_msg = f"{condition} {shelter_advice}".strip()

        print(f"\n[Alert Service] Disaster Agent detected active hazard: {hazard_type} (Severity: {severity})")
        print(f"[Alert Service] Automatically evaluating alert dispatch with duplicate protection...")

        return cls.dispatch_alert(
            latitude=lat,
            longitude=lon,
            radius_km=radius_km,
            severity=severity,
            title=title,
            message=full_msg,
            alert_type=alert_type,
            is_test=False,
        )


# Global Singleton Alert Service
alert_service = AlertService()

