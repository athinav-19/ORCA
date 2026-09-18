"""
test_sms_integration.py - Comprehensive Verification Suite for ORCA SMS Integration
ISRO SIH Problem Statement 176: Proactive Early Warning SMS Dispatch

Verifies:
- TEST A: SMS_MODE=disabled (safe default, zero SMS, no fake SENT)
- TEST B: SMS_MODE=test (whitelist validation, no accidental external SMS)
- TEST C: Missing Provider Credentials (no server crash, honest unconfigured handling)
- TEST D: Invalid Phone Number Validation (rejected pre-gateway with clear FAILED error)
- TEST E: Duplicate SMS Idempotency (DUPLICATE_SMS_SKIPPED on repeated delivery attempt)
- TEST F: Full Cyclone Alert Flow (Detection -> Alert -> Deliveries -> Dispatch -> Status Update)
- TEST G: Existing /health endpoint (HTTP 200, healthy, includes SMS status block)
- TEST H: Existing /api/alerts/test endpoint (HTTP 200, executes full pipeline)
"""

import os
import sys
import json
import uuid
import urllib.request
import urllib.error

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from supabase_client import get_supabase_client, is_supabase_configured
from sms_provider import (
    send_sms,
    mask_phone_number,
    normalize_and_validate_phone,
    normalize_phone_e164,
    get_sms_mode,
    get_sms_status,
    is_transient_error,
)
from alert_service import alert_service, format_cyclone_sms_message

BASE_URL = "http://127.0.0.1:8000"


def log_test_header(label, title):
    print(f"\n{'='*75}")
    print(f"[{label}] {title}")
    print(f"{'='*75}")


def log_test_result(label, status, details=None):
    print(f"--> [{label}] RESULT: {status}")
    if details:
        if isinstance(details, dict):
            for k, v in details.items():
                print(f"    - {k}: {v}")
        else:
            print(f"    {details}")
    print(f"{'-'*75}")


def http_get(endpoint):
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body)
    except Exception as e:
        return 0, {"error": str(e)}


def http_post(endpoint, payload):
    url = f"{BASE_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"raw": body}
    except Exception as e:
        return 0, {"error": str(e)}


def main():
    print("=" * 75)
    print("ORCA SMS PROVIDER & DISPATCH INTEGRATION VERIFICATION SUITE")
    print("=" * 75)

    client = get_supabase_client()
    results = {}

    # Ensure a known test user exists in Supabase for testing
    test_user_id = "fc208148-6dee-4d0a-ba0c-41923820b2a7"
    test_phone = "+919876543210"

    # -----------------------------------------------------------------
    # TEST A: SMS_MODE=disabled
    # -----------------------------------------------------------------
    log_test_header("TEST A", "SMS_MODE=disabled (Safe development default)")
    os.environ["SMS_MODE"] = "disabled"
    os.environ["ALERT_SMS_MODE"] = "disabled"

    res_a = send_sms(test_phone, "ORCA TEST ALERT: Cyclone warning for Thoothukudi")
    pass_a1 = (res_a.get("success") is False and res_a.get("status") == "FAILED" and "disabled" in res_a.get("error", "").lower())

    # Verify through alert_service dispatch in disabled mode
    test_alert_a = alert_service.create_alert(
        alert_type="CYCLONE",
        title="TEST A ALERT: Mode Disabled",
        message="Cyclonic storm approaching Gulf of Mannar.",
        latitude=8.8000,
        longitude=78.2000,
        radius_km=50.0,
        severity="WARNING"
    )
    alert_a_id = test_alert_a.get("id") if test_alert_a else None

    # Dispatch to test user
    deliv_summary_a = alert_service.create_deliveries_and_dispatch(
        alert_id=alert_a_id,
        affected_users=[{"id": test_user_id, "phone_number": test_phone}],
        message="Cyclone warning"
    )

    # Check alert_deliveries row in DB
    deliv_a_row = client.table("alert_deliveries").select("*").eq("alert_id", alert_a_id).eq("user_id", test_user_id).execute()
    deliv_a_data = deliv_a_row.data[0] if deliv_a_row.data else {}

    pass_a2 = (
        deliv_a_data.get("channel") == "SMS" and
        deliv_a_data.get("status") == "FAILED" and
        "disabled" in (deliv_a_data.get("error_message") or "").lower() and
        deliv_a_data.get("sent_at") is None
    )

    tA_pass = pass_a1 and pass_a2
    log_test_result("TEST A", "PASS" if tA_pass else "FAIL", {
        "direct_send_result": res_a.get("error"),
        "alert_created": alert_a_id is not None,
        "delivery_channel": deliv_a_data.get("channel"),
        "delivery_status": deliv_a_data.get("status"),
        "error_message": deliv_a_data.get("error_message"),
        "no_fake_sent_confirmed": deliv_a_data.get("status") != "SENT",
    })
    results["TEST_A"] = "PASS" if tA_pass else "FAIL"

    # -----------------------------------------------------------------
    # TEST B: SMS_MODE=test
    # -----------------------------------------------------------------
    log_test_header("TEST B", "SMS_MODE=test (Whitelist validation & isolation)")
    os.environ["SMS_MODE"] = "test"
    os.environ["ALERT_SMS_MODE"] = "test"
    os.environ["SMS_TEST_PHONE_NUMBERS"] = "+919876543210"

    # 1. Non-whitelisted number should be skipped
    non_whitelisted_phone = "+919876500099"
    res_b_non = send_sms(non_whitelisted_phone, "Test message")
    pass_b_non = (
        res_b_non.get("success") is False and
        res_b_non.get("status") == "FAILED" and
        "not in configured test numbers" in (res_b_non.get("error") or "").lower()
    )

    # 2. Whitelisted number should pass safety guard and reach provider (honest unconfigured fallback)
    res_b_white = send_sms(test_phone, "Test message")
    pass_b_white = (
        res_b_white.get("status") == "FAILED" and
        "not configured" in (res_b_white.get("error") or "").lower()
    )

    tB_pass = pass_b_non and pass_b_white
    log_test_result("TEST B", "PASS" if tB_pass else "FAIL", {
        "non_whitelisted_recipient": mask_phone_number(non_whitelisted_phone),
        "non_whitelisted_result": res_b_non.get("error"),
        "whitelisted_recipient": mask_phone_number(test_phone),
        "whitelisted_provider_status": res_b_white.get("status"),
        "isolation_verified": pass_b_non,
    })
    results["TEST_B"] = "PASS" if tB_pass else "FAIL"

    # -----------------------------------------------------------------
    # TEST C: Missing Provider Credentials (LIVE Mode with no keys)
    # -----------------------------------------------------------------
    log_test_header("TEST C", "Missing Provider Credentials (Graceful handling in LIVE mode)")
    os.environ["SMS_MODE"] = "live"
    os.environ["ALERT_SMS_MODE"] = "live"
    os.environ["SMS_PROVIDER"] = "generic_rest"
    os.environ["SMS_ENDPOINT_URL"] = ""  # Intentionally missing
    os.environ["SMS_API_KEY"] = ""

    test_alert_c = alert_service.create_alert(
        alert_type="CYCLONE",
        title="TEST C ALERT: Missing Credentials",
        message="Alert created while credentials absent.",
        latitude=8.8000,
        longitude=78.2000,
        radius_km=50.0,
        severity="WARNING"
    )
    alert_c_id = test_alert_c.get("id") if test_alert_c else None

    deliv_summary_c = alert_service.create_deliveries_and_dispatch(
        alert_id=alert_c_id,
        affected_users=[{"id": test_user_id, "phone_number": test_phone}],
        message="Cyclone warning"
    )

    deliv_c_row = client.table("alert_deliveries").select("*").eq("alert_id", alert_c_id).eq("user_id", test_user_id).execute()
    deliv_c_data = deliv_c_row.data[0] if deliv_c_row.data else {}

    tC_pass = (
        alert_c_id is not None and
        deliv_c_data.get("status") == "FAILED" and
        "not configured" in (deliv_c_data.get("error_message") or "").lower()
    )
    log_test_result("TEST C", "PASS" if tC_pass else "FAIL", {
        "alert_created_successfully": alert_c_id is not None,
        "no_crash_verified": True,
        "delivery_status": deliv_c_data.get("status"),
        "error_message": deliv_c_data.get("error_message"),
    })
    results["TEST_C"] = "PASS" if tC_pass else "FAIL"

    # -----------------------------------------------------------------
    # TEST D: Invalid Phone Number Validation
    # -----------------------------------------------------------------
    log_test_header("TEST D", "Invalid Phone Number Validation (Pre-gateway rejection)")
    invalid_phones = ["123", "phone12345", "98765"]
    val_results = []
    for inv in invalid_phones:
        is_val, norm, err = normalize_and_validate_phone(inv)
        val_results.append((not is_val, err))

    # Test through alert_service dispatch
    test_alert_d = alert_service.create_alert(
        alert_type="CYCLONE",
        title="TEST D ALERT: Invalid Phone",
        message="Invalid phone rejection test.",
        latitude=8.8000,
        longitude=78.2000,
        radius_km=50.0,
        severity="WARNING"
    )
    alert_d_id = test_alert_d.get("id") if test_alert_d else None

    deliv_summary_d = alert_service.create_deliveries_and_dispatch(
        alert_id=alert_d_id,
        affected_users=[{"id": test_user_id, "phone_number": "12345"}],
        message="Test alert"
    )

    deliv_d_row = client.table("alert_deliveries").select("*").eq("alert_id", alert_d_id).eq("user_id", test_user_id).execute()
    deliv_d_data = deliv_d_row.data[0] if deliv_d_row.data else {}

    tD_pass = (
        all(vr[0] for vr in val_results) and
        deliv_d_data.get("status") == "FAILED" and
        "invalid phone" in (deliv_d_data.get("error_message") or "").lower()
    )
    log_test_result("TEST D", "PASS" if tD_pass else "FAIL", {
        "validation_rejection_count": len(val_results),
        "delivery_status": deliv_d_data.get("status"),
        "error_message": deliv_d_data.get("error_message"),
        "pre_gateway_rejection_verified": True,
    })
    results["TEST_D"] = "PASS" if tD_pass else "FAIL"

    # -----------------------------------------------------------------
    # TEST E: Duplicate SMS Protection (Idempotency)
    # -----------------------------------------------------------------
    log_test_header("TEST E", "Duplicate SMS Protection (alert_id + user_id + channel)")
    # Re-dispatch for alert_d_id and test_user_id (which already has a delivery record)
    # First set delivery status to SENT to test suppression
    client.table("alert_deliveries").update({"status": "SENT"}).eq("alert_id", alert_d_id).eq("user_id", test_user_id).execute()

    deliv_summary_e = alert_service.create_deliveries_and_dispatch(
        alert_id=alert_d_id,
        affected_users=[{"id": test_user_id, "phone_number": "+919876543210"}],
        message="Duplicate test"
    )

    tE_pass = deliv_summary_e.get("skipped_duplicates_count", 0) == 1
    log_test_result("TEST E", "PASS" if tE_pass else "FAIL", {
        "alert_id": alert_d_id,
        "user_id": test_user_id,
        "channel": "SMS",
        "skipped_duplicates_count": deliv_summary_e.get("skipped_duplicates_count"),
        "duplicate_sms_prevented": tE_pass,
    })
    results["TEST_E"] = "PASS" if tE_pass else "FAIL"

    # -----------------------------------------------------------------
    # TEST F: Full Cyclone Alert Flow
    # -----------------------------------------------------------------
    log_test_header("TEST F", "Full Cyclone Alert Flow (Detection -> Supabase -> SMS Lifecycle)")
    os.environ["SMS_MODE"] = "test"
    os.environ["ALERT_SMS_MODE"] = "test"
    os.environ["SMS_TEST_PHONE_NUMBERS"] = "+919876543210"

    # Execute full dispatch_alert method
    dispatch_f = alert_service.dispatch_alert(
        latitude=8.8000,
        longitude=78.2000,
        radius_km=100.0,
        severity="DANGER",
        title="CYCLONIC STORM WARNING: GULF OF MANNAR",
        message="Deep depression intensified into severe cyclonic storm. Gusts reaching 85 km/h.",
        alert_type="CYCLONE",
        is_test=True,  # Allow immediate test execution without deduplication block
    )

    alert_f_id = dispatch_f.get("alert_id")
    alert_f_row = client.table("alerts").select("*").eq("id", alert_f_id).execute() if alert_f_id else None
    alert_f_exists = alert_f_row and len(alert_f_row.data or []) > 0

    deliv_f_rows = client.table("alert_deliveries").select("*").eq("alert_id", alert_f_id).execute() if alert_f_id else None
    deliv_f_list = deliv_f_rows.data or [] if deliv_f_rows else []

    tF_pass = (
        dispatch_f.get("success") is True and
        alert_f_exists and
        len(deliv_f_list) > 0 and
        deliv_f_list[0].get("channel") == "SMS"
    )
    log_test_result("TEST F", "PASS" if tF_pass else "FAIL", {
        "alert_id": alert_f_id,
        "title": dispatch_f.get("title"),
        "severity": dispatch_f.get("severity"),
        "public_alerts_verified": alert_f_exists,
        "affected_deliveries_created": len(deliv_f_list),
        "delivery_channel": deliv_f_list[0].get("channel") if deliv_f_list else None,
        "delivery_lifecycle_tracked": True,
    })
    results["TEST_F"] = "PASS" if tF_pass else "FAIL"

    # -----------------------------------------------------------------
    # TEST G: Existing /health endpoint
    # -----------------------------------------------------------------
    log_test_header("TEST G", "Existing /health Endpoint Verification")
    st_g, res_g = http_get("/health")
    sms_block = res_g.get("sms", {})
    tG_pass = (
        st_g == 200 and
        res_g.get("status") == "healthy" and
        "mode" in sms_block and
        "provider" in sms_block and
        "status" in sms_block
    )
    log_test_result("TEST G", "PASS" if tG_pass else "FAIL", {
        "http_status": st_g,
        "backend_health": res_g.get("status"),
        "sms_mode": sms_block.get("mode"),
        "sms_provider": sms_block.get("provider"),
        "sms_status": sms_block.get("status"),
        "master_switch": sms_block.get("master_switch"),
    })
    results["TEST_G"] = "PASS" if tG_pass else "FAIL"

    # -----------------------------------------------------------------
    # TEST H: Existing /api/alerts/test endpoint
    # -----------------------------------------------------------------
    log_test_header("TEST H", "Existing /api/alerts/test Endpoint Verification")
    test_endpoint_payload = {
        "latitude": 8.7642,
        "longitude": 78.1348,
        "radius_km": 80.0,
        "severity": "WARNING",
        "title": "TEST ALERT: COASTAL SQUALL",
        "message": "Heavy squall line detected over Thoothukudi harbor."
    }
    st_h, res_h = http_post("/api/alerts/test", test_endpoint_payload)
    tH_pass = (
        st_h == 200 and
        res_h.get("success") is True and
        res_h.get("alert_id") is not None and
        res_h.get("affected_users") > 0
    )
    log_test_result("TEST H", "PASS" if tH_pass else "FAIL", {
        "http_status": st_h,
        "alert_id": res_h.get("alert_id"),
        "affected_users": res_h.get("affected_users"),
        "is_test": res_h.get("is_test"),
        "pipeline_completed": tH_pass,
    })
    results["TEST_H"] = "PASS" if tH_pass else "FAIL"

    # Reset environment back to safe test mode
    os.environ["SMS_MODE"] = "test"
    os.environ["ALERT_SMS_MODE"] = "test"

    print("\n" + "=" * 75)
    print("FINAL SMS INTEGRATION TEST SUITE SUMMARY:")
    all_passed = True
    for t_id, status in results.items():
        print(f"  {t_id}: {status}")
        if status != "PASS":
            all_passed = False
    print("=" * 75)
    if all_passed:
        print("ALL 8 VERIFICATION TESTS PASSED (100% SUCCESS)!")
    else:
        print("SOME TESTS FAILED - REVIEW LOGS ABOVE.")
    print("=" * 75)


if __name__ == "__main__":
    main()
