"""
test_routing_srilanka_landmask.py - Comprehensive Verification for Maritime Routing & Landmask Guardrails

Tests:
1. Country-to-port alias mapping ("srilanka", "sri lanka", "ceylon", "colombo") -> Colombo Port (6.9428, 79.8412).
2. Direct route to Sri Lanka from Tuticorin/Gulf of Mannar does NOT trigger LAND_INTERSECTION_ERROR.
3. Zero-distance queries (e.g. Tuticorin to Tuticorin) return STATIONARY, NOT LAND_INTERSECTION_ERROR.
4. Unknown destination queries return UNKNOWN_DESTINATION / DATA_UNAVAILABLE, NOT LAND_INTERSECTION_ERROR.
5. Genuine overland cross-peninsula routes (e.g. Kochi to Chennai) STILL correctly trigger LAND_INTERSECTION_ERROR.
"""

import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from gis_agent import GisAgent
from decision_engine import RiskAnalysisAgent, SafeAlternative, resolve_target_coordinates

def test_alias_mapping():
    print("\n--- TEST 1: Country-to-Port Alias Mapping ---")
    gis = GisAgent()
    
    # 1. "sri lanka"
    loc1 = gis.resolve_location("sri lanka")
    assert loc1["lat"] == 6.9428 and loc1["lon"] == 79.8412, f"Failed sri lanka: {loc1}"
    assert loc1["name"] == "Colombo Port"
    assert loc1.get("is_unknown") is False
    print("[OK] 'sri lanka' resolves to Colombo Port (6.9428, 79.8412)")

    # 2. "srilanka"
    loc2 = gis.resolve_location("srilanka")
    assert loc2["lat"] == 6.9428 and loc2["lon"] == 79.8412, f"Failed srilanka: {loc2}"
    assert loc2["name"] == "Colombo Port"
    print("[OK] 'srilanka' resolves to Colombo Port (6.9428, 79.8412)")

    # 3. "ceylon"
    loc3 = gis.resolve_location("ceylon")
    assert loc3["lat"] == 6.9428 and loc3["lon"] == 79.8412, f"Failed ceylon: {loc3}"
    print("[OK] 'ceylon' resolves to Colombo Port (6.9428, 79.8412)")

    # 4. SafeAlternative resolver
    safe_alt = SafeAlternative()
    coords_sl = safe_alt._resolve_coordinates("sri lanka")
    assert coords_sl == (6.9428, 79.8412), f"Failed SafeAlternative sri lanka: {coords_sl}"
    coords_sl2 = safe_alt._resolve_coordinates("srilanka")
    assert coords_sl2 == (6.9428, 79.8412), f"Failed SafeAlternative srilanka: {coords_sl2}"
    print("[OK] SafeAlternative._resolve_coordinates properly resolves Sri Lanka aliases")

    # 5. resolve_target_coordinates
    tgt = resolve_target_coordinates({}, "what is the wave height near sri lanka?")
    assert tgt == (6.9428, 79.8412), f"Failed resolve_target_coordinates: {tgt}"
    print("[OK] resolve_target_coordinates properly resolves Sri Lanka in query string")


def test_srilanka_open_water_routing():
    print("\n--- TEST 2: Sri Lanka Open Water Route (No False LAND_INTERSECTION_ERROR) ---")
    gis = GisAgent()

    # Land intersection check between Tuticorin and Colombo Port
    intersects = gis.check_land_intersection(8.7642, 78.1348, 6.9428, 79.8412)
    assert not intersects, "Tuticorin to Colombo Port should NOT intersect land polygon!"
    print("[OK] check_land_intersection(Tuticorin -> Colombo) is False")

    # generate_safe_sea_route
    route = gis.generate_safe_sea_route(origin="tuticorin", destination="sri lanka")
    assert route.get("route_status") != "LAND_INTERSECTION_ERROR", f"Unexpected land error: {route}"
    assert route.get("route_status") == "SAFE_PASSAGE_PLAN", f"Route status: {route.get('route_status')}"
    assert route.get("total_distance_nm", 0) > 50.0, f"Distance too small: {route.get('total_distance_nm')}"
    print(f"[OK] generate_safe_sea_route(tuticorin -> sri lanka) generated {route['total_distance_nm']} NM passage without land error")

    # execute_task with Sri Lanka query
    task_res = gis.execute_task(
        location="thoothukudi",
        time_frame="Current",
        task_instructions="Give me navigation route from thoothukudi to sri lanka",
        expected_format="TEXT_SUMMARY",
        persona="CAPTAIN",
    )
    assert task_res.get("status") != "LAND_INTERSECTION_ERROR", f"Task returned land error: {task_res}"
    assert task_res.get("error") != "LAND_INTERSECTION_ERROR"
    print("[OK] gis.execute_task('from thoothukudi to sri lanka') succeeded without land error")


def test_zero_distance_stationary():
    print("\n--- TEST 3: Zero-Distance / Stationary Guardrail ---")
    gis = GisAgent()

    # check_land_intersection zero-distance
    intersects = gis.check_land_intersection(8.7642, 78.1348, 8.7642, 78.1348)
    assert not intersects, "Zero-distance check should return False"
    print("[OK] check_land_intersection for identical coords is False")

    # generate_safe_sea_route for identical ports
    route = gis.generate_safe_sea_route(origin="tuticorin", destination="tuticorin")
    assert route.get("route_status") == "STATIONARY", f"Expected STATIONARY, got {route.get('route_status')}"
    assert route.get("total_distance_nm") == 0.0
    print("[OK] generate_safe_sea_route(tuticorin -> tuticorin) returned STATIONARY with 0.0 NM")

    # execute_task for zero-distance
    task_res = gis.execute_task(
        location="tuticorin",
        time_frame="Current",
        task_instructions="Plan passage from tuticorin to tuticorin",
        expected_format="TEXT_SUMMARY",
        persona="FISHERMAN",
    )
    assert task_res.get("status") != "LAND_INTERSECTION_ERROR", f"Returned land error for stationary: {task_res}"
    print("[OK] gis.execute_task('from tuticorin to tuticorin') returned without land error")


def test_unknown_destination():
    print("\n--- TEST 4: Unknown Destination Guardrail ---")
    gis = GisAgent()

    # resolve_location on unknown port
    unknown_loc = gis.resolve_location("port_of_nowhere_xyz")
    assert unknown_loc.get("is_unknown") is True, f"Expected is_unknown=True, got {unknown_loc}"
    print("[OK] resolve_location on unknown port sets is_unknown=True")

    # generate_safe_sea_route to unknown destination
    route = gis.generate_safe_sea_route(origin="tuticorin", destination="port_of_nowhere_xyz")
    assert route.get("route_status") == "UNKNOWN_DESTINATION", f"Expected UNKNOWN_DESTINATION, got {route.get('route_status')}"
    assert route.get("error") != "LAND_INTERSECTION_ERROR", "Unknown destination must NOT trigger LAND_INTERSECTION_ERROR"
    print("[OK] generate_safe_sea_route(tuticorin -> port_of_nowhere_xyz) returned UNKNOWN_DESTINATION")

    # execute_task to unknown destination
    task_res = gis.execute_task(
        location="tuticorin",
        time_frame="Current",
        task_instructions="route from tuticorin to port_of_nowhere_xyz",
        expected_format="TEXT_SUMMARY",
        persona="FISHERMAN",
    )
    assert task_res.get("status") == "DATA_UNAVAILABLE", f"Expected DATA_UNAVAILABLE, got {task_res.get('status')}"
    assert task_res.get("error") == "UNKNOWN_DESTINATION", f"Expected UNKNOWN_DESTINATION, got {task_res.get('error')}"
    print("[OK] gis.execute_task to unknown destination returned DATA_UNAVAILABLE / UNKNOWN_DESTINATION, NOT LAND_INTERSECTION_ERROR")


def test_genuine_land_intersection():
    print("\n--- TEST 5: Genuine Cross-Peninsula Overland Detection ---")
    gis = GisAgent()

    # Kochi (West coast, 9.93, 76.27) to Chennai (East coast, 13.08, 80.27)
    intersects = gis.check_land_intersection(9.9312, 76.2673, 13.0827, 80.2707)
    assert intersects, "Kochi to Chennai overland line MUST return True"
    print("[OK] check_land_intersection(Kochi -> Chennai) is True")

    # Mangalore (West coast, 12.92, 74.81) to Vizag (East coast, 17.68, 83.21)
    intersects_m_v = gis.check_land_intersection(12.9250, 74.8150, 17.6868, 83.2185)
    assert intersects_m_v, "Mangalore to Vizag overland line MUST return True"
    print("[OK] check_land_intersection(Mangalore -> Vizag) is True")

    # generate_safe_sea_route
    route = gis.generate_safe_sea_route(origin="kochi", destination="chennai")
    assert route.get("route_status") == "LAND_INTERSECTION_ERROR", f"Expected LAND_INTERSECTION_ERROR, got {route.get('route_status')}"
    print("[OK] generate_safe_sea_route(kochi -> chennai) returned LAND_INTERSECTION_ERROR")

    # Decision Engine guardrail check
    risk_agent = RiskAnalysisAgent()
    eval_res = risk_agent.evaluate_threats(
        aggregated_data={"GIS_AGENT": route},
        normalized_query="route from kochi to chennai",
        persona="CAPTAIN",
    )
    assert eval_res.get("status") == "LAND_INTERSECTION_ERROR", f"Expected LAND_INTERSECTION_ERROR in risk agent, got {eval_res.get('status')}"
    assert eval_res.get("land_intersection_error") is True
    print("[OK] RiskAnalysisAgent properly evaluated LAND_INTERSECTION_ERROR for cross-peninsula route")


if __name__ == "__main__":
    print("=================================================================")
    print("RUNNING ORCA MARITIME ROUTING & LANDMASK GUARDRAIL TESTS")
    print("=================================================================")
    try:
        test_alias_mapping()
        test_srilanka_open_water_routing()
        test_zero_distance_stationary()
        test_unknown_destination()
        test_genuine_land_intersection()
        print("\n=================================================================")
        print("ALL 5 MARITIME ROUTING & LANDMASK TESTS PASSED SUCCESSFULLY! [SUCCESS]")
        print("=================================================================\n")
    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILURE: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

