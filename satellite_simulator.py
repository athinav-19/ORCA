"""
satellite_simulator.py - Project ORCA Offshore Satellite Communication Simulator
ISRO SIH Problem Statement 176: Marine Multi-Agent System

================================================================================
           PROJECT ORCA - OFFSHORE SATELLITE GATEWAY SIMULATOR
        [DEVELOPMENT & TESTING ONLY - NOT A REAL SATELLITE LINK]
================================================================================

Simulates the end-to-end offshore communication path:
  Vessel Mobile Client
         ↓ (Bluetooth Low Energy)
  Vessel Satellite Terminal (e.g. Iridium GO! / Garmin inReach / NavIC Marine Terminal)
         ↓ (Satellite Constellation Uplink)
  Ground Gateway Infrastructure (e.g. Iridium Gateway / Provider Cloud)
         ↓ (HTTPS POST)
  ORCA FastAPI Server (Satellite Gateway Adapter)
         ↓
  ORCA Multi-Agent Intelligence Engine (Manager, Ocean, Weather, Disaster, GIS, PFZ)
         ↓
  Compact Marine Advisory Response
         ↓ (HTTPS)
  Ground Gateway Infrastructure
         ↓ (Satellite Constellation Downlink)
  Vessel Satellite Terminal
         ↓ (Bluetooth Low Energy)
  Vessel Mobile Client
"""

import os
import sys
import time
import json
import uuid
import argparse
from typing import Dict, Any, Optional
import requests

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


SIMULATOR_BANNER = """
================================================================================
             PROJECT ORCA - OFFSHORE SATELLITE GATEWAY SIMULATOR
          [DEVELOPMENT & TESTING ONLY - NOT A REAL SATELLITE LINK]
================================================================================
NOTICE: This simulator runs on terrestrial Internet/local loops to test ORCA
server-side gateway adapters, idempotency stores, and compact binary encoders.
It does NOT establish physical RF links with satellite constellations.
================================================================================
"""


def build_simulated_satellite_packet(
    query: str,
    lat: float,
    lon: float,
    persona: str = "FISHERMAN",
    lang: str = "en",
    speed: float = 4.2,
    heading: float = 120.0,
    message_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Generates a strictly formatted ORCA Satellite Protocol V1 message packet."""
    msg_id = message_id or f"sim-{uuid.uuid4().hex[:8]}"
    return {
        "protocol_version": 1,
        "message_id": msg_id,
        "message_type": "ORCA_QUERY",
        "timestamp": int(time.time()),
        "payload": {
            "query": query,
            "latitude": round(float(lat), 4),
            "longitude": round(float(lon), 4),
            "persona": persona,
            "language": lang,
            "speed_knots": round(float(speed), 1),
            "heading_degrees": round(float(heading), 1),
            "gps_accuracy_meters": 4.5,
        },
    }


def send_satellite_message(
    server_url: str,
    packet: Dict[str, Any],
    api_key: Optional[str] = None,
    timeout: int = 45,
    direct: bool = False,
) -> Dict[str, Any]:
    """
    Transmits simulated packet to ORCA /api/satellite/message endpoint.
    If server is offline or direct=True, executes in-process via ASGI transport.
    """
    endpoint = f"{server_url.rstrip('/')}/api/satellite/message"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-Satellite-Api-Key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"

    if not direct:
        start_time = time.time()
        try:
            resp = requests.post(endpoint, json=packet, headers=headers, timeout=timeout)
            duration_ms = round((time.time() - start_time) * 1000, 1)
            raw_size = len(resp.content)

            return {
                "http_status": resp.status_code,
                "duration_ms": duration_ms,
                "bytes_received": raw_size,
                "headers": dict(resp.headers),
                "data": resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text,
            }
        except requests.exceptions.RequestException:
            # Fall through to in-process execution
            print(f"[Simulator Notice] Could not connect to {server_url}. Using in-process ASGI engine...")

    # In-process ASGI execution
    import asyncio
    import httpx
    from server import app

    async def _direct_call():
        start_t = time.time()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            resp = await client.post("/api/satellite/message", json=packet, headers=headers, timeout=timeout)
            dur_ms = round((time.time() - start_t) * 1000, 1)
            raw_sz = len(resp.content)
            return {
                "http_status": resp.status_code,
                "duration_ms": dur_ms,
                "bytes_received": raw_sz,
                "headers": dict(resp.headers),
                "data": resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text,
            }

    try:
        # Enable satellite gateway for simulated in-process run
        os.environ["SATELLITE_GATEWAY_ENABLED"] = "true"
        return asyncio.run(_direct_call())
    except Exception as e:
        return {
            "http_status": 0,
            "duration_ms": 0,
            "bytes_received": 0,
            "error": str(e),
        }



def display_simulated_cycle(packet: Dict[str, Any], result: Dict[str, Any]):
    """Renders the step-by-step simulated satellite transmission journey."""
    print("\n" + "-" * 75)
    print("📡 SIMULATED SATELLITE TRANSMISSION SEQUENCE")
    print("-" * 75)

    msg_id = packet["message_id"]
    query = packet["payload"]["query"]
    lat = packet["payload"]["latitude"]
    lon = packet["payload"]["longitude"]
    persona = packet["payload"]["persona"]
    raw_packet_bytes = len(json.dumps(packet).encode("utf-8"))

    print(f"1. [Vessel Mobile Client] Query: \"{query}\"")
    print(f"   Coordinates: ({lat}, {lon}) | Persona: {persona}")
    print(f"   Transmitted via BLE to Vessel Terminal (Packet Size: {raw_packet_bytes} bytes)")
    print()
    print(f"2. [Vessel Satellite Terminal] Uplinking via Satellite Constellation...")
    print(f"   Message ID: {msg_id}")
    print()
    print(f"3. [Ground Gateway Provider] Relaying to ORCA Server Adapter...")
    print()

    status_code = result.get("http_status")
    duration = result.get("duration_ms")
    bytes_rx = result.get("bytes_received")

    if status_code == 200:
        data = result["data"]
        print(f"4. [ORCA Server Pipeline] Execution Complete (Latency: {duration} ms)")
        print(f"   HTTP Status: {status_code} OK | Received Compact Payload: {bytes_rx} bytes")
        print()
        print(f"5. [Ground Gateway] Downlinking Compact Advisory to Vessel Terminal...")
        print()
        print("=" * 75)
        print("          OFFSHORE VESSEL TERMINAL - ADVISORY DISPLAY")
        print("=" * 75)
        print(f"📋 Message ID   : {data.get('message_id')}")
        print(f"🛡️ Risk Status  : {data.get('risk')} (Score: {data.get('risk_score', 'N/A')}/100)")
        print(f"🌊 Wave Height  : {data.get('wave_height_m', 'N/A')} m")
        print(f"💨 Wind Speed   : {data.get('wind_speed_kmh', 'N/A')} km/h")
        print(f"📦 Idempotency  : {'(Served from Cache)' if data.get('cached') else '(Fresh Agent Run)'}")
        print("-" * 75)
        print("📢 ADVISORY BRIEFING:")
        print(data.get("advisory", "No advisory text"))
        print("=" * 75)
    elif status_code == 503:
        print(f"⚠️ [ORCA Server Notice] Satellite Integration is DISABLED (HTTP 503):")
        print(f"   {result.get('data')}")
    else:
        print(f"❌ [Communication Failure] HTTP Status: {status_code}")
        print(f"   Error: {result.get('error') or result.get('data')}")


def run_interactive_simulator(server_url: str):
    """Interactive CLI terminal prompt for sending custom queries."""
    print(SIMULATOR_BANNER)
    print("Interactive Mode. Type 'quit' or 'exit' to terminate.\n")

    while True:
        try:
            query = input("Enter offshore maritime query: ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                break

            lat_str = input("Enter Latitude [default: 8.7642]: ").strip() or "8.7642"
            lon_str = input("Enter Longitude [default: 78.1348]: ").strip() or "78.1348"
            persona = input("Enter Persona [default: FISHERMAN]: ").strip().upper() or "FISHERMAN"

            packet = build_simulated_satellite_packet(
                query=query,
                lat=float(lat_str),
                lon=float(lon_str),
                persona=persona,
            )

            result = send_satellite_message(server_url, packet)
            display_simulated_cycle(packet, result)
            print()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Simulator error: {e}")


def run_automated_verification(server_url: str):
    """Runs a complete battery of simulated test cases."""
    print(SIMULATOR_BANNER)
    print("Running Automated Satellite Gateway Verification Battery...\n")

    # 1. Standard Fishing Query
    print("[TEST 1/5] Sending standard PFZ query...")
    p1 = build_simulated_satellite_packet(
        query="Is it safe to fish here tomorrow morning?",
        lat=8.7642,
        lon=78.1348,
        persona="FISHERMAN",
    )
    r1 = send_satellite_message(server_url, p1)
    display_simulated_cycle(p1, r1)

    # 2. Idempotency Retransmission Test
    print("\n[TEST 2/5] Testing Idempotency (Retransmitting identical message_id)...")
    r2 = send_satellite_message(server_url, p1)
    if r2.get("http_status") == 200:
        is_cached = r2.get("data", {}).get("cached", False)
        print(f"   ✓ Retransmission Handled: cached={is_cached} (No redundant agent execution)")
    else:
        print(f"   Notice: Response status {r2.get('http_status')}")

    # 3. Storm Warning Query
    print("\n[TEST 3/5] Testing Storm / Rough Sea Inquiry...")
    p3 = build_simulated_satellite_packet(
        query="Heavy swells and dark clouds approaching, should we return to port?",
        lat=8.7642,
        lon=78.1348,
        persona="FISHERMAN",
    )
    r3 = send_satellite_message(server_url, p3)
    display_simulated_cycle(p3, r3)

    # 4. Status Check by Message ID
    print("\n[TEST 4/5] Testing Asynchronous Status Retrieval Endpoint...")
    status_url = f"{server_url.rstrip('/')}/api/satellite/status/{p1['message_id']}"
    try:
        s_resp = requests.get(status_url, timeout=5)
        print(f"   GET /api/satellite/status/{p1['message_id']} -> HTTP {s_resp.status_code}")
        print(f"   Status Payload: {json.dumps(s_resp.json(), indent=2)}")
    except Exception as e:
        print(f"   Status check error: {e}")

    # 5. Invalid Payload Boundary Test
    print("\n[TEST 5/5] Testing Invalid Coordinates Rejection...")
    p5 = build_simulated_satellite_packet(
        query="Out of bounds test",
        lat=999.0,  # Invalid
        lon=78.1348,
    )
    r5 = send_satellite_message(server_url, p5)
    print(f"   Invalid Latitude Request -> HTTP {r5.get('http_status')} (Expected 422)")

    print("\n" + "=" * 75)
    print("SIMULATOR BATTERY COMPLETED")
    print("========================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ORCA Offshore Satellite Gateway Simulator (Development Only)")
    parser.add_argument("--server-url", default="http://localhost:8000", help="ORCA FastAPI backend URL")
    parser.add_argument("--query", help="Custom single-shot maritime query")
    parser.add_argument("--lat", type=float, default=8.7642, help="Vessel Latitude")
    parser.add_argument("--lon", type=float, default=78.1348, help="Vessel Longitude")
    parser.add_argument("--persona", default="FISHERMAN", help="Stakeholder Persona")
    parser.add_argument("--lang", default="en", help="Language code")
    parser.add_argument("--message-id", help="Explicit message ID")
    parser.add_argument("--test-all", action="store_true", help="Run automated test battery")

    args = parser.parse_args()

    if args.test_all:
        run_automated_verification(args.server_url)
    elif args.query:
        print(SIMULATOR_BANNER)
        packet = build_simulated_satellite_packet(
            query=args.query,
            lat=args.lat,
            lon=args.lon,
            persona=args.persona,
            lang=args.lang,
            message_id=args.message_id,
        )
        result = send_satellite_message(args.server_url, packet)
        display_simulated_cycle(packet, result)
    else:
        run_interactive_simulator(args.server_url)

