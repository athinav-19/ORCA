# Project ORCA: Offshore Satellite Communication Architecture & Integration Guide
**ISRO SIH Problem Statement 176: Marine Multi-Agent System**

---

## 1. Architectural Overview & Boundary Separation

Project ORCA provides real-time marine intelligence to coastal and offshore stakeholders. When a vessel departs coastal cellular range, normal 4G/5G mobile Internet is lost. To maintain navigational safety, flood warnings, storm alerts, and Potential Fishing Zone (PFZ) guidance, ORCA integrates with offshore vessel satellite communication terminals.

> [!IMPORTANT]
> **Fundamental Architectural Boundary:**
> - The Android mobile device **never** communicates directly with orbital satellites. A standard smartphone lacks the specialized phased-array antenna, RF power amplifiers, and satellite modem hardware required for direct space communication.
> - The Android app connects to an onboard **Vessel Satellite Terminal** (e.g., Iridium GO! Exec, Garmin inReach Marine, Inmarsat Fleet One, or NavIC marine terminal) via a local link (typically **Bluetooth Low Energy** or onboard Wi-Fi).
> - The vessel terminal transmits packets over the satellite constellation link to the provider's **Ground Gateway Infrastructure**.
> - The satellite provider's ground gateway delivers the request via terrestrial HTTPS to the **ORCA Server** endpoint (`POST /api/satellite/message` or `POST /api/satellite/webhook`).
> - The ORCA server processes the query using its **existing, unmodified multi-agent pipeline** and returns a bandwidth-minimized **Compact Response** through the same path back to the vessel terminal.

```
+-----------------------------------------------------------------------------+
|                                OFFSHORE VESSEL                              |
|                                                                             |
|   +-------------------+                     +---------------------------+   |
|   |  Android ORCA App | <--- Bluetooth ---> | Vessel Satellite Terminal |   |
|   +-------------------+                     +---------------------------+   |
+-----------------------------------------------------------│-----------------+
                                                            │
                                                Satellite Uplink / Downlink
                                                  (Iridium / NavIC / Inmarsat)
                                                            │
                                                            ▼
                                              +---------------------------+
                                              | Satellite Provider Ground |
                                              |   Gateway Infrastructure  |
                                              +---------------------------+
                                                            │
                                                Secure HTTPS API / Webhook
                                                            │
                                                            ▼
+-----------------------------------------------------------------------------+
|                                 ORCA SERVER                                 |
|                                                                             |
|   +---------------------------------------------------------------------+   |
|   |             Offshore Satellite Gateway Adapter Layer                |   |
|   |  - Authentication (API Key / HMAC SHA256 Signature)                 |   |
|   |  - Payload Size Limits (HTTP 413 guard: default 4096 bytes)         |   |
|   |  - Message Idempotency Store (Prevents redundant agent runs)        |   |
|   |  - Compact Response Serializer (Bandwidth-minimized packets)        |   |
|   +-----------------------------------│---------------------------------+   |
|                                       │                                     |
|                                       ▼                                     |
|   +---------------------------------------------------------------------+   |
|   |              Existing ORCA Multi-Agent Pipeline (Unchanged)         |   |
|   |                                                                     |   |
|   |                    Manager / Agentic Orchestrator                   |   |
|   |                                   │                                 |   |
|   |         ┌───────────┬─────────────┼─────────────┬───────────┐       |   |
|   |         ▼           ▼             ▼             ▼           ▼       |   |
|   |     PFZ Agent  Ocean Agent  Weather Agent  Disaster Agent GIS Agent |   |
|   |         └───────────┴─────────────┼─────────────┴───────────┘       |   |
|   |                                   ▼                                 |   |
|   |                          Risk Analysis Agent                        |   |
|   |                                   ▼                                 |   |
|   |                            Reasoning Agent                          |   |
+-----------------------------------------------------------------------------+
```

---

## 2. Real-World Integration Statement

> "The exact implementation of `SatelliteGatewayAdapter` depends on the satellite terminal/provider used by the vessel. ORCA does not assume a specific satellite vendor or proprietary protocol."

ORCA defines a clean, extensible application-layer protocol and adapter boundary:
- **`SatelliteTransport`**: Abstract interface specifying `send(message)`, `receive()`, and `acknowledge(message_id)`.
- **`SatelliteGatewayAdapter`**: Production-ready gateway adapter handling outbound HTTPS dispatch, timeout management, retries, and acknowledgement.
- **`SatelliteGatewayPoller`**: Background worker for satellite providers requiring polling rather than push webhooks.

This allows ORCA to interface transparently with:
1. Ground station REST APIs (e.g. Iridium CloudConnect, Inmarsat API)
2. MQTT/Broker-based maritime gateways
3. Indigenous ISRO NavIC / GSAT two-way messaging ground systems
4. Direct satellite receiver sidecars

---

## 3. Relation to Offline Mobile Mode

It is essential to distinguish between the two mobile operating states:

| Mode | Connectivity State | Processing Location | Data Used |
| :--- | :--- | :--- | :--- |
| **Offline Mobile Mode** | No Internet **AND** No Vessel Satellite Terminal | On-device Android client SQLite database & offline maps | Pre-cached PFZ polygons and coastal charts synced while in harbor |
| **Offshore Satellite Mode** | No Cellular Internet, **BUT** Connected to Vessel Terminal via Bluetooth | ORCA Server via Satellite Ground Gateway | Real-time dual-agency ISRO MOSDAC + Copernicus satellite telemetry |

> [!CAUTION]
> If a fisherman's smartphone has **neither cellular coverage nor a connected satellite terminal**, the server cannot receive the request. The Android app automatically falls back to its offline local mode.

---

## 4. Satellite API Endpoints

### 4.1 Direct Gateway Query
- **Endpoint:** `POST /api/satellite/message` (or `POST /api/v1/satellite/message`)
- **Headers:**
  - `Content-Type: application/json`
  - `X-Satellite-Api-Key: <api_key>` (Optional, if authentication is configured)
- **Request Envelope:**
```json
{
  "protocol_version": 1,
  "message_id": "sat-msg-0912-abc",
  "message_type": "ORCA_QUERY",
  "timestamp": 1780000000,
  "payload": {
    "query": "Is it safe to fish here tomorrow morning?",
    "latitude": 8.7642,
    "longitude": 78.1348,
    "persona": "FISHERMAN",
    "language": "en",
    "speed_knots": 4.2,
    "heading_degrees": 120.0,
    "gps_accuracy_meters": 4.5
  }
}
```
- **Response Envelope (Bandwidth-Minimized Compact Payload):**
```json
{
  "protocol_version": 1,
  "message_id": "sat-msg-0912-abc",
  "status": "SUCCESS",
  "risk": "SAFE",
  "risk_score": 16.8,
  "advisory": "Route analysis complete. You are cleared for transit to your designated maritime operating sector with favorable navigational conditions observed across coastal and offshore zones. The primary operating destination is established at Latitude 9.0932° N, Longitude 78.3218° E...",
  "latitude": 8.7642,
  "longitude": 78.1348,
  "timestamp": 1780000015,
  "data_age_minutes": 15,
  "wave_height_m": 1.2,
  "wind_speed_kmh": 14.0,
  "cached": false
}
```

### 4.2 Asynchronous Webhook Receiver
- **Endpoint:** `POST /api/satellite/webhook`
- **Headers:**
  - `Content-Type: application/json`
  - `X-Satellite-Signature: sha256=<hmac_hex>` (Validated against `SATELLITE_WEBHOOK_SECRET`)
- **Behavior:**
  Processes message, stores response in idempotency cache, dispatches outbound transmission to satellite provider gateway, and immediately returns HTTP 200 acknowledgment:
```json
{
  "protocol_version": 1,
  "message_id": "sat-msg-0912-abc",
  "status": "ACK",
  "timestamp": 1780000015
}
```

### 4.3 Asynchronous Message Status Check
- **Endpoint:** `GET /api/satellite/status/{message_id}`
- **Response:**
```json
{
  "message_id": "sat-msg-0912-abc",
  "status": "COMPLETED",
  "received_at": 1780000000.5,
  "updated_at": 1780000002.1,
  "retries": 0,
  "response": { ... },
  "error": null
}
```

---

## 5. Security & Bandwidth Optimization

1. **Payload Size Guard:** Rejects messages exceeding `SATELLITE_MAX_PAYLOAD_SIZE` (default 4096 bytes) with HTTP 413 Payload Too Large.
2. **Compact Serializer:** Strips heavy GeoJSON geometries, raster charts, base64 audio binaries, and prompt suggestions to minimize satellite packet billing.
3. **Idempotency Deduplication:** If satellite network retransmissions deliver the same `message_id` multiple times, the server serves the stored response immediately without executing the agent pipeline again.
4. **Credential Safety:** Zero credentials logged in logs; API keys and secrets automatically redacted.

---

## 6. Configuration Reference

Set in `.env`:
```env
# Enable or disable satellite gateway integration
SATELLITE_GATEWAY_ENABLED=true

# Remote satellite provider gateway URL for outbound transmission
SATELLITE_GATEWAY_URL=https://api.satellite-provider.com/v1/orca

# API token for authenticating with provider gateway
SATELLITE_GATEWAY_API_KEY=

# Secret for verifying incoming provider webhooks
SATELLITE_WEBHOOK_SECRET=

# Message send timeout in seconds
SATELLITE_MESSAGE_TIMEOUT=30

# Maximum permitted payload size in bytes
SATELLITE_MAX_PAYLOAD_SIZE=4096

# Polling frequency in seconds (if using pull-based gateway)
SATELLITE_POLL_INTERVAL_SECONDS=60
```

---

## 7. Testing with the Development Simulator

To test the complete offshore gateway lifecycle during local development without physical satellite hardware:

```bash
# Run interactive simulation
python satellite_simulator.py --server-url http://localhost:8000

# Run automated 5-test battery
python satellite_simulator.py --server-url http://localhost:8000 --test-all

# Run custom single query
python satellite_simulator.py --query "Is it safe to fish near Thoothukudi tomorrow?" --lat 8.7642 --lon 78.1348
```

