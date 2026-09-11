"""
transport - Project ORCA Communication & Gateway Layer
ISRO SIH Problem Statement 176: Marine Multi-Agent System

Exports core models, adapters, transports, and idempotency utilities for
supporting offshore satellite communication.
"""

from transport.models import (
    SatellitePayload,
    SatelliteMessage,
    SatelliteCompactResponse,
    SatelliteMessageRecord,
    SatelliteMessageType,
    SatelliteProtocolVersion,
    ProcessingStatus,
)
from transport.adapter import OrcaRequestAdapter, OrcaResponseAdapter
from transport.idempotency import SatelliteIdempotencyStore, idempotency_store
from transport.security import (
    validate_api_key,
    validate_hmac_signature,
    validate_payload_size,
    validate_timestamp,
    redact_secrets,
)
from transport.satellite import (
    SatelliteTransport,
    SatelliteGatewayAdapter,
    gateway_adapter,
)
from transport.poller import SatelliteGatewayPoller

__all__ = [
    "SatellitePayload",
    "SatelliteMessage",
    "SatelliteCompactResponse",
    "SatelliteMessageRecord",
    "SatelliteMessageType",
    "SatelliteProtocolVersion",
    "ProcessingStatus",
    "OrcaRequestAdapter",
    "OrcaResponseAdapter",
    "SatelliteIdempotencyStore",
    "idempotency_store",
    "validate_api_key",
    "validate_hmac_signature",
    "validate_payload_size",
    "validate_timestamp",
    "redact_secrets",
    "SatelliteTransport",
    "SatelliteGatewayAdapter",
    "gateway_adapter",
    "SatelliteGatewayPoller",
]

