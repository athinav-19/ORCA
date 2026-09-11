"""
transport/idempotency.py - Satellite Message Deduplication & Idempotency Store
ISRO SIH Problem Statement 176: Marine Multi-Agent System

Ensures reliable, idempotent request handling for high-latency offshore satellite links:
- Prevents redundant agent re-execution upon duplicate message transmission
- Tracks message processing states: RECEIVED, PROCESSING, COMPLETED, FAILED
- Thread-safe in-memory cache with optional JSON file persistence
"""

import os
import json
import time
import threading
from typing import Dict, Any, Optional, Tuple
from transport.models import (
    SatelliteMessageRecord,
    ProcessingStatus,
    SatelliteCompactResponse,
)


class SatelliteIdempotencyStore:
    """
    Thread-safe store tracking satellite message status and cached responses.
    """

    def __init__(self, persistence_file: Optional[str] = None, ttl_seconds: float = 86400.0):
        self._lock = threading.Lock()
        self._records: Dict[str, SatelliteMessageRecord] = {}
        self._persistence_file = persistence_file or os.path.join(".", "data", "satellite_idempotency_store.json")
        self._ttl_seconds = ttl_seconds
        self._load_from_disk()

    def _load_from_disk(self):
        """Loads non-expired records from disk on startup if file exists."""
        if not os.path.exists(self._persistence_file):
            return
        try:
            with open(self._persistence_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            now = time.time()
            with self._lock:
                for k, v in data.items():
                    # Prune records older than TTL
                    if now - v.get("updated_at", 0) < self._ttl_seconds:
                        self._records[k] = SatelliteMessageRecord(**v)
        except Exception as e:
            print(f"[Satellite Idempotency] Notice: Could not load persistence file: {e}")

    def _save_to_disk(self):
        """Persists records to disk safely."""
        try:
            os.makedirs(os.path.dirname(self._persistence_file), exist_ok=True)
            with self._lock:
                serialized = {k: v.model_dump() for k, v in self._records.items()}
            temp_file = f"{self._persistence_file}.tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(serialized, f, indent=2)
            os.replace(temp_file, self._persistence_file)
        except Exception:
            pass  # Non-fatal file save warning

    def get(self, message_id: str) -> Optional[SatelliteMessageRecord]:
        """Retrieves a message record by ID."""
        with self._lock:
            return self._records.get(message_id)

    def register(self, message_id: str, request_payload: Optional[Dict[str, Any]] = None) -> Tuple[SatelliteMessageRecord, bool]:
        """
        Registers a message_id.
        Returns (record, is_new):
        - is_new=True if newly registered.
        - is_new=False if this message_id already exists in store (duplicate).
        """
        with self._lock:
            existing = self._records.get(message_id)
            if existing:
                existing.retries += 1
                existing.updated_at = time.time()
                return existing, False

            record = SatelliteMessageRecord(
                message_id=message_id,
                status=ProcessingStatus.RECEIVED,
                received_at=time.time(),
                updated_at=time.time(),
                request_payload=request_payload,
            )
            self._records[message_id] = record
            return record, True

    def mark_processing(self, message_id: str):
        """Updates record status to PROCESSING."""
        with self._lock:
            record = self._records.get(message_id)
            if record:
                record.status = ProcessingStatus.PROCESSING
                record.updated_at = time.time()

    def mark_completed(self, message_id: str, compact_response: SatelliteCompactResponse):
        """Marks record as COMPLETED and stores the generated compact response."""
        with self._lock:
            record = self._records.get(message_id)
            if record:
                record.status = ProcessingStatus.COMPLETED
                record.updated_at = time.time()
                record.compact_response = compact_response.model_dump()
        self._save_to_disk()

    def mark_failed(self, message_id: str, error_message: str):
        """Marks record as FAILED with an error message."""
        with self._lock:
            record = self._records.get(message_id)
            if record:
                record.status = ProcessingStatus.FAILED
                record.updated_at = time.time()
                record.error = error_message
        self._save_to_disk()

    def get_completed_response(self, message_id: str) -> Optional[SatelliteCompactResponse]:
        """Returns the completed compact response if available."""
        with self._lock:
            record = self._records.get(message_id)
            if record and record.status == ProcessingStatus.COMPLETED and record.compact_response:
                resp_data = dict(record.compact_response)
                resp_data["cached"] = True
                return SatelliteCompactResponse(**resp_data)
        return None

    def count(self) -> int:
        """Returns number of active records."""
        with self._lock:
            return len(self._records)

    def clear(self):
        """Clears in-memory records (mainly used in test setups)."""
        with self._lock:
            self._records.clear()


# Global Singleton Idempotency Store
idempotency_store = SatelliteIdempotencyStore()

