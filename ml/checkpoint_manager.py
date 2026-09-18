"""
checkpoint_manager.py - Pipeline State & Checkpointing for ORCA ML
SIH 2026 Problem Statement SIH26176

Provides thread-safe state persistence, step validation, and live progress reporting
for overnight automated training runs.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional

CHECKPOINT_FILE = os.path.join("ml", "checkpoint.json")
PROGRESS_FILE = os.path.join("ml", "training_progress.json")
STATUS_FILE = os.path.join("ml", "status.json")
LOGS_DIR = "logs"

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs("ml", exist_ok=True)


def get_logger(name: str, log_filename: str) -> logging.Logger:
    """Creates a standardized file + console logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File Handler
    log_path = os.path.join(LOGS_DIR, log_filename)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console Stream Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


class CheckpointManager:
    """Tracks completed stages and allows seamless resumption."""

    def __init__(self, checkpoint_path: str = CHECKPOINT_FILE):
        self.checkpoint_path = checkpoint_path
        self.state: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "initialized_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "current_phase": "NOT_STARTED",
            "completed_steps": {},
            "model_status": {
                "disaster": "PENDING",
                "weather": "PENDING",
                "pfz": "PENDING",
            },
            "metrics": {},
        }

    def save(self):
        self.state["last_updated"] = datetime.now().isoformat()
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        self.export_status_files()

    def is_step_completed(self, step_name: str) -> bool:
        return self.state.get("completed_steps", {}).get(step_name, False)

    def mark_step_completed(self, step_name: str, details: Optional[Dict[str, Any]] = None):
        if "completed_steps" not in self.state:
            self.state["completed_steps"] = {}
        self.state["completed_steps"][step_name] = {
            "completed": True,
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
        }
        self.save()

    def set_phase(self, phase_name: str):
        self.state["current_phase"] = phase_name
        self.save()

    def update_model_status(self, model_name: str, status: str, metrics: Optional[Dict[str, Any]] = None):
        if "model_status" not in self.state:
            self.state["model_status"] = {}
        self.state["model_status"][model_name] = status
        if metrics:
            if "metrics" not in self.state:
                self.state["metrics"] = {}
            self.state["metrics"][model_name] = metrics
        self.save()

    def export_status_files(self):
        """Writes human-readable summary files for morning inspection."""
        progress_data = {
            "title": "ORCA Multi-Agent ML Training Pipeline - Status Report",
            "current_phase": self.state.get("current_phase", "UNKNOWN"),
            "last_updated": self.state.get("last_updated"),
            "models": self.state.get("model_status", {}),
            "completed_steps_count": len(self.state.get("completed_steps", {})),
            "completed_steps": list(self.state.get("completed_steps", {}).keys()),
            "metrics_summary": self.state.get("metrics", {}),
        }
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress_data, f, indent=2)

        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "phase": self.state.get("current_phase"),
                "status": "RUNNING" if self.state.get("current_phase") != "COMPLETED" else "SUCCESS",
                "models": self.state.get("model_status", {}),
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)

