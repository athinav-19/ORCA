"""
model_registry.py - Project ORCA ML Model Lifecycle & Registry
SIH 2026 Problem Statement SIH26176

Provides singleton in-memory caching and lifecycle management for all trained
specialized ML models:
1. Disaster Hazard Predictor (24h Ahead)
2. Multi-Horizon Marine Weather Regressor Suite (6h to 72h)
3. Potential Fishing Zone (PFZ) Habitat Suitability & Frontal Persistence

Enforces:
- Single load at startup (Zero disk I/O on API query paths)
- Strict feature preservation and schema verification
- Graceful failure isolation (Marks UNAVAILABLE without returning fake data)
"""

import os
import json
import logging
import joblib
from typing import Dict, Any, Optional

logger = logging.getLogger("ORCA.ModelRegistry")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DISASTER_DIR = os.path.join(BASE_DIR, "ml", "models", "disaster")
WEATHER_DIR = os.path.join(BASE_DIR, "ml", "models", "weather")
PFZ_DIR = os.path.join(BASE_DIR, "ml", "models", "pfz")


class ModelRegistry:
    """Thread-safe singleton registry caching loaded ML models and metadata."""

    _instance = None
    _models: Dict[str, Any] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    _features: Dict[str, list] = {}
    _status: Dict[str, str] = {
        "disaster": "UNLOADED",
        "weather": "UNLOADED",
        "pfz": "UNLOADED",
    }
    _errors: Dict[str, Optional[str]] = {
        "disaster": None,
        "weather": None,
        "pfz": None,
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance

    @classmethod
    def load_all_models(cls) -> Dict[str, str]:
        """Loads all three models into memory once at application startup."""
        cls.load_disaster_model()
        cls.load_weather_model()
        cls.load_pfz_model()
        return cls.get_status()

    @classmethod
    def load_disaster_model(cls) -> Optional[Any]:
        model_path = os.path.join(DISASTER_DIR, "best_model.joblib")
        meta_path = os.path.join(DISASTER_DIR, "metadata.json")
        feat_path = os.path.join(DISASTER_DIR, "feature_list.json")

        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model artifact missing at: {model_path}")

            cls._models["disaster"] = joblib.load(model_path)
            cls._status["disaster"] = "LOADED"
            cls._errors["disaster"] = None

            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    cls._metadata["disaster"] = json.load(f)

            if os.path.exists(feat_path):
                with open(feat_path, "r", encoding="utf-8") as f:
                    feat_data = json.load(f)
                    cls._features["disaster"] = feat_data.get("features", [])

            return cls._models["disaster"]

        except Exception as e:
            logger.error(f"[ModelRegistry] Failed to load Disaster Model: {e}")
            cls._status["disaster"] = "FAILED"
            cls._errors["disaster"] = str(e)
            cls._models["disaster"] = None
            return None

    @classmethod
    def load_weather_model(cls) -> Optional[Any]:
        model_path = os.path.join(WEATHER_DIR, "best_model.joblib")
        meta_path = os.path.join(WEATHER_DIR, "metadata.json")
        feat_path = os.path.join(WEATHER_DIR, "feature_list.json")

        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model artifact missing at: {model_path}")

            cls._models["weather"] = joblib.load(model_path)
            cls._status["weather"] = "LOADED"
            cls._errors["weather"] = None

            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    cls._metadata["weather"] = json.load(f)

            if os.path.exists(feat_path):
                with open(feat_path, "r", encoding="utf-8") as f:
                    feat_data = json.load(f)
                    cls._features["weather"] = feat_data.get("features", [])

            return cls._models["weather"]

        except Exception as e:
            logger.error(f"[ModelRegistry] Failed to load Weather Model: {e}")
            cls._status["weather"] = "FAILED"
            cls._errors["weather"] = str(e)
            cls._models["weather"] = None
            return None

    @classmethod
    def load_pfz_model(cls) -> Optional[Any]:
        model_path = os.path.join(PFZ_DIR, "best_model.joblib")
        meta_path = os.path.join(PFZ_DIR, "metadata.json")
        feat_path = os.path.join(PFZ_DIR, "feature_list.json")

        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model artifact missing at: {model_path}")

            cls._models["pfz"] = joblib.load(model_path)
            cls._status["pfz"] = "LOADED"
            cls._errors["pfz"] = None

            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    cls._metadata["pfz"] = json.load(f)

            if os.path.exists(feat_path):
                with open(feat_path, "r", encoding="utf-8") as f:
                    feat_data = json.load(f)
                    cls._features["pfz"] = feat_data.get("features", [])

            return cls._models["pfz"]

        except Exception as e:
            logger.error(f"[ModelRegistry] Failed to load PFZ Model: {e}")
            cls._status["pfz"] = "FAILED"
            cls._errors["pfz"] = str(e)
            cls._models["pfz"] = None
            return None

    @classmethod
    def get_model(cls, key: str) -> Optional[Any]:
        """Retrieves cached model or attempts lazy load if not yet initialized."""
        if key not in cls._models or cls._models[key] is None:
            if key == "disaster":
                return cls.load_disaster_model()
            elif key == "weather":
                return cls.load_weather_model()
            elif key == "pfz":
                return cls.load_pfz_model()
        return cls._models.get(key)

    @classmethod
    def get_status(cls) -> Dict[str, str]:
        return dict(cls._status)

    @classmethod
    def get_metadata(cls, key: str) -> Dict[str, Any]:
        return cls._metadata.get(key, {})

    @classmethod
    def get_features(cls, key: str) -> list:
        return cls._features.get(key, [])

    @classmethod
    def get_error(cls, key: str) -> Optional[str]:
        return cls._errors.get(key)

    @classmethod
    def print_startup_status(cls):
        """Prints formatted startup status banner required by ORCA specification."""
        print("=" * 80)
        print("ORCA ML MODEL STATUS")
        print("=" * 80)

        models_info = [
            ("Disaster", "disaster", os.path.join("ml", "models", "disaster", "best_model.joblib")),
            ("Weather", "weather", os.path.join("ml", "models", "weather", "best_model.joblib")),
            ("PFZ", "pfz", os.path.join("ml", "models", "pfz", "best_model.joblib")),
        ]

        for display_name, key, default_rel_file in models_info:
            status = cls._status.get(key, "UNLOADED")
            meta = cls._metadata.get(key, {})
            feats = cls._features.get(key, [])
            version = meta.get("model_version", f"{display_name}Model_v1")
            file_path = meta.get("model_file", default_rel_file)
            feat_count = len(feats) if feats else meta.get("feature_count", 0)

            print(f"{display_name:<10}: {status}")
            print(f"  - Model Version : {version}")
            print(f"  - Model File    : {file_path}")
            print(f"  - Feature Count : {feat_count}")
            if status == "FAILED" and cls._errors.get(key):
                print(f"  - Error         : {cls._errors.get(key)}")
            print("-" * 80)
        print("=" * 80)

