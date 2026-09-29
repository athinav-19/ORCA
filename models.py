"""
models.py - Project ORCA Marine Stakeholder Schemas & Persona Definitions
ISRO SIH Problem Statement 176: Marine Multi-Agent System

Defines explicit enumerations, Pydantic schemas, and dynamic resolution utilities
for all 5 supported marine stakeholder personas:
1. FISHERMAN
2. MARITIME_AUTHORITY
3. DISASTER_MANAGEMENT
4. RESEARCHER
5. MARITIME_OPERATOR
"""

import re
from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, field_validator


class StakeholderPersona(str, Enum):
    FISHERMAN = "FISHERMAN"
    MARITIME_AUTHORITY = "MARITIME_AUTHORITY"
    DISASTER_MANAGEMENT = "DISASTER_MANAGEMENT"
    RESEARCHER = "RESEARCHER"
    MARITIME_OPERATOR = "MARITIME_OPERATOR"


# Legacy alias mapping for backward compatibility
PERSONA_ALIASES: Dict[str, StakeholderPersona] = {
    "AUTHORITY": StakeholderPersona.MARITIME_AUTHORITY,
    "COAST_GUARD": StakeholderPersona.MARITIME_AUTHORITY,
    "NAVY": StakeholderPersona.MARITIME_AUTHORITY,
    "PATROL": StakeholderPersona.MARITIME_AUTHORITY,
    "DISASTER": StakeholderPersona.DISASTER_MANAGEMENT,
    "NDRF": StakeholderPersona.DISASTER_MANAGEMENT,
    "SDMA": StakeholderPersona.DISASTER_MANAGEMENT,
    "RESEARCH": StakeholderPersona.RESEARCHER,
    "SCIENTIST": StakeholderPersona.RESEARCHER,
    "OCEANOGRAPHER": StakeholderPersona.RESEARCHER,
    "OPERATOR": StakeholderPersona.MARITIME_OPERATOR,
    "SHIPPING": StakeholderPersona.MARITIME_OPERATOR,
    "CARGO": StakeholderPersona.MARITIME_OPERATOR,
    "PORT": StakeholderPersona.MARITIME_OPERATOR,
    "FISHERY": StakeholderPersona.FISHERMAN,
    "ANGLER": StakeholderPersona.FISHERMAN,
    "TRAWLER": StakeholderPersona.FISHERMAN,
}

# Official Decision Dossier Badges with exact icons and descriptions
PERSONA_BADGES: Dict[str, str] = {
    StakeholderPersona.MARITIME_AUTHORITY.value: "🛡️ MARITIME AUTHORITY (Surveillance, Coast Guard & Enforcement)",
    StakeholderPersona.RESEARCHER.value: "🔬 RESEARCHER (Oceanographic & Climate Science)",
    StakeholderPersona.DISASTER_MANAGEMENT.value: "🚨 DISASTER MANAGEMENT (Evacuation & Crisis Response)",
    StakeholderPersona.MARITIME_OPERATOR.value: "🚢 MARITIME OPERATOR (Commercial Shipping & Port Operations)",
    StakeholderPersona.FISHERMAN.value: "🎣 FISHERMAN (Artisanal / Commercial Fishery)",
}

# Persona numbered shortcuts for CLI and UI menus
PERSONA_MENU_OPTIONS = [
    ("1", StakeholderPersona.FISHERMAN, "🎣 FISHERMAN (Artisanal / Commercial Fishery)"),
    ("2", StakeholderPersona.MARITIME_AUTHORITY, "🛡️ MARITIME AUTHORITY (Surveillance, Coast Guard & Enforcement)"),
    ("3", StakeholderPersona.DISASTER_MANAGEMENT, "🚨 DISASTER MANAGEMENT (Evacuation & Crisis Response)"),
    ("4", StakeholderPersona.RESEARCHER, "🔬 RESEARCHER (Oceanographic & Climate Science)"),
    ("5", StakeholderPersona.MARITIME_OPERATOR, "🚢 MARITIME OPERATOR (Commercial Shipping & Port Operations)"),
]


# Fuel consumption rates (Liters per Nautical Mile) calibrated per stakeholder persona
PERSONA_FUEL_CONSUMPTION_RATES: Dict[StakeholderPersona, float] = {
    StakeholderPersona.FISHERMAN: 1.5,
    StakeholderPersona.RESEARCHER: 8.0,
    StakeholderPersona.DISASTER_MANAGEMENT: 5.0,
    StakeholderPersona.MARITIME_AUTHORITY: 25.0,
    StakeholderPersona.MARITIME_OPERATOR: 150.0,
}


def get_fuel_consumption_rate(persona: Optional[Any]) -> float:
    """
    Returns the fuel consumption rate in Liters per Nautical Mile (L/NM)
    for the specified stakeholder persona.
    Defaults to 1.5 L/NM (FISHERMAN craft).
    """
    p_enum = resolve_persona(persona) or StakeholderPersona.FISHERMAN
    return PERSONA_FUEL_CONSUMPTION_RATES.get(p_enum, 1.5)


def resolve_persona(val: Optional[str]) -> Optional[StakeholderPersona]:
    """
    Resolves a string, integer shortcut ('1'-'5'), or alias to a valid StakeholderPersona Enum.
    Returns None if the value cannot be resolved.
    """
    if val is None:
        return None
    cleaned = str(val).strip().upper()
    if not cleaned:
        return None

    # Check numeric menu shortcuts
    numeric_map = {
        "1": StakeholderPersona.FISHERMAN,
        "2": StakeholderPersona.MARITIME_AUTHORITY,
        "3": StakeholderPersona.DISASTER_MANAGEMENT,
        "4": StakeholderPersona.RESEARCHER,
        "5": StakeholderPersona.MARITIME_OPERATOR,
    }
    if cleaned in numeric_map:
        return numeric_map[cleaned]

    # Exact enum match
    try:
        return StakeholderPersona(cleaned)
    except ValueError:
        pass

    # Normalized alias match
    normalized_key = cleaned.replace(" ", "_").replace("-", "_")
    if normalized_key in PERSONA_ALIASES:
        return PERSONA_ALIASES[normalized_key]

    for alias_key, persona_enum in PERSONA_ALIASES.items():
        if alias_key in normalized_key:
            return persona_enum

    return None


def classify_persona_intent(query_text: str) -> Optional[StakeholderPersona]:
    """
    Dynamically infers the likely marine stakeholder persona from query keywords and semantics
    when user_context.persona is unspecified.
    """
    q = (query_text or "").lower()

    # 1. DISASTER_MANAGEMENT Keywords (High Priority for Life-Safety)
    disaster_keywords = [
        "cyclone", "evacuation", "storm surge", "tsunami", "flood", "inundation",
        "relief camp", "shelter allocation", "ndrf", "sdma", "coastal warning",
        "emergency shelter", "disaster response", "life safety", "high damage",
        "landfall", "eye of cyclone"
    ]
    if any(k in q for k in disaster_keywords):
        return StakeholderPersona.DISASTER_MANAGEMENT

    # 2. MARITIME_AUTHORITY Keywords (Security, Enforcement, Patrol, SAR)
    authority_keywords = [
        "patrol", "coast guard", "surveillance", "interception", "sar",
        "search and rescue", "border security", "jurisdictional", "eez perimeter",
        "foreign vessel", "border clearance", "maritime boundary", "law enforcement",
        "patrol boat", "intercept", "defense station", "naval"
    ]
    if any(k in q for k in authority_keywords):
        return StakeholderPersona.MARITIME_AUTHORITY

    # 3. RESEARCHER Keywords (Biophysical Oceanography, MOSDAC Satellites)
    researcher_keywords = [
        "chlorophyll", "kd_490", "kd490", "tsm", "total suspended matter",
        "diffuse attenuation", "altimeter", "saral", "altika", "ssha",
        "netcdf", "biophysical", "transect", "sampling", "irradiance",
        "turbidity", "ocean color", "telemetry", "geostrophic", "scientific"
    ]
    if any(k in q for k in researcher_keywords):
        return StakeholderPersona.RESEARCHER

    # 4. MARITIME_OPERATOR Keywords (Commercial Shipping, Port Approaches, Fairways)
    operator_keywords = [
        "fairway", "shipping lane", "sloc", "tss", "tanker", "container ship",
        "cargo", "deep draft", "port approach", "fuel consumption", "knots",
        "transit speed", "pilot station", "commercial fairway", "berth", "anchorage"
    ]
    if any(k in q for k in operator_keywords):
        return StakeholderPersona.MARITIME_OPERATOR

    # 5. FISHERMAN Keywords (Potential Fishing Zones, Catch, Trawling)
    fisherman_keywords = [
        "fish", "fishing", "pfz", "catch", "tuna", "sardine", "mackerel",
        "seerfish", "hilsa", "trevally", "ribbonfish", "net", "boat",
        "trawler", "gillnet", "shoal", "fishing harbor", "sea rough"
    ]
    if any(k in q for k in fisherman_keywords):
        return StakeholderPersona.FISHERMAN

    return None


# =====================================================================
# Location Context Schemas (Genuine Location-Aware Architecture)
# =====================================================================

class LocationSource(str, Enum):
    USER_QUERY = "USER_QUERY"
    EXPLICIT_QUERY = "EXPLICIT_QUERY"
    USER_GPS = "USER_GPS"
    DEVICE_GPS = "DEVICE_GPS"
    ROUTE = "ROUTE"
    ROUTE_ORIGIN = "ROUTE_ORIGIN"
    ROUTE_DESTINATION = "ROUTE_DESTINATION"
    DEFAULT = "DEFAULT"
    SYSTEM = "SYSTEM"
    NONE = "NONE"


class LocationStatus(str, Enum):
    RESOLVED = "RESOLVED"
    LOCATION_REQUIRED = "LOCATION_REQUIRED"
    LOCATION_RESOLUTION_FAILED = "LOCATION_RESOLUTION_FAILED"
    LOCATION_DATA_MISMATCH = "LOCATION_DATA_MISMATCH"


class LocationContext(BaseModel):
    location_type: str = Field("COASTAL_PORT", description="COASTAL_PORT | COORDINATES | ROUTE | MARITIME_SECTOR | RELATIVE_GPS | UNKNOWN")
    name: Optional[str] = Field(None, description="Standardized geographic feature or port name (e.g. Mumbai, Chennai)")
    location_name: Optional[str] = Field(None, description="Standardized geographic feature or port name (e.g. Mumbai, Chennai)")
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0, description="Resolved decimal latitude")
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0, description="Resolved decimal longitude")
    source: LocationSource = Field(LocationSource.NONE, description="Resolution source: USER_QUERY | USER_GPS | ROUTE_ORIGIN | ROUTE_DESTINATION | SYSTEM | NONE")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Geocoding confidence score")
    origin: Optional[Dict[str, Any]] = Field(None, description="Origin for route queries {name, latitude, longitude}")
    destination: Optional[Dict[str, Any]] = Field(None, description="Destination for route queries {name, latitude, longitude}")
    radius_km: Optional[float] = Field(None, description="Spatial search radius in km")
    country: Optional[str] = Field("India", description="Country name")
    region: Optional[str] = Field(None, description="Maritime sector or coastal state (e.g. Maharashtra Coast)")
    explicit_in_query: bool = Field(False, description="True if explicitly specified in user query text")
    status: LocationStatus = Field(LocationStatus.RESOLVED, description="Resolution status: RESOLVED | LOCATION_REQUIRED | LOCATION_RESOLUTION_FAILED")
    error_message: Optional[str] = Field(None, description="Explanation when location resolution fails or is required")

    model_config = {"extra": "allow"}

    def __init__(self, **data: Any):
        if "name" in data and not data.get("location_name"):
            data["location_name"] = data["name"]
        elif "location_name" in data and not data.get("name"):
            data["name"] = data["location_name"]
        super().__init__(**data)
        if self.name and not self.location_name:
            self.location_name = self.name
        elif self.location_name and not self.name:
            self.name = self.location_name

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_dict(self) -> Dict[str, Any]:
        resolved_name = self.name or self.location_name
        return {
            "location_type": self.location_type,
            "location_name": resolved_name,
            "name": resolved_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "source": self.source.value if isinstance(self.source, LocationSource) else str(self.source),
            "confidence": self.confidence,
            "origin": self.origin,
            "destination": self.destination,
            "radius_km": self.radius_km,
            "country": self.country,
            "region": self.region,
            "explicit_in_query": self.explicit_in_query,
            "status": self.status.value if isinstance(self.status, LocationStatus) else str(self.status),
            "error_message": self.error_message,
        }


# =====================================================================
# Pydantic Schemas for Request & Telemetry
# =====================================================================

class TelemetryData(BaseModel):
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0, description="Vessel GPS latitude (None if no device GPS fix)")
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0, description="Vessel GPS longitude (None if no device GPS fix)")
    speed_knots: Optional[float] = 0.0
    heading_degrees: Optional[float] = 120.0
    gps_accuracy_meters: Optional[float] = 4.5


class QueryRequest(BaseModel):
    query: Optional[str] = Field(None, description="Stakeholder natural language query in any supported Indian language or English")
    query_text: Optional[str] = Field(None, description="Stakeholder query text alias")
    persona: Optional[str] = Field(
        None,
        description="Explicit persona: FISHERMAN | MARITIME_AUTHORITY | DISASTER_MANAGEMENT | RESEARCHER | MARITIME_OPERATOR (dynamic if omitted)"
    )
    session_id: Optional[str] = Field("sess_marine_ui", description="Session identifier for multi-turn conversational memory")
    telemetry: Optional[TelemetryData] = None
    input_type: Optional[str] = Field("TEXT", description="TEXT | AUDIO")
    raw_audio_base64: Optional[str] = Field(None, description="Base64 encoded audio string for speech-to-text")
    source_language_code: Optional[str] = Field(None, description="ISO 639-1 language code (e.g. en, ta, hi, te, ml, bn)")
    language_preference: Optional[str] = Field(None, description="Preferred language code alias (e.g. en, ta, hi, te, ml, bn)")
    lang: Optional[str] = Field(None, description="Client requested ISO 639-1 language code (e.g. en, ta, hi, te, ml, gu, kn, or)")


class AdvisoryResponse(BaseModel):
    recommendation: Optional[str] = Field("Safe to proceed. Direct corridor to PFZ Alpha open.", description="Card recommendation text")
    wave_height: Optional[str] = Field(None, description="Wave height")
    wind_speed: Optional[str] = Field(None, description="Wind speed")
    key_advisories: Optional[List[str]] = Field(default_factory=lambda: [
        "🎯 Target: PFZ Hotspot",
        "🐟 High-Yield Catch: Tuna, Mackerel",
        "🧭 Nav Brief: ~2h 49m at 8.0 kt",
        "🛡️ Safety: Clear of IMBL & Shipping Fairways",
        "☀️ INSAT-3DR Solar: Auxiliary endurance verified"
    ])
    chat_text: str = Field("", description="Main conversational text advisory for display and speech")
    threat_status: Optional[str] = Field(None, description="Evaluated threat status: SAFE, CAUTION, NO-GO")
    risk_score: Optional[float] = Field(None, description="Evaluated composite risk score")
    native_advisory_text: Optional[str] = Field(None, description="Advisory text translated into regional language")
    show_route: Optional[bool] = Field(False, description="Flag indicating whether to display navigation route")

    model_config = {"extra": "allow"}


class OrcaResponse(BaseModel):
    status: Optional[str] = Field("success", description="Status string for Android client compatibility")
    success: bool = Field(True, description="Query execution status flag")
    reply: Optional[str] = Field(None, description="Mirrored advisory text for Android clients")
    response: Optional[str] = Field(None, description="Mirrored advisory text for Android clients")
    message: Optional[str] = Field(None, description="Mirrored advisory text for Android clients")
    chat_text: Optional[str] = Field(None, description="Top-level conversational advisory text")
    advisory: Optional[Union[Dict[str, Any], AdvisoryResponse, str]] = Field(None, description="Structured advisory object or string")
    advisory_details: Optional[Any] = Field(None, description="Structured advisory object")
    threat_status: Optional[str] = Field(None, description="Evaluated threat status: SAFE, CAUTION, NO-GO")
    risk_score: Optional[float] = Field(None, description="Evaluated composite risk score")
    native_advisory_text: Optional[str] = Field(None, description="Top-level native advisory text")
    session_id: Optional[str] = Field(None, description="Session ID")
    source_language: Optional[str] = Field(None, description="Source or preferred language code")
    language_name: Optional[str] = Field(None, description="Name of language")
    risk_assessment: Optional[Dict[str, Any]] = Field(None, description="Composite risk assessment")
    safe_sea_route: Optional[Dict[str, Any]] = Field(None, description="Navigational waypoints and clearance")
    show_route: Optional[bool] = Field(False, description="Flag indicating whether to display navigation route on client map")
    original_query: Optional[str] = Field(None, description="Original raw user query")
    effective_query: Optional[str] = Field(None, description="Internal effective English query")
    green_marine_energy: Optional[Dict[str, Any]] = Field(None, description="Solar irradiance and zero-emission stats")
    prompt_suggestions: Optional[List[str]] = Field(None, description="Suggested prompt follow-ups")
    satellite_provenance: Optional[Dict[str, Any]] = Field(None, description="Dual-agency satellite telemetry provenance")
    location_context: Optional[Dict[str, Any]] = Field(None, description="Resolved shared location context")
    location: Optional[Dict[str, Any]] = Field(None, description="Standardized geographic location details")
    origin: Optional[Dict[str, Any]] = Field(None, description="Resolved route departure origin")
    destination: Optional[Dict[str, Any]] = Field(None, description="Resolved route arrival destination")
    intent: Optional[str] = Field(None, description="Canonical query intent category")
    summary: Optional[str] = Field(None, description="Concise 1-2 sentence query summary")
    conditions: Optional[Dict[str, Any]] = Field(None, description="Relevant observed/forecast conditions")
    risk: Optional[Dict[str, Any]] = Field(None, description="Standardized risk summary {level, score}")
    recommendation: Optional[str] = Field(None, description="Actionable recommendation")
    provenance: Optional[List[Dict[str, Any]]] = Field(None, description="Detailed data provenance list")
    visualization: Optional[Dict[str, Any]] = Field(None, description="Visualization metadata (ROUTE, BOUNDARY) or null")
    data_quality: Optional[Union[Dict[str, Any], str]] = Field(None, description="Data freshness and quality status")
    timestamp: Optional[str] = Field(None, description="ISO timestamp of response generation")
    model_versions: Optional[Dict[str, Any]] = Field(None, description="Active ML and heuristic model versions")
    error_code: Optional[str] = Field(None, description="Standardized error code")
    error: Optional[str] = Field(None, description="Error description or code")

    model_config = {"extra": "allow"}


# =====================================================================
# Supabase User, Location & Alert Schemas
# =====================================================================

class UserRole(str, Enum):
    FISHERMAN = "FISHERMAN"
    RESEARCHER = "RESEARCHER"
    AUTHORITY = "AUTHORITY"
    MARITIME_OPERATOR = "MARITIME_OPERATOR"


class UserRegisterRequest(BaseModel):
    id: Optional[str] = Field(None, description="Optional persistent user UUID identifier")
    name: str = Field(..., min_length=1, description="Full name of the marine stakeholder")
    role: str = Field(..., description="Role: FISHERMAN, RESEARCHER, AUTHORITY, MARITIME_OPERATOR")
    phone_number: str = Field(..., description="Phone number, e.g. +919876543210")
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0, description="Initial GPS latitude")
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0, description="Initial GPS longitude")
    current_lat: Optional[float] = Field(None, ge=-90.0, le=90.0, description="Alias for latitude")
    current_lon: Optional[float] = Field(None, ge=-180.0, le=180.0, description="Alias for longitude")
    notification_enabled: bool = Field(True, description="Enable notifications")
    sms_alerts_enabled: bool = Field(True, description="Enable SMS alerts")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Name cannot be empty")
        return s

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        cleaned = v.strip().upper().replace(" ", "_").replace("-", "_")
        if cleaned in ("MARITIME_AUTHORITY", "AUTHORITY", "COAST_GUARD", "NAVY"):
            return UserRole.AUTHORITY.value
        elif cleaned in ("FISHERMAN", "FISHERY"):
            return UserRole.FISHERMAN.value
        elif cleaned in ("RESEARCHER", "RESEARCH", "SCIENTIST"):
            return UserRole.RESEARCHER.value
        elif cleaned in ("MARITIME_OPERATOR", "OPERATOR", "SHIPPING"):
            return UserRole.MARITIME_OPERATOR.value
        elif cleaned in [r.value for r in UserRole]:
            return cleaned
        raise ValueError(f"Invalid role: '{v}'. Must be one of: {', '.join([r.value for r in UserRole])}")

    @field_validator("phone_number")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        cleaned = v.strip().replace(" ", "").replace("-", "")
        digits = re.sub(r"\D", "", cleaned)
        if len(digits) < 10:
            raise ValueError("Phone number must contain at least 10 digits")
        if not re.match(r"^\+?[0-9]{10,15}$", cleaned):
            raise ValueError("Invalid phone number format. Expected e.g. +919876543210 or 9876543210")
        return cleaned

    def get_lat(self) -> Optional[float]:
        return self.latitude if self.latitude is not None else self.current_lat

    def get_lon(self) -> Optional[float]:
        return self.longitude if self.longitude is not None else self.current_lon


class UserLocationUpdateRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier (UUID)")
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Updated latitude")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Updated longitude")
    accuracy_meters: Optional[float] = Field(None, ge=0.0, description="GPS accuracy in meters")
    recorded_at: Optional[str] = Field(None, description="ISO timestamp of location capture")

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("user_id cannot be empty")
        return s


class TestAlertRequest(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Alert epicenter latitude")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Alert epicenter longitude")
    radius_km: float = Field(100.0, gt=0.0, le=5000.0, description="Hazard radius in kilometers")
    severity: str = Field("WARNING", description="Severity level: INFO, WARNING, DANGER, CRITICAL")
    message: str = Field(..., min_length=1, description="Alert warning message")
    title: Optional[str] = Field("TEST ALERT - Cyclone warning", description="Alert headline title")
    hazard_type: Optional[str] = Field("CYCLONE", description="Hazard category: CYCLONE, SQUALL, TSUNAMI, TEST")
    alert_type: Optional[str] = Field(None, description="Alias for hazard_type")

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if cleaned not in ("INFO", "WARNING", "DANGER", "CRITICAL"):
            return "WARNING"
        return cleaned



