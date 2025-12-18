"""
Data models for OSINT AI System
"""

from datetime import datetime 
from typing import Optional, List, Dict, Any 
from enum import Enum 
from pydantic import BaseModel, Field, validator
from uuid import UUID, uuid4 



class ThreatCategory(str, Enum):
    #Enumeration of threat categories
    MILITARY_BUILDUP = "military_buildup"
    AIRSPACE_VIOLATION = "airspace_violation"
    NAVAL_INCIDENT = "naval_incident"
    BORDER_CONFLICT = "border_conflict"
    DRONE_ATTACK = "drone_attack"
    SANCTIONS = "sanctions"
    TERRORISM = "terrorism"
    CIVIL_UNREST = "civil_unrest"
    CYBER_OPERATION = "cyber_operation"
    WMD_ACTIVITY = "wmd_activity"
    HUMANITARIAN_CRISIS = "humanitarian_crisis"
    ENERGY_SECURITY = "energy_security"

class AlertLevel(str, Enum):
    #Alert severity level
    CRITICAL = "critical" #>70% escalation probability
    HIGH = "high" #50-70%
    MEDIUM = "medium" #30-50%
    LOW = "low" #<30%
    INFO = "info" #Informational only 

class EventSource(str, Enum):
    #Data source types 
    GDELT = "gdelt"
    NEWSAPI = "newsapi"
    EVENTREGISTRY = "eventregistry"
    TWITTER = "twitter"
    RSS = "rss"
    MANUAL = "manual"

class Entity(BaseModel):
    #Named entity extracted from text
    text: str
    type: str #Person, Org, Gpe, Loc, etc
    confidence: float = Field(ge = 0.0, le = 1.0)
    start_char: int
    end_char: int 
    canonical_name: Optional[str] = None #Resolved entity name 
    wikidata_id: Optional[str] = None 

class GeopoliticalEvent(BaseModel):
    #Core event model
    event_id: UUID = Field(default_factory = uuid4)
    timestamp: datetime 
    source: EventSource
    source_url: str 

    #Content
    title: str
    description: str 
    full_text: Optional[str] = None 
    language: str = "en"

    #Classification
    threat_category: Optional[ThreatCategory] = None 
    threat_confidence: float = Field(default = 0.0, ge = 0.0, le = 1.0)

    #Geospatial
    locations: List[str] = Field(default_factory = list)
    countries: List[str] = Field(default_factory = list)
    coordinates: Optional[Dict[str, float]] = None #{"lat": x, "lon": y}
    region: Optional[str] = None 

    #Entities
    entities: List[Entity] = Field(default_factory = list)
    actors: List[str] = Field(default_factory = list) #Key actors involved 

    #Metadata
    keywords: List[str] = Field(default_factory = list)
    sentiment_score: Optional[float] = None #-1 to 1
    importance_score: float = Field(default = 0.5, ge = 0.0, le = 1.0)

    #Processing
    embedding: Optional[List[float]] = None 
    cluster_id: Optional[int] = None 
    is_duplicate: bool = False
    duplicate_of: Optional[UUID] = None 

    #Timestamps
    processed_at: datetime = Field(default_factory = datetime.utcnow)
    indexed_at: Optional[datetime] = None 

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }

class EventCluster(BaseModel):
    #Cluster of related events
    cluster_id: int 
    events: List[UUID] = Field(default_factory = list)
    event_count: int = 0

    #Cluster characteristics
    primary_threat: ThreatCategory
    region: str
    start_time: datetime
    end_time: datetime 

    #Computed features
    intensity: float = Field(ge = 0.0, le = 1.0) #Event frequency
    geographic_spread: float = 0.0 #Spatial dispersion
    escalation_trajectory: str = "stable" #increasing/decreasing/stable 

    #Narrative
    summary: str 
    key_actors: List[str] = Field(default_factory = list) 

    class Config: 
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class EscalationPrediction(BaseModel):
    #Model output for escalation risk 
    cluster_id: int 
    prediction_time: datetime = Field(default_factory = datetime.utcnow)

    #Prediction
    escalation_probability: float = Field(ge = 0.0, le = 1.0)
    confidence: float = Field(ge = 0.0, le = 1.0)
    time_horizon_days: int = 30 

    #Alert
    alert_level: AlertLevel 

    #Features used
    features: Dict[str, Any] = Field(default_factory = dict) 

    #Explanation
    reasoning: str 
    historical_parallels: List[str] = Field(default_factory = list) 
    risk_factors: List[str] = Field(default_factory = list) 
    mitigating_factors: List[str] = Field(default_factory = list) 

    class Config: 
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ThreatAlert(BaseModel):
    #Generated alert for stakeholders
    alert_id: UUID = Field(default_factory = uuid4)
    cluster_id: int 
    alert_level: AlertLevel 

    #Content
    title: str
    summary: str 
    detailed_analysis: str 

    #Context
    threat_category: ThreatCategory
    region: str 
    escalation_probability: float 

    #Evidence
    supporting_events: List[UUID] = Field(default_factory = list) 
    source_urls: List[str] = Field(default_factory = list) 

    #Timing
    created_at: datetime = Field(default_factory = datetime.utcnow)
    expires_at: Optional[datetime] = None 

    #Distribution
    sent_to: List[str] = Field(default_factory = list)
    acknowledged: bool = False 

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }

class IntelligenceBrief(BaseModel):
    #Daily Intelligence Brief
    brief_id: UUID = Field(default_factory = uuid4)
    date: datetime 

    #Sections
    executive_summary: str 
    critical_alerts: List[ThreatAlert] = Field(default_factory = list) 
    watch_list: List[Dict[str, Any]] = Field(default_factory = list)
    declining_tensions: List[Dict[str, Any]] = Field(default_factory = list)

    #Regional overviews
    regional_summaries: Dict[str, str] = Field(default_factory = dict)

    #Statistics
    total_events_processed: int = 0
    new_alerts: int = 0
    ongoing_situations: int = 0

    #Metadata
    generated_at: datetime = Field(default_factory = datetime.utcnow)
    generated_by: str = "OSINT-AI-System"
    version: str = "1.0"

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }

class PatternMatch(BaseModel):
    #Historical pattern matching result
    event_id: UUID
    historical_event_id: str 
    similarity_score: float = Field(ge = 0.0, le = 1.0)

    #Historical context 
    historical_data: datetime 
    historical_summary: str 
    outcome: str #What happened after the historical event 

    #Relevance
    relevance_explanation: str 
    lessons_learned: str

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }

class AgentState(BaseModel):
    #State object for LangGraph workflow
    #input
    raw_events: List[Dict[str, Any]] = Field(default_factory = list)

    #Processed events
    events: List[GeopoliticalEvent] = Field(default_factory = list)
    clusters: List[EventCluster] = Field(default_factory = list) 

    #Analysis
    predictions: List[EscalationPrediction] = Field(default_factory = list) 
    pattern_matches: List[PatternMatch] = Field(default_factory = list) 

    #Outputs
    alerts: List[ThreatAlert] = Field(default_factory = list) 
    brief: Optional[IntelligenceBrief] = None 

    #Workflow control 
    current_step: str = "ingestion"
    errors: List[str] = Field(default_factory = list)
    metadata: Dict[str, Any] = Field(default_factory = dict)

    class Config:
        arbitrary_types_allowed = True

class SystemMetrics(BaseModel):
    #System performance metrics
    timestamp: datetime = Field(default_factory = datetime.utcnow)

    #Throughput
    events_processed_24h: int = 0
    events_per_minute: float = 0.0

    #Quality
    classification_accuracy: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0

    #Latency
    avg_processing_time_ms: float = 0.0
    p95_processing_time_ms: float = 0.0
    p99_processing_time_ms: float = 0.0

    #Alerts
    critical_alerts_24h: int = 0
    high_alerts_24h: int = 0

    #Resources
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0

    #LLM
    llm_tokens_used_24h: int = 0
    llm_api_errors_24h: int = 0

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
