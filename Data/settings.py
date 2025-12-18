"""
Configuration management for OSINT AI System
"""

import os
from pathlib import Path
from typing import Optional 
from pydantic_settings import BaseSettings
from pydantic import Field 

class Settings(BaseSettings):
    #Application settings with environment variable support

    #Application
    APP_NAME: str = "OSINT-AI-Early-Warning-System"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False 

    #API Keys - LLM Providers
    ANTHROPIC_API_KEY: str = Field(..., description = "Anthropic API key for Claude")
    OPENAI_API_KEY: str = Field(..., description = "OpenAI API key")

    #API Keys - Data Sources
    NEWSAPI_KEY: Optional[str] = None
    EVENTREGISTRY_KEY: Optional[str] = None
    TWITTER_BEARER_TOKEN: Optional[str] = None 
    GDELT_API_KEY: Optional[str] = None

    #Vector Store
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: str = "us-west1-gcp"
    PINECONE_INDEX_NAME: str = "osing-events"

    #Databases
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "osint_ai"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = Field(..., description = "PostgreSQL password")

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = Field(..., description = "Neo4j password")

    #AWS
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    S3_BUCKET_NAME: str = "osint-ai-documents"

    #Kafka/Kinesis
    USE_KAFKA: bool = True 
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC_EVENTS: str = "geopolitical-events"

    KINESIS_STREAM_NAME: str = "geopolitical-events"

    #LangSmith (Monitoring)
    LANGCHAIN_TRACING_V2: bool = True 
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: Optional[str] = None 
    LANGCHAIN_PROJECT: str = "osint-ai-geopolitical"

    #Model Configuration
    PRIMARY_LLM_MODEL: str = "claude-3-5-haiku-20241022"
    FAST_LLM_MODEL: str = "claude-3-5-haiku-20241022"
    EMBEDDING_MODEL: str = "text-embedding-3-large"

    #LLM Parameters
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 4096

    #Data Ingestions
    GDELT_POLL_INTERVAL: int = 900 #15 minutes
    NEWS_API_POLL_INTERVAL: int = 3600 #1 hour
    MAX_ARTICLES_PER_FETCH: int = 100

    #Events Processing
    SIMILARITY_THRESHOLD: float = 0.05 #For deduplication
    CLUSTER_MIN_SAMPLES: int = 3
    CLUSTER_EPS: float = 0.3 

    #Threat Classification
    THREAT_CATEGORIES: list = Field(
        default=[
            "military_buildup",
            "airspace_violation",
            "naval_incident",
            "border_conflict",
            "drone_attack",
            "sanctions",
            "terrorism",
            "civil_unrest",
            "cyber_operation",
            "wmd_activity",
            "humanitarian_crisis",
            "energy_security"
        ]
    )

    #Escalation Thresholds
    CRITICAL_THRESHOLD: float = 0.70 #Escalation probability
    HIGH_THRESHOLD: float = 0.50
    MEDIUM_THRESHOLD: float = 0.30 

    #Alert Configuration
    ENABLE_ALERTS: bool = True 
    ALERT_WEBHOOK_URL: Optional[str] = None 
    ALERT_EMAIL_TO: Optional[str] = None 

    #API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4

    #Monitoring
    PROMETHEUS_PORT: int = 9090
    SENTRY_DSN: Optional[str] = None 

    #Paths
    DATA_DIR: Path = Path("data")
    CACHE_DIR: Path = Path("cache")
    MODELS_DIR: Path = Path("models")
    LOGS_DIR: Path = Path("logs")

    # Optional external URLs (present in .env for convenience)
    DATABASE_URL: Optional[str] = None
    REDIS_URL: Optional[str] = None
    VLLM_BASE_URL: Optional[str] = None
    SGLANG_BASE_URL: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True 

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #Create directories if they don't exist 
        for dir_path in [self.DATA_DIR, self.CACHE_DIR, self.MODELS_DIR, self.LOGS_DIR]:
            dir_path.mkdir(parents = True, exist_ok = True)

    @property 
    def postgres_url(self) -> str:
        #Get PostgreSQL connection URL
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def redis_url(self) -> str: 
        #Get Redis connection URL
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
#Global settings instance
settings = Settings() 

#Threat category metadata
THREAT_METADATA = {
    "military_buildup": {
        "severity_base": 0.7,
        "keywords": ["troop", "deployment", "military", "exercise", "mobilization"],
        "actors": ["state_military", "foreign_force"]
    },
    "airspace_violation": {
        "severity_base": 0.6,
        "keywords": ["airspace", "intercept", "fighter", "bomber", "incursion"],
        "actors": ["air_force", "aviation"]
    },
    "naval_incident": {
        "severity_base": 0.65,
        "keywords": ["naval", "warship", "destroyer", "carrier", "maritime"],
        "actors": ["navy", "coast_guard"]
    },
    "border_conflict": {
        "severity_base": 0.75,
        "keywords": ["border", "crossing", "frontier", "demarcation", "territorial"],
        "actors": ["border_force", "paramilitary"]
    },
    "drone_attack": {
        "severity_base": 0.8,
        "keywords": ["drone", "uav", "unmanned", "strike", "kamikaze"],
        "actors": ["militia", "insurgent", "state_military"]
    },
    "sanctions": {
        "severity_base": 0.4,
        "keywords": ["sanction", "embargo", "restriction", "penalty", "ban"],
        "actors": ["government", "un", "eu"]
    },
    "terrorism": {
        "severity_base": 0.9,
        "keywords": ["terror", "attack", "bombing", "suicide", "hostage"],
        "actors": ["terrorist_group", "extremist"]
    },
    "civil_unrest": {
        "severity_base": 0.5,
        "keywords": ["protest", "riot", "unrest", "demonstration", "coup"],
        "actors": ["civilian", "opposition", "military"]
    },
    "cyber_operation": {
        "severity_base": 0.6,
        "keywords": ["cyber", "hack", "ransomware", "infrastructure", "attack"],
        "actors": ["state_actor", "hacker_group"]
    },
    "wmd_activity": {
        "severity_base": 0.95,
        "keywords": ["nuclear", "missile", "wmd", "proliferation", "test"],
        "actors": ["state_military", "rogue_state"]
    },
    "humanitarian_crisis": {
        "severity_base": 0.7,
        "keywords": ["refugee", "famine", "genocide", "displacement", "crisis"],
        "actors": ["un", "ngo", "government"]
    },
    "energy_security": {
        "severity_base": 0.55,
        "keywords": ["pipeline", "energy", "oil", "gas", "opec"],
        "actors": ["state_company", "cartel"]
    }
}


# Geographic regions of interest
REGIONS = {
    "south_china_sea": {
        "bbox": [99.0, -4.0, 125.0, 23.0],  # [min_lon, min_lat, max_lon, max_lat]
        "priority": "critical",
        "keywords": ["south china sea", "spratly", "paracel", "scarborough", "taiwan strait"]
    },
    "middle_east": {
        "bbox": [34.0, 12.0, 63.0, 42.0],
        "priority": "critical",
        "keywords": ["syria", "iraq", "iran", "israel", "yemen", "persian gulf"]
    },
    "ukraine_russia": {
        "bbox": [22.0, 44.0, 40.0, 52.0],
        "priority": "critical",
        "keywords": ["ukraine", "russia", "crimea", "donbas", "kherson"]
    },
    "korean_peninsula": {
        "bbox": [124.0, 33.0, 131.0, 43.0],
        "priority": "high",
        "keywords": ["north korea", "south korea", "dmz", "pyongyang"]
    },
    "indo_pakistan": {
        "bbox": [68.0, 23.0, 97.0, 37.0],
        "priority": "high",
        "keywords": ["kashmir", "loc", "india", "pakistan", "china border"]
    },
    "sahel": {
        "bbox": [-17.0, 10.0, 15.0, 23.0],
        "priority": "medium",
        "keywords": ["mali", "niger", "burkina faso", "wagner", "sahel"]
    },
    "south_caucasus": {
        "bbox": [38.0, 38.0, 50.0, 44.0],
        "priority": "medium",
        "keywords": ["armenia", "azerbaijan", "nagorno-karabakh", "georgia"]
    },
    "horn_of_africa": {
        "bbox": [32.0, -5.0, 52.0, 18.0],
        "priority": "medium",
        "keywords": ["ethiopia", "somalia", "eritrea", "sudan", "tigray"]
    }
}
