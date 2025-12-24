"""
Application configuration with Pydantic settings.
Handles environment variables and validation.
Supports Docker Secrets for secure credential management.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import Optional
from app.utils.secrets import get_secret


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    internal_api_key: Optional[str] = Field(None, env="INTERNAL_API_KEY")
    environment: str = Field("local", env="ENVIRONMENT")
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_chat_model: str = Field("gpt-4o", env="OPENAI_CHAT_MODEL")
    openai_embedding_model: str = Field(
        "text-embedding-3-small",
        env="OPENAI_EMBEDDING_MODEL"
    )
    
    # Qdrant Configuration
    qdrant_host: str = Field("localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(6333, env="QDRANT_PORT")
    qdrant_collection_name: str = Field("documents", env="QDRANT_COLLECTION_NAME")
    qdrant_api_key: Optional[str] = Field(None, env="QDRANT_API_KEY")
    
    # Redis Configuration
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    redis_db: int = Field(0, env="REDIS_DB")
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    redis_ttl_seconds: int = Field(
        604800,  # 7 days default
        env="REDIS_TTL_SECONDS",
        description="Time-to-live for conversation sessions in seconds"
    )
    
    # LangGraph Configuration
    enable_langgraph_persistence: bool = Field(
        True,
        env="ENABLE_LANGGRAPH_PERSISTENCE",
        description="Enable persistent conversation memory with Redis"
    )
    max_agent_iterations: int = Field(
        10,
        env="MAX_AGENT_ITERATIONS",
        description="Maximum reasoning iterations to prevent infinite loops"
    )
    
    # LangSmith Configuration (NEW - Enhanced)
    langchain_tracing_v2: bool = Field(
        False,
        env="LANGCHAIN_TRACING_V2",
        description="Enable LangSmith tracing for observability"
    )
    langchain_api_key: Optional[str] = Field(
        None,
        env="LANGCHAIN_API_KEY",
        description="LangSmith API key from https://smith.langchain.com"
    )
    langchain_project: str = Field(
        "agentic-backend",
        env="LANGCHAIN_PROJECT",
        description="LangSmith project name"
    )
    langsmith_endpoint: str = Field(
        "https://api.smith.langchain.com",
        env="LANGSMITH_ENDPOINT",
        description="LangSmith API endpoint"
    )
    langsmith_enable_feedback: bool = Field(
        True,
        env="LANGSMITH_ENABLE_FEEDBACK",
        description="Enable user feedback collection"
    )
    langsmith_enable_cost_tracking: bool = Field(
        True,
        env="LANGSMITH_ENABLE_COST_TRACKING",
        description="Enable automatic cost calculation"
    )
    
    # RAG Configuration
    rag_top_k: int = Field(5, env="RAG_TOP_K")
    rag_score_threshold: float = Field(0.7, env="RAG_SCORE_THRESHOLD")
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(1000, env="RATE_LIMIT_PER_HOUR")
    
    # Background Jobs
    celery_broker_url: Optional[str] = Field(None, env="CELERY_BROKER_URL")
    celery_result_backend: Optional[str] = Field(None, env="CELERY_RESULT_BACKEND")
    
    # Monitoring & Alerting (NEW)
    enable_performance_monitoring: bool = Field(
        True,
        env="ENABLE_PERFORMANCE_MONITORING",
        description="Track performance metrics"
    )
    slow_request_threshold_ms: int = Field(
        5000,
        env="SLOW_REQUEST_THRESHOLD_MS",
        description="Log warning for requests slower than this (milliseconds)"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()