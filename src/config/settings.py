"""
Configuration management using Pydantic Settings.
Demonstrates production-ready configuration patterns.
"""

from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings with type validation and environment support.
    
    Demonstrates:
    - Type-safe configuration
    - Environment-based settings
    - Validation with Pydantic
    - Production best practices
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Keys
    openai_api_key: str = Field(..., description="OpenAI API key")
    google_api_key: Optional[str] = Field(None, description="Google Gemini API key (optional)")
    
    # Application Settings
    app_name: str = Field("Pharmaceutical Intelligence AI", description="Application name")
    environment: Literal["development", "staging", "production"] = Field(
        "development", 
        description="Runtime environment"
    )
    debug: bool = Field(False, description="Debug mode")
    
    # Model Configuration
    default_llm_model: str = Field("gpt-4o", description="Default LLM model")
    llm_temperature: float = Field(0.7, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(2048, ge=1, le=8192, description="Max output tokens")
    
    # Embedding Configuration  
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model for vector search"
    )
    embedding_dimension: int = Field(384, description="Embedding vector dimension")
    
    # Vector Store Configuration
    vector_store_type: Literal["chroma", "faiss"] = Field("chroma", description="Vector store backend")
    chroma_persist_directory: str = Field("./data/chroma_db", description="ChromaDB persistence path")
    collection_name: str = Field("pharmaceutical_docs", description="Vector collection name")
    
    # Retrieval Configuration
    retrieval_top_k: int = Field(5, ge=1, le=20, description="Number of documents to retrieve")
    rerank_top_k: int = Field(3, ge=1, le=10, description="Number of documents after re-ranking")
    chunk_size: int = Field(512, ge=100, le=2000, description="Document chunk size")
    chunk_overlap: int = Field(50, ge=0, le=500, description="Overlap between chunks")
    
    # API Configuration
    api_host: str = Field("0.0.0.0", description="API host")
    api_port: int = Field(8000, ge=1000, le=65535, description="API port")
    api_workers: int = Field(1, ge=1, le=8, description="Number of API workers")
    
    # Rate Limiting
    rate_limit_requests: int = Field(100, description="Requests per minute")
    rate_limit_tokens: int = Field(100000, description="Tokens per minute")
    
    # Monitoring
    enable_metrics: bool = Field(True, description="Enable Prometheus metrics")
    enable_tracing: bool = Field(True, description="Enable request tracing")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        "INFO",
        description="Logging level"
    )
    
    # Compliance
    enable_compliance_check: bool = Field(True, description="Enable regulatory compliance checking")
    enable_safety_validation: bool = Field(True, description="Enable safety validation")
    
    # Evaluation
    enable_evaluation: bool = Field(False, description="Enable automatic evaluation")
    evaluation_sample_rate: float = Field(0.1, ge=0.0, le=1.0, description="% of requests to evaluate")
    



# Global settings instance
settings = Settings()
