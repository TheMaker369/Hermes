"""
Configuration management for the Hermes AI System.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """System-wide configuration settings."""

    # API Keys
    openai_api_key: str = ""
    firecrawl_api_key: str = ""

    # Feature Flags
    allow_remote: bool = True
    allow_openai: bool = False
    allow_external: bool = True
    local_only_llm: bool = True

    # Storage Configuration
    chroma_remote: bool = False
    chroma_url: str = "https://your-chroma-cloud-instance.com"
    chroma_path: str = "./memory_storage"

    # Performance Settings
    timeout_seconds: int = 5
    research_depth: int = 2
    research_breadth: int = 3

    # Model Settings
    o3_model: str = "o3-mini-high"
    default_llm_model: str = "deepseek-r1"
    fallback_llm_model: str = "mistral"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
