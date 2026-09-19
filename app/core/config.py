# app/core/config.py
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "ContextEngine"
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434" 
    QDRANT_HOST: str = "127.0.0.1"
    QDRANT_PORT: int = 6333
    REDIS_HOST: str = "127.0.0.1"
    REDIS_PORT: int = 6380

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()


# Force container names strictly if executing inside a container sandbox environment
if os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER") == "true":
    settings.OLLAMA_BASE_URL = "http://docker.internal"
    settings.QDRANT_HOST = "qdrant"
    settings.QDRANT_PORT = 6333
    settings.REDIS_HOST = "redis"
    settings.REDIS_PORT = 6379  # 🌟 Essential: Mapped internally to port 6379 inside the Docker bridge network
