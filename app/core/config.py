"""Application settings, loaded from environment variables and an optional .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "ContextEngine"
    LOG_LEVEL: str = "INFO"

    # Ollama (LLM + embeddings)
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    LLM_MODEL: str = "llama3.1"
    EMBEDDING_MODEL: str = "mxbai-embed-large"
    EMBEDDING_DIMENSIONS: int = 1024
    LLM_TIMEOUT_SECONDS: float = 45.0

    # Qdrant (vector store)
    QDRANT_HOST: str = "127.0.0.1"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "context_engine_docs"

    # Redis Stack (conversation checkpoints)
    REDIS_HOST: str = "127.0.0.1"
    REDIS_PORT: int = 6380

    # Agent behaviour
    RETRIEVAL_K: int = 4
    MAX_RETRIEVAL_RETRIES: int = 1  # query rewrites before falling back to web search
    WEB_SEARCH_RESULTS: int = 3
    GRADER_CONCURRENCY: int = 4  # parallel relevance checks; Ollama must allow parallel requests

    # API
    # The UI's Vite dev server runs on 3000 (see vite.config.js); 5173 is Vite's default.
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

    @property
    def redis_url(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}"


settings = Settings()
