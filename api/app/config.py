import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Configuration de l'application chargée depuis les variables d'environnement.
    """
    # PostgreSQL
    LLM_USER: str
    LLM_PASSWORD: str
    POSTGRES_DB: str
    DB_HOST: str = "postgres-db"
    DB_PORT: int = 5432

    # ChromaDB
    CHROMA_HOST: str
    CHROMA_PORT: int
    CHROMA_COLLECTION: str

    # Ollama
    OLLAMA_HOST: str
    OLLAMA_PORT: int
    LLM_MODEL: str

    # Embedding Model
    EMBEDDING_MODEL_NAME: str

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.LLM_USER}:{self.LLM_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.POSTGRES_DB}"

    @property
    def OLLAMA_BASE_URL(self) -> str:
        return f"http://{self.OLLAMA_HOST}:{self.OLLAMA_PORT}"

# Instance unique des paramètres pour toute l'application
settings = Settings()