import os
from urllib.parse import quote_plus
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Configuration de l'application chargée depuis les variables d'environnement.
    """
    # --- CORRECTION ---
    # Ajouter les variables pour l'utilisateur admin de PostgreSQL
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str

    # PostgreSQL (Utilisateur LLM)
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
        # URL-encode credentials to support special characters such as @, :, /, +, !, $
        user = quote_plus(self.LLM_USER)
        pwd = quote_plus(self.LLM_PASSWORD)
        host = self.DB_HOST
        port = self.DB_PORT
        db = self.POSTGRES_DB
        return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"

    @property
    def ADMIN_DATABASE_URL(self) -> str:
        # Admin URL for schema introspection and initialization tasks
        admin_user = quote_plus(self.POSTGRES_USER)
        admin_pwd = quote_plus(self.POSTGRES_PASSWORD)
        host = self.DB_HOST
        port = self.DB_PORT
        db = self.POSTGRES_DB
        return f"postgresql://{admin_user}:{admin_pwd}@{host}:{port}/{db}"

    @property
    def OLLAMA_BASE_URL(self) -> str:
        return f"http://{self.OLLAMA_HOST}:{self.OLLAMA_PORT}"

# Instance unique des paramètres pour toute l'application
settings = Settings()