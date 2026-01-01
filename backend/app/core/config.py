from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path

# Get the project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "Hybrid RAG System"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-it-in-prod")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    ADMIN_REGISTRATION_SECRET: str = os.getenv("ADMIN_REGISTRATION_SECRET", "super-secret-admin-key")

    # Ollama Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "qwen3:1.7b")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    # DB and Storage - Use absolute paths
    DATABASE_URL: str = f"sqlite+aiosqlite:///{ROOT_DIR}/data/database.db"
    STORAGE_PATH: str = str(ROOT_DIR / "data" / "documents")
    FAISS_INDEX_PATH: str = str(ROOT_DIR / "data" / "faiss_index")
    LOG_FILE: str = str(ROOT_DIR / "logs" / "app.log")

    class Config:
        case_sensitive = True

settings = Settings()
