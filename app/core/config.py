from pydantic_settings import BaseSettings,SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str="RAG ENGINE"
    ENVIRONMENT: str="development"
    PORT: int = 8000
    DEBUG: bool = True

    REDIS_URL: str="redis://localhost:6379/0"
    QDRANT_URL: str="http://localhost:6333"
    
    model_config=SettingsConfigDict(env_file=".env",env_file_encoding="utf-8")

settings=Settings()