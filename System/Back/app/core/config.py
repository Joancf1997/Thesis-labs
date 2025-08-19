# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "New AI Assistant"
    API_V1_STR: str = "/api/v1"
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "joseandres"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = "news_AI"
    DATABASE_URL : str = "postgresql://joseandres:@localhost/news_AI"

    # replaces the old inner Config class
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
