# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "My Backend API"
    API_V1_STR: str = "/api/v1"
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "myuser"
    POSTGRES_PASSWORD: str = "mypassword"
    POSTGRES_DB: str = "mydatabase"
    DATABASE_URL: str = "postgresql+psycopg2://myuser:mypassword@localhost:5432/mydatabase"

    # replaces the old inner Config class
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
