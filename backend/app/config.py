from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application configuration from environment variables."""

    # Supabase
    supabase_url: str = "postgresql://localhost:5432/postgres"
    supabase_key: str = "test_key"
    supabase_service_role_key: str = "test_service_role_key"

    # Google AI
    gemini_api_key: str = "test_gemini_key"
    gemini_model: str = "gemini-2.0-flash-exp"

    # Messenger/WhatsApp
    messenger_verify_token: str = ""
    messenger_app_secret: str = ""

    # App
    environment: str = "development"
    log_level: str = "info"

    model_config = {"env_file": ".env", "case_sensitive": False}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
