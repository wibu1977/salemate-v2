import os
from app.config import Settings, get_settings


def test_settings_loads():
    """Test that settings load from environment."""
    os.environ["SUPABASE_URL"] = "https://test.supabase.co"
    os.environ["SUPABASE_KEY"] = "test-key"
    os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "test-service-key"
    os.environ["GEMINI_API_KEY"] = "test-gemini-key"

    settings = get_settings()
    assert settings.supabase_url == "https://test.supabase.co"
    assert settings.supabase_key == "test-key"
    assert settings.gemini_model == "gemini-2.0-flash-exp"


def test_get_settings_cached():
    """Test that get_settings returns cached instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
