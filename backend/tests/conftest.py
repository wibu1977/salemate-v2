"""
Shared pytest fixtures for all tests.

This module provides reusable fixtures for testing Sellora backend.
Fixures include test data samples, database mocks, HTTP test clients,
and other common test utilities.
"""

import asyncio
import pytest
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear settings cache to allow tests to set environment variables.

    This fixture runs before each test to ensure tests can control
    environment variables without interference from previously cached settings.
    """
    import app.config
    app.config.get_settings.cache_clear()


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of default event loop for test session.

    This fixture ensures that all async tests use the same event loop,
    which is important for cleanup and resource management.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db_pool() -> AsyncGenerator[AsyncMock, None]:
    """Create a test database pool (mocked for testing).

    In production, this would connect to a test database.
    For Phase 1 tests, we use mocked connections in individual tests.
    This fixture is available for integration tests that need database setup.

    Yields:
        AsyncMock: A mocked database pool instance
    """
    # Set default environment variables for database if not set by tests
    import os
    if "SUPABASE_URL" not in os.environ:
        os.environ["SUPABASE_URL"] = "postgresql://test/test"
    if "SUPABASE_KEY" not in os.environ:
        os.environ["SUPABASE_KEY"] = "test_key"
    if "SUPABASE_SERVICE_ROLE_KEY" not in os.environ:
        os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "test_service_role_key"

    # Return a mock pool instead of trying to connect
    pool = AsyncMock()
    pool.init = AsyncMock()
    pool.close = AsyncMock()
    pool.acquire = AsyncMock()
    yield pool


@pytest.fixture
def test_client() -> TestClient:
    """Create a test client for FastAPI.

    This fixture provides a FastAPI TestClient configured to make requests
    to the test FastAPI application without starting a real server.

    Note: This fixture will import the app module, so any tests
    that need to control environment variables should set them before
    using this fixture or import app directly.

    Yields:
        TestClient: An HTTP client configured for testing app
    """
    # Set default environment variables if not set by tests
    import os
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = "test_gemini_key"
    if "MESSENGER_VERIFY_TOKEN" not in os.environ:
        os.environ["MESSENGER_VERIFY_TOKEN"] = "test_verify_token"
    if "MESSENGER_APP_SECRET" not in os.environ:
        os.environ["MESSENGER_APP_SECRET"] = "test_app_secret"
    if "SUPABASE_URL" not in os.environ:
        os.environ["SUPABASE_URL"] = "postgresql://test/test"
    if "SUPABASE_KEY" not in os.environ:
        os.environ["SUPABASE_KEY"] = "test_key"
    if "SUPABASE_SERVICE_ROLE_KEY" not in os.environ:
        os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "test_service_role_key"

    from app.main import app
    return TestClient(app)


@pytest.fixture
def sample_message_data() -> dict:
    """Sample message data for testing.

    Returns a dictionary representing a typical message that would be
    stored in the messages table.

    Returns:
        dict: Sample message data with message_id, conversation_id, shop_id, etc.
    """
    return {
        "message_id": "test_msg_001",
        "conversation_id": "test_conv_001",
        "shop_id": "550e8400-e29b-41d4-a716-446655440000",
        "sender_id": "customer_001",
        "sender_type": "customer",
        "channel": "messenger",
        "content_type": "text",
        "content": {"text": "Hello, I'm interested in your products"},
        "sent_at": "2024-01-15T10:30:00Z",
    }


@pytest.fixture
def sample_messenger_webhook() -> dict:
    """Sample Messenger webhook payload for testing.

    Returns a dictionary representing a typical webhook payload from
    the Facebook Messenger Platform.

    Returns:
        dict: Sample webhook payload with entry, messaging objects, etc.
    """
    return {
        "object": "page",
        "entry": [{
            "id": "123456789",
            "time": 1705314600,
            "messaging": [{
                "sender": {"id": "1234567890"},
                "recipient": {"id": "9876543210"},
                "timestamp": 1705314600000,
                "message": {
                    "mid": "msg_id_001",
                    "text": "How much does your product cost?"
                }
            }]
        }]
    }


@pytest.fixture
def sample_product_data() -> dict:
    """Sample product data for testing.

    Returns a dictionary representing a typical product that would be
    stored in the product catalog.

    Returns:
        dict: Sample product data with name, description, price, variants, tags
    """
    return {
        "name": "Test Product",
        "description": "A test product for testing purposes",
        "price": 29.99,
        "currency": "USD",
        "variants": ["Small", "Medium", "Large"],
        "tags": ["test", "sample"],
    }


@pytest.fixture
def mock_db() -> AsyncMock:
    """Mock database connection for testing.

    Returns an AsyncMock that can be used to mock database operations
    without connecting to a real database.

    Returns:
        AsyncMock: Mock database connection with common methods mocked
    """
    conn = AsyncMock()
    conn.execute = AsyncMock()
    conn.fetch = AsyncMock()
    conn.fetchval = AsyncMock()
    conn.fetchrow = AsyncMock()
    conn.executemany = AsyncMock()
    conn.is_closed = False
    return conn
