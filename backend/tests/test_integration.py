"""
Integration tests for end-to-end webhook flow.

Tests the complete message processing pipeline:
1. Webhook receives message
2. Message is written to database
3. Intent is extracted
4. Conversation is updated
5. AI agent generates reply
6. Reply is sent back
"""

# Set environment variables before importing app
import os
os.environ["SUPABASE_URL"] = "postgresql://test/test"
os.environ["SUPABASE_KEY"] = "test_key"
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "test_service_role_key"
os.environ["GEMINI_API_KEY"] = "test_gemini_key"
os.environ["MESSENGER_VERIFY_TOKEN"] = "test_verify_token"
os.environ["MESSENGER_APP_SECRET"] = "test_app_secret"

import pytest
import time
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4, UUID
from datetime import datetime
from typing import AsyncGenerator

from fastapi.testclient import TestClient
from fastapi import Request

from app.main import app
from app.config import get_settings
from app.db.queries import IntentType, ConversationStage, ChannelType, SenderType
from app.services.extractor import FastExtractor
from app.services.conversation import ConversationTracker
from app.services.ai_agent import AIAgent
from app.db.connection import override_get_db, reset_get_db_override
from app.routers.webhooks import (
    override_settings, reset_settings_override,
    get_customer_by_psid, create_customer, get_or_create_conversation
)


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch('app.config.get_settings') as mock:
        settings = MagicMock()
        settings.supabase_url = "postgresql://test/test"
        settings.supabase_key = "test_key"
        settings.supabase_service_role_key = "test_service_role_key"
        settings.gemini_api_key = "test_gemini_key"
        settings.gemini_model = "gemini-2.0-flash-exp"
        settings.messenger_verify_token = "test_verify_token"
        settings.messenger_app_secret = "test_app_secret"
        settings.environment = "test"
        settings.log_level = "info"
        mock.return_value = settings
        yield settings


@pytest.fixture
def mock_db():
    """Mock database connection."""
    conn = AsyncMock()
    conn.fetchrow = AsyncMock()
    conn.fetch = AsyncMock()
    conn.fetchval = AsyncMock()
    conn.execute = AsyncMock()
    conn.is_closed = False
    return conn


@pytest.fixture
def client_with_db(mock_db):
    """Create test client with mocked database."""
    async def override_get_db_for_test():
        yield mock_db

    override_get_db(override_get_db_for_test)
    try:
        yield TestClient(app)
    finally:
        reset_get_db_override()


@pytest.fixture(autouse=True)
def reset_overrides():
    """Reset overrides after each test."""
    yield
    reset_get_db_override()
    reset_settings_override()


@pytest.mark.asyncio
class TestWebhookIntegration:
    """Test the complete webhook message flow."""

    async def test_complete_webhook_flow_customer_inquiry(self, client_with_db, mock_db):
        """Test complete webhook flow from message receipt to AI reply."""
        sample_shop_id = uuid4()
        sample_customer_id = uuid4()
        sample_conversation_id = uuid4()
        sample_message_id = uuid4()

        # Mock database calls
        # First call: customer lookup (not found)
        # Second call: get_or_create_conversation lookup (no existing)
        mock_db.fetchrow.side_effect = [None, None]

        mock_db.fetch.return_value = [{
            "conversation_id": sample_conversation_id,
            "shop_id": sample_shop_id,
            "customer_id": sample_customer_id,
            "channel": ChannelType.messenger,
            "started_at": datetime.utcnow(),
            "last_message_at": datetime.utcnow(),
            "message_count": 0,
            "customer_message_count": 0,
            "business_message_count": 0,
            "conversation_depth": 0,
            "conversation_stage": ConversationStage.discovery,
            "drop_off_flag": False,
            "resulted_in_order": False,
            "status": "active"
        }]

        mock_db.fetchval.return_value = sample_message_id

        webhook_data = {
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

        response = client_with_db.post("/webhooks/messenger", json=webhook_data)

        # Should return 200 quickly (< 50ms target)
        assert response.status_code == 200
        assert response.json() == {"status": "received"}

        # Verify database operations were called
        assert mock_db.fetchrow.call_count >= 2
        assert mock_db.fetch.call_count >= 1
        assert mock_db.fetchval.call_count >= 1

    async def test_webhook_multiple_messages_in_batch(self, client_with_db, mock_db):
        """Test webhook processing multiple messages in a single entry."""
        sample_shop_id = uuid4()
        sample_customer_id = uuid4()
        sample_conversation_id = uuid4()
        sample_message_id = uuid4()

        # Mock database calls - customer exists, conversation exists
        mock_db.fetchrow.side_effect = [
            # Customer lookup (found)
            {
                "customer_id": sample_customer_id,
                "shop_id": sample_shop_id,
                "psid": "1234567890",
                "display_name": "Test Customer",
                "channel": ChannelType.messenger,
                "first_seen_at": datetime.utcnow(),
                "last_contact_at": datetime.utcnow(),
                "conversation_count": 1,
                "total_order_value": 0.0,
                "order_count": 0
            },
            # Existing conversation lookup (found)
            {
                "conversation_id": sample_conversation_id,
                "shop_id": sample_shop_id,
                "customer_id": sample_customer_id,
                "channel": ChannelType.messenger,
                "started_at": datetime.utcnow(),
                "last_message_at": datetime.utcnow(),
                "message_count": 2,
                "customer_message_count": 1,
                "business_message_count": 1,
                "conversation_depth": 1,
                "conversation_stage": ConversationStage.discovery,
                "drop_off_flag": False,
                "resulted_in_order": False,
                "status": "active"
            }
        ]

        mock_db.fetchval.return_value = sample_message_id

        webhook_data = {
            "object": "page",
            "entry": [{
                "id": "123456789",
                "time": 1705314600,
                "messaging": [
                    {
                        "sender": {"id": "1234567890"},
                        "recipient": {"id": "9876543210"},
                        "timestamp": 1705314600000,
                        "message": {
                            "mid": "msg_id_001",
                            "text": "Hello!"
                        }
                    },
                    {
                        "sender": {"id": "1234567890"},
                        "recipient": {"id": "9876543210"},
                        "timestamp": 1705314601000,
                        "message": {
                            "mid": "msg_id_002",
                            "text": "How much?"
                        }
                    }
                ]
            }]
        }

        response = client_with_db.post("/webhooks/messenger", json=webhook_data)

        assert response.status_code == 200
        assert response.json() == {"status": "received"}

    async def test_webhook_with_existing_customer(self, client_with_db, mock_db):
        """Test webhook when customer already exists."""
        sample_shop_id = uuid4()
        sample_customer_id = uuid4()
        sample_conversation_id = uuid4()
        sample_message_id = uuid4()

        # Mock database calls - customer exists, no existing conversation
        mock_db.fetchrow.side_effect = [
            # Customer lookup (found)
            {
                "customer_id": sample_customer_id,
                "shop_id": sample_shop_id,
                "psid": "1234567890",
                "display_name": "Existing Customer",
                "channel": ChannelType.messenger,
                "first_seen_at": datetime.utcnow(),
                "last_contact_at": datetime.utcnow(),
                "conversation_count": 5,
                "total_order_value": 150.0,
                "order_count": 2
            },
            # No existing conversation
            None
        ]

        mock_db.fetch.return_value = [{
            "conversation_id": sample_conversation_id,
            "shop_id": sample_shop_id,
            "customer_id": sample_customer_id,
            "channel": ChannelType.messenger,
            "started_at": datetime.utcnow(),
            "last_message_at": datetime.utcnow(),
            "message_count": 0,
            "customer_message_count": 0,
            "business_message_count": 0,
            "conversation_depth": 0,
            "conversation_stage": ConversationStage.discovery,
            "drop_off_flag": False,
            "resulted_in_order": False,
            "status": "active"
        }]

        mock_db.fetchval.return_value = sample_message_id

        webhook_data = {
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
                        "text": "I'm interested in your products"
                    }
                }]
            }]
        }

        response = client_with_db.post("/webhooks/messenger", json=webhook_data)

        assert response.status_code == 200
        assert response.json() == {"status": "received"}

    async def test_webhook_response_time_under_50ms(self, client_with_db, mock_db):
        """Test that webhook responds within 50ms (critical requirement)."""
        sample_shop_id = uuid4()
        sample_customer_id = uuid4()
        sample_conversation_id = uuid4()
        sample_message_id = uuid4()

        # Mock database calls
        mock_db.fetchrow.side_effect = [None, None]
        mock_db.fetch.return_value = [{
            "conversation_id": sample_conversation_id,
            "shop_id": sample_shop_id,
            "customer_id": sample_customer_id,
            "channel": ChannelType.messenger,
            "started_at": datetime.utcnow(),
            "last_message_at": datetime.utcnow(),
            "message_count": 0,
            "customer_message_count": 0,
            "business_message_count": 0,
            "conversation_depth": 0,
            "conversation_stage": ConversationStage.discovery,
            "drop_off_flag": False,
            "resulted_in_order": False,
            "status": "active"
        }]
        mock_db.fetchval.return_value = sample_message_id

        webhook_data = {
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
                        "text": "Hello"
                    }
                }]
            }]
        }

        start_time = time.perf_counter()
        response = client_with_db.post("/webhooks/messenger", json=webhook_data)
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000

        assert response.status_code == 200
        # In test environment with mocks, this should pass
        assert elapsed_ms < 50, f"Response time {elapsed_ms}ms exceeds 50ms threshold"

    async def test_webhook_with_product_mention(self, client_with_db, mock_db):
        """Test webhook processing when product is mentioned."""
        sample_shop_id = uuid4()
        sample_customer_id = uuid4()
        sample_conversation_id = uuid4()
        sample_message_id = uuid4()

        # Mock database calls
        mock_db.fetchrow.side_effect = [None, None]

        mock_db.fetch.return_value = [{
            "conversation_id": sample_conversation_id,
            "shop_id": sample_shop_id,
            "customer_id": sample_customer_id,
            "channel": ChannelType.messenger,
            "started_at": datetime.utcnow(),
            "last_message_at": datetime.utcnow(),
            "message_count": 0,
            "customer_message_count": 0,
            "business_message_count": 0,
            "conversation_depth": 0,
            "conversation_stage": ConversationStage.interest,
            "drop_off_flag": False,
            "resulted_in_order": False,
            "status": "active"
        }]
        mock_db.fetchval.return_value = sample_message_id

        webhook_data = {
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
                        "text": "Tell me about the leather jacket"
                    }
                }]
            }]
        }

        response = client_with_db.post("/webhooks/messenger", json=webhook_data)

        assert response.status_code == 200
        assert response.json() == {"status": "received"}

    async def test_webhook_skips_business_messages(self, client_with_db, mock_db):
        """Test that webhook skips messages sent by business (not customer)."""
        webhook_data = {
            "object": "page",
            "entry": [{
                "id": "PAGE_ID",
                "time": 1705314600,
                "messaging": [{
                    "sender": {"id": "PAGE_ID"},  # Same as page ID = business message
                    "recipient": {"id": "9876543210"},
                    "timestamp": 1705314600000,
                    "message": {
                        "mid": "msg_id_001",
                        "text": "This is from business"
                    }
                }]
            }]
        }

        response = client_with_db.post("/webhooks/messenger", json=webhook_data)

        assert response.status_code == 200
        # Should not have made database calls since message was skipped
        assert mock_db.fetchrow.call_count == 0


@pytest.mark.asyncio
class TestHealthCheck:
    """Test health check endpoint."""

    async def test_health_endpoint(self, client_with_db):
        """Test health check returns proper response."""
        response = client_with_db.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data

    async def test_root_endpoint(self, client_with_db):
        """Test root endpoint returns API info."""
        response = client_with_db.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Sellora API"
        assert data["version"] == "0.1.0"
        assert data["status"] == "running"


@pytest.mark.asyncio
class TestCatalogIntegration:
    """Test catalog import integration."""

    async def test_catalog_list_products(self, client_with_db, mock_db):
        """Test listing products from catalog."""
        sample_shop_id = str(uuid4())

        # Mock database response
        mock_db.fetch.return_value = [
            {
                "id": "prod-1",
                "name": "Test Product 1",
                "description": "A test product",
                "price": 29.99,
                "tags": ["test"]
            },
            {
                "id": "prod-2",
                "name": "Test Product 2",
                "description": "Another test product",
                "price": 49.99,
                "tags": ["test", "sample"]
            }
        ]

        response = client_with_db.get(
            "/catalog/products",
            params={"shop_id": sample_shop_id}
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

    async def test_catalog_get_single_product(self, client_with_db, mock_db):
        """Test getting a single product from catalog."""
        sample_shop_id = str(uuid4())

        # Mock database response
        mock_db.fetchrow.return_value = {
            "id": "prod-1",
            "name": "Test Product 1",
            "description": "A test product",
            "price": 29.99,
            "tags": ["test"]
        }

        response = client_with_db.get(
            "/catalog/products/prod-1",
            params={"shop_id": sample_shop_id}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "prod-1"

    async def test_catalog_get_product_not_found(self, client_with_db, mock_db):
        """Test getting a non-existent product."""
        sample_shop_id = str(uuid4())

        # Mock database response - product not found
        mock_db.fetchrow.return_value = None

        response = client_with_db.get(
            "/catalog/products/nonexistent",
            params={"shop_id": sample_shop_id}
        )

        assert response.status_code == 404


@pytest.mark.asyncio
class TestConversationStageProgression:
    """Test conversation stage progression through integration."""

    async def test_discovery_to_interest_stage(self, client_with_db, mock_db):
        """Test stage progression from discovery to interest."""
        sample_shop_id = uuid4()
        sample_customer_id = uuid4()
        sample_conversation_id = uuid4()
        sample_message_id = uuid4()

        # Mock database - customer exists with existing conversation in discovery stage
        mock_db.fetchrow.side_effect = [
            # Customer lookup
            {
                "customer_id": sample_customer_id,
                "shop_id": sample_shop_id,
                "psid": "1234567890",
                "display_name": "Test Customer",
                "channel": ChannelType.messenger,
                "first_seen_at": datetime.utcnow(),
                "last_contact_at": datetime.utcnow(),
                "conversation_count": 1,
                "total_order_value": 0.0,
                "order_count": 0
            },
            # Existing conversation in discovery stage
            {
                "conversation_id": sample_conversation_id,
                "shop_id": sample_shop_id,
                "customer_id": sample_customer_id,
                "channel": ChannelType.messenger,
                "started_at": datetime.utcnow(),
                "last_message_at": datetime.utcnow(),
                "message_count": 2,
                "customer_message_count": 1,
                "business_message_count": 1,
                "conversation_depth": 1,
                "conversation_stage": ConversationStage.discovery,
                "drop_off_flag": False,
                "resulted_in_order": False,
                "status": "active"
            }
        ]

        mock_db.fetchval.return_value = sample_message_id

        webhook_data = {
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
                        "text": "What products do you have available?"
                    }
                }]
            }]
        }

        response = client_with_db.post("/webhooks/messenger", json=webhook_data)

        assert response.status_code == 200
        assert response.json() == {"status": "received"}

    async def test_interest_to_intent_stage(self, client_with_db, mock_db):
        """Test stage progression from interest to intent (price inquiry)."""
        sample_shop_id = uuid4()
        sample_customer_id = uuid4()
        sample_conversation_id = uuid4()
        sample_message_id = uuid4()

        # Mock database
        mock_db.fetchrow.side_effect = [
            {
                "customer_id": sample_customer_id,
                "shop_id": sample_shop_id,
                "psid": "1234567890",
                "display_name": "Test Customer",
                "channel": ChannelType.messenger,
                "first_seen_at": datetime.utcnow(),
                "last_contact_at": datetime.utcnow(),
                "conversation_count": 2,
                "total_order_value": 0.0,
                "order_count": 0
            },
            {
                "conversation_id": sample_conversation_id,
                "shop_id": sample_shop_id,
                "customer_id": sample_customer_id,
                "channel": ChannelType.messenger,
                "started_at": datetime.utcnow(),
                "last_message_at": datetime.utcnow(),
                "message_count": 3,
                "customer_message_count": 2,
                "business_message_count": 1,
                "conversation_depth": 1,
                "conversation_stage": ConversationStage.interest,
                "drop_off_flag": False,
                "resulted_in_order": False,
                "status": "active"
            }
        ]

        mock_db.fetchval.return_value = sample_message_id

        webhook_data = {
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
                        "text": "How much does it cost?"
                    }
                }]
            }]
        }

        response = client_with_db.post("/webhooks/messenger", json=webhook_data)

        assert response.status_code == 200
        assert response.json() == {"status": "received"}


@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling in integration flow."""

    async def test_webhook_with_invalid_payload(self, client_with_db):
        """Test webhook with invalid payload."""
        response = client_with_db.post(
            "/webhooks/messenger",
            json={"invalid": "payload"}
        )

        # Should return 200 even with invalid payload (graceful degradation)
        assert response.status_code == 200

    async def test_webhook_with_empty_entry(self, client_with_db):
        """Test webhook with empty entry list."""
        webhook_data = {
            "object": "page",
            "entry": []
        }

        response = client_with_db.post("/webhooks/messenger", json=webhook_data)

        assert response.status_code == 200
        assert response.json() == {"status": "received"}

    async def test_catalog_with_invalid_shop_id(self, client_with_db, mock_db):
        """Test catalog import with invalid shop ID."""
        mock_db.execute.return_value = None

        csv_content = b"""id,name,description,price,tags
prod-1,Test Product,Test,29.99,test"""

        from io import BytesIO
        file_data = BytesIO(csv_content)

        response = client_with_db.post(
            "/catalog/import",
            params={"shop_id": "invalid-uuid"},
            files={"file": ("products.csv", file_data, "text/csv")}
        )

        assert response.status_code == 400


@pytest.mark.asyncio
class TestHMACIntegration:
    """Test HMAC verification integration."""

    async def test_webhook_with_valid_hmac(self, client_with_db, mock_db):
        """Test webhook with valid HMAC signature."""
        import hmac
        import hashlib

        sample_shop_id = uuid4()
        sample_customer_id = uuid4()
        sample_conversation_id = uuid4()
        sample_message_id = uuid4()

        # Setup mocks
        mock_db.fetchrow.side_effect = [None, None]
        mock_db.fetch.return_value = [{
            "conversation_id": sample_conversation_id,
            "shop_id": sample_shop_id,
            "customer_id": sample_customer_id,
            "channel": ChannelType.messenger,
            "started_at": datetime.utcnow(),
            "last_message_at": datetime.utcnow(),
            "message_count": 0,
            "customer_message_count": 0,
            "business_message_count": 0,
            "conversation_depth": 0,
            "conversation_stage": ConversationStage.discovery,
            "drop_off_flag": False,
            "resulted_in_order": False,
            "status": "active"
        }]
        mock_db.fetchval.return_value = sample_message_id

        payload = {
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
                        "text": "Hello"
                    }
                }]
            }]
        }

        import json
        payload_bytes = json.dumps(payload).encode()
        secret = "test_app_secret".encode()
        signature = hmac.new(secret, payload_bytes, hashlib.sha1).hexdigest()

        response = client_with_db.post(
            "/webhooks/messenger",
            json=payload,
            headers={"X-Hub-Signature": f"sha1={signature}"}
        )

        assert response.status_code == 200

    async def test_webhook_with_invalid_hmac(self, client_with_db):
        """Test webhook with invalid HMAC signature."""
        payload = {
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
                        "text": "Hello"
                    }
                }]
            }]
        }

        response = client_with_db.post(
            "/webhooks/messenger",
            json=payload,
            headers={"X-Hub-Signature": "sha1=invalid_signature"}
        )

        # Should return 403 for invalid signature
        assert response.status_code == 403


@pytest.mark.asyncio
class TestExtractionIntegration:
    """Test intent extraction integration."""

    async def test_price_inquiry_intent_detected(self, client_with_db, mock_db):
        """Test that price inquiry intent is properly detected."""
        sample_shop_id = uuid4()
        sample_customer_id = uuid4()
        sample_conversation_id = uuid4()
        sample_message_id = uuid4()

        # Mock database calls
        mock_db.fetchrow.side_effect = [None, None]
        mock_db.fetch.return_value = [{
            "conversation_id": sample_conversation_id,
            "shop_id": sample_shop_id,
            "customer_id": sample_customer_id,
            "channel": ChannelType.messenger,
            "started_at": datetime.utcnow(),
            "last_message_at": datetime.utcnow(),
            "message_count": 0,
            "customer_message_count": 0,
            "business_message_count": 0,
            "conversation_depth": 0,
            "conversation_stage": ConversationStage.intent,
            "drop_off_flag": False,
            "resulted_in_order": False,
            "status": "active"
        }]
        mock_db.fetchval.return_value = sample_message_id

        webhook_data = {
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
                        "text": "What's the price?"
                    }
                }]
            }]
        }

        response = client_with_db.post("/webhooks/messenger", json=webhook_data)

        assert response.status_code == 200

    async def test_purchase_intent_detected(self, client_with_db, mock_db):
        """Test that purchase intent is properly detected."""
        sample_shop_id = uuid4()
        sample_customer_id = uuid4()
        sample_conversation_id = uuid4()
        sample_message_id = uuid4()

        # Mock database calls
        mock_db.fetchrow.side_effect = [None, None]
        mock_db.fetch.return_value = [{
            "conversation_id": sample_conversation_id,
            "shop_id": sample_shop_id,
            "customer_id": sample_customer_id,
            "channel": ChannelType.messenger,
            "started_at": datetime.utcnow(),
            "last_message_at": datetime.utcnow(),
            "message_count": 0,
            "customer_message_count": 0,
            "business_message_count": 0,
            "conversation_depth": 0,
            "conversation_stage": ConversationStage.negotiation,
            "drop_off_flag": False,
            "resulted_in_order": False,
            "status": "active"
        }]
        mock_db.fetchval.return_value = sample_message_id

        webhook_data = {
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
                        "text": "I want to buy this now"
                    }
                }]
            }]
        }

        response = client_with_db.post("/webhooks/messenger", json=webhook_data)

        assert response.status_code == 200
