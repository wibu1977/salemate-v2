"""Tests for conftest.py shared fixtures.

Tests that the shared fixtures in conftest.py work correctly and can be
imported by other test modules.

Note: Tests that use test_client fixture will have env vars set by conftest.
"""

import pytest
from uuid import UUID
from datetime import datetime


class TestSampleMessageDataFixture:
    """Test the sample_message_data fixture."""

    def test_sample_message_data_structure(self, sample_message_data):
        """Test that sample_message_data has correct structure."""
        assert "message_id" in sample_message_data
        assert "conversation_id" in sample_message_data
        assert "shop_id" in sample_message_data
        assert "sender_id" in sample_message_data
        assert "sender_type" in sample_message_data
        assert "channel" in sample_message_data
        assert "content_type" in sample_message_data
        assert "content" in sample_message_data
        assert "sent_at" in sample_message_data

    def test_sample_message_data_values(self, sample_message_data):
        """Test that sample_message_data has valid values."""
        assert sample_message_data["message_id"] == "test_msg_001"
        assert sample_message_data["conversation_id"] == "test_conv_001"
        assert sample_message_data["sender_id"] == "customer_001"
        assert sample_message_data["sender_type"] == "customer"
        assert sample_message_data["channel"] == "messenger"
        assert sample_message_data["content_type"] == "text"
        assert sample_message_data["content"]["text"] == "Hello, I'm interested in your products"
        assert sample_message_data["sent_at"] == "2024-01-15T10:30:00Z"

    def test_sample_message_data_shop_id_is_uuid(self, sample_message_data):
        """Test that shop_id is a valid UUID string."""
        # Should be a valid UUID
        UUID(sample_message_data["shop_id"])


class TestSampleMessengerWebhookFixture:
    """Test the sample_messenger_webhook fixture."""

    def test_sample_messenger_webhook_structure(self, sample_messenger_webhook):
        """Test that sample_messenger_webhook has correct structure."""
        assert "object" in sample_messenger_webhook
        assert "entry" in sample_messenger_webhook
        assert isinstance(sample_messenger_webhook["entry"], list)
        assert len(sample_messenger_webhook["entry"]) > 0

    def test_sample_messenger_webhook_values(self, sample_messenger_webhook):
        """Test that sample_messenger_webhook has valid values."""
        assert sample_messenger_webhook["object"] == "page"
        entry = sample_messenger_webhook["entry"][0]
        assert "id" in entry
        assert "time" in entry
        assert "messaging" in entry
        assert isinstance(entry["messaging"], list)
        assert len(entry["messaging"]) > 0

    def test_sample_messenger_webhook_message_structure(self, sample_messenger_webhook):
        """Test that the message within the webhook has correct structure."""
        messaging = sample_messenger_webhook["entry"][0]["messaging"][0]
        assert "sender" in messaging
        assert "recipient" in messaging
        assert "timestamp" in messaging
        assert "message" in messaging
        assert "mid" in messaging["message"]
        assert "text" in messaging["message"]


class TestSampleProductDataFixture:
    """Test the sample_product_data fixture."""

    def test_sample_product_data_structure(self, sample_product_data):
        """Test that sample_product_data has correct structure."""
        assert "name" in sample_product_data
        assert "description" in sample_product_data
        assert "price" in sample_product_data
        assert "currency" in sample_product_data
        assert "variants" in sample_product_data
        assert "tags" in sample_product_data

    def test_sample_product_data_values(self, sample_product_data):
        """Test that sample_product_data has valid values."""
        assert sample_product_data["name"] == "Test Product"
        assert sample_product_data["description"] == "A test product for testing purposes"
        assert sample_product_data["price"] == 29.99
        assert sample_product_data["currency"] == "USD"
        assert isinstance(sample_product_data["variants"], list)
        assert isinstance(sample_product_data["tags"], list)

    def test_sample_product_data_variants(self, sample_product_data):
        """Test that sample_product_data has correct variants."""
        assert "Small" in sample_product_data["variants"]
        assert "Medium" in sample_product_data["variants"]
        assert "Large" in sample_product_data["variants"]

    def test_sample_product_data_tags(self, sample_product_data):
        """Test that sample_product_data has correct tags."""
        assert "test" in sample_product_data["tags"]
        assert "sample" in sample_product_data["tags"]


class TestTestClientFixture:
    """Test the test_client fixture."""

    def test_test_client_exists(self, test_client):
        """Test that test_client fixture exists and is callable."""
        assert test_client is not None
        assert hasattr(test_client, "get")
        assert hasattr(test_client, "post")
        assert hasattr(test_client, "put")
        assert hasattr(test_client, "delete")

    def test_test_client_can_make_requests(self, test_client):
        """Test that test_client can make requests to the app."""
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestDbPoolFixture:
    """Test the db_pool fixture."""

    @pytest.mark.asyncio
    async def test_db_pool_exists(self, db_pool):
        """Test that db_pool fixture exists."""
        assert db_pool is not None
        # Note: db_pool is a DatabasePool instance
        # In test environment, it should be initialized but may not connect to a real DB
