import pytest
from datetime import datetime, timedelta
from uuid import uuid4, UUID
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from app.services.conversation import ConversationTracker
from app.db.queries import IntentType, ConversationStage, ChannelType, SenderType


@pytest.fixture
def mock_db():
    """Mock database connection."""
    return AsyncMock()


@pytest.fixture
def tracker(mock_db):
    """Create ConversationTracker instance with mock DB."""
    return ConversationTracker(mock_db)


@pytest.fixture
def sample_shop_id():
    """Sample shop ID."""
    return uuid4()


@pytest.fixture
def sample_customer_id():
    """Sample customer ID."""
    return uuid4()


@pytest.fixture
def sample_conversation_id():
    """Sample conversation ID."""
    return uuid4()


@pytest.fixture
def sample_message_id():
    """Sample message ID."""
    return uuid4()


class TestCreateCustomer:
    """Test customer creation helper."""

    async def test_create_customer(self, tracker, mock_db, sample_shop_id):
        """Test creating a new customer."""
        psid = "123456789"
        display_name = "Test Customer"
        locale = "en_US"
        channel = ChannelType.messenger

        new_customer_id = uuid4()

        # Mock DB response - fetchrow is used with RETURNING
        mock_db.fetchrow.return_value = {
            "customer_id": new_customer_id,
            "shop_id": sample_shop_id,
            "psid": psid,
            "display_name": display_name,
            "channel": channel,
            "first_seen_at": datetime.utcnow(),
            "last_contact_at": None,
            "conversation_count": 0,
            "total_order_value": 0.0,
            "order_count": 0
        }

        result = await tracker.create_customer(
            shop_id=sample_shop_id,
            psid=psid,
            display_name=display_name,
            locale=locale,
            channel=channel
        )

        # Verify INSERT was called via fetchrow (with RETURNING)
        mock_db.fetchrow.assert_called_once()
        query = mock_db.fetchrow.call_args[0][0]
        assert "INSERT INTO customers" in query

        # Verify result contains customer_id
        assert result is not None
        assert result["customer_id"] == new_customer_id


class TestGetCustomerByPsid:
    """Test customer lookup by PSID."""

    async def test_get_customer_by_psid(self, tracker, mock_db, sample_shop_id, sample_customer_id):
        """Test retrieving customer by PSID."""
        psid = "123456789"

        # Reset mock call count
        mock_db.fetchrow.reset_mock()

        # Mock DB response
        mock_db.fetchrow.return_value = {
            "customer_id": sample_customer_id,
            "shop_id": sample_shop_id,
            "psid": psid,
            "display_name": "Test Customer",
            "channel": ChannelType.messenger,
            "first_seen_at": datetime.utcnow(),
            "last_contact_at": None,
            "conversation_count": 5,
            "total_order_value": 150.0,
            "order_count": 2
        }

        result = await tracker.get_customer_by_psid(sample_shop_id, psid)

        # Verify SELECT query was called
        mock_db.fetchrow.assert_called_once()
        query = mock_db.fetchrow.call_args[0][0]
        assert "SELECT" in query
        assert "customers" in query
        assert "psid = $2" in query.lower() or "psid=$2" in query.lower()

        # Verify result
        assert result is not None
        assert result["customer_id"] == sample_customer_id
        assert result["psid"] == psid


class TestGetOrCreateConversation:
    """Test conversation retrieval and creation."""

    async def test_get_or_create_new_conversation(self, tracker, mock_db, sample_shop_id, sample_customer_id):
        """Test creating a new conversation when none exists within idle window."""
        channel = ChannelType.messenger
        psid = "123456789"

        # First call: No recent conversation found (returns None)
        mock_db.fetchrow.return_value = None

        # Mock the INSERT for new conversation
        new_conversation_id = uuid4()
        mock_db.fetch.return_value = [{
            "conversation_id": new_conversation_id,
            "shop_id": sample_shop_id,
            "customer_id": sample_customer_id,
            "channel": channel,
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

        result = await tracker.get_or_create_conversation(
            shop_id=sample_shop_id,
            customer_id=sample_customer_id,
            channel=channel,
            psid=psid
        )

        # Verify query for existing conversation was made
        assert mock_db.fetchrow.call_count >= 1

        # Verify result contains conversation_id
        assert result is not None
        assert "conversation_id" in result

    async def test_get_or_create_existing_conversation(self, tracker, mock_db, sample_shop_id, sample_customer_id):
        """Test retrieving existing conversation within idle window."""
        channel = ChannelType.messenger
        psid = "123456789"
        existing_conversation_id = uuid4()
        now = datetime.utcnow()

        # Existing conversation within idle window
        mock_db.fetchrow.return_value = {
            "conversation_id": existing_conversation_id,
            "shop_id": sample_shop_id,
            "customer_id": sample_customer_id,
            "channel": channel,
            "started_at": now - timedelta(hours=4),
            "last_message_at": now - timedelta(hours=2),
            "message_count": 3,
            "customer_message_count": 2,
            "business_message_count": 1,
            "conversation_depth": 1,
            "conversation_stage": ConversationStage.interest,
            "drop_off_flag": False,
            "resulted_in_order": False,
            "status": "active"
        }

        result = await tracker.get_or_create_conversation(
            shop_id=sample_shop_id,
            customer_id=sample_customer_id,
            channel=channel,
            psid=psid
        )

        # Verify SELECT query was called (not INSERT)
        mock_db.fetchrow.assert_called_once()
        mock_db.execute.assert_not_called()

        # Verify result contains existing conversation
        assert result is not None
        assert result["conversation_id"] == existing_conversation_id


class TestRecordMessage:
    """Test message recording."""

    async def test_record_message_customer(self, tracker, mock_db, sample_shop_id, sample_customer_id, sample_conversation_id):
        """Test recording a customer message."""
        sender_type = SenderType.customer
        channel = ChannelType.messenger
        content = "How much is this product?"
        sent_at = datetime.utcnow()

        # Mock INSERT for message
        new_message_id = uuid4()
        mock_db.fetchval.return_value = new_message_id

        result = await tracker.record_message(
            shop_id=sample_shop_id,
            conversation_id=sample_conversation_id,
            customer_id=sample_customer_id,
            sender_type=sender_type,
            channel=channel,
            content=content,
            sent_at=sent_at
        )

        # Verify INSERT for message was called
        mock_db.fetchval.assert_called_once()
        query = mock_db.fetchval.call_args[0][0]
        assert "INSERT INTO messages" in query

        # Verify UPDATE for conversation was called
        assert mock_db.execute.call_count >= 1

        # Verify result contains message_id
        assert result is not None
        assert result["message_id"] == new_message_id

    async def test_record_message_business(self, tracker, mock_db, sample_shop_id, sample_customer_id, sample_conversation_id):
        """Test recording a business message."""
        sender_type = SenderType.business
        channel = ChannelType.messenger
        content = "That product is $49.99"
        sent_at = datetime.utcnow()

        # Mock INSERT for message
        new_message_id = uuid4()
        mock_db.fetchval.return_value = new_message_id

        result = await tracker.record_message(
            shop_id=sample_shop_id,
            conversation_id=sample_conversation_id,
            customer_id=sample_customer_id,
            sender_type=sender_type,
            channel=channel,
            content=content,
            sent_at=sent_at
        )

        # Verify result contains message_id
        assert result is not None
        assert result["message_id"] == new_message_id


class TestUpdateStage:
    """Test conversation stage progression."""

    async def test_update_stage_intent(self, tracker, mock_db, sample_conversation_id):
        """Test stage progression for price/purchase intents."""
        intent_type = IntentType.purchase_intent
        current_stage = ConversationStage.discovery

        # Mock current conversation state
        mock_db.fetchrow.return_value = {
            "conversation_id": sample_conversation_id,
            "conversation_stage": current_stage
        }

        await tracker.update_stage(sample_conversation_id, intent_type)

        # Verify UPDATE was called
        mock_db.execute.assert_called_once()
        query = mock_db.execute.call_args[0][0]
        assert "UPDATE conversations" in query
        assert "conversation_stage" in query

    async def test_update_stage_interest(self, tracker, mock_db, sample_conversation_id):
        """Test stage progression for product/availability intents."""
        intent_type = IntentType.product_inquiry
        current_stage = ConversationStage.discovery

        # Mock current conversation state
        mock_db.fetchrow.return_value = {
            "conversation_id": sample_conversation_id,
            "conversation_stage": current_stage
        }

        await tracker.update_stage(sample_conversation_id, intent_type)

        # Verify UPDATE was called
        mock_db.execute.assert_called_once()


class TestAddProductMention:
    """Test product mention tracking."""

    async def test_add_product_mention(self, tracker, mock_db, sample_conversation_id):
        """Test adding a product to conversation's mentioned products."""
        product_id = "PROD-123"

        # Mock current products_mentioned
        mock_db.fetchval.return_value = ["PROD-456"]

        await tracker.add_product_mention(sample_conversation_id, product_id)

        # Verify UPDATE was called
        mock_db.execute.assert_called_once()
        query = mock_db.execute.call_args[0][0]
        assert "UPDATE conversations" in query
        assert "products_mentioned" in query


class TestGetConversation:
    """Test conversation retrieval."""

    async def test_get_conversation(self, tracker, mock_db, sample_conversation_id):
        """Test retrieving full conversation details."""
        sample_shop_id = uuid4()
        sample_customer_id = uuid4()

        mock_db.fetchrow.return_value = {
            "conversation_id": sample_conversation_id,
            "shop_id": sample_shop_id,
            "customer_id": sample_customer_id,
            "channel": ChannelType.messenger,
            "started_at": datetime.utcnow() - timedelta(hours=5),
            "last_message_at": datetime.utcnow() - timedelta(minutes=30),
            "message_count": 5,
            "customer_message_count": 3,
            "business_message_count": 2,
            "conversation_depth": 2,
            "conversation_stage": ConversationStage.interest,
            "intent_score": 0.75,
            "drop_off_flag": False,
            "resulted_in_order": False,
            "status": "active"
        }

        result = await tracker.get_conversation(sample_conversation_id)

        # Verify SELECT query
        mock_db.fetchrow.assert_called_once()
        query = mock_db.fetchrow.call_args[0][0]
        assert "SELECT" in query
        assert "conversations" in query

        # Verify result
        assert result is not None
        assert result["conversation_id"] == sample_conversation_id
        assert result["conversation_stage"] == ConversationStage.interest


class TestGetRecentMessages:
    """Test recent message retrieval."""

    async def test_get_recent_messages(self, tracker, mock_db, sample_conversation_id):
        """Test retrieving recent messages for context window."""
        sample_shop_id = uuid4()
        sample_customer_id = uuid4()

        mock_db.fetch.return_value = [
            {
                "message_id": uuid4(),
                "shop_id": sample_shop_id,
                "conversation_id": sample_conversation_id,
                "customer_id": sample_customer_id,
                "sender_type": SenderType.customer,
                "channel": ChannelType.messenger,
                "content": "Hello",
                "sent_at": datetime.utcnow() - timedelta(minutes=10),
            },
            {
                "message_id": uuid4(),
                "shop_id": sample_shop_id,
                "conversation_id": sample_conversation_id,
                "customer_id": sample_customer_id,
                "sender_type": SenderType.business,
                "channel": ChannelType.messenger,
                "content": "Hi there!",
                "sent_at": datetime.utcnow() - timedelta(minutes=5),
            }
        ]

        result = await tracker.get_recent_messages(sample_conversation_id, limit=10)

        # Verify SELECT query
        mock_db.fetch.assert_called_once()
        query = mock_db.fetch.call_args[0][0]
        assert "SELECT" in query
        assert "messages" in query
        assert "ORDER BY sent_at DESC" in query

        # Verify result
        assert result is not None
        assert len(result) == 2
        assert result[0]["sender_type"] == SenderType.customer


class TestCheckDropOff:
    """Test drop-off detection."""

    async def test_check_drop_off(self, tracker, mock_db):
        """Test drop-off detection for conversations."""
        sample_shop_id = uuid4()

        # Reset mock call count
        mock_db.fetch.reset_mock()
        mock_db.execute.reset_mock()

        mock_db.fetch.return_value = [
            {
                "conversation_id": uuid4(),
                "shop_id": sample_shop_id,
                "customer_id": uuid4(),
                "conversation_stage": ConversationStage.negotiation,
                "last_message_at": datetime.utcnow() - timedelta(hours=9)
            },
            {
                "conversation_id": uuid4(),
                "shop_id": sample_shop_id,
                "customer_id": uuid4(),
                "conversation_stage": ConversationStage.interest,
                "last_message_at": datetime.utcnow() - timedelta(hours=10)
            }
        ]

        result = await tracker.check_drop_off(sample_shop_id)

        # Verify SELECT query
        mock_db.fetch.assert_called_once()
        query = mock_db.fetch.call_args[0][0]
        assert "SELECT" in query
        assert "conversations" in query

        # Verify UPDATE for drop_off_flag was called (one per dropped conversation)
        # The check_drop_off method calls _mark_dropped_off for each dropped conversation
        assert mock_db.execute.call_count == 2

        # Verify result
        assert result is not None
        assert len(result) == 2


class TestMarkConverted:
    """Test conversation conversion marking."""

    async def test_mark_converted(self, tracker, mock_db, sample_conversation_id):
        """Test marking a conversation as converted."""
        await tracker.mark_converted(sample_conversation_id)

        # Verify UPDATE was called
        mock_db.execute.assert_called_once()
        query = mock_db.execute.call_args[0][0]
        assert "UPDATE conversations" in query
        assert "resulted_in_order" in query
        assert "conversation_stage" in query

        # Check that resulted_in_order is set to True
        update_call = mock_db.execute.call_args[0][0]
        assert "true" in update_call.lower()

        # Check that stage is set to converted
        assert "'converted'" in update_call.lower() or '"converted"' in update_call.lower()
