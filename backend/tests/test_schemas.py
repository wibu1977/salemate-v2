from uuid import uuid4
from datetime import datetime
from app.models.schemas import (
    MessageCreate,
    Message,
    ExtractedSignalCreate,
    ConversationCreate,
    CustomerCreate,
    MessengerWebhookPayload,
)


def test_message_create_valid():
    """Test MessageCreate validation."""
    msg = MessageCreate(
        shop_id=uuid4(),
        conversation_id=uuid4(),
        customer_id=uuid4(),
        sender_type="customer",
        channel="messenger",
        content="hello",
        sent_at=datetime.now(),
    )
    assert msg.content_type == "text"


def test_extracted_signal_strength_bounds():
    """Test that intent_strength is bounded 0-1."""
    from pydantic import ValidationError
    try:
        ExtractedSignalCreate(
            shop_id=uuid4(),
            message_id=uuid4(),
            conversation_id=uuid4(),
            customer_id=uuid4(),
            intent_type="price_inquiry",
            intent_strength=1.5,  # Invalid
        )
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass


def test_conversation_stage_enum():
    """Test that ConversationStage has expected values."""
    from app.db.queries import ConversationStage
    assert ConversationStage.discovery == "discovery"
    assert ConversationStage.intent == "intent"
