from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID
from datetime import datetime
from typing import Optional, List
from app.db.queries import ChannelType, SenderType, ContentType, ConversationStage, IntentType


# Request/Response Schemas
class MessageCreate(BaseModel):
    """Schema for creating a new message."""
    shop_id: UUID
    conversation_id: UUID
    customer_id: UUID
    sender_type: SenderType
    channel: ChannelType
    content: str
    content_type: ContentType = ContentType.text
    sent_at: datetime
    platform_msg_id: Optional[str] = None


class Message(BaseModel):
    """Message response schema."""
    model_config = ConfigDict(from_attributes=True)

    message_id: UUID
    shop_id: UUID
    conversation_id: UUID
    customer_id: UUID
    sender_type: SenderType
    channel: ChannelType
    content: str
    content_type: ContentType
    sent_at: datetime
    day_of_week: int
    hour_of_day: int
    platform_msg_id: Optional[str] = None
    read_at: Optional[datetime] = None


class ExtractedSignalCreate(BaseModel):
    """Schema for creating extracted signals."""
    shop_id: UUID
    message_id: UUID
    conversation_id: UUID
    customer_id: UUID
    intent_type: IntentType
    intent_strength: float = Field(ge=0.0, le=1.0)
    product_mentioned: Optional[str] = None
    product_raw: Optional[str] = None
    variant_mentioned: Optional[str] = None
    price_mentioned: bool = False
    quantity_mentioned: bool = False
    extraction_method: str = "rule_based"


class ExtractedSignal(BaseModel):
    """Extracted signal response schema."""
    model_config = ConfigDict(from_attributes=True)

    signal_id: UUID
    shop_id: UUID
    message_id: UUID
    conversation_id: UUID
    customer_id: UUID
    extracted_at: datetime
    intent_type: IntentType
    intent_type_refined: Optional[IntentType] = None
    intent_strength: float
    product_mentioned: Optional[str] = None
    product_raw: Optional[str] = None
    extraction_method: str


class ConversationCreate(BaseModel):
    """Schema for creating a conversation."""
    shop_id: UUID
    customer_id: UUID
    channel: ChannelType
    started_at: datetime


class ConversationUpdate(BaseModel):
    """Schema for updating a conversation."""
    last_message_at: Optional[datetime] = None
    message_count: Optional[int] = None
    customer_message_count: Optional[int] = None
    business_message_count: Optional[int] = None
    conversation_stage: Optional[ConversationStage] = None
    status: Optional[str] = None
    products_mentioned: Optional[List[str]] = None


class Conversation(BaseModel):
    """Conversation response schema."""
    model_config = ConfigDict(from_attributes=True)

    conversation_id: UUID
    shop_id: UUID
    customer_id: UUID
    channel: ChannelType
    started_at: datetime
    last_message_at: datetime
    message_count: int
    customer_message_count: int
    business_message_count: int
    conversation_depth: int
    conversation_stage: ConversationStage
    intent_score: Optional[float] = None
    drop_off_flag: bool = False
    resulted_in_order: bool = False
    status: str


class CustomerCreate(BaseModel):
    """Schema for creating a customer."""
    shop_id: UUID
    psid: str
    display_name: Optional[str] = None
    locale: Optional[str] = None
    channel: ChannelType
    first_seen_at: datetime


class Customer(BaseModel):
    """Customer response schema."""
    model_config = ConfigDict(from_attributes=True)

    customer_id: UUID
    shop_id: UUID
    psid: str
    display_name: Optional[str] = None
    channel: ChannelType
    first_seen_at: datetime
    last_contact_at: Optional[datetime] = None
    conversation_count: int = 0
    total_order_value: float = 0.0
    order_count: int = 0
    intent_score_latest: Optional[float] = None
    churn_risk_score: Optional[float] = None
    priority_score: Optional[float] = None


# Webhook Schemas
class MessengerWebhookEntry(BaseModel):
    """Single entry from Messenger webhook."""
    id: str
    time: int
    messaging: List[dict]


class MessengerWebhookPayload(BaseModel):
    """Full Messenger webhook payload."""
    object: str
    entry: List[MessengerWebhookEntry]


class MessengerMessage(BaseModel):
    """Message content from Messenger."""
    mid: str
    text: Optional[str] = None


class MessengerSender(BaseModel):
    """Sender info from Messenger."""
    id: str


# AI Agent Schemas
class AIRequest(BaseModel):
    """Request to AI agent."""
    message: str
    conversation_history: List[dict]
    customer_profile: dict
    retrieved_context: List[dict]
    system_prompt: str


class AIResponse(BaseModel):
    """Response from AI agent."""
    reply: str
    confidence: float
    suggested_products: Optional[List[str]] = None
