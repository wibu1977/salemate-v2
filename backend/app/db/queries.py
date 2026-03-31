from enum import Enum

# Enums matching database schema
class ChannelType(str, Enum):
    messenger = "messenger"
    whatsapp = "whatsapp"


class SenderType(str, Enum):
    customer = "customer"
    business = "business"


class ContentType(str, Enum):
    text = "text"
    image = "image"
    audio = "audio"
    video = "video"
    file = "file"
    template = "template"


class ConversationStage(str, Enum):
    discovery = "discovery"
    interest = "interest"
    intent = "intent"
    negotiation = "negotiation"
    converted = "converted"
    dormant = "dormant"


class IntentType(str, Enum):
    price_inquiry = "price_inquiry"
    product_inquiry = "product_inquiry"
    availability_inquiry = "availability_inquiry"
    purchase_intent = "purchase_intent"
    complaint = "complaint"
    general_chat = "general_chat"
    unknown = "unknown"
