from typing import Dict, List, Set
from app.db.queries import IntentType

# Default keyword dictionaries (English only - NO Vietnamese or Thai as per spec)
DEFAULT_KEYWORDS: Dict[IntentType, Set[str]] = {
    IntentType.price_inquiry: {
        "price", "cost", "how", "much", "how much", "how many", "expensive", "cheap",
        "discount", "sale", "promo", "offer", "deal", "amount", "rate", "charge", "pay",
    },
    IntentType.product_inquiry: {
        "tell me about", "what is", "describe", "specs", "details", "features",
        "information", "show", "display", "look at", "color", "size", "variant",
        "material", "style", "design", "brand", "model",
    },
    IntentType.availability_inquiry: {
        "in stock", "available", "have", "stock", "inventory", "ready", "still",
        "can i get", "do you have", "out of stock", "sold out",
    },
    IntentType.purchase_intent: {
        "want to buy", "order", "purchase", "i'll take", "buy", "get",
        "checkout", "pay", "transaction", "cho myself", "take one",
    },
    IntentType.complaint: {
        "wrong", "broken", "damaged", "late", "refund", "complaint",
        "problem", "issue", "error", "mistake", "disappointed", "unsatisfied",
    },
}


# Price patterns for regex (exactly 5 patterns as per spec)
PRICE_PATTERNS: List[str] = [
    r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # $1,000.00
    r'\€\d+(?:,\d{3})*(?:\.\d{2})?',  # €20
    r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:usd|eur|gbp|vnd|thb)',  # 100 usd - simple
    r'\d+k\s*(?:vnd)?',  # 100k vnd
    r'\d+\.\d+\s*(?:million|k|b)',  # 1.5 million
]


# Quantity patterns for regex (exactly 4 patterns as per spec)
QUANTITY_PATTERNS: List[str] = [
    r'\b\d+\s*(?:pieces?|items?|units?|qty|quantity)\b',  # numeric + quantity words
    r'\b(?:order|buy|get|take)\s+\d+\b',  # order/buy/get/take + number
    r'\b\d+\s*(?:cái|chiếc|bộ|thùng)\b',  # Vietnamese
    r'\b\d+\s*(?:ชิ้น|ชุด|กล่อง)\b',  # Thai
]


def get_keywords_for_intent(intent: IntentType) -> Set[str]:
    """Get keyword set for a given intent type."""
    return DEFAULT_KEYWORDS.get(intent, set())


def get_all_intents() -> List[IntentType]:
    """Get all available intent types."""
    return list(DEFAULT_KEYWORDS.keys())
