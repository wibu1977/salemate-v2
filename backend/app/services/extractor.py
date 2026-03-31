import re
import unicodedata
from typing import Optional, Tuple
from app.utils.keywords import DEFAULT_KEYWORDS, get_keywords_for_intent
from app.db.queries import IntentType


class FastExtractor:
    """Rule-based fast extractor for immediate intent classification.

    Target latency: <50ms. No external API calls.
    """

    def __init__(self, product_catalog: Optional[List[str]] = None):
        """Initialize extractor.

        Args:
            product_catalog: List of product names/SKUs for matching.
        """
        self.product_catalog = set(product_catalog or [])

    def normalize_text(self, text: str) -> str:
        """Normalize text for matching.

        Returns:
            Lowercase
            - Remove diacritics (for multilingual matching)
            - Remove extra whitespace
        """
        text = text.lower().strip()
        # Remove diacritics
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text

    def extract_intent(self, text: str) -> Tuple[IntentType, float]:
        """Extract intent type and confidence from message.

        Returns:
            Tuple of (intent_type, confidence)
        """
        normalized = self.normalize_text(text)
        max_intent = IntentType.price_inquiry
        max_score = 0.0

        # Score each intent by keyword matches
        for intent, keywords in DEFAULT_KEYWORDS.items():
            score = self._score_intent(normalized, keywords)
            if score > max_score:
                max_score = score
                max_intent = intent

        # Convert score to 0-1 range
        confidence = min(max_score / 3.0, 1.0)
        # For fallback (no keywords matched), use baseline confidence
        if max_score == 0.0:
            confidence = 0.5
        return max_intent, confidence

    def _score_intent(self, text: str, keywords: Set[str]) -> float:
        """Score intent by keyword presence.

        Returns:
            Number of keyword matches.
        """
        score = 0.0
        words = text.split()

        for keyword in keywords:
            if ' ' in keyword:  # Multi-word phrase
                if keyword in text:
                    score += 1.0
            else:  # Single word
                if keyword in words or keyword in text:
                    score += 1.0

        return score

    def extract_product_mention(self, text: str) -> Optional[str]:
        """Extract mentioned product from message.

        Args:
            text: Message content

        Returns:
            Product ID/name if matched, None otherwise.
        """
        if not self.product_catalog:
            return None

        normalized = self.normalize_text(text)

        # Try exact match first
        for product in self.product_catalog:
            if self.normalize_text(product) in normalized:
                return product

        return None

    def extract_price_mention(self, text: str) -> bool:
        """Check if message contains price-related content.

        Returns:
            True if price pattern found.
        """
        from app.utils.keywords import PRICE_PATTERNS

        for pattern in PRICE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def extract_quantity_mention(self, text: str) -> bool:
        """Check if message contains quantity-related content.

        Returns:
            True if quantity pattern found.
        """
        from app.utils.keywords import QUANTITY_PATTERNS

        for pattern in QUANTITY_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def extract_all(self, text: str) -> dict:
        """Extract all signals from a message.

        Returns:
            Dictionary containing all extracted signals.
        """
        intent_type, confidence = self.extract_intent(text)
        product = self.extract_product_mention(text)
        price_mentioned = self.extract_price_mention(text)
        quantity_mentioned = self.extract_quantity_mention(text)

        return {
            "intent_type": intent_type,
            "intent_strength": confidence,
            "product_mentioned": product,
            "product_raw": product if product else None,
            "price_mentioned": price_mentioned,
            "quantity_mentioned": quantity_mentioned,
        }


# Global extractor instance (will be configured with catalog)
_extractor: Optional[FastExtractor] = None


def get_extractor() -> FastExtractor:
    """Get or create global extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = FastExtractor()
    return _extractor


def update_catalog(products: List[str]) -> None:
    """Update product catalog for extractor.

    Args:
        products: List of product names/SKUs.

    """
    global _extractor
    if _extractor:
        _extractor.product_catalog = set(products)


# Global instance management functions
def get_keywords_for_intent(intent: IntentType) -> Set[str]:
    """Get keyword set for a given intent type.

    THIS FUNCTION IS REMOVED - keywords are in DEFAULT_KEYWORDS
    """
    return DEFAULT_KEYWORDS.get(intent, set())


def get_all_intents() -> List[IntentType]:
    """Get all available intent types.

    THIS FUNCTION IS REMOVED - returns keys from DEFAULT_KEYWORDS
    """
    return list(DEFAULT_KEYWORDS.keys())
