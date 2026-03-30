import pytest
from app.services.extractor import FastExtractor
from app.db.queries import IntentType


def test_extract_intent_price_inquiry():
    """Test price intent extraction."""
    extractor = FastExtractor()

    intent, confidence = extractor.extract_intent("How much is this jacket?")
    assert intent == IntentType.price_inquiry
    assert confidence > 0.5


def test_extract_intent_purchase_intent():
    """Test purchase intent extraction."""
    extractor = FastExtractor()

    intent, confidence = extractor.extract_intent("I want to buy 2 of these")
    assert intent == IntentType.purchase_intent
    assert confidence > 0.5


def test_extract_intent_multilingual():
    """Test multilingual intent extraction (Vietnamese, Thai)."""
    extractor = FastExtractor()

    # Vietnamese
    intent, _ = extractor.extract_intent("Áo này giá bao nhiêu?")
    assert intent == IntentType.price_inquiry  # price_inquiry is correct fallback
    assert _ > 0.3

    # Thai
    intent, _ = extractor.extract_intent("ราคาเท่าไหร่")
    assert intent == IntentType.price_inquiry
    assert _ > 0.3


def test_extract_product_mention():
    """Test product mention extraction."""
    extractor = FastExtractor(["Wool Jacket", "Cotton Shirt"])

    assert extractor.extract_product_mention("I'm looking for a wool jacket") == "Wool Jacket"
    assert extractor.extract_product_mention("Show me the cotton shirt") == "Cotton Shirt"
    assert extractor.extract_product_mention("Just browsing") is None


def test_extract_price_mention():
    """Test price mention extraction."""
    extractor = FastExtractor()

    assert extractor.extract_price_mention("$49.99") is True
    assert extractor.extract_price_mention("100k vnd") is True
    assert extractor.extract_price_mention("€20") is True
    assert extractor.extract_price_mention("1.5 million") is True
    assert extractor.extract_price_mention("no price here") is False


def test_extract_quantity_mention():
    """Test quantity mention extraction."""
    extractor = FastExtractor()

    assert extractor.extract_quantity_mention("I want 2 pieces") is True
    assert extractor.extract_quantity_mention("Order 3 items") is True
    assert extractor.extract_quantity_mention("Just one") is False


def test_normalize_text():
    """Test text normalization."""
    from app.services.extractor import FastExtractor

    extractor = FastExtractor()

    assert extractor.normalize_text("ÁO Khoác  This is a TEST") == "ao khoac this is a test"
    assert extractor.normalize_text("  hello  ") == "hello"
    assert extractor.normalize_text("CAFFE") == "caffe"


def test_extract_all():
    """Test complete extraction."""
    from app.services.extractor import FastExtractor

    extractor = FastExtractor(["Wool Jacket"])

    result = extractor.extract_all("I want to buy 2 wool jackets for $100 each")

    assert result["intent_type"] == IntentType.purchase_intent
    assert result["intent_strength"] > 0.5
    assert result["product_mentioned"] == "Wool Jacket"
    assert result["price_mentioned"] is True
    assert result["quantity_mentioned"] is True
    assert "intent_strength" in result
    assert "product_mentioned" in result
    assert "product_raw" in result
    assert "price_mentioned" in result
    assert "quantity_mentioned" in result


def test_get_extractor_cached():
    """Test that get_extractor returns cached instance."""
    from app.services.extractor import get_extractor

    e1 = get_extractor()
    e2 = get_extractor()
    assert e1 is e2  # Singleton pattern
