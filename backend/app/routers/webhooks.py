"""Webhook endpoints for Messenger and WhatsApp.

This module provides critical webhook receivers that must respond within 50ms while:
- Verifying HMAC signatures for security
- Writing all necessary database records
- Extracting signals with FastExtractor
- Tracking conversations with ConversationTracker

Phase 1: Messenger is fully implemented, WhatsApp has placeholder endpoints.
"""

import hmac
import hashlib
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4

from fastapi import APIRouter, Request, HTTPException, status, Depends, Query
from fastapi.responses import Response, PlainTextResponse
from pydantic import BaseModel

from app.config import get_settings
from app.db.connection import get_db
from app.db.queries import IntentType, ConversationStage, ChannelType, SenderType
from app.services.conversation import ConversationTracker
from app.services.extractor import FastExtractor, get_extractor


logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()

# Global extractor instance (initialized with empty catalog for Phase 1)
extractor = get_extractor()

# Default shop ID for Phase 1 (in production, this comes from page mapping)
DEFAULT_SHOP_ID = uuid4()

# For testing: allow settings override
_settings_override = None


def get_current_settings():
    """Get current settings, allowing for test override."""
    global _settings_override
    return _settings_override if _settings_override else settings


def override_settings(settings_obj):
    """Override settings for testing."""
    global _settings_override
    _settings_override = settings_obj


def reset_settings_override():
    """Reset settings override."""
    global _settings_override
    _settings_override = None


# ============================================================================
# HMAC Verification (Security)
# ============================================================================

def verify_hmac_signature(
    x_hub_signature: Optional[str],
    payload: str,
    app_secret: str
) -> bool:
    """Verify HMAC signature from Facebook/WhatsApp.

    Args:
        x_hub_signature: X-Hub-Signature header value (e.g., "sha1=...")
        payload: Raw request body as string
        app_secret: App secret from settings

    Returns:
        True if signature is valid, False otherwise
    """
    if not x_hub_signature:
        return False

    # Parse signature
    try:
        algorithm, signature = x_hub_signature.split('=', 1)
    except ValueError:
        return False

    # Generate expected signature
    expected_signature = hmac.new(
        app_secret.encode(),
        payload.encode(),
        getattr(hashlib, algorithm)
    ).hexdigest()

    # Compare signatures using constant-time comparison
    return hmac.compare_digest(expected_signature, signature)


# ============================================================================
# Helper Functions
# ============================================================================

async def get_customer_by_psid(
    db,
    shop_id: UUID,
    psid: str
) -> Optional[Dict[str, Any]]:
    """Get customer by platform sender ID.

    Args:
        db: Database connection
        shop_id: Shop ID
        psid: Platform Sender ID

    Returns:
        Customer dict if found, None otherwise
    """
    query = """
        SELECT
            customer_id, shop_id, psid, display_name, channel,
            first_seen_at, last_contact_at, conversation_count,
            total_order_value, order_count
        FROM customers
        WHERE shop_id = $1 AND psid = $2
    """
    return await db.fetchrow(query, shop_id, psid)


async def create_customer(
    db,
    shop_id: UUID,
    psid: str,
    display_name: Optional[str] = None,
    channel: ChannelType = ChannelType.messenger
) -> Dict[str, Any]:
    """Create a new customer.

    Args:
        db: Database connection
        shop_id: Shop ID
        psid: Platform Sender ID
        display_name: Customer display name
        channel: Communication channel

    Returns:
        Created customer dict
    """
    now = datetime.utcnow()

    query = """
        INSERT INTO customers (
            shop_id, psid, display_name, channel,
            first_seen_at, conversation_count, total_order_value, order_count
        ) VALUES ($1, $2, $3, $4, $5, 0, 0.0, 0)
        RETURNING
            customer_id, shop_id, psid, display_name, channel,
            first_seen_at, last_contact_at, conversation_count,
            total_order_value, order_count
    """
    return await db.fetchrow(query, shop_id, psid, display_name, channel, now)


async def get_or_create_conversation(
    db,
    shop_id: UUID,
    customer_id: UUID,
    channel: ChannelType,
    psid: str
) -> Dict[str, Any]:
    """Get existing conversation or create new one.

    Uses 8-hour idle window to determine if a new conversation
    should be started. If no message in the last 8 hours, creates new.

    Args:
        db: Database connection
        shop_id: Shop ID
        customer_id: Customer ID
        channel: Communication channel
        psid: Platform Sender ID

    Returns:
        Conversation dict with conversation_id and metadata
    """
    from datetime import timedelta

    IDLE_WINDOW_HOURS = 8

    # Look for recent conversation within idle window
    idle_threshold = datetime.utcnow() - timedelta(hours=IDLE_WINDOW_HOURS)

    query = """
        SELECT
            conversation_id, shop_id, customer_id, channel,
            started_at, last_message_at, message_count,
            customer_message_count, business_message_count,
            conversation_depth, conversation_stage,
            drop_off_flag, resulted_in_order, status
        FROM conversations
        WHERE
            shop_id = $1 AND customer_id = $2 AND channel = $3
            AND status = 'active'
            AND last_message_at > $4
        ORDER BY last_message_at DESC
        LIMIT 1
    """

    existing = await db.fetchrow(
        query, shop_id, customer_id, channel, idle_threshold
    )

    if existing:
        return dict(existing)

    # Create new conversation
    now = datetime.utcnow()
    conversation_id = uuid4()

    query = """
        INSERT INTO conversations (
            conversation_id, shop_id, customer_id, channel,
            started_at, last_message_at, message_count,
            customer_message_count, business_message_count,
            conversation_depth, conversation_stage,
            drop_off_flag, resulted_in_order, status
        ) VALUES ($1, $2, $3, $4, $5, $6, 0, 0, 0, 0, $7, false, false, 'active')
        RETURNING
            conversation_id, shop_id, customer_id, channel,
            started_at, last_message_at, message_count,
            customer_message_count, business_message_count,
            conversation_depth, conversation_stage,
            drop_off_flag, resulted_in_order, status
    """

    result = await db.fetch(
        query,
        conversation_id,
        shop_id,
        customer_id,
        channel,
        now,
        now,
        ConversationStage.discovery,
    )

    return dict(result[0]) if result else {}


def _build_product_patterns(products: List[str]) -> List[str]:
    """Build regex patterns for product catalog matching.

    This is a placeholder for Phase 2 integration with product catalog.

    Args:
        products: List of product names

    Returns:
        List of regex patterns for matching
    """
    patterns = []
    for product in products:
        # Simple pattern: match product name case-insensitively
        pattern = product.lower()
        patterns.append(pattern)
    return patterns


async def process_messenger_message(
    db,
    shop_id: UUID,
    psid: str,
    content: str,
    msg_id: str,
    sent_at: datetime
) -> Dict[str, Any]:
    """Process a Messenger message with full integration.

    This function completes ALL critical operations before webhook returns:
    1. Get or create customer
    2. Get or create conversation
    3. Extract intent with FastExtractor
    4. Record message
    5. Update conversation stage based on intent
    6. Track product mentions (if detected)

    Args:
        db: Database connection
        shop_id: Shop ID
        psid: Platform Sender ID
        content: Message content
        msg_id: Platform message ID
        sent_at: Message timestamp

    Returns:
        Dict with conversation_id and processing results
    """
    # Step 1: Get or create customer
    customer = await get_customer_by_psid(db, shop_id, psid)
    if not customer:
        customer = await create_customer(db, shop_id, psid, channel=ChannelType.messenger)
    customer_id = customer["customer_id"]

    # Step 2: Get or create conversation
    conversation = await get_or_create_conversation(
        db, shop_id, customer_id, ChannelType.messenger, psid
    )
    conversation_id = conversation["conversation_id"]

    # Step 3: Extract intent with FastExtractor (<50ms target)
    extraction = extractor.extract_all(content)
    intent_type = extraction["intent_type"]
    confidence = extraction["intent_strength"]
    product_mentioned = extraction.get("product_mentioned")

    # Step 4: Record message
    message_id = uuid4()

    query = """
        INSERT INTO messages (
            message_id, shop_id, conversation_id, customer_id,
            sender_type, channel, content, content_type, sent_at,
            platform_msg_id, day_of_week, hour_of_day
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        RETURNING message_id
    """

    message_id = await db.fetchval(
        query,
        message_id,
        shop_id,
        conversation_id,
        customer_id,
        SenderType.customer,
        ChannelType.messenger,
        content,
        "text",
        sent_at,
        msg_id,
        sent_at.weekday(),
        sent_at.hour,
    )

    # Step 5: Update conversation metadata
    await db.execute(
        """
        UPDATE conversations
        SET
            last_message_at = $1,
            message_count = message_count + 1,
            customer_message_count = customer_message_count + 1,
            conversation_depth = LEAST(customer_message_count + 1, business_message_count)
        WHERE conversation_id = $2
        """,
        sent_at,
        conversation_id
    )

    # Step 6: Update conversation stage based on intent
    current_stage = ConversationStage(conversation["conversation_stage"])
    tracker = ConversationTracker(db)

    # Map intent to stage
    stage_mapping = {
        IntentType.general_chat: ConversationStage.discovery,
        IntentType.unknown: ConversationStage.discovery,
        IntentType.complaint: ConversationStage.discovery,
        IntentType.product_inquiry: ConversationStage.interest,
        IntentType.availability_inquiry: ConversationStage.interest,
        IntentType.price_inquiry: ConversationStage.intent,
        IntentType.purchase_intent: ConversationStage.negotiation,
    }
    target_stage = stage_mapping.get(intent_type, ConversationStage.discovery)

    stage_order = [
        ConversationStage.discovery,
        ConversationStage.interest,
        ConversationStage.intent,
        ConversationStage.negotiation,
        ConversationStage.converted,
        ConversationStage.dormant,
    ]

    try:
        current_idx = stage_order.index(current_stage)
        target_idx = stage_order.index(target_stage)
        if target_idx > current_idx:
            await db.execute(
                """
                UPDATE conversations
                SET conversation_stage = $1
                WHERE conversation_id = $2
                """,
                target_stage,
                conversation_id
            )
    except ValueError:
        pass

    # Step 7: Track product mentions (if detected)
    if product_mentioned:
        # Get current products_mentioned
        current_products = await db.fetchval(
            "SELECT products_mentioned FROM conversations WHERE conversation_id = $1",
            conversation_id
        )
        current_products = current_products if current_products else []

        if product_mentioned not in current_products:
            current_products.append(product_mentioned)
            await db.execute(
                """
                UPDATE conversations
                SET products_mentioned = $1
                WHERE conversation_id = $2
                """,
                current_products,
                conversation_id
            )

    return {
        "conversation_id": conversation_id,
        "customer_id": customer_id,
        "intent_type": intent_type,
        "intent_strength": confidence,
        "product_mentioned": product_mentioned,
    }


# ============================================================================
# Messenger Webhook Endpoints
# ============================================================================

@router.get("/messenger")
async def messenger_webhook_verification(
    hub_mode: str = Query(..., alias="hub.mode"),
    hub_challenge: str = Query(..., alias="hub.challenge"),
    hub_verify_token: str = Query(..., alias="hub.verify_token")
) -> PlainTextResponse:
    """Verify Messenger webhook subscription.

    Facebook sends a GET request with hub.verify_token to verify
    that this endpoint is valid. We must respond with hub.challenge.

    Args:
        hub_mode: Must be "subscribe"
        hub_challenge: Challenge string to echo back
        hub_verify_token: Verification token from app settings

    Returns:
        Plain text response with hub.challenge or error
    """

    # Verify the token
    current_settings = get_current_settings()
    if hub_verify_token != current_settings.messenger_verify_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid verification token"
        )

    # Verify mode
    if hub_mode != "subscribe":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid hub.mode"
        )

    # Respond with challenge (success)
    return PlainTextResponse(content=hub_challenge)


@router.post("/messenger")
async def messenger_webhook_message(
    request: Request,
    db=Depends(get_db)
) -> Dict[str, str]:
    """Receive Messenger webhook events.

    CRITICAL: Must respond within 50ms to avoid webhook timeout.
    All database writes and extractions complete before response.

    Args:
        request: FastAPI request object
        db: Database connection

    Returns:
        {"status": "received"}
    """
    # Get raw payload for HMAC verification
    body = await request.body()
    payload_str = body.decode("utf-8")

    # Verify HMAC signature (security)
    x_hub_signature = request.headers.get("X-Hub-Signature")
    current_settings = get_current_settings()
    if x_hub_signature and not verify_hmac_signature(
        x_hub_signature,
        payload_str,
        current_settings.messenger_app_secret
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid signature"
        )

    # Parse JSON payload
    payload = await request.json()

    # Process entries (Facebook may batch multiple events)
    for entry in payload.get("entry", []):
        for messaging_event in entry.get("messaging", []):
            sender_psid = messaging_event.get("sender", {}).get("id")

            # Only process messages (not postbacks, deliveries, etc. for Phase 1)
            if "message" in messaging_event:
                message_data = messaging_event["message"]

                # Skip if this is a message from business (not customer)
                if sender_psid == entry.get("id"):
                    continue

                # Extract message content
                content = message_data.get("text", "")
                msg_id = message_data.get("mid", "")
                timestamp_ms = messaging_event.get("timestamp", 0)
                sent_at = datetime.fromtimestamp(timestamp_ms / 1000)

                # Process message with full integration
                await process_messenger_message(
                    db,
                    DEFAULT_SHOP_ID,
                    sender_psid,
                    content,
                    msg_id,
                    sent_at
                )

    return {"status": "received"}


# ============================================================================
# WhatsApp Webhook Endpoints (Phase 1 Placeholders)
# ============================================================================

@router.get("/whatsapp")
async def whatsapp_webhook_verification(
    hub_mode: str = Query(..., alias="hub.mode"),
    hub_challenge: str = Query(..., alias="hub.challenge"),
    hub_verify_token: str = Query(..., alias="hub.verify_token")
) -> PlainTextResponse:
    """Verify WhatsApp webhook subscription (Phase 1 placeholder).

    Phase 1: Placeholder endpoint for WhatsApp verification.
    Full implementation in Phase 2.

    Args:
        hub_mode: Must be "subscribe"
        hub_challenge: Challenge string to echo back
        hub_verify_token: Verification token from app settings

    Returns:
        Plain text response with hub.challenge or error
    """

    # Phase 1: Simple verification response
    current_settings = get_current_settings()
    if hub_verify_token != current_settings.messenger_verify_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid verification token"
        )

    if hub_mode != "subscribe":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid hub.mode"
        )

    return PlainTextResponse(content=hub_challenge)


@router.post("/whatsapp")
async def whatsapp_webhook_message(
    request: Request
) -> Dict[str, str]:
    """Receive WhatsApp webhook events (Phase 1 placeholder).

    Phase 1: Placeholder endpoint for WhatsApp messages.
    Full implementation in Phase 2.

    Args:
        request: FastAPI request object

    Returns:
        {"status": "received", "note": "Phase 1 placeholder"}
    """
    return {"status": "received", "note": "Phase 1 placeholder"}
