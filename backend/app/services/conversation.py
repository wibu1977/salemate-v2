"""Conversation lifecycle management service.

This module provides the ConversationTracker class for managing conversation
thread lifecycle, including creation, message recording, stage progression,
product mention tracking, and drop-off detection.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4

from app.db.queries import (
    IntentType,
    ConversationStage,
    ChannelType,
    SenderType,
)

# Constants
IDLE_WINDOW_HOURS = 8


class ConversationTracker:
    """Manage conversation lifecycle and state.

    This class handles:
    - Getting or creating conversations with idle window management
    - Recording messages and updating conversation metadata
    - Stage progression based on detected intents
    - Product mention tracking
    - Drop-off detection for re-engagement
    - Conversion tracking
    """

    def __init__(self, db_connection):
        """Initialize ConversationTracker.

        Args:
            db_connection: Database connection (asyncpg.Connection or mock)
        """
        self.db = db_connection

    async def get_customer_by_psid(
        self,
        shop_id: UUID,
        psid: str,
    ) -> Optional[Dict[str, Any]]:
        """Get customer by PSID.

        Args:
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
        return await self.db.fetchrow(query, shop_id, psid)

    async def create_customer(
        self,
        shop_id: UUID,
        psid: str,
        display_name: Optional[str] = None,
        locale: Optional[str] = None,
        channel: ChannelType = ChannelType.messenger,
    ) -> Dict[str, Any]:
        """Create a new customer.

        Args:
            shop_id: Shop ID
            psid: Platform Sender ID
            display_name: Customer display name
            locale: Customer locale
            channel: Communication channel

        Returns:
            Created customer dict
        """
        now = datetime.utcnow()

        query = """
            INSERT INTO customers (
                shop_id, psid, display_name, locale, channel,
                first_seen_at, conversation_count, total_order_value, order_count
            ) VALUES ($1, $2, $3, $4, $5, $6, 0, 0.0, 0)
            RETURNING
                customer_id, shop_id, psid, display_name, channel,
                first_seen_at, last_contact_at, conversation_count,
                total_order_value, order_count
        """
        return await self.db.fetchrow(
            query, shop_id, psid, display_name, locale, channel, now
        )

    async def get_or_create_conversation(
        self,
        shop_id: UUID,
        customer_id: UUID,
        channel: ChannelType,
        psid: str,
    ) -> Dict[str, Any]:
        """Get existing conversation or create new one.

        Uses IDLE_WINDOW_HOURS (8 hours) to determine if a new conversation
        should be started. If no message in the last 8 hours, creates new.

        Args:
            shop_id: Shop ID
            customer_id: Customer ID
            channel: Communication channel
            psid: Platform Sender ID

        Returns:
            Conversation dict with conversation_id and metadata
        """
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

        existing = await self.db.fetchrow(
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

        result = await self.db.fetch(
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

    async def record_message(
        self,
        shop_id: UUID,
        conversation_id: UUID,
        customer_id: UUID,
        sender_type: SenderType,
        channel: ChannelType,
        content: str,
        sent_at: datetime,
        content_type: str = "text",
        platform_msg_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a message and update conversation metadata.

        Args:
            shop_id: Shop ID
            conversation_id: Conversation ID
            customer_id: Customer ID
            sender_type: customer or business
            channel: Communication channel
            content: Message content
            sent_at: Message timestamp
            content_type: Message content type
            platform_msg_id: Platform message ID

        Returns:
            Dict with message_id and updated conversation info
        """
        # Insert message
        message_id = uuid4()

        query = """
            INSERT INTO messages (
                message_id, shop_id, conversation_id, customer_id,
                sender_type, channel, content, content_type, sent_at,
                platform_msg_id, day_of_week, hour_of_day
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            RETURNING message_id
        """

        message_id = await self.db.fetchval(
            query,
            message_id,
            shop_id,
            conversation_id,
            customer_id,
            sender_type,
            channel,
            content,
            content_type,
            sent_at,
            platform_msg_id,
            sent_at.weekday(),
            sent_at.hour,
        )

        # Update conversation metadata
        await self._update_conversation_after_message(
            conversation_id, sender_type, sent_at
        )

        return {"message_id": message_id}

    async def _update_conversation_after_message(
        self,
        conversation_id: UUID,
        sender_type: SenderType,
        sent_at: datetime,
    ) -> None:
        """Update conversation metadata after recording a message.

        Args:
            conversation_id: Conversation ID
            sender_type: customer or business
            sent_at: Message timestamp
        """
        if sender_type == SenderType.customer:
            # Customer message: increment customer_message_count
            query = """
                UPDATE conversations
                SET
                    last_message_at = $1,
                    message_count = message_count + 1,
                    customer_message_count = customer_message_count + 1,
                    conversation_depth = LEAST(customer_message_count + 1, business_message_count)
                WHERE conversation_id = $2
            """
        else:
            # Business message: increment business_message_count
            query = """
                UPDATE conversations
                SET
                    last_message_at = $1,
                    message_count = message_count + 1,
                    business_message_count = business_message_count + 1,
                    conversation_depth = LEAST(customer_message_count, business_message_count + 1)
                WHERE conversation_id = $2
            """

        await self.db.execute(query, sent_at, conversation_id)

    async def update_stage(
        self,
        conversation_id: UUID,
        intent_type: IntentType,
        current_stage: Optional[ConversationStage] = None,
    ) -> None:
        """Update conversation stage based on detected intent.

        Stage progression is one-way:
        discovery -> interest -> intent -> negotiation -> converted -> dormant

        Args:
            conversation_id: Conversation ID
            intent_type: Detected intent type
            current_stage: Current conversation stage (optional, fetched if not provided)
        """
        if current_stage is None:
            query = """
                SELECT conversation_stage FROM conversations
                WHERE conversation_id = $1
            """
            row = await self.db.fetchrow(query, conversation_id)
            if row:
                current_stage = ConversationStage(row["conversation_stage"])
            else:
                current_stage = ConversationStage.discovery

        # Determine target stage based on intent
        target_stage = self._map_intent_to_stage(intent_type)

        # Only update if target stage is later in progression
        if self._should_update_stage(current_stage, target_stage):
            query = """
                UPDATE conversations
                SET conversation_stage = $1
                WHERE conversation_id = $2
            """
            await self.db.execute(query, target_stage, conversation_id)

    def _map_intent_to_stage(self, intent_type: IntentType) -> ConversationStage:
        """Map intent type to conversation stage.

        Args:
            intent_type: Detected intent type

        Returns:
            Target conversation stage
        """
        stage_mapping = {
            IntentType.general_chat: ConversationStage.discovery,
            IntentType.unknown: ConversationStage.discovery,
            IntentType.complaint: ConversationStage.discovery,
            IntentType.product_inquiry: ConversationStage.interest,
            IntentType.availability_inquiry: ConversationStage.interest,
            IntentType.price_inquiry: ConversationStage.intent,
            IntentType.purchase_intent: ConversationStage.negotiation,
        }
        return stage_mapping.get(intent_type, ConversationStage.discovery)

    def _should_update_stage(
        self, current: ConversationStage, target: ConversationStage
    ) -> bool:
        """Check if stage should be updated (one-way progression).

        Args:
            current: Current stage
            target: Target stage

        Returns:
            True if target is later in progression
        """
        stage_order = [
            ConversationStage.discovery,
            ConversationStage.interest,
            ConversationStage.intent,
            ConversationStage.negotiation,
            ConversationStage.converted,
            ConversationStage.dormant,
        ]

        try:
            current_idx = stage_order.index(current)
            target_idx = stage_order.index(target)
            return target_idx > current_idx
        except ValueError:
            return False

    async def add_product_mention(
        self,
        conversation_id: UUID,
        product_id: str,
    ) -> None:
        """Add product to conversation's mentioned products list.

        Avoids duplicates by checking existing mentions.

        Args:
            conversation_id: Conversation ID
            product_id: Product ID to add
        """
        # Get current products_mentioned
        query = """
            SELECT products_mentioned FROM conversations
            WHERE conversation_id = $1
        """
        row = await self.db.fetchval(query, conversation_id)

        current_products = row if row else []
        if product_id in current_products:
            return  # Already mentioned

        # Add product to list
        current_products.append(product_id)

        query = """
            UPDATE conversations
            SET products_mentioned = $1
            WHERE conversation_id = $2
        """
        await self.db.execute(query, current_products, conversation_id)

    async def get_conversation(
        self,
        conversation_id: UUID,
    ) -> Optional[Dict[str, Any]]:
        """Get full conversation details.

        Args:
            conversation_id: Conversation ID

        Returns:
            Conversation dict or None if not found
        """
        query = """
            SELECT
                conversation_id, shop_id, customer_id, channel,
                started_at, last_message_at, message_count,
                customer_message_count, business_message_count,
                conversation_depth, conversation_stage,
                intent_score, drop_off_flag, resulted_in_order, status
            FROM conversations
            WHERE conversation_id = $1
        """
        row = await self.db.fetchrow(query, conversation_id)
        return dict(row) if row else None

    async def get_recent_messages(
        self,
        conversation_id: UUID,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get recent messages for context window.

        Args:
            conversation_id: Conversation ID
            limit: Maximum number of messages to return

        Returns:
            List of message dicts, ordered by sent_at DESC
        """
        query = """
            SELECT
                message_id, shop_id, conversation_id, customer_id,
                sender_type, channel, content, sent_at
            FROM messages
            WHERE conversation_id = $1
            ORDER BY sent_at DESC
            LIMIT $2
        """
        rows = await self.db.fetch(query, conversation_id, limit)
        return [dict(row) for row in rows]

    async def check_drop_off(
        self,
        shop_id: UUID,
    ) -> List[Dict[str, Any]]:
        """Find conversations where customer dropped off.

        Drop-off is defined as:
        - Last message was more than IDLE_WINDOW_HOURS (8) ago
        - Status is 'active'
        - Stage is NOT 'dormant' or 'converted'

        This is meant to be run as an hourly SQL job for re-engagement.

        Args:
            shop_id: Shop ID

        Returns:
            List of dropped-off conversation dicts
        """
        idle_threshold = datetime.utcnow() - timedelta(hours=IDLE_WINDOW_HOURS)

        # Find dropped-off conversations
        query = """
            SELECT
                conversation_id, shop_id, customer_id,
                conversation_stage, last_message_at
            FROM conversations
            WHERE
                shop_id = $1
                AND status = 'active'
                AND drop_off_flag = false
                AND conversation_stage NOT IN ('dormant', 'converted')
                AND last_message_at < $2
        """
        dropped = await self.db.fetch(query, shop_id, idle_threshold)

        dropped_list = [dict(row) for row in dropped]

        # Mark them as dropped off
        for conv in dropped_list:
            await self._mark_dropped_off(conv["conversation_id"])

        return dropped_list

    async def _mark_dropped_off(
        self,
        conversation_id: UUID,
    ) -> None:
        """Mark a conversation as dropped off.

        Args:
            conversation_id: Conversation ID
        """
        query = """
            UPDATE conversations
            SET drop_off_flag = true, conversation_stage = 'dormant'
            WHERE conversation_id = $1
        """
        await self.db.execute(query, conversation_id)

    async def mark_converted(
        self,
        conversation_id: UUID,
    ) -> None:
        """Mark a conversation as converted when order is completed.

        Sets resulted_in_order to true and conversation_stage to 'converted'.

        Args:
            conversation_id: Conversation ID
        """
        query = """
            UPDATE conversations
            SET resulted_in_order = true, conversation_stage = 'converted'
            WHERE conversation_id = $1
        """
        await self.db.execute(query, conversation_id)
