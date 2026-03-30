"""AI Agent with RAG for generating replies."""
import asyncio
import httpx
from typing import List, Dict, Any, Optional
import asyncpg

from app.config import get_settings
from app.db.connection import DatabasePool


class AIAgent:
    """AI agent with RAG-based reply generation.

    Responsibilities:
    - Context window assembly (system_prompt + customer profile + retrieved embeddings + conversation history)
    - Product embedding retrieval from pgvector for RAG
    - Reply generation using Gemini 2.0 Flash

    Important: This agent does NOT perform scoring or make final decisions.
    All scoring (intent_score, churn_risk, priority_score) is computed deterministically in Python.
    """

    # Gemini 2.0 Flash API endpoint
    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"

    # Context window token limits (approximate)
    MAX_SYSTEM_PROMPT_TOKENS = 300
    MAX_CUSTOMER_PROFILE_TOKENS = 200
    MAX_RETRIEVED_CHUNKS_TOKENS = 400
    MAX_CONVERSATION_HISTORY_TOKENS = 600
    MAX_CURRENT_MESSAGE_TOKENS = 100

    def __init__(
        self,
        http_client: Optional[httpx.AsyncClient] = None,
        embeddings_service: Optional[Any] = None
    ):
        """Initialize AI agent.

        Args:
            http_client: Optional HTTP client for API calls.
            embeddings_service: Optional embeddings service for RAG.
        """
        self.http_client = http_client or httpx.AsyncClient(timeout=30.0)
        self._embeddings_service = embeddings_service
        self._conn: Optional[asyncpg.Connection] = None
        self._settings = get_settings()

    @property
    def embeddings_service(self):
        """Lazy load embeddings service."""
        if self._embeddings_service is None:
            from app.services.embeddings import get_embeddings_service
            self._embeddings_service = get_embeddings_service()
        return self._embeddings_service

    async def _ensure_connection(self) -> asyncpg.Connection:
        """Lazy connection initialization.

        Returns:
            Active database connection.
        """
        if self._conn is None or self._conn.is_closed:
            self._conn = await DatabasePool.acquire().__aenter__()
        return self._conn

    async def _close(self) -> None:
        """Close database connection."""
        if self._conn and not self._conn.is_closed:
            await self._conn.close()
            self._conn = None

    async def _get_relevant_chunks(
        self,
        message_id: str,
        shop_id: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve top-3 product embeddings using cosine similarity.

        Args:
            message_id: Message ID to get content from.
            shop_id: Shop ID for multi-tenancy.
            top_k: Maximum number of chunks to retrieve (default: 3).

        Returns:
            List of dictionaries with chunk_text, source_id, and similarity.
        """
        try:
            conn = await self._ensure_connection()

            # Get message content for embedding
            message_row = await conn.fetchrow(
                "SELECT content FROM messages WHERE message_id = $1",
                message_id
            )

            if not message_row:
                return []

            message_content = message_row["content"]

            # Generate embedding for the message
            query_vector = await self.embeddings_service.get_embedding(message_content)

            # Retrieve similar chunks from pgvector
            similar_chunks = await self.embeddings_service.cosine_similarity(
                query_vector=query_vector,
                top_k=top_k,
                shop_id=shop_id,
                source_type="product"
            )

            return [
                {
                    "chunk_text": chunk["chunk_text"],
                    "source_id": chunk["source_id"],
                    "similarity": chunk["similarity"]
                }
                for chunk in similar_chunks
            ]
        except Exception as e:
            # Log error in production; return empty list for now
            return []

    async def _get_customer_profile_block(
        self,
        customer_id: str,
        shop_id: str
    ) -> str:
        """Format customer profile from customers/user_metrics tables.

        The profile includes pre-computed metrics from the database.

        Args:
            customer_id: Customer ID.
            shop_id: Shop ID for multi-tenancy.

        Returns:
            Formatted customer profile block string.
        """
        try:
            conn = await self._ensure_connection()

            # Join customers with user_metrics for comprehensive profile
            profile = await conn.fetchrow("""
                SELECT
                    c.display_name,
                    COALESCE(u.preferred_inbox_hours, ARRAY[]::int[]) as preferred_inbox_hours,
                    COALESCE(u.preferred_days, ARRAY[]::text[]) as preferred_days,
                    COALESCE(u.top_products_mentioned, ARRAY[]::text[]) as top_products_mentioned,
                    COALESCE(c.order_count, 0) as order_count,
                    COALESCE(c.avg_order_value, 0.0) as avg_order_value,
                    COALESCE(c.intent_score_latest, 0.0) as intent_score_latest,
                    COALESCE(c.churn_risk_score, 0.0) as churn_risk_score,
                    CASE
                        WHEN c.churn_risk_score < 0.3 THEN 'Low'
                        WHEN c.churn_risk_score < 0.6 THEN 'Medium'
                        ELSE 'High'
                    END as churn_label,
                    COALESCE(u.segment_label, 'Unknown') as segment_label
                FROM customers c
                LEFT JOIN user_metrics u ON c.customer_id = u.customer_id AND u.shop_id = $2
                WHERE c.customer_id = $1 AND c.shop_id = $2
            """, customer_id, shop_id)

            if not profile:
                return "CUSTOMER PROFILE: New customer (no data available yet)"

            # Format according to spec
            return (
                f"CUSTOMER PROFILE (pre-computed — do not modify):\n"
                f"Name: {profile['display_name'] or 'Unknown'} | "
                f"Active hours: {', '.join(map(str, profile['preferred_inbox_hours'][:5])) or 'N/A'} | "
                f"Active days: {', '.join(profile['preferred_days'][:3]) or 'N/A'}\n"
                f"Interested in: {', '.join(profile['top_products_mentioned'][:3]) or 'None yet'}\n"
                f"Orders: {profile['order_count']} · avg ${profile['avg_order_value']:.2f}\n"
                f"Intent score: {profile['intent_score_latest']:.2f} | "
                f"Churn risk: {profile['churn_risk_score']:.2f} ({profile['churn_label']})\n"
                f"Segment: {profile['segment_label']}"
            )
        except Exception as e:
            # Return minimal profile on error
            return "CUSTOMER PROFILE: Unable to load profile data"

    async def _assemble_context(
        self,
        shop_id: str,
        customer_id: str,
        conversation_id: str,
        message_id: str,
        current_message: str,
        system_prompt: str
    ) -> Dict[str, Any]:
        """Assemble full context window for AI generation.

        Context components:
        - system_prompt: Shop's AI system prompt (~300 tokens)
        - customer_profile: Pre-computed profile from customers/user_metrics (~200 tokens)
        - retrieved_chunks: Top-3 product chunks from pgvector (~400 tokens)
        - conversation_history: Last 20 messages from messages table (~600 tokens)
        - current_message: The incoming message text (~100 tokens)

        Args:
            shop_id: Shop ID for multi-tenancy.
            customer_id: Customer ID.
            conversation_id: Conversation ID.
            message_id: Message ID for RAG.
            current_message: The incoming message text.
            system_prompt: Shop's AI system prompt.

        Returns:
            Dictionary with all context components.
        """
        conn = await self._ensure_connection()

        # Get customer profile
        customer_profile = await self._get_customer_profile_block(customer_id, shop_id)

        # Get relevant product chunks
        retrieved_chunks = await self._get_relevant_chunks(message_id, shop_id)

        # Get last 20 messages from conversation history (most recent first, then reverse)
        history_rows = await conn.fetch("""
            SELECT
                message_id,
                sender_type,
                content,
                sent_at
            FROM messages
            WHERE conversation_id = $1 AND shop_id = $2
            ORDER BY sent_at DESC
            LIMIT 20
        """, conversation_id, shop_id)

        # Convert to list
        history_rows = list(history_rows)

        # For robustness (handles mocks that don't respect ORDER BY):
        # Sort by sent_at descending and take last 20, then reverse for chronological order
        if len(history_rows) > 20:
            # Try to sort by sent_at (handles both datetime and string formats)
            def get_sent_at_key(row):
                sent_at = row.get("sent_at")
                if isinstance(sent_at, str):
                    return sent_at  # ISO string sort works correctly
                elif hasattr(sent_at, 'isoformat'):
                    return sent_at.isoformat()
                else:
                    return str(sent_at)

            # Sort by sent_at descending to get most recent first
            history_rows.sort(key=get_sent_at_key, reverse=True)
            # Take the last 20 (most recent)
            history_rows = history_rows[:20]

        # Reverse to get chronological order (oldest first)
        history_rows = list(reversed(history_rows))

        conversation_history = []
        for row in history_rows:
            sent_at = row["sent_at"]
            # Handle both datetime objects and ISO string values
            if sent_at and not isinstance(sent_at, str):
                sent_at = sent_at.isoformat() if hasattr(sent_at, 'isoformat') else str(sent_at)
            conversation_history.append({
                "message_id": str(row["message_id"]),
                "role": row["sender_type"],
                "content": row["content"],
                "sent_at": sent_at
            })

        # Assemble full context
        return {
            "system_prompt": system_prompt,
            "customer_profile": customer_profile,
            "retrieved_chunks": retrieved_chunks,
            "conversation_history": conversation_history,
            "current_message": current_message
        }

    async def generate_reply(
        self,
        shop_id: str,
        customer_id: str,
        conversation_id: str,
        message_id: str,
        current_message: str,
        system_prompt: str
    ) -> Dict[str, Any]:
        """Generate AI reply based on conversation context.

        Uses Gemini 2.0 Flash with RAG context for intelligent responses.

        Args:
            shop_id: Shop ID for multi-tenancy.
            customer_id: Customer ID.
            conversation_id: Conversation ID.
            message_id: Message ID for RAG.
            current_message: The incoming message text.
            system_prompt: Shop's AI system prompt.

        Returns:
            Dictionary with reply text and optional metadata.
        """
        try:
            # Assemble context window
            context = await self._assemble_context(
                shop_id=shop_id,
                customer_id=customer_id,
                conversation_id=conversation_id,
                message_id=message_id,
                current_message=current_message,
                system_prompt=system_prompt
            )

            # Build the prompt for Gemini
            prompt_parts = [
                {"text": context["system_prompt"]},
                {"text": "\n\n"},
                {"text": context["customer_profile"]},
                {"text": "\n\n"},
            ]

            # Add retrieved product chunks if available
            if context["retrieved_chunks"]:
                prompt_parts.append({"text": "RELEVANT PRODUCTS:\n"})
                for i, chunk in enumerate(context["retrieved_chunks"], 1):
                    prompt_parts.append({"text": f"{i}. {chunk['chunk_text']}\n"})
                prompt_parts.append({"text": "\n"})

            # Add conversation history
            if context["conversation_history"]:
                prompt_parts.append({"text": "RECENT CONVERSATION:\n"})
                for msg in context["conversation_history"][-20:]:  # Limit to last 20
                    role = msg["role"]
                    prefix = "Customer" if role == "customer" else "Business"
                    prompt_parts.append({"text": f"{prefix}: {msg['content']}\n"})
                prompt_parts.append({"text": "\n"})

            # Add current message
            prompt_parts.append({"text": f"CUSTOMER SAYS: {current_message}\n\n"})
            prompt_parts.append({"text": "Your reply:"})

            # Call Gemini API
            response = await self.http_client.post(
                f"{self.GEMINI_API_URL}?key={self._settings.gemini_api_key}",
                json={
                    "contents": [{"parts": prompt_parts}],
                    "generationConfig": {
                        "temperature": 0.7,
                        "topK": 40,
                        "topP": 0.95,
                        "maxOutputTokens": 500
                    }
                }
            )
            response.raise_for_status()
            data = response.json()

            # Extract reply text
            reply = ""
            candidates = data.get("candidates", [])
            if candidates and candidates[0].get("content"):
                parts = candidates[0]["content"].get("parts", [])
                if parts and parts[0].get("text"):
                    reply = parts[0]["text"].strip()

            # Extract any mentioned products from retrieved chunks
            suggested_products = []
            for chunk in context["retrieved_chunks"]:
                product_name = chunk.get("source_id", "")
                if product_name and product_name not in suggested_products:
                    suggested_products.append(product_name)

            return {
                "reply": reply,
                "confidence": 0.8,  # Placeholder - can be computed from model's response
                "suggested_products": suggested_products[:5] if suggested_products else None
            }

        except httpx.HTTPStatusError:
            # Return fallback response on API error
            return {
                "reply": "I apologize, but I'm having trouble connecting right now. Please try again shortly or contact our support team.",
                "confidence": 0.0,
                "suggested_products": None
            }
        except Exception as e:
            # Return fallback response on any error
            return {
                "reply": "I apologize, but I'm having trouble processing your request. Please try again.",
                "confidence": 0.0,
                "suggested_products": None
            }


# Global service instance
_service: Optional[AIAgent] = None


def get_ai_agent() -> AIAgent:
    """Get or create global AI agent instance."""
    global _service
    if _service is None:
        _service = AIAgent()
    return _service
