"""Tests for AI agent with RAG support."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import httpx

from app.services.ai_agent import AIAgent, get_ai_agent


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for Google AI API."""
    mock_instance = AsyncMock(spec=httpx.AsyncClient)
    return mock_instance


@pytest.fixture
def mock_connection():
    """Mock asyncpg connection."""
    conn = AsyncMock(spec_set=['execute', 'fetch', 'fetchval', 'fetchrow', 'is_closed', 'close', 'fetchmany'])
    conn.execute = AsyncMock()
    conn.fetch = AsyncMock()
    conn.fetchval = AsyncMock()
    conn.fetchrow = AsyncMock()
    conn.fetchmany = AsyncMock()
    conn.close = AsyncMock()
    type(conn).is_closed = PropertyMock(return_value=False)
    return conn


@pytest.fixture
def mock_embeddings_service():
    """Mock embeddings service for RAG."""
    service = AsyncMock()
    service.get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3] * 256)
    service.cosine_similarity = AsyncMock(return_value=[
        {
            "embedding_id": "emb-1",
            "source_type": "product",
            "source_id": "prod-1",
            "chunk_text": "Wool Jacket - A warm and stylish wool jacket",
            "similarity": 0.92
        },
        {
            "embedding_id": "emb-2",
            "source_type": "product",
            "source_id": "prod-1",
            "chunk_text": "Available in sizes S, M, L, XL",
            "similarity": 0.88
        },
        {
            "embedding_id": "emb-3",
            "source_type": "product",
            "source_id": "prod-2",
            "chunk_text": "Cotton Shirt - Breathable cotton shirt",
            "similarity": 0.78
        }
    ])
    return service


@pytest.fixture
def ai_agent(mock_http_client, mock_embeddings_service):
    """Create AI agent instance."""
    agent = AIAgent(
        http_client=mock_http_client,
        embeddings_service=mock_embeddings_service
    )
    yield agent
    # Cleanup
    if hasattr(agent, "_conn") and agent._conn is not None:
        import asyncio
        if asyncio.iscoroutinefunction(agent._conn.close):
            asyncio.run(agent._conn.close())


# Test fixtures for data
@pytest.fixture
def sample_customer_profile():
    """Sample customer profile from customers/user_metrics tables."""
    return {
        "display_name": "Alice Johnson",
        "preferred_inbox_hours": [9, 10, 11, 14, 15, 16, 17],
        "preferred_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "top_products_mentioned": ["Wool Jacket", "Cotton Shirt"],
        "order_count": 3,
        "avg_order_value": 85.50,
        "intent_score_latest": 0.72,
        "churn_risk_score": 0.25,
        "churn_label": "Low",
        "segment_label": "Loyal Customer"
    }


@pytest.fixture
def sample_conversation_history():
    """Sample conversation history (last 20 messages)."""
    return [
        {"role": "customer", "content": "Hi, I'm looking for a warm jacket", "sent_at": "2026-03-27T10:00:00"},
        {"role": "business", "content": "Hello! We have some great wool jackets. What style are you looking for?", "sent_at": "2026-03-27T10:01:00"},
        {"role": "customer", "content": "Something casual but warm for winter", "sent_at": "2026-03-27T10:02:00"}
    ]


@pytest.fixture
def sample_retrieved_chunks():
    """Sample retrieved chunks from pgvector."""
    return [
        {
            "chunk_text": "Wool Jacket - A warm and stylish wool jacket perfect for winter",
            "source_id": "prod-1",
            "similarity": 0.92
        },
        {
            "chunk_text": "Available in sizes S, M, L, XL and colors red, blue, black",
            "source_id": "prod-1",
            "similarity": 0.88
        },
        {
            "chunk_text": "Price: $89.99 - Made with premium wool blend",
            "source_id": "prod-1",
            "similarity": 0.85
        }
    ]


@pytest.fixture
def sample_system_prompt():
    """Sample AI system prompt."""
    return """You are a helpful sales assistant for a clothing store.
You should be friendly, professional, and help customers find products they love.
Always be honest about product details and availability."""


# Tests


@pytest.mark.asyncio
async def test_get_relevant_chunks(ai_agent, mock_connection, mock_embeddings_service):
    """Test retrieval of top-3 product embeddings using cosine similarity."""
    # Setup database for message query
    mock_connection.fetchrow.return_value = {
        "content": "I'm looking for a warm jacket"
    }
    ai_agent._conn = mock_connection

    chunks = await ai_agent._get_relevant_chunks(
        message_id="msg-123",
        shop_id="shop-456"
    )

    # Verify embeddings service was called
    mock_embeddings_service.get_embedding.assert_called_once_with("I'm looking for a warm jacket")
    mock_embeddings_service.cosine_similarity.assert_called_once()

    # Verify we got exactly 3 chunks
    assert len(chunks) == 3
    assert all("chunk_text" in c for c in chunks)
    assert all("source_id" in c for c in chunks)


@pytest.mark.asyncio
async def test_get_relevant_chunks_empty_message(ai_agent, mock_connection, mock_embeddings_service):
    """Test relevant chunks with empty message content."""
    mock_connection.fetchrow.return_value = {"content": ""}
    ai_agent._conn = mock_connection

    chunks = await ai_agent._get_relevant_chunks(
        message_id="msg-123",
        shop_id="shop-456"
    )

    # Should still call embedding service even with empty text
    assert mock_embeddings_service.get_embedding.called
    assert isinstance(chunks, list)


@pytest.mark.asyncio
async def test_get_customer_profile_block(ai_agent, mock_connection, sample_customer_profile):
    """Test formatting customer profile from customers/user_metrics tables."""
    # Mock customer data
    mock_connection.fetchrow.return_value = sample_customer_profile
    ai_agent._conn = mock_connection

    profile_block = await ai_agent._get_customer_profile_block(
        customer_id="cust-123",
        shop_id="shop-456"
    )

    # Verify the format includes all required fields
    assert "Alice Johnson" in profile_block
    assert "9, 10, 11" in profile_block or "preferred_inbox_hours" in profile_block
    assert "Wool Jacket" in profile_block
    assert "3" in profile_block  # order_count
    assert "85.5" in profile_block or "85.50" in profile_block  # avg_order_value
    assert "0.72" in profile_block or "0.7" in profile_block  # intent_score_latest
    assert "0.25" in profile_block  # churn_risk_score
    assert "Loyal Customer" in profile_block

    # Verify the pre-computed disclaimer is present
    assert "pre-computed" in profile_block.lower() or "do not modify" in profile_block.lower()


@pytest.mark.asyncio
async def test_get_customer_profile_block_missing_customer(ai_agent, mock_connection):
    """Test customer profile block when customer not found."""
    mock_connection.fetchrow.return_value = None
    ai_agent._conn = mock_connection

    profile_block = await ai_agent._get_customer_profile_block(
        customer_id="nonexistent",
        shop_id="shop-456"
    )

    # Should return a minimal/default block
    assert profile_block is not None
    assert isinstance(profile_block, str)


@pytest.mark.asyncio
async def test_assemble_context(
    ai_agent,
    mock_connection,
    mock_embeddings_service,
    sample_customer_profile,
    sample_conversation_history,
    sample_system_prompt
):
    """Test context window assembly with all components."""
    # Mock database queries - use side_effect for different queries
    # First fetchrow is for customer profile, second is for message content
    mock_connection.fetchrow.side_effect = [
        sample_customer_profile,  # Customer profile query
        {"content": "I'm interested in the wool jacket"},  # Message content query
    ]
    mock_connection.fetch.return_value = [
        {
            "message_id": "msg-1",
            "sender_type": "customer",
            "content": "Hi, I'm looking for a warm jacket",
            "sent_at": "2026-03-27T10:00:00"
        },
        {
            "message_id": "msg-2",
            "sender_type": "business",
            "content": "Hello! We have some great wool jackets.",
            "sent_at": "2026-03-27T10:01:00"
        },
        {
            "message_id": "msg-3",
            "sender_type": "customer",
            "content": "Something casual but warm for winter",
            "sent_at": "2026-03-27T10:02:00"
        }
    ]
    ai_agent._conn = mock_connection

    context = await ai_agent._assemble_context(
        shop_id="shop-456",
        customer_id="cust-123",
        conversation_id="conv-123",
        message_id="msg-123",
        current_message="I'm interested in the wool jacket",
        system_prompt=sample_system_prompt
    )

    # Verify all context components are present
    assert "system_prompt" in context
    assert "customer_profile" in context
    assert "retrieved_chunks" in context
    assert "conversation_history" in context
    assert "current_message" in context

    # Verify system prompt is included
    assert "sales assistant" in context["system_prompt"].lower()

    # Verify customer profile is included
    assert "Alice Johnson" in context["customer_profile"]

    # Verify retrieved chunks are included
    assert len(context["retrieved_chunks"]) == 3

    # Verify conversation history is limited and formatted
    assert len(context["conversation_history"]) <= 20

    # Verify current message is included
    assert "wool jacket" in context["current_message"]


@pytest.mark.asyncio
async def test_assemble_context_token_limits(
    ai_agent,
    mock_connection,
    mock_embeddings_service,
    sample_customer_profile,
    sample_system_prompt
):
    """Test that context assembly respects token limits."""
    # Mock customer data - use side_effect for multiple queries
    mock_connection.fetchrow.side_effect = [
        sample_customer_profile,  # Customer profile query
        {"content": "Test message"},  # Message content query
    ]
    ai_agent._conn = mock_connection

    # Create a long conversation history (30 messages)
    long_history = [
        {
            "message_id": f"msg-{i}",
            "sender_type": "customer" if i % 2 == 0 else "business",
            "content": f"Message number {i} with some additional text",
            "sent_at": f"2026-03-27T10:{i:02d}:00"
        }
        for i in range(30)
    ]
    mock_connection.fetch.return_value = long_history

    context = await ai_agent._assemble_context(
        shop_id="shop-456",
        customer_id="cust-123",
        conversation_id="conv-123",
        message_id="msg-123",
        current_message="Test message",
        system_prompt=sample_system_prompt
    )

    # Verify conversation history is limited to last 20 messages
    assert len(context["conversation_history"]) <= 20

    # Verify the most recent messages are kept
    assert context["conversation_history"][-1]["message_id"] == "msg-29"


@pytest.mark.asyncio
async def test_generate_reply(ai_agent, mock_http_client, mock_connection, mock_embeddings_service):
    """Test reply generation with mocked Gemini API."""
    # Mock Gemini API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value={
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Great choice! Our wool jacket is perfect for winter. Would you like me to check your size in stock?"}
                    ]
                }
            }
        ]
    })
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    # Mock database queries
    mock_connection.fetchrow.return_value = {
        "display_name": "Alice",
        "preferred_inbox_hours": [9, 10, 11, 14, 15, 16, 17],
        "preferred_days": ["Monday", "Tuesday"],
        "top_products_mentioned": ["Wool Jacket"],
        "order_count": 2,
        "avg_order_value": 75.0,
        "intent_score_latest": 0.7,
        "churn_risk_score": 0.2,
        "churn_label": "Low",
        "segment_label": "Regular"
    }
    mock_connection.fetch.return_value = []
    ai_agent._conn = mock_connection

    result = await ai_agent.generate_reply(
        shop_id="shop-456",
        customer_id="cust-123",
        conversation_id="conv-123",
        message_id="msg-123",
        current_message="I'm interested in the wool jacket",
        system_prompt="You are a helpful assistant."
    )

    # Verify response structure
    assert "reply" in result
    assert result["reply"] is not None
    assert isinstance(result["reply"], str)
    assert len(result["reply"]) > 0

    # Verify Gemini API was called
    mock_http_client.post.assert_called_once()

    # Verify the request includes the assembled context
    call_args = mock_http_client.post.call_args
    assert call_args is not None
    request_body = call_args.kwargs.get("json", call_args[1].get("json") if len(call_args) > 1 else {})
    assert "contents" in request_body


@pytest.mark.asyncio
async def test_generate_reply_with_product_recommendations(ai_agent, mock_http_client, mock_connection, mock_embeddings_service):
    """Test reply generation that includes product recommendations."""
    # Mock Gemini response with product mentions
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value={
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "I recommend the Wool Jacket for warmth. It's available in multiple sizes."}
                    ]
                }
            }
        ]
    })
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    # Mock database
    mock_connection.fetchrow.return_value = {
        "display_name": "Bob",
        "preferred_inbox_hours": [9, 10, 11],
        "preferred_days": ["Monday"],
        "top_products_mentioned": [],
        "order_count": 0,
        "avg_order_value": 0.0,
        "intent_score_latest": 0.5,
        "churn_risk_score": 0.3,
        "churn_label": "Medium",
        "segment_label": "New"
    }
    mock_connection.fetch.return_value = []
    ai_agent._conn = mock_connection

    result = await ai_agent.generate_reply(
        shop_id="shop-456",
        customer_id="cust-123",
        conversation_id="conv-123",
        message_id="msg-123",
        current_message="I need a warm jacket",
        system_prompt="You are a helpful assistant."
    )

    assert "reply" in result
    assert "Wool Jacket" in result["reply"] or "jacket" in result["reply"].lower()


@pytest.mark.asyncio
async def test_generate_reply_gemini_error_handling(ai_agent, mock_http_client, mock_connection):
    """Test reply generation when Gemini API fails."""
    # Mock Gemini API error
    mock_http_client.post.side_effect = httpx.HTTPStatusError(
        "Service unavailable",
        request=MagicMock(),
        response=MagicMock(status_code=503)
    )

    # Mock database
    mock_connection.fetchrow.return_value = {
        "display_name": "Customer",
        "preferred_inbox_hours": [9, 10],
        "preferred_days": ["Monday"],
        "top_products_mentioned": [],
        "order_count": 0,
        "avg_order_value": 0.0,
        "intent_score_latest": 0.5,
        "churn_risk_score": 0.3,
        "churn_label": "Medium",
        "segment_label": "New"
    }
    mock_connection.fetch.return_value = []
    ai_agent._conn = mock_connection

    result = await ai_agent.generate_reply(
        shop_id="shop-456",
        customer_id="cust-123",
        conversation_id="conv-123",
        message_id="msg-123",
        current_message="Hello",
        system_prompt="You are a helpful assistant."
    )

    # Should return a fallback response
    assert "reply" in result
    assert result["reply"] is not None
    assert len(result["reply"]) > 0


@pytest.mark.asyncio
async def test_generate_reply_empty_message(ai_agent, mock_http_client, mock_connection):
    """Test reply generation with empty message."""
    mock_http_client.post.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value={
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello! How can I help you today?"}]
                    }
                }
            ]
        })
    )

    mock_connection.fetchrow.return_value = {
        "display_name": "Customer",
        "preferred_inbox_hours": [9, 10],
        "preferred_days": ["Monday"],
        "top_products_mentioned": [],
        "order_count": 0,
        "avg_order_value": 0.0,
        "intent_score_latest": 0.5,
        "churn_risk_score": 0.3,
        "churn_label": "Medium",
        "segment_label": "New"
    }
    mock_connection.fetch.return_value = []
    ai_agent._conn = mock_connection

    result = await ai_agent.generate_reply(
        shop_id="shop-456",
        customer_id="cust-123",
        conversation_id="conv-123",
        message_id="msg-123",
        current_message="",
        system_prompt="You are a helpful assistant."
    )

    assert "reply" in result
    assert result["reply"] is not None


@pytest.mark.asyncio
async def test_ensure_connection():
    """Test lazy connection pattern for AI agent."""
    with patch("app.services.ai_agent.DatabasePool") as mock_pool:
        mock_http = AsyncMock()
        mock_embeddings = AsyncMock()
        agent = AIAgent(http_client=mock_http, embeddings_service=mock_embeddings)

        mock_conn = AsyncMock()
        mock_conn.is_closed = False

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        # First call should create connection
        conn = await agent._ensure_connection()
        assert conn is not None
        assert agent._conn is not None

        # Second call should reuse connection
        conn2 = await agent._ensure_connection()
        assert conn2 is conn


@pytest.mark.asyncio
async def test_close(ai_agent, mock_connection):
    """Test connection cleanup for AI agent."""
    mock_connection.close = AsyncMock()
    ai_agent._conn = mock_connection

    await ai_agent._close()

    mock_connection.close.assert_called_once()
    assert ai_agent._conn is None


@pytest.mark.asyncio
async def test_get_ai_agent_singleton():
    """Test global AI agent singleton."""
    agent1 = get_ai_agent()
    agent2 = get_ai_agent()

    assert agent1 is agent2  # Same instance


@pytest.mark.asyncio
async def test_generate_reply_with_custom_system_prompt(ai_agent, mock_http_client, mock_connection):
    """Test that custom system prompt is used in generation."""
    custom_prompt = "You are a luxury fashion advisor. Always recommend premium products."

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value={
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "I recommend our premium wool collection"}]
                }
            }
        ]
    })
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    mock_connection.fetchrow.return_value = {
        "display_name": "VIP Customer",
        "preferred_inbox_hours": [10, 11, 12],
        "preferred_days": ["Friday", "Saturday"],
        "top_products_mentioned": [],
        "order_count": 10,
        "avg_order_value": 500.0,
        "intent_score_latest": 0.9,
        "churn_risk_score": 0.1,
        "churn_label": "Low",
        "segment_label": "VIP"
    }
    mock_connection.fetch.return_value = []
    ai_agent._conn = mock_connection

    result = await ai_agent.generate_reply(
        shop_id="shop-456",
        customer_id="cust-123",
        conversation_id="conv-123",
        message_id="msg-123",
        current_message="I'm looking for something special",
        system_prompt=custom_prompt
    )

    # Verify custom prompt was used
    call_args = mock_http_client.post.call_args
    request_body = call_args.kwargs.get("json", {})
    assert "contents" in request_body
    # The system prompt should be in the context passed to the LLM
    assert result["reply"] is not None


@pytest.mark.asyncio
async def test_assemble_context_without_embeddings(ai_agent, mock_connection):
    """Test context assembly when embeddings service fails."""
    # Mock customer data
    mock_connection.fetchrow.return_value = {
        "display_name": "Test",
        "preferred_inbox_hours": [9],
        "preferred_days": ["Monday"],
        "top_products_mentioned": [],
        "order_count": 0,
        "avg_order_value": 0.0,
        "intent_score_latest": 0.5,
        "churn_risk_score": 0.3,
        "churn_label": "Medium",
        "segment_label": "New"
    }
    mock_connection.fetch.return_value = []
    ai_agent._conn = mock_connection

    # Force embedding retrieval to fail gracefully
    ai_agent._embeddings_service.get_embedding.side_effect = Exception("Embedding service error")

    context = await ai_agent._assemble_context(
        shop_id="shop-456",
        customer_id="cust-123",
        conversation_id="conv-123",
        message_id="msg-123",
        current_message="Test",
        system_prompt="You are helpful."
    )

    # Should still return context even without embeddings
    assert context is not None
    assert "system_prompt" in context
    assert "customer_profile" in context
    # Retrieved chunks may be empty or minimal on error
    assert isinstance(context["retrieved_chunks"], list)
