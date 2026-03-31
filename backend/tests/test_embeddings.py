"""Tests for embeddings service with pgvector support."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import numpy as np

from app.services.embeddings import EmbeddingsService


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for Google AI API."""
    mock_instance = AsyncMock()
    mock_instance.post = AsyncMock()
    return mock_instance


@pytest.fixture
def mock_connection():
    """Mock asyncpg connection."""
    conn = AsyncMock(spec_set=['execute', 'fetch', 'fetchval', 'fetchrow', 'is_closed', 'close', 'executemany'])
    conn.execute = AsyncMock()
    conn.fetch = AsyncMock()
    conn.fetchval = AsyncMock()
    conn.fetchrow = AsyncMock()
    conn.close = AsyncMock()
    conn.executemany = AsyncMock()
    # Add is_closed property mock (not callable)
    type(conn).is_closed = PropertyMock(return_value=False)
    return conn


@pytest.fixture
def embeddings_service(mock_http_client):
    """Create embeddings service instance."""
    service = EmbeddingsService(http_client=mock_http_client)
    yield service
    # Cleanup
    if hasattr(service, "_conn") and service._conn is not None:
        import asyncio
        if asyncio.iscoroutinefunction(service._conn.close):
            asyncio.run(service._conn.close())


@pytest.mark.asyncio
async def test_get_embedding_generation(embeddings_service, mock_http_client):
    """Test embedding generation with mocked Google AI."""
    # Mock Google AI API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value={
        "embedding": {
            "values": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
    })
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    vector = await embeddings_service.get_embedding("test text")

    assert len(vector) == 5
    assert all(isinstance(x, float) for x in vector)
    mock_http_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_store_embedding(embeddings_service, mock_connection):
    """Test storing embeddings in database."""
    mock_connection.execute.return_value = None
    embeddings_service._conn = mock_connection

    embedding_id = await embeddings_service.store_embedding(
        embedding_id="test-id-1",
        shop_id="shop-123",
        source_type="product",
        source_id="prod-456",
        chunk_text="A beautiful wool jacket",
        vector=[0.1, 0.2, 0.3]
    )

    assert embedding_id == "test-id-1"
    mock_connection.execute.assert_called_once()
    # Verify pgvector format
    call_args = mock_connection.execute.call_args
    call_str = str(call_args)
    assert "[0.1,0.2,0.3]" in call_str


@pytest.mark.asyncio
async def test_cosine_similarity(embeddings_service, mock_connection):
    """Test cosine similarity calculation."""
    # Mock database response for pgvector query
    mock_connection.fetch.return_value = [
        {
            "embedding_id": "emb-1",
            "source_type": "product",
            "source_id": "prod-1",
            "chunk_text": "Similar product",
            "similarity": 0.95
        },
        {
            "embedding_id": "emb-2",
            "source_type": "product",
            "source_id": "prod-2",
            "chunk_text": "Less similar",
            "similarity": 0.75
        }
    ]
    embeddings_service._conn = mock_connection

    results = await embeddings_service.cosine_similarity(
        query_vector=[0.1, 0.2, 0.3],
        top_k=2
    )

    assert len(results) == 2
    assert results[0]["similarity"] >= results[1]["similarity"]
    mock_connection.fetch.assert_called_once()


@pytest.mark.asyncio
async def test_batch_store_embeddings(embeddings_service, mock_connection):
    """Test batch embedding storage."""
    mock_connection.executemany.return_value = None
    embeddings_service._conn = mock_connection

    embeddings_list = [
        {
            "embedding_id": "emb-1",
            "shop_id": "shop-123",
            "source_type": "product",
            "source_id": "prod-1",
            "chunk_text": "Chunk 1",
            "vector": [0.1, 0.2, 0.3]
        },
        {
            "embedding_id": "emb-2",
            "shop_id": "shop-123",
            "source_type": "product",
            "source_id": "prod-1",
            "chunk_text": "Chunk 2",
            "vector": [0.4, 0.5, 0.6]
        }
    ]

    result = await embeddings_service.batch_store_embeddings(embeddings_list)

    assert result["success_count"] == 2
    assert result["failed_count"] == 0
    mock_connection.executemany.assert_called_once()


@pytest.mark.asyncio
async def test_update_product_catalog_embeddings(embeddings_service, mock_connection, mock_http_client):
    """Test product catalog embedding updates."""
    # Mock embedding API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value={
        "embedding": {"values": [0.1, 0.2, 0.3]}
    })
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    mock_connection.execute.return_value = None
    embeddings_service._conn = mock_connection

    chunks = [
        "Wool Jacket - A warm and stylish wool jacket",
        "Wool Jacket - Available in sizes S, M, L, XL"
    ]

    result = await embeddings_service.update_product_catalog_embeddings(
        product_id="prod-123",
        shop_id="shop-456",
        chunks=chunks
    )

    assert result["product_id"] == "prod-123"
    assert result["chunks_stored"] == 2
    assert mock_http_client.post.call_count == 2


@pytest.mark.asyncio
async def test_ensure_connection():
    """Test lazy connection pattern."""
    with patch("app.services.embeddings.DatabasePool") as mock_pool:
        mock_http_client = AsyncMock()
        service = EmbeddingsService(http_client=mock_http_client)

        mock_conn = AsyncMock()
        mock_conn.is_closed = False

        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        # First call should create connection
        conn = await service._ensure_connection()
        assert conn is not None
        assert service._conn is not None

        # Second call should reuse connection
        conn2 = await service._ensure_connection()
        assert conn2 is conn


@pytest.mark.asyncio
async def test_close(embeddings_service, mock_connection):
    """Test connection cleanup."""
    mock_connection.close = AsyncMock()
    embeddings_service._conn = mock_connection

    await embeddings_service._close()

    mock_connection.close.assert_called_once()
    assert embeddings_service._conn is None


@pytest.mark.asyncio
async def test_get_embedding_empty_text(embeddings_service):
    """Test embedding generation with empty text."""
    vector = await embeddings_service.get_embedding("")

    # Empty text should still return a vector of correct dimension
    assert vector is not None
    assert len(vector) == 768  # Default embedding dimension
    assert all(x == 0.0 for x in vector)


@pytest.mark.asyncio
async def test_cosine_similarity_with_filters(embeddings_service, mock_connection):
    """Test cosine similarity with source type filtering."""
    mock_connection.fetch.return_value = [
        {
            "embedding_id": "emb-1",
            "source_type": "policy",
            "source_id": "policy-1",
            "chunk_text": "Policy text",
            "similarity": 0.88
        }
    ]
    embeddings_service._conn = mock_connection

    results = await embeddings_service.cosine_similarity(
        query_vector=[0.1, 0.2, 0.3],
        top_k=5,
        source_type="policy"
    )

    assert len(results) == 1
    assert results[0]["source_type"] == "policy"
    mock_connection.fetch.assert_called_once()


@pytest.mark.asyncio
async def test_cosine_similarity_with_shop_filter(embeddings_service, mock_connection):
    """Test cosine similarity with shop ID filtering."""
    mock_connection.fetch.return_value = [
        {
            "embedding_id": "emb-1",
            "source_type": "product",
            "source_id": "prod-1",
            "chunk_text": "Product text",
            "similarity": 0.92
        }
    ]
    embeddings_service._conn = mock_connection

    results = await embeddings_service.cosine_similarity(
        query_vector=[0.1, 0.2, 0.3],
        top_k=5,
        shop_id="shop-123"
    )

    assert len(results) == 1
    mock_connection.fetch.assert_called_once()


@pytest.mark.asyncio
async def test_batch_store_embeddings_with_partial_failure(embeddings_service, mock_connection):
    """Test batch storage with some failures."""
    # Setup executemany to fail first
    mock_connection.executemany.side_effect = Exception("Batch error")
    # For fallback to individual inserts - first succeeds, second fails
    execute_results = [None, Exception("Database error")]
    mock_connection.execute = AsyncMock(side_effect=execute_results)
    embeddings_service._conn = mock_connection

    embeddings_list = [
        {
            "embedding_id": "emb-1",
            "shop_id": "shop-123",
            "source_type": "product",
            "source_id": "prod-1",
            "chunk_text": "Chunk 1",
            "vector": [0.1, 0.2, 0.3]
        },
        {
            "embedding_id": "emb-2",
            "shop_id": "shop-123",
            "source_type": "product",
            "source_id": "prod-1",
            "chunk_text": "Chunk 2",
            "vector": [0.4, 0.5, 0.6]
        }
    ]

    result = await embeddings_service.batch_store_embeddings(embeddings_list)

    # Should have some results even with partial failure
    assert "success_count" in result
    assert "failed_count" in result
    assert result["success_count"] == 1
    assert result["failed_count"] == 1


@pytest.mark.asyncio
async def test_search_products(embeddings_service, mock_connection, mock_http_client):
    """Test product search functionality."""
    # Mock embedding API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value={
        "embedding": {"values": [0.1, 0.2, 0.3]}
    })
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    # Mock database response
    mock_connection.fetch.return_value = [
        {
            "embedding_id": "emb-1",
            "source_type": "product",
            "source_id": "prod-1",
            "chunk_text": "Wool Jacket",
            "similarity": 0.92
        },
        {
            "embedding_id": "emb-2",
            "source_type": "product",
            "source_id": "prod-1",
            "chunk_text": "Available in red, blue",
            "similarity": 0.85
        },
        {
            "embedding_id": "emb-3",
            "source_type": "product",
            "source_id": "prod-2",
            "chunk_text": "Cotton Shirt",
            "similarity": 0.78
        }
    ]
    embeddings_service._conn = mock_connection

    results = await embeddings_service.search_products(
        query_text="warm jacket",
        shop_id="shop-123",
        top_k=2
    )

    assert len(results) == 2
    assert results[0]["product_id"] == "prod-1"
    assert results[0]["similarity"] == 0.92
    assert len(results[0]["chunks"]) == 2


@pytest.mark.asyncio
async def test_cleanup_product_embeddings(embeddings_service, mock_connection):
    """Test cleanup of product embeddings."""
    mock_connection.execute.return_value = "DELETE 3"
    embeddings_service._conn = mock_connection

    count = await embeddings_service.cleanup_product_embeddings("prod-123")

    assert count == 3
    mock_connection.execute.assert_called_once()


@pytest.mark.asyncio
async def test_get_global_embeddings_service():
    """Test global embeddings service singleton."""
    from app.services.embeddings import get_embeddings_service

    service1 = get_embeddings_service()
    service2 = get_embeddings_service()

    assert service1 is service2  # Same instance


@pytest.mark.asyncio
async def test_cosine_similarity_min_similarity_filter(embeddings_service, mock_connection):
    """Test cosine similarity with minimum similarity filter."""
    # Return results with varying similarity scores
    mock_connection.fetch.return_value = [
        {
            "embedding_id": "emb-1",
            "source_type": "product",
            "source_id": "prod-1",
            "chunk_text": "High similarity",
            "similarity": 0.90
        },
        {
            "embedding_id": "emb-2",
            "source_type": "product",
            "source_id": "prod-2",
            "chunk_text": "Low similarity",
            "similarity": 0.50
        }
    ]
    embeddings_service._conn = mock_connection

    results = await embeddings_service.cosine_similarity(
        query_vector=[0.1, 0.2, 0.3],
        top_k=5,
        min_similarity=0.75
    )

    assert len(results) == 1
    assert results[0]["similarity"] == 0.90
