"""Embeddings service with pgvector support for product recommendations."""
import asyncio
import httpx
from typing import List, Dict, Optional, Any
import asyncpg

from app.config import get_settings
from app.db.connection import DatabasePool


class EmbeddingsService:
    """Service for generating and managing embeddings with pgvector.

    Supports:
    - Generating embeddings using Google GenerativeAI API
    - Storing embeddings in product_embeddings table
    - Vector similarity search for recommendations
    """

    # Default embedding dimension (text-embedding-004)
    EMBEDDING_DIM = 768

    # Google AI API endpoint
    GEMINI_EMBEDDING_URL = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"

    def __init__(self, http_client: Optional[httpx.AsyncClient] = None):
        """Initialize embeddings service.

        Args:
            http_client: Optional HTTP client for API calls.
        """
        self.http_client = http_client or httpx.AsyncClient(timeout=30.0)
        self._conn: Optional[asyncpg.Connection] = None
        self._settings = get_settings()

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

    async def get_embedding(self, text: str, api_key: Optional[str] = None) -> List[float]:
        """Generate embedding vector using Google GenerativeAI API.

        Args:
            text: Text to embed.
            api_key: Optional API key (uses default from settings if not provided).

        Returns:
            List of float values representing the embedding vector.
        """
        if not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.EMBEDDING_DIM

        key = api_key or self._settings.gemini_api_key

        try:
            response = await self.http_client.post(
                f"{self.GEMINI_EMBEDDING_URL}?key={key}",
                json={
                    "content": {
                        "parts": [{"text": text}]
                    },
                    "model": "models/text-embedding-004"
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["embedding"]["values"]
        except Exception as e:
            # In production, log and consider fallback
            # For now, return zero vector to prevent blocking
            return [0.0] * self.EMBEDDING_DIM

    async def store_embedding(
        self,
        embedding_id: str,
        shop_id: str,
        source_type: str,
        source_id: str,
        chunk_text: str,
        vector: List[float],
    ) -> str:
        """Store a single embedding in the database.

        Args:
            embedding_id: Unique identifier for the embedding.
            shop_id: Shop ID for multi-tenancy.
            source_type: Type of source (product, policy, faq, summary, campaign).
            source_id: ID of the source object.
            chunk_text: Text content that was embedded.
            vector: Embedding vector as list of floats.

        Returns:
            The embedding_id that was stored.
        """
        conn = await self._ensure_connection()

        # Convert vector to pgvector format
        vector_str = f"[{','.join(str(x) for x in vector)}]"

        await conn.execute(
            """
            INSERT INTO product_embeddings
            (embedding_id, shop_id, source_type, source_id, chunk_text, vector)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (embedding_id) DO UPDATE SET
                shop_id = EXCLUDED.shop_id,
                source_type = EXCLUDED.source_type,
                source_id = EXCLUDED.source_id,
                chunk_text = EXCLUDED.chunk_text,
                vector = EXCLUDED.vector
            """,
            embedding_id, shop_id, source_type, source_id, chunk_text, vector_str
        )

        return embedding_id

    async def batch_store_embeddings(
        self,
        embeddings_list: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Store multiple embeddings efficiently.

        Args:
            embeddings_list: List of dictionaries with keys:
                - embedding_id: str
                - shop_id: str
                - source_type: str
                - source_id: str
                - chunk_text: str
                - vector: List[float]

        Returns:
            Dictionary with success_count and failed_count.
        """
        conn = await self._ensure_connection()

        success_count = 0
        failed_count = 0

        try:
            # Try bulk insert first
            records = []
            for emb in embeddings_list:
                vector_str = f"[{','.join(str(x) for x in emb['vector'])}]"
                records.append((
                    emb['embedding_id'],
                    emb['shop_id'],
                    emb['source_type'],
                    emb['source_id'],
                    emb['chunk_text'],
                    vector_str
                ))

            await conn.executemany(
                """
                INSERT INTO product_embeddings
                (embedding_id, shop_id, source_type, source_id, chunk_text, vector)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (embedding_id) DO UPDATE SET
                    shop_id = EXCLUDED.shop_id,
                    source_type = EXCLUDED.source_type,
                    source_id = EXCLUDED.source_id,
                    chunk_text = EXCLUDED.chunk_text,
                    vector = EXCLUDED.vector
                """,
                records
            )
            success_count = len(embeddings_list)

        except Exception:
            # Fall back to individual inserts
            for emb in embeddings_list:
                try:
                    await self.store_embedding(
                        embedding_id=emb['embedding_id'],
                        shop_id=emb['shop_id'],
                        source_type=emb['source_type'],
                        source_id=emb['source_id'],
                        chunk_text=emb['chunk_text'],
                        vector=emb['vector']
                    )
                    success_count += 1
                except Exception:
                    failed_count += 1

        return {
            "success_count": success_count,
            "failed_count": failed_count
        }

    async def update_product_catalog_embeddings(
        self,
        product_id: str,
        shop_id: str,
        chunks: List[str],
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Store all product chunks as individual embedding rows.

        Args:
            product_id: Product ID to update embeddings for.
            shop_id: Shop ID for multi-tenancy.
            chunks: List of text chunks to embed (name, description, variants, etc.).
            api_key: Optional API key for embeddings.

        Returns:
            Dictionary with product_id, chunks_stored, and any errors.
        """
        key = api_key or self._settings.gemini_api_key

        # Delete existing embeddings for this product
        conn = await self._ensure_connection()
        await conn.execute(
            "DELETE FROM product_embeddings WHERE source_id = $1 AND source_type = 'product'",
            product_id
        )

        # Generate and store new embeddings
        chunks_stored = 0
        errors = []

        for idx, chunk in enumerate(chunks):
            try:
                vector = await self.get_embedding(chunk, api_key=key)
                embedding_id = f"{product_id}_chunk_{idx}"

                await self.store_embedding(
                    embedding_id=embedding_id,
                    shop_id=shop_id,
                    source_type="product",
                    source_id=product_id,
                    chunk_text=chunk,
                    vector=vector
                )
                chunks_stored += 1
            except Exception as e:
                errors.append(f"Chunk {idx}: {str(e)}")

        return {
            "product_id": product_id,
            "chunks_stored": chunks_stored,
            "errors": errors
        }

    async def cosine_similarity(
        self,
        query_vector: List[float],
        top_k: int = 3,
        shop_id: Optional[str] = None,
        source_type: Optional[str] = None,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Find similar items using cosine similarity.

        Args:
            query_vector: Query embedding vector.
            top_k: Maximum number of results to return.
            shop_id: Optional filter by shop ID.
            source_type: Optional filter by source type (product, policy, faq, etc.).
            min_similarity: Minimum similarity threshold (0-1).

        Returns:
            List of dictionaries containing embedding_id, source_type, source_id,
            chunk_text, and similarity score.
        """
        conn = await self._ensure_connection()

        # Convert query vector to pgvector format
        vector_str = f"[{','.join(str(x) for x in query_vector)}]"

        # Build WHERE clause
        where_conditions = ["1=1"]
        params = [vector_str, top_k]

        if shop_id:
            where_conditions.append("shop_id = $" + str(len(params) + 1))
            params.append(shop_id)

        if source_type:
            where_conditions.append("source_type = $" + str(len(params) + 1))
            params.append(source_type)

        where_clause = " AND ".join(where_conditions)

        # pgvector cosine similarity query
        query = f"""
            SELECT
                embedding_id,
                source_type,
                source_id,
                chunk_text,
                1 - (vector <=> $1::vector) as similarity
            FROM product_embeddings
            WHERE {where_clause}
            ORDER BY vector <=> $1::vector ASC
            LIMIT $2
        """

        rows = await conn.fetch(query, *params)

        # Filter by min_similarity and format results
        results = []
        for row in rows:
            similarity = float(row["similarity"])
            if similarity >= min_similarity:
                results.append({
                    "embedding_id": row["embedding_id"],
                    "source_type": row["source_type"],
                    "source_id": row["source_id"],
                    "chunk_text": row["chunk_text"],
                    "similarity": similarity
                })

        return results

    async def search_products(
        self,
        query_text: str,
        shop_id: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Search for products by semantic similarity.

        Args:
            query_text: Query text to embed and search.
            shop_id: Shop ID for multi-tenancy.
            top_k: Maximum number of products to return.

        Returns:
            List of unique products with aggregated similarity scores.
        """
        # Generate query embedding
        query_vector = await self.get_embedding(query_text)

        # Find similar chunks
        similar_chunks = await self.cosine_similarity(
            query_vector=query_vector,
            top_k=top_k * 3,  # Get more chunks to aggregate
            shop_id=shop_id,
            source_type="product"
        )

        # Aggregate by product_id (source_id)
        product_scores: Dict[str, Dict[str, Any]] = {}
        for chunk in similar_chunks:
            product_id = chunk["source_id"]
            if product_id not in product_scores:
                product_scores[product_id] = {
                    "product_id": product_id,
                    "similarity": 0.0,
                    "chunks": []
                }
            product_scores[product_id]["similarity"] = max(
                product_scores[product_id]["similarity"],
                chunk["similarity"]
            )
            product_scores[product_id]["chunks"].append(chunk["chunk_text"])

        # Sort by similarity and return top_k
        sorted_products = sorted(
            product_scores.values(),
            key=lambda x: x["similarity"],
            reverse=True
        )

        return sorted_products[:top_k]

    async def cleanup_product_embeddings(self, product_id: str) -> int:
        """Remove all embeddings for a product.

        Args:
            product_id: Product ID to clean up.

        Returns:
            Number of embeddings deleted.
        """
        conn = await self._ensure_connection()

        result = await conn.execute(
            "DELETE FROM product_embeddings WHERE source_id = $1 AND source_type = 'product'",
            product_id
        )

        # Parse result string to get count
        # Format: "DELETE <count>"
        return int(result.split()[-1]) if result else 0


# Global service instance
_service: Optional[EmbeddingsService] = None


def get_embeddings_service() -> EmbeddingsService:
    """Get or create global embeddings service instance."""
    global _service
    if _service is None:
        _service = EmbeddingsService()
    return _service
