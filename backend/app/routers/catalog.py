"""Catalog import endpoints for product catalog management.

This module provides endpoints for importing product catalogs via CSV or JSON files.
The import process:
1. Parses the uploaded file (CSV or JSON)
2. Stores products in the products table
3. Triggers background embedding generation for RAG
4. Returns 200 immediately (within 50ms) with processing results

Phase 1: Products stored in products table, embeddings indexed in background.
Phase 2: Dedicated product_embeddings table will replace products table.
"""

import asyncio
import csv as csv_module
import io
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from uuid import UUID

from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends, Query, Form
from pydantic import BaseModel, Field, field_validator

from app.db.connection import get_db
from app.services.embeddings import get_embeddings_service


logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# Pydantic Models
# ============================================================================

class ProductImport(BaseModel):
    """Schema for a single product during import."""
    id: str = Field(..., description="Product unique identifier")
    name: str = Field(..., min_length=1, max_length=255, description="Product name")
    description: str = Field(..., min_length=1, max_length=2000, description="Product description")
    price: float = Field(..., gt=0, description="Product price")
    tags: List[str] = Field(default_factory=list, description="Product tags/categories")

    @field_validator('tags', mode='before')
    @classmethod
    def parse_tags(cls, v):
        """Parse tags from string or list."""
        if isinstance(v, str):
            # Split by comma and strip whitespace
            return [tag.strip() for tag in v.split(',') if tag.strip()]
        return v or []


class ProductImportResponse(BaseModel):
    """Response from product import operation."""
    indexed: int = Field(..., ge=0, description="Number of products successfully indexed")
    failed: int = Field(..., ge=0, description="Number of products that failed to import")
    time_seconds: float = Field(..., ge=0, description="Time taken for import in seconds")


# ============================================================================
# CSV Parsing Functions
# ============================================================================

def parse_csv_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse a single CSV row into a product dict.

    Args:
        row: Dictionary from CSV reader with column headers as keys

    Returns:
        Product dict with parsed values, or None if row is invalid
    """
    try:
        product_id = row.get("id", "")
        name = row.get("name", "")
        description = row.get("description", "")
        price_value = row.get("price")
        tags_value = row.get("tags", "")

        # Convert to strings for validation
        product_id_str = str(product_id).strip() if product_id is not None else ""
        name_str = str(name).strip() if name is not None else ""
        description_str = str(description).strip() if description is not None else ""

        # Validate required fields
        if not product_id_str or not name_str or not description_str or price_value is None:
            return None

        # Parse price
        try:
            if isinstance(price_value, (int, float)):
                price = float(price_value)
            else:
                price_str = str(price_value).replace('$', '').replace(',', '').strip()
                price = float(price_str)
        except (ValueError, AttributeError):
            return None

        # Validate price
        if price <= 0:
            return None

        # Parse tags
        tags_str = str(tags_value).strip() if tags_value is not None else ""
        tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()] if tags_str else []

        return {
            "id": product_id_str,
            "name": name_str,
            "description": description_str,
            "price": price,
            "tags": tags
        }
    except Exception as e:
        logger.debug(f"Error parsing CSV row: {e}")
        return None


def parse_csv_content(content: str) -> List[Dict[str, Any]]:
    """Parse CSV content string into list of product dicts.

    Args:
        content: Raw CSV content as string

    Returns:
        List of valid product dicts (invalid rows are skipped)
    """
    if not content or not content.strip():
        return []

    products = []
    reader = csv_module.DictReader(io.StringIO(content))

    for row in reader:
        product = parse_csv_row(row)
        if product:
            products.append(product)

    return products


def parse_json_content(content: str) -> List[Dict[str, Any]]:
    """Parse JSON content string into list of product dicts.

    Args:
        content: Raw JSON content as string

    Returns:
        List of valid product dicts (invalid items are skipped)
    """
    import json

    if not content or not content.strip():
        return []

    try:
        data = json.loads(content)
        if not isinstance(data, list):
            return []

        products = []
        for item in data:
            try:
                # Create ProductImport model to validate
                product = ProductImport(**item)
                products.append(product.model_dump())
            except Exception as e:
                logger.debug(f"Error parsing JSON item: {e}")
                continue

        return products
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return []


# ============================================================================
# Embedding Indexing (Background)
# ============================================================================

async def index_catalog_embeddings(products: List[Dict[str, Any]], shop_id: str) -> None:
    """Index products in embeddings store for RAG.

    Runs in background to avoid blocking the import response.

    Args:
        products: List of product dicts with id, name, description, price, tags
        shop_id: Shop ID for multi-tenancy
    """
    if not products:
        return

    service = get_embeddings_service()

    for product in products:
        try:
            product_id = product["id"]
            name = product["name"]
            description = product["description"]
            tags = product.get("tags", [])
            price = product.get("price", 0)

            # Build chunks for embedding (name, description, tags, price info)
            chunks = [
                f"{name}: {description}",
                f"Price: ${price:.2f}",
                f"Tags: {', '.join(tags)}" if tags else "No tags"
            ]

            # Add tag-specific chunks for better matching
            for tag in tags:
                chunks.append(f"Category: {tag}")

            await service.update_product_catalog_embeddings(
                product_id=product_id,
                shop_id=shop_id,
                chunks=chunks
            )

            logger.info(f"Indexed product {product_id} for embeddings")

        except Exception as e:
            logger.error(f"Error indexing product {product.get('id')}: {e}")


# ============================================================================
# Database Operations
# ============================================================================

async def store_products(
    db,
    shop_id: UUID,
    products: List[Dict[str, Any]]
) -> Dict[str, int]:
    """Store products in database.

    Args:
        db: Database connection
        shop_id: Shop ID for multi-tenancy
        products: List of product dicts to store

    Returns:
        Dict with indexed count and failed count
    """
    indexed = 0
    failed = 0

    for product in products:
        try:
            product_id = product["id"]

            # Check for duplicate
            existing = await db.fetchrow(
                "SELECT id FROM products WHERE shop_id = $1 AND id = $2",
                shop_id,
                product_id
            )

            if existing:
                # Update existing product
                await db.execute(
                    """
                    UPDATE products
                    SET name = $1, description = $2, price = $3, tags = $4, updated_at = $5
                    WHERE shop_id = $6 AND id = $7
                    """,
                    product["name"],
                    product["description"],
                    product["price"],
                    product.get("tags", []),
                    datetime.now(timezone.utc),
                    shop_id,
                    product_id
                )
            else:
                # Insert new product
                await db.execute(
                    """
                    INSERT INTO products (id, shop_id, name, description, price, tags, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (shop_id, id) DO UPDATE SET
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        price = EXCLUDED.price,
                        tags = EXCLUDED.tags,
                        updated_at = EXCLUDED.updated_at
                    """,
                    product_id,
                    shop_id,
                    product["name"],
                    product["description"],
                    product["price"],
                    product.get("tags", []),
                    datetime.now(timezone.utc),
                    datetime.now(timezone.utc)
                )

            indexed += 1

        except Exception as e:
            logger.error(f"Error storing product {product.get('id')}: {e}")
            failed += 1

    return {"indexed": indexed, "failed": failed}


# ============================================================================
# Import Endpoint
# ============================================================================

@router.post("/import", response_model=ProductImportResponse)
async def import_catalog(
    file: UploadFile = File(..., description="CSV or JSON file with product data"),
    shop_id: str = Query(..., description="Shop ID for multi-tenancy"),
    db=Depends(get_db)
) -> ProductImportResponse:
    """Import product catalog from CSV or JSON file.

    The endpoint:
    1. Parses the uploaded file (CSV or JSON)
    2. Stores products in the database synchronously
    3. Queues background embedding generation
    4. Returns 200 within 50ms with import results

    CSV Format:
        id,name,description,price,tags
        prod-1,Wool Jacket,A warm jacket,99.99,winter,coat

    JSON Format:
        [
            {"id": "prod-1", "name": "Wool Jacket", "description": "A warm jacket", "price": 99.99, "tags": ["winter"]},
            ...
        ]

    Args:
        file: Uploaded CSV or JSON file
        shop_id: Shop ID for multi-tenancy
        db: Database connection

    Returns:
        ProductImportResponse with counts and timing
    """
    import time
    start_time = time.perf_counter()

    # Validate shop_id is a valid UUID
    try:
        shop_uuid = UUID(shop_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid shop_id format"
        )

    # Read file content
    content_bytes = await file.read()
    content = content_bytes.decode('utf-8')

    # Parse based on content type
    filename = file.filename or ""
    content_type = file.content_type or ""

    if "json" in filename or "json" in content_type:
        products = parse_json_content(content)
    else:
        # Default to CSV
        products = parse_csv_content(content)

    if not products:
        return ProductImportResponse(
            indexed=0,
            failed=0,
            time_seconds=(time.perf_counter() - start_time)
        )

    # Store products in database (synchronous)
    result = await store_products(db, shop_uuid, products)

    # Trigger background embedding indexing (non-blocking)
    asyncio.create_task(index_catalog_embeddings(products, shop_id))

    elapsed = time.perf_counter() - start_time

    logger.info(
        f"Catalog import complete: {result['indexed']} indexed, "
        f"{result['failed']} failed for shop {shop_id}"
    )

    return ProductImportResponse(
        indexed=result["indexed"],
        failed=result["failed"],
        time_seconds=elapsed
    )


@router.get("/products", response_model=List[ProductImport])
async def list_products(
    shop_id: str = Query(..., description="Shop ID for multi-tenancy"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of products to return"),
    db=Depends(get_db)
) -> List[ProductImport]:
    """List products for a shop.

    Args:
        shop_id: Shop ID for multi-tenancy
        limit: Maximum number of products to return
        db: Database connection

    Returns:
        List of products
    """
    # Validate shop_id is a valid UUID
    try:
        shop_uuid = UUID(shop_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid shop_id format"
        )

    rows = await db.fetch(
        """
        SELECT id, name, description, price, tags
        FROM products
        WHERE shop_id = $1
        ORDER BY created_at DESC
        LIMIT $2
        """,
        shop_uuid,
        limit
    )

    return [
        ProductImport(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            price=row["price"],
            tags=row["tags"] or []
        )
        for row in rows
    ]


@router.get("/products/{product_id}", response_model=ProductImport)
async def get_product(
    product_id: str,
    shop_id: str = Query(..., description="Shop ID for multi-tenancy"),
    db=Depends(get_db)
) -> ProductImport:
    """Get a single product by ID.

    Args:
        product_id: Product ID
        shop_id: Shop ID for multi-tenancy
        db: Database connection

    Returns:
        Product details
    """
    # Validate shop_id is a valid UUID
    try:
        shop_uuid = UUID(shop_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid shop_id format"
        )

    row = await db.fetchrow(
        """
        SELECT id, name, description, price, tags
        FROM products
        WHERE shop_id = $1 AND id = $2
        """,
        shop_uuid,
        product_id
    )

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found"
        )

    return ProductImport(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        price=row["price"],
        tags=row["tags"] or []
    )


@router.delete("/products/{product_id}")
async def delete_product(
    product_id: str,
    shop_id: str = Query(..., description="Shop ID for multi-tenancy"),
    db=Depends(get_db)
) -> Dict[str, str]:
    """Delete a product and its embeddings.

    Args:
        product_id: Product ID to delete
        shop_id: Shop ID for multi-tenancy
        db: Database connection

    Returns:
        Deletion confirmation
    """
    # Validate shop_id is a valid UUID
    try:
        shop_uuid = UUID(shop_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid shop_id format"
        )

    # Delete product from database
    result = await db.execute(
        """
        DELETE FROM products
        WHERE shop_id = $1 AND id = $2
        """,
        shop_uuid,
        product_id
    )

    # Cleanup embeddings in background
    service = get_embeddings_service()
    asyncio.create_task(service.cleanup_product_embeddings(product_id))

    return {"status": "deleted", "product_id": product_id}
