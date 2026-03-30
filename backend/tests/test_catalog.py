"""Tests for catalog import endpoints.

Tests the product catalog import functionality that:
- Parses CSV files with product data
- Stores products in the database
- Integrates with embeddings service for indexing
- Returns 200 immediately (within 50ms) for webhook-like operations
"""

# Set environment variables before importing app
import os
os.environ["SUPABASE_URL"] = "postgresql://test/test"
os.environ["SUPABASE_KEY"] = "test_key"
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "test_service_role_key"
os.environ["GEMINI_API_KEY"] = "test_gemini_key"
os.environ["MESSENGER_VERIFY_TOKEN"] = "test_verify_token"
os.environ["MESSENGER_APP_SECRET"] = "test_app_secret"

import pytest
import asyncio
import io
import csv as csv_module
from unittest.mock import AsyncMock, Mock, patch, MagicMock, PropertyMock
from uuid import uuid4
from datetime import datetime
from typing import AsyncGenerator, List, Dict, Any

from fastapi.testclient import TestClient
from fastapi import UploadFile

from app.main import app
from app.config import get_settings
from app.db.connection import override_get_db, reset_get_db_override
from app.routers.catalog import parse_csv_row, parse_csv_content, index_catalog_embeddings


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch('app.config.get_settings') as mock:
        settings = Mock()
        settings.supabase_url = "postgresql://test/test"
        settings.supabase_key = "test_key"
        settings.supabase_service_role_key = "test_service_role_key"
        settings.gemini_api_key = "test_gemini_key"
        settings.gemini_model = "gemini-2.0-flash-exp"
        settings.messenger_verify_token = "test_verify_token"
        settings.messenger_app_secret = "test_app_secret"
        settings.environment = "test"
        settings.log_level = "info"
        mock.return_value = settings
        yield settings


@pytest.fixture
def mock_db():
    """Mock database connection."""
    conn = AsyncMock()
    conn.execute = AsyncMock()
    conn.fetch = AsyncMock()
    conn.fetchval = AsyncMock()
    conn.fetchrow = AsyncMock()
    conn.executemany = AsyncMock()
    conn.is_closed = False
    return conn


@pytest.fixture(autouse=True)
def reset_overrides():
    """Reset overrides after each test."""
    yield
    reset_get_db_override()


@pytest.fixture
def client_with_db(mock_db):
    """Create test client with mocked database."""
    async def override_get_db_for_test():
        yield mock_db

    override_get_db(override_get_db_for_test)
    try:
        yield TestClient(app)
    finally:
        reset_get_db_override()


@pytest.fixture
def sample_shop_id():
    """Sample shop ID."""
    return uuid4()


class TestCSVParsing:
    """Test CSV row parsing functionality."""

    def test_parse_csv_row_valid(self):
        """Test parsing a valid CSV row."""
        row = {
            "id": "prod-123",
            "name": "Wool Jacket",
            "description": "A warm and stylish wool jacket",
            "price": "99.99",
            "tags": "winter,coat,outerwear"
        }

        product = parse_csv_row(row)

        assert product["id"] == "prod-123"
        assert product["name"] == "Wool Jacket"
        assert product["description"] == "A warm and stylish wool jacket"
        assert product["price"] == 99.99
        assert product["tags"] == ["winter", "coat", "outerwear"]

    def test_parse_csv_row_price_as_float(self):
        """Test parsing price that's already a float."""
        row = {
            "id": "prod-456",
            "name": "Cotton Shirt",
            "description": "A comfortable cotton shirt",
            "price": 29.99,
            "tags": "summer,casual"
        }

        product = parse_csv_row(row)

        assert product["price"] == 29.99
        assert isinstance(product["price"], float)

    def test_parse_csv_row_empty_tags(self):
        """Test parsing row with empty tags."""
        row = {
            "id": "prod-789",
            "name": "Plain T-Shirt",
            "description": "A basic t-shirt",
            "price": "15.00",
            "tags": ""
        }

        product = parse_csv_row(row)

        assert product["tags"] == []

    def test_parse_csv_row_missing_optional_fields(self):
        """Test parsing row with missing optional fields."""
        row = {
            "id": "prod-000",
            "name": "Simple Product",
            "description": "A simple product",
            "price": "10.00"
        }

        product = parse_csv_row(row)

        assert product["tags"] == []

    def test_parse_csv_row_invalid_price(self):
        """Test parsing row with invalid price."""
        row = {
            "id": "prod-invalid",
            "name": "Invalid Price Product",
            "description": "Product with invalid price",
            "price": "not_a_number",
            "tags": ""
        }

        # Should handle gracefully - return None or raise validation error
        product = parse_csv_row(row)
        # Invalid price should result in None
        assert product is None

    def test_parse_csv_row_missing_required_field(self):
        """Test parsing row with missing required field."""
        row = {
            "id": "prod-missing",
            "name": "Missing Field Product",
            "description": "Product missing price",
            "tags": ""
        }

        # Missing price should result in None
        product = parse_csv_row(row)
        assert product is None


class TestCSVContentParsing:
    """Test CSV content parsing functionality."""

    def test_parse_csv_content_valid(self):
        """Test parsing valid CSV content."""
        csv_content = """id,name,description,price,tags
prod-1,Wool Jacket,A warm jacket,99.99,winter,coat
prod-2,Cotton Shirt,A comfortable shirt,29.99,summer,casual"""

        products = parse_csv_content(csv_content)

        assert len(products) == 2
        assert products[0]["id"] == "prod-1"
        assert products[0]["name"] == "Wool Jacket"
        assert products[0]["price"] == 99.99
        assert products[1]["id"] == "prod-2"
        assert products[1]["name"] == "Cotton Shirt"

    def test_parse_csv_content_empty(self):
        """Test parsing empty CSV content."""
        csv_content = ""

        products = parse_csv_content(csv_content)

        assert products == []

    def test_parse_csv_content_headers_only(self):
        """Test parsing CSV with only headers."""
        csv_content = "id,name,description,price,tags"

        products = parse_csv_content(csv_content)

        assert products == []

    def test_parse_csv_content_with_invalid_rows(self):
        """Test parsing CSV with some invalid rows."""
        csv_content = """id,name,description,price,tags
prod-1,Valid Product,A valid product,99.99,tag1
prod-2,Invalid Product,Invalid price,not_a_number,tag2
prod-3,Another Valid,Another valid,49.99,tag3"""

        products = parse_csv_content(csv_content)

        # Should return 2 valid products, skipping the invalid one
        assert len(products) == 2
        assert products[0]["id"] == "prod-1"
        assert products[1]["id"] == "prod-3"

    def test_parse_csv_content_quoted_fields(self):
        """Test parsing CSV with quoted fields containing commas."""
        csv_content = '''id,name,description,price,tags
prod-1,"Jacket, Wool","A jacket with wool",99.99,winter
prod-2,"T-Shirt, Cotton","A cotton t-shirt",19.99,casual'''

        products = parse_csv_content(csv_content)

        assert len(products) == 2
        assert products[0]["name"] == "Jacket, Wool"
        assert products[0]["description"] == "A jacket with wool"


class TestCatalogImportEndpoint:
    """Test catalog import endpoint."""

    def test_import_csv_valid_format(self, client_with_db, mock_db):
        """Test importing CSV with valid format."""
        csv_content = """id,name,description,price,tags
prod-1,Wool Jacket,A warm jacket,99.99,winter,coat
prod-2,Cotton Shirt,A comfortable shirt,29.99,summer,casual"""

        # Mock database operations
        mock_db.execute.return_value = None
        mock_db.fetch.return_value = []  # No duplicate products

        csv_file = io.BytesIO(csv_content.encode('utf-8'))
        csv_file.name = "products.csv"

        response = client_with_db.post(
            "/catalog/import",
            files={"file": ("products.csv", csv_file, "text/csv")},
            params={"shop_id": str(uuid4())}
        )

        # Should return 200 immediately
        assert response.status_code == 200
        data = response.json()
        assert "indexed" in data
        assert "failed" in data
        assert "time_seconds" in data

    def test_import_csv_duplicate_handling(self, client_with_db, mock_db):
        """Test handling of duplicate product IDs."""
        csv_content = """id,name,description,price,tags
prod-1,Wool Jacket,A warm jacket,99.99,winter
prod-1,Cotton Shirt,A shirt (duplicate ID),29.99,casual"""

        # Mock database operations - product exists
        mock_db.fetchrow.return_value = {"id": "prod-1"}  # Product exists
        mock_db.execute.return_value = None

        csv_file = io.BytesIO(csv_content.encode('utf-8'))
        csv_file.name = "products.csv"

        response = client_with_db.post(
            "/catalog/import",
            files={"file": ("products.csv", csv_file, "text/csv")},
            params={"shop_id": str(uuid4())}
        )

        assert response.status_code == 200
        data = response.json()
        # Should handle duplicates gracefully

    def test_import_csv_missing_fields(self, client_with_db, mock_db):
        """Test CSV with missing required fields."""
        csv_content = """id,name,description,tags
prod-1,Wool Jacket,A warm jacket,winter"""

        mock_db.execute.return_value = None
        mock_db.fetch.return_value = []

        csv_file = io.BytesIO(csv_content.encode('utf-8'))
        csv_file.name = "products.csv"

        response = client_with_db.post(
            "/catalog/import",
            files={"file": ("products.csv", csv_file, "text/csv")},
            params={"shop_id": str(uuid4())}
        )

        assert response.status_code == 200
        data = response.json()
        # Should report failures for invalid rows

    def test_import_endpoint_no_data(self, client_with_db, mock_db):
        """Test importing an empty file."""
        csv_content = ""

        mock_db.execute.return_value = None
        mock_db.fetch.return_value = []

        csv_file = io.BytesIO(csv_content.encode('utf-8'))
        csv_file.name = "products.csv"

        response = client_with_db.post(
            "/catalog/import",
            files={"file": ("products.csv", csv_file, "text/csv")},
            params={"shop_id": str(uuid4())}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["indexed"] == 0

    def test_import_endpoint_json_format(self, client_with_db, mock_db):
        """Test importing products in JSON format."""
        json_content = """[
            {"id": "prod-1", "name": "Wool Jacket", "description": "A warm jacket", "price": 99.99, "tags": ["winter"]},
            {"id": "prod-2", "name": "Cotton Shirt", "description": "A comfortable shirt", "price": 29.99, "tags": ["summer"]}
        ]"""

        mock_db.execute.return_value = None
        mock_db.fetch.return_value = []

        json_file = io.BytesIO(json_content.encode('utf-8'))
        json_file.name = "products.json"

        response = client_with_db.post(
            "/catalog/import",
            files={"file": ("products.json", json_file, "application/json")},
            params={"shop_id": str(uuid4())}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["indexed"] == 2

    def test_import_endpoint_response_time(self, client_with_db, mock_db):
        """Test that import endpoint returns quickly (<50ms)."""
        import time

        csv_content = """id,name,description,price,tags
prod-1,Wool Jacket,A warm jacket,99.99,winter"""

        mock_db.execute.return_value = None
        mock_db.fetch.return_value = []

        csv_file = io.BytesIO(csv_content.encode('utf-8'))
        csv_file.name = "products.csv"

        start_time = time.perf_counter()
        response = client_with_db.post(
            "/catalog/import",
            files={"file": ("products.csv", csv_file, "text/csv")},
            params={"shop_id": str(uuid4())}
        )
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000

        assert response.status_code == 200
        # Should respond quickly even with background processing
        assert elapsed_ms < 50, f"Response time {elapsed_ms}ms exceeds 50ms threshold"


class TestEmbeddingIntegration:
    """Test embeddings service integration."""

    @pytest.mark.asyncio
    async def test_index_catalog_embeddings(self):
        """Test indexing products for embeddings."""
        products = [
            {
                "id": "prod-1",
                "name": "Wool Jacket",
                "description": "A warm and stylish wool jacket",
                "price": 99.99,
                "tags": ["winter", "coat"]
            },
            {
                "id": "prod-2",
                "name": "Cotton Shirt",
                "description": "A comfortable cotton shirt for summer",
                "price": 29.99,
                "tags": ["summer", "casual"]
            }
        ]
        shop_id = str(uuid4())

        call_count = []

        # Mock the embeddings service
        with patch('app.routers.catalog.get_embeddings_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service

            # Mock update_product_catalog_embeddings
            async def mock_update_embeddings(product_id, shop_id, chunks, api_key=None):
                call_count.append(product_id)
                return {"product_id": product_id, "chunks_stored": len(chunks), "errors": []}

            mock_service.update_product_catalog_embeddings = mock_update_embeddings

            await index_catalog_embeddings(products, shop_id)

            # Should have called update for each product
            assert len(call_count) == 2

    @pytest.mark.asyncio
    async def test_index_catalog_embeddings_chunking(self):
        """Test that product data is properly chunked for embeddings."""
        products = [
            {
                "id": "prod-chunk",
                "name": "Test Product with Long Description",
                "description": "This is a longer description that should be chunked properly for embedding generation. " * 5,
                "price": 99.99,
                "tags": ["tag1", "tag2", "tag3"]
            }
        ]
        shop_id = str(uuid4())

        chunks_created = []

        # Mock the embeddings service
        with patch('app.routers.catalog.get_embeddings_service') as mock_get_service:
            mock_service = AsyncMock()

            async def mock_update_embeddings(product_id, shop_id, chunks, api_key=None):
                chunks_created.extend(chunks)
                return {"product_id": product_id, "chunks_stored": len(chunks), "errors": []}

            mock_service.update_product_catalog_embeddings = mock_update_embeddings
            mock_get_service.return_value = mock_service

            await index_catalog_embeddings(products, shop_id)

            # Should have created multiple chunks (name, description, tags, variants info)
            assert len(chunks_created) > 1


class TestProductImportModels:
    """Test Pydantic models for product import."""

    def test_product_import_model_valid(self):
        """Test ProductImport model with valid data."""
        from app.routers.catalog import ProductImport

        product = ProductImport(
            id="prod-123",
            name="Wool Jacket",
            description="A warm jacket",
            price=99.99,
            tags=["winter", "coat"]
        )

        assert product.id == "prod-123"
        assert product.name == "Wool Jacket"
        assert product.price == 99.99
        assert product.tags == ["winter", "coat"]

    def test_product_import_model_optional_fields(self):
        """Test ProductImport model with optional fields."""
        from app.routers.catalog import ProductImport

        product = ProductImport(
            id="prod-123",
            name="Simple Product",
            description="A simple product",
            price=10.00
        )

        assert product.tags == []

    def test_product_import_response_model(self):
        """Test ProductImportResponse model."""
        from app.routers.catalog import ProductImportResponse

        response = ProductImportResponse(
            indexed=10,
            failed=2,
            time_seconds=0.045
        )

        assert response.indexed == 10
        assert response.failed == 2
        assert response.time_seconds == 0.045


class TestBackgroundProcessing:
    """Test background processing of embeddings."""

    @pytest.mark.asyncio
    async def test_background_embeddings_does_not_block(self, client_with_db, mock_db):
        """Test that embedding indexing runs in background and doesn't block response."""
        csv_content = """id,name,description,price,tags
prod-1,Wool Jacket,A warm jacket,99.99,winter"""

        mock_db.execute.return_value = None
        mock_db.fetch.return_value = []

        csv_file = io.BytesIO(csv_content.encode('utf-8'))
        csv_file.name = "products.csv"

        # Create mock that delays for embedding service
        with patch('app.routers.catalog.get_embeddings_service') as mock_get_service:
            mock_service = AsyncMock()

            async def delayed_update(*args, **kwargs):
                await asyncio.sleep(0.1)  # Simulate slow embedding generation
                return {"product_id": "prod-1", "chunks_stored": 1, "errors": []}

            mock_service.update_product_catalog_embeddings = delayed_update
            mock_get_service.return_value = mock_service

            import time
            start_time = time.perf_counter()

            response = client_with_db.post(
                "/catalog/import",
                files={"file": ("products.csv", csv_file, "text/csv")},
                params={"shop_id": str(uuid4())}
            )

            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time) * 1000

            # Response should return quickly even with delayed embedding service
            assert response.status_code == 200
            # Background processing should not block
            assert elapsed_ms < 50, f"Response took {elapsed_ms}ms, should be <50ms"
