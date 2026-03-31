# Sellora Backend API

AI-Powered Chat-Commerce Platform Backend Service

## Project Overview

**Sellora** is a multi-tenant SaaS platform that turns Facebook Messenger and WhatsApp into automated sales channels. Store owners connect their messaging accounts, configure their product catalog and store policy, and receive an AI agent that consults customers, recommends products, handles checkout, and re-engages buyers.

**Key Principle**: In chat-commerce, conversation = the customer journey. All analytics, scoring, and outreach decisions are derived exclusively from conversation data (messages, timestamps, conversation flow).

---

## Technology Stack

| Layer | Technology |
| --- | --- |
| API Framework | FastAPI |
| Database | PostgreSQL via Supabase with pgvector |
| Messaging APIs | Meta Messenger API + WhatsApp Business API |
| AI Model | Gemini 2.0 Flash |
| Embedding Model | Qwen3-Embedding-8B (pgvector) |
| HTTP Client | httpx |
| Background Jobs | Cron (initial) / Celery + Redis (scale) |
| Payments | Paddle (billing & checkout) |

---

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration and settings
│   ├── db/
│   │   ├── __init__.py
│   │   ├── connection.py     # Database connection pool
│   │   └── queries.py      # Database queries and enums
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── webhooks.py      # Messenger/WhatsApp webhook endpoints
│   │   └── catalog.py      # Product catalog import/listing
│   ├── services/
│   │   ├── __init__.py
│   │   ├── extractor.py     # Fast intent/signal extraction
│   │   ├── conversation.py  # Conversation state tracking
│   │   ├── embeddings.py    # Embedding generation/indexing
│   │   ├── ai_agent.py     # RAG-based AI agent
│   │   └── outreach.py     # Outreach campaign service
│   └── schemas/
│       ├── __init__.py
│       ├── message.py
│       ├── conversation.py
│       └── catalog.py
├── tests/
│   ├── conftest.py          # Shared pytest fixtures
│   ├── test_config.py
│   ├── test_schemas.py
│   ├── test_extractor.py
│   ├── test_conversation.py
│   ├── test_webhooks.py
│   ├── test_embeddings.py
│   ├── test_ai_agent.py
│   └── test_catalog.py
├── pyproject.toml           # Project configuration (single source of truth)
├── requirements.txt         # Generated from pyproject.toml (for deployment)
├── .env.example           # Environment variables template
└── README.md             # This file
```

---

## Installation

### Prerequisites

- Python 3.11 or higher
- A PostgreSQL database (via Supabase)
- A Gemini API key for AI operations

### Setup

1. Clone the repository and navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install the project in editable mode with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

   Or install from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy the environment variables template:
   ```bash
   cp .env.example .env
   ```

4. Configure your environment variables in `.env` (see below).

---

## Environment Configuration

Create a `.env` file based on `.env.example` with the following required variables:

```bash
# Supabase (PostgreSQL database)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# Google AI (Gemini)
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.0-flash-exp

# Messenger/WhatsApp (to be configured per shop)
MESSENGER_VERIFY_TOKEN=your-verify-token
MESSENGER_APP_SECRET=your-app-secret

# Application
ENVIRONMENT=development
LOG_LEVEL=info
```

---

## Running the Server

### Local Development

Start the server with auto-reload enabled:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Production

Start the server with multiple workers:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Direct Python

```bash
python -m app.main
```

---

## API Endpoints

### Health & Status

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/` | Root endpoint - returns API info |
| GET | `/health` | Health check - returns service status |

**Health Check Response:**
```json
{
  "status": "healthy",
  "service": "sellora-api"
}
```

### Webhooks

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/webhooks/messenger` | Verify Messenger webhook (Meta challenge) |
| POST | `/webhooks/messenger` | Receive Messenger webhook events |
| GET | `/webhooks/whatsapp` | Verify WhatsApp webhook (Phase 2) |
| POST | `/webhooks/whatsapp` | Receive WhatsApp webhook events (Phase 2) |

### Product Catalog

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/catalog/import` | Import product catalog (CSV/JSON) |
| GET | `/catalog/products` | List products for a shop |

---

## Architecture Notes

### Dual Processing Architecture

**Fast Pass** (Sync, <50ms):
- Keyword/rule-based Python extraction
- Intent type, product mentioned, drop-off flag
- Runs immediately on webhook receipt

**Enrichment Pass** (Async, within 60s):
- Optional Gemini classification
- Refined intent, sentiment, entity normalization
- Queued as background job

### Database Layer

- **Append-only messages**: Never modify `messages` table
- **Derived tables**: Computed from raw message data
- **Multi-tenancy**: All queries respect `shop_id` via RLS

### RAG Pipeline

Context assembly per message:
1. System prompt (~300 tokens)
2. Customer profile (~200 tokens)
3. Retrieved product chunks (~400 tokens)
4. Conversation history - last 20 messages (~600 tokens)
5. Current message (~100 tokens)

---

## Testing

Run all tests:

```bash
pytest -v
```

Run tests with coverage:

```bash
pytest --cov=app --cov-report=html
```

Run specific test file:

```bash
pytest tests/test_webhooks.py -v
```

Run specific test function:

```bash
pytest tests/test_extractor.py::TestFastExtractor::test_extract_intent_price_inquiry -v
```

### Test Fixtures

The `tests/conftest.py` file provides shared fixtures:

- `sample_message_data` - Sample message for testing
- `sample_messenger_webhook` - Sample webhook payload
- `sample_product_data` - Sample product data
- `mock_db` - Mocked database connection
- `test_client` - FastAPI test client
- `db_pool` - Mocked database pool

---

## Development Notes

### Key Architectural Patterns

1. **Dependency Injection**: Use FastAPI's dependency system for database connections
2. **Async/Await**: All I/O operations are asynchronous
3. **Type Hints**: Full type annotations using Python 3.11+ syntax
4. **Environment-based Config**: Settings loaded via `pydantic-settings`
5. **Append-only Design**: Messages never modified, only appended
6. **Deterministic Scoring**: All scoring logic in Python, never LLM

### Performance Requirements

- Webhook response: <50ms (critical path)
- AI enrichment: within 60s (background)
- Database queries: <100ms typical

### Error Handling

- HTTP exceptions return JSON with `detail` field
- Unexpected errors logged and return 500
- Background job failures logged but don't block responses

---

## Dependency Management

This project uses `pyproject.toml` as the **single source of truth** for dependencies.

### Installing Dependencies

**Recommended** (from pyproject.toml):
```bash
pip install -e ".[dev]"
```

**Alternative** (from requirements.txt):
```bash
pip install -r requirements.txt
```

### Adding Dependencies

Edit `pyproject.toml` directly, then regenerate `requirements.txt`:
```bash
pip freeze > requirements.txt
```

Or use pip-tools for better control:
```bash
pip-compile pyproject.toml --extra dev
```

---

## Deployment

### Environment Variables

Ensure all required environment variables are set in your deployment environment:
- `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_SERVICE_ROLE_KEY`
- `GEMINI_API_KEY`, `GEMINI_MODEL`
- `MESSENGER_VERIFY_TOKEN`, `MESSENGER_APP_SECRET`

### Database Migration

For Phase 1, tables are created via Supabase directly. Future versions will include Alembic migrations.

### Monitoring

- Health check endpoint: `/health`
- Logs: Configurable via `LOG_LEVEL` environment variable
- Error tracking: To be added with Sentry (Phase 4)

---

## License

[To be determined]

---

## Contributing

[To be determined]
