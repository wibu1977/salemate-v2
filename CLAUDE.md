# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Sellora** is a multi-tenant SaaS platform that turns Facebook Messenger and WhatsApp into automated sales channels. Store owners connect their messaging accounts, configure their product catalog and store policy, and receive an AI agent that consults customers, recommends products, handles checkout, and re-engages buyers.

**Key Principle**: In chat-commerce, conversation = the customer journey. All analytics, scoring, and outreach decisions are derived exclusively from conversation data (messages, timestamps, conversation flow).

---

## Technology Stack

| Layer | Technology |
| --- | --- |
| Backend API | Python + FastAPI |
| Frontend Dashboard | Next.js (Vercel) |
| Database | PostgreSQL via Supabase — single project, multi-tenant RLS |
| Messaging APIs | Meta Messenger API + WhatsApp Business API |
| AI Model | Gemini 2.0 Flash — classify, extract entities, generate replies only |
| Embedding Model | Qwen3-Embedding-8B (pgvector) |
| Background Jobs | Cron (initial) / Celery + Redis (scale) |
| Payments | Paddle (billing & checkout) |

---

## Architecture: Strict LLM Role Boundary

| What the LLM DOES | What the LLM MUST NEVER DO |
| --- | --- |
| Classify message intent (enhancement to rules) | Compute intent\_score or any sub-feature (Q, E, P, F, D) |
| Extract entity names (products, variants, prices) | Make final outreach decisions |
| Generate personalized replies and outreach drafts | Modify any numeric field in the database |
| Explain scores in plain language | Run synchronously in the message-response critical path |

**Dual Processing Architecture**:
1. **Fast Pass** (Sync, <50ms): Keyword/rule-based Python extraction — intent\_type, product\_mentioned, drop\_off\_flag
2. **Enrichment Pass** (Async, within 60s): Optional Gemini classification — refined intent, sentiment, entity normalization

---

## Database Schema (Multi-Tenancy)

All tables include `shop_id` (UUID, non-nullable, indexed). RLS policies restrict all reads/writes to rows where `shop_id` matches the authenticated session claim.

### Core Tables

| Table | Purpose | Key Notes |
| --- | --- | --- |
| `shops` | Tenant configuration | Stores intent\_weights ({w\_Q, w\_E, w\_P, w\_F, w\_D}), ai\_system\_prompt, channels, plan\_tier |
| `messages` | Raw chat log | **Append-only**. Never modified. All derived tables computed from this. Indexes: `shop_id`, `conversation_id`, `sent_at`, `day_of_week`, `hour_of_day` |
| `conversations` | Thread-level aggregation | conversation\_stage, intent\_score, drop\_off\_flag, products\_mentioned, resulted\_in\_order |
| `extracted_signals` | Intelligence layer | Two-pass write: fast extractor populates rule-based fields, AI enricher updates refined fields. Core input for all scoring |
| `customers` | Customer profiles | preferred\_inbox\_hours[], preferred\_days[], top\_products\_mentioned[], churn\_risk\_score, priority\_score |
| `user_metrics` | Pre-computed analytics | Written nightly. Inputs for scoring formulas. Separated from customers table for historical snapshots |
| `orders` | Order tracking | Links to conversation\_id. Supports paddle, shopify, woocommerce, payment\_link |
| `product_embeddings` | RAG vector store | pgvector(768). ivfflat cosine index. Stores product/policy/faq/summary chunks |
| `outreach_campaigns` | Outreach drafts | status=pending\_approval | approved | rejected | sent | replied | ignored |
| `decision_log` | Audit trail | Every trigger evaluation logged. trigger\_type, threshold\_crossed, action\_taken |

---

## Message Flow: Webhook to Reply

1. Meta/WhatsApp webhook POST → FastAPI endpoint
2. Write raw message to `messages` (<5ms)
3. Fast extractor runs sync: keyword scan → writes `extracted_signals` (<50ms total)
4. Update `conversations`: message\_count++, last\_message\_at, recompute conversation\_stage
5. RAG agent assembles context: system\_prompt + customer\_profile\_block + retrieved chunks + last 20 messages
6. Gemini generates reply → written to `messages`, sent via API
7. Background: AI enricher job queued (within 60s). Updates `extracted_signals` with refined classification

---

## Scoring Formulas (Deterministic, Never LLM)

### Intent Score (Replaces v2 RFM)
```
intent_score = 0.30×Q + 0.25×E + 0.20×P + 0.15×F + 0.10×D
```

All features normalized 0-1 across shop customers via min-max scaling (computed nightly).

| Feature | Name | Input | Meaning |
| --- | --- | --- | --- |
| Q | Query intent | avg\_intent\_strength\_7d × (1 + purchase\_intent\_ratio\_7d) | Purchase-related question strength |
| E | Engagement | (avg\_conversation\_depth\_7d / max) × (1 − norm(avg\_response\_speed\_7d)) | Deep exchanges + fast replies |
| P | Product interest | (unique\_products\_mentioned\_7d + product\_repeat\_max\_7d) / normalizer | Returns to specific products |
| F | Frequency | conversations\_last\_7d / max | Multiple recent threads |
| D | Drop-off | −(drop\_off\_count\_7d / conversations\_last\_7d) | Penalty applied as subtraction |

**Constraint**: All feature values read from `user_metrics` (pre-computed nightly). Scoring function is pure Python — no DB reads or API calls.

### Churn Risk Score
```
churn_score = w1×norm(days_since_last_contact)
            + w2×norm(1 / interaction_frequency_30d)
            + w3×norm(1 / max(activity_trend, 0.01))
```

Default weights: w1=0.50, w2=0.30, w3=0.20

### Priority Score
```
priority_score = churn_risk_score × expected_value
expected_value = avg_order_value × purchase_likelihood
purchase_likelihood = 1 / (1 + exp(−5 × (intent_score_latest − churn_score)))
```

---

## Trigger Engine (Hourly)

### Trigger Catalogue

| Trigger | Condition | Priority Modifier | Outreach Type |
| --- | --- | --- | --- |
| price\_inquiry\_dropoff | stage=intent AND drop\_off\_flag=true AND ≤24h since drop\_off | ×2.0 | Follow-up offer / urgency |
| repeated\_product\_view | product\_repeat\_max\_7d ≥ 3 AND no order in 7 days | ×1.5 | Recommendation / social proof |
| high\_intent\_score | intent\_score > 0.70 (edge trigger) | ×1.3 | Conversion push |
| drop\_off\_reengagement | drop\_off\_flag=true AND 24h–72h since drop\_off | ×1.0 | Re-engagement |
| promo\_match | product in top\_products\_mentioned AND product on promotion | ×1.3 | Promotional offer |
| post\_purchase\_followup | order\_completed 7 days ago, no subsequent message | ×0.8 | Upsell / review request |
| churn\_threshold | churn\_score crosses 0.70 (edge trigger) | ×1.0 | Win-back |

### Fatigue Controls

| Control | Default | Configurable? |
| --- | --- | --- |
| Weekly outreach cap | 2 messages/customer/week | Yes (shops.outreach\_weekly\_cap) |
| Cooldown period | 48h between sends | Yes (shops.outreach\_cooldown\_hours) |
| Pending suppression | No draft while one pending\_approval | Always enforced |
| Post-send silence | No re-evaluation for 24h after send | Always enforced |

---

## RAG Pipeline

### What Gets Embedded
| Source | Chunking | Purpose |
| --- | --- | --- |
| Product catalog | 1 chunk per product: name + description + price + variants + tags | Semantic product matching |
| Store policy / FAQ | 500-token chunks with 50-token overlap | Policy Q&A |
| Conversation summaries | Nightly Gemini: 2–3 sentences per closed session | Past context beyond 20-message window |
| High-performing campaigns | Full text + conversion\_rate | Style reference for outreach |

### Context Window Assembly (Per Message)
| Block | Source | Max Tokens |
| --- | --- | --- |
| System prompt | shops.ai\_system\_prompt | ~300 |
| Customer profile | customers + user\_metrics pre-computed fields | ~200 |
| Retrieved product chunks | pgvector top-3 cosine similarity | ~400 |
| Conversation history | Last 20 messages from `messages` | ~600 |
| Current message | Incoming message text | ~100 |

Customer profile block format:
```
CUSTOMER PROFILE (pre-computed — do not modify):
Name: {display_name} | Active hours: {preferred_inbox_hours}
Interested in: {top_products_mentioned}
Orders: {order_count} · avg {avg_order_value}
Intent score: {intent_score_latest} | Churn risk: {churn_risk_score} ({churn_label})
Segment: {segment_label}
```

---

## Conversation Intelligence

### Conversation Stage (Deterministic State Machine)
| Stage | Transition | Action |
| --- | --- | --- |
| discovery | First message, no strong intent | Let agent engage |
| interest | product\_inquiry or availability\_inquiry detected | Agent handles |
| intent | price\_inquiry or purchase\_intent detected | Monitor, trigger on drop-off |
| negotiation | Multiple price\_inquiry/quantity signals | Conversion push if drop-off |
| converted | order\_completed exists | Post-purchase follow-up (7 days) |
| dormant | No message > idle\_window AND stage was intent/negotiation | Re-engagement outreach |

### Drop-off Detection (Deterministic SQL)
```sql
UPDATE conversations SET drop_off_flag = true, drop_off_at = NOW()
WHERE status = 'idle'
  AND resulted_in_order = false
  AND last_message_at < NOW() - INTERVAL '4 hours'
  AND conversation_id IN (
    SELECT DISTINCT conversation_id FROM extracted_signals
    WHERE intent_type IN ('price_inquiry','purchase_intent','availability_inquiry')
      AND message_id = (
        SELECT message_id FROM messages
        WHERE conversation_id = extracted_signals.conversation_id
        ORDER BY sent_at DESC LIMIT 1
      )
  );
```

---

## Billing Tiers

| Tier | Price | Channels | Included |
| --- | --- | --- | --- |
| Starter | $29/mo | 1 Messenger page | AI agent · checkout · CRM list · 500 outreach/mo · basic triggers · rule-based extraction |
| Growth | $79/mo | Up to 3 (Messenger + WhatsApp) | All Starter + full analytics · intent scoring · all triggers · AI enrichment · decision log |
| Pro | $179/mo | Up to 10 | All Growth + custom intent weights · tunable keyword dicts · Qwen3 embedding migration · priority support |

---

## Build Phases (Specification v3.0)

### Phase 1 — Foundation (weeks 1–4)
- Supabase: all tables with RLS policies
- FastAPI: webhook receiver, message write
- Fast extractor: rule-based Python
- Conversation tracker
- Basic Gemini agent with RAG
- Product catalog import + embedding indexing

### Phase 2 — Commerce (weeks 5–8)
- WhatsApp Business API integration
- Conversation stage state machine
- Drop-off detection (hourly SQL)
- Paddle checkout + order tracking
- AI enrichment pass (async)
- Nightly session summarization

### Phase 3 — Scoring & Triggers (weeks 9–12)
- Nightly scoring job: user\_metrics, intent score, churn formula, priority, inbox-time
- Trigger engine (hourly): full catalogue, decision log, fatigue controls
- Outreach drafting with template retrieval
- CRM dashboard: customer list with scores, segments, priority

### Phase 4 — Outreach & Intelligence (weeks 13–16)
- Outreach approval queue and scheduler
- Campaign performance tracking
- K-means segmentation job
- Decision log view for tuning
- Paddle SaaS billing

### Phase 5 — Scale (weeks 17–20)
- Celery + Redis for background jobs
- Custom keyword dictionary editor per shop
- Intent weight tuning UI
- Composite DB indexes

---

## Phase 1 Status

Phase 1 Foundation is **COMPLETE** as of 2026-03-30.

### Components Delivered
- ✅ Project structure with pyproject.toml and dependencies
- ✅ Database connection pool with asyncpg
- ✅ Configuration management via pydantic-settings
- ✅ Pydantic schemas (17 models for messages, conversations, customers, webhooks, AI)
- ✅ Database schema with 9 tables and RLS policies
- ✅ Fast extractor (rule-based, 5 intent types)
- ✅ Conversation tracker (stage progression, idle window detection)
- ✅ FastAPI webhook receiver (<50ms response target)
- ✅ pgvector embeddings service (cosine similarity search)
- ✅ AI agent with RAG (context assembly, Gemini integration)
- ✅ Product catalog import endpoints (CSV/JSON support)
- ✅ Configuration files (README.md, conftest.py, requirements.txt)
- ✅ Integration tests (20 end-to-end tests)
- ✅ Smoke test script for deployment verification

### Test Coverage
- Total tests: 139
- Unit tests: 119 (covering all services and routers)
- Integration tests: 20 (webhook flow, health checks, catalog)
- Smoke test: standalone deployment verification

### Next Steps
- Phase 2: Commerce (WhatsApp, checkout, AI enrichment, drop-off detection)
- Deploy to Railway/Render
- Configure Supabase project (run migrations)
- Set up Meta app for webhooks

---

## Deployment

| Service | Provider | Est. Monthly |
| --- | --- | --- |
| Database | Supabase | ~$25 |
| Backend | Railway/Render (FastAPI) | ~$20–30 |
| Frontend | Vercel (Next.js) | ~$20 |
| AI API | Gemini 2.0 Flash | ~$10–30 (50 shops) |
| Embedding | text-embedding-004 | $0–10 |
| **Total (50 shops)** | | ~$75–115/mo |

---

## Key Constraints & Reminders

1. **Messages are append-only**: Never modify `messages` table rows. All derived tables computed from this source of truth.
2. **LLM for enhancement only**: All scoring, churn detection, and trigger evaluation must be deterministic Python. LLM only used for classification refinement and generation.
3. **Multi-tenancy via RLS**: All queries must respect `shop_id`. RLS policies enforce tenant isolation at the database layer.
4. **Fast path <50ms**: Webhook must return 200 within 50ms. Use rule-based extraction immediately, defer AI enrichment to background.
5. **Hard constraint**: If Gemini is unavailable or times out, fast extractor result stands. System must never block message delivery.
