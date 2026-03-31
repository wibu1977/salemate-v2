**Sellora**

AI-Powered Chat-Commerce Platform · Messenger & WhatsApp

Technical Product Specification | v3.0 — Chat-Commerce Redesign

| **Field** | **Value** |
| --- | --- |
| Version | 3.0 — Chat-commerce redesign (supersedes v2.0) |
| Key change | Replaced web-analytics / RFM model with conversation-driven intelligence |
| Backend | Python + FastAPI (scoring, NLP, jobs) + Next.js (dashboard) |
| Database | PostgreSQL via Supabase — single project, multi-tenant RLS |
| Messaging | Meta Messenger API + WhatsApp Business API |
| AI model | Gemini 3 Flash — classify · extract entities · generate replies only |
| Scoring engine | Deterministic formulas in Python backend — never LLM |
| Embedding model |  Qwen3-Embedding-8B  |
| Deployment | Vercel (frontend) · Railway/Render (backend) · Supabase (DB) |
| Target budget | $100–200 / month early stage |
| Date | March 26, 2026 |

|  |
| --- |
| **What changed from v2.0**  The RFM / web-event model has been retired. In chat-commerce, there are no page views, add-to-cart events, or checkout funnels — the entire customer journey lives in the conversation. v3.0 rebuilds analytics around four new layers: (1) raw message storage, (2) a conversation intelligence extraction layer, (3) a chat-native intent scoring formula, (4) a trigger engine that reads conversation stage and intent score rather than web events. The backend moves from Node.js to Python + FastAPI for NLP compatibility. |

# **1 Product Overview**

Sellora is a multi-tenant SaaS platform that turns Facebook Messenger and WhatsApp into fully automated sales channels. Store owners connect their messaging accounts, configure their product catalog and store policy, and immediately get an AI agent that consults customers, recommends products, handles checkout, and re-engages buyers based on their conversation behavior.

The fundamental design principle of v3.0 is:

|  |
| --- |
| **Conversation = the customer journey**  In chat-commerce, there are no web pages to track. Every signal — interest, intent, hesitation, drop-off — is encoded in the message exchange itself. All analytics, scoring, and outreach decisions are derived exclusively from conversation data: messages, timestamps, and conversation flow. |

## **1.1 Strict LLM role boundary**

|  |
| --- |
| **What the LLM does**  Classify message intent (as enhancement to rule-based system) · extract entity names (products, variants, prices) · generate personalized replies and outreach drafts · explain scores in plain language. |

|  |
| --- |
| **What the LLM must never do**  Compute intent\_score or any sub-feature (Q, E, P, F, D) · make final outreach decisions · modify any numeric field in the database · run synchronously in the message-response critical path. |

## **1.2 Dual processing architecture**

Signal extraction runs in two passes to balance latency and quality:

| **Pass** | **Runs** | **Method** | **Output** |
| --- | --- | --- | --- |
| Fast | Synchronously on message receive (< 50ms) | Keyword / rule-based Python | intent\_type · product\_mentioned · drop\_off\_flag (time-based) |
| Enrichment | Async background job (within 60s) | Optional Gemini classification | Refined intent\_type · sentiment · entity normalization |

# **2 System Architecture**

## **2.1 Component overview**

| **Component** | **Technology** | **Role** |
| --- | --- | --- |
| Webhook receiver | FastAPI (Python) | Receives Messenger / WhatsApp webhook events. Writes raw message to DB. Dispatches fast extraction. Returns 200 immediately. |
| Fast extractor | Python rule engine | Synchronous. Extracts intent\_type, product\_mentioned from keywords. Writes extracted\_signals row. |
| AI enricher | Gemini 2.0 Flash (async) | Background. Refines intent classification, normalizes entity names. Updates extracted\_signals. |
| Scoring job | Python cron (nightly) | Computes intent\_score, churn\_risk, priority\_score, preferred\_inbox\_hours. Writes to user\_metrics. |
| Trigger engine | Python cron (hourly) | Evaluates trigger conditions. Drafts campaigns. Writes decision\_log. |
| RAG agent | Gemini 2.0 Flash | Handles live customer replies using retrieved catalog + policy chunks + conversation history. |
| Dashboard | Next.js + Vercel | CRM list, conversation view, outreach approval queue, analytics. |
| Background queue | Cron (initial) / Celery+Redis (scale) | Async enrichment jobs, email notifications, scheduled sends. |

## **2.2 Message flow — from webhook to reply**

1. Meta / WhatsApp webhook POST arrives at FastAPI endpoint.
2. Webhook handler writes raw message to messages table (< 5ms).
3. Fast extractor runs synchronously: keyword scan → writes extracted\_signals row with intent\_type, product\_mentioned (< 50ms total).
4. Conversation record is updated: message\_count++, last\_message\_at, conversation\_stage recomputed.
5. RAG agent assembles context window: system\_prompt + customer\_profile\_block + retrieved chunks + last 20 messages.
6. Gemini generates reply. Reply written to messages table (sender = business). Sent via API.
7. Background: AI enricher job queued. Runs within 60s. Updates extracted\_signals with refined classification and sentiment.

## **2.3 Infrastructure and deployment**

| **Service** | **Provider** | **Monthly cost (est.)** |
| --- | --- | --- |
| Database | Supabase (single project, all shops) | ~$25 |
| Backend API | Railway or Render (Python FastAPI) | ~$20–30 |
| Frontend | Vercel (Next.js) | ~$20 |
| AI API | Gemini 2.0 Flash (Google AI) | ~$10–30 at 50 shops |
| Embedding API | text-embedding-004 (free quota initially) | $0–10 |
| Total at 50 shops |  | ~$75–115/month |

# **3 Database Schema**

|  |
| --- |
| **Multi-tenancy**  All tables include shop\_id (UUID, non-nullable, indexed). RLS policies restrict all reads and writes to rows where shop\_id matches the authenticated session claim. No cross-tenant data access is possible at the query layer. |

## **3.1 shops**

| **Field** | **Type** | **Description** |
| --- | --- | --- |
| shop\_id | UUID PK | Primary tenant identifier. |
| owner\_email | text | Login via Supabase Auth. |
| channels | jsonb | Array of {channel\_type: messenger|whatsapp, page\_id, access\_token}. |
| plan\_tier | enum | starter | growth | pro |
| paddle\_subscription\_id | text | Active Paddle subscription. |
| ai\_system\_prompt | text | Agent persona, tone, and store policy. |
| outreach\_weekly\_cap | int | Max outreach per customer per week. Default: 2. |
| outreach\_cooldown\_hours | int | Min hours between outreach to same customer. Default: 48. |
| drop\_off\_window\_hours | int | Hours after high-intent message before drop\_off\_flag fires. Default: 4. |
| intent\_weights | jsonb | { w\_Q, w\_E, w\_P, w\_F, w\_D }. Tunable per shop. Defaults: 0.30,0.25,0.20,0.15,0.10. |
| created\_at | timestamptz | Account creation. |

## **3.2 messages (raw chat log)**

|  |
| --- |
| **Append-only**  Messages are never modified after insert. They are the source of truth. All derived tables are computed from this table. |

| **Field** | **Type** | **Description** |
| --- | --- | --- |
| message\_id | UUID PK | Primary key. |
| shop\_id | UUID FK·idx | Tenant identifier. |
| conversation\_id | UUID FK | References conversations. |
| customer\_id | UUID FK | References customers. |
| sender\_type | enum | customer | business |
| channel | enum | messenger | whatsapp |
| content | text | Raw message text. |
| content\_type | enum | text | image | audio | video | file | template |
| sent\_at | timestamptz·idx | Exact send time. Primary index for all time-based queries. |
| day\_of\_week | int 0–6 | Extracted from sent\_at. Indexed. |
| hour\_of\_day | int 0–23 | Extracted from sent\_at. Indexed. |
| platform\_msg\_id | text | Native ID from Meta / WhatsApp API for deduplication. |
| read\_at | timestamptz | When business message was read by customer (delivery receipt). |

## **3.3 conversations (thread-level aggregation)**

One row per conversation thread. A new conversation is created when a customer sends a message after a configurable idle window (default: 8 hours since last message in thread).

| **Field** | **Type** | **Description** |
| --- | --- | --- |
| conversation\_id | UUID PK | Primary key. |
| shop\_id | UUID FK·idx | Tenant identifier. |
| customer\_id | UUID FK | References customers. |
| channel | enum | messenger | whatsapp |
| started\_at | timestamptz | First message timestamp. |
| last\_message\_at | timestamptz | Most recent message. Updated on each new message. |
| message\_count | int | Total messages in thread. |
| customer\_message\_count | int | Customer-sent messages only. |
| business\_message\_count | int | Business-sent messages only. |
| conversation\_depth | int | Back-and-forth exchange count (min of customer/business counts). |
| avg\_response\_speed\_s | numeric | Mean seconds between customer message and next business reply. |
| conversation\_stage | enum | discovery | interest | intent | negotiation | converted | dormant. See Section 4.3. |
| intent\_score | numeric 0–1 | Latest computed intent score for this conversation. See Section 5. |
| drop\_off\_flag | bool | True if customer stopped replying after high-intent message. See Section 4.2. |
| drop\_off\_at | timestamptz | When drop-off was detected. |
| resulted\_in\_order | bool | True if an order\_completed event exists for this conversation. |
| products\_mentioned | text[] | Deduplicated list of product IDs mentioned in thread. |
| status | enum | active | idle | closed |

## **3.4 extracted\_signals (conversation intelligence layer)**

|  |
| --- |
| **Core of the intelligence layer**  One row per message, written in two passes. The fast extractor populates rule-based fields immediately on message receive. The AI enricher updates refined fields asynchronously within 60 seconds. This table is what all scoring formulas read. |

| **Field** | **Type** | **Description** |
| --- | --- | --- |
| signal\_id | UUID PK | Primary key. |
| shop\_id | UUID FK·idx | Tenant identifier. |
| message\_id | UUID FK | References messages. |
| conversation\_id | UUID FK | References conversations. |
| customer\_id | UUID FK | References customers. |
| extracted\_at | timestamptz | When fast extractor ran. |
| enriched\_at | timestamptz nullable | When AI enricher ran. Null if not yet enriched. |
| intent\_type | enum | Rule-based: price\_inquiry | product\_inquiry | availability\_inquiry | purchase\_intent | complaint | general\_chat | unknown |
| intent\_type\_refined | enum nullable | AI-refined intent. Same enum. Overwrites intent\_type in scoring if populated. |
| intent\_strength | numeric 0–1 | Confidence of intent\_type. Rule-based: 1.0 if exact keyword match, 0.6 if partial. AI pass: model confidence. |
| product\_mentioned | text nullable | Product ID matched from catalog. Rule-based: keyword lookup. AI pass: entity normalization. |
| product\_raw | text nullable | Raw product string from message before normalization. |
| variant\_mentioned | text nullable | Size, color, or variant detail mentioned. |
| price\_mentioned | bool | True if message contains price-related keywords. |
| quantity\_mentioned | bool | True if message contains quantity or order-size language. |
| sentiment | enum nullable | AI pass only: positive | neutral | negative |
| sentiment\_score | numeric nullable | AI pass only: -1.0 to 1.0. |
| extraction\_method | enum | rule\_based | ai\_enhanced |

## **3.5 customers**

| **Field** | **Type** | **Description** |
| --- | --- | --- |
| customer\_id | UUID PK | Primary key. |
| shop\_id | UUID FK·idx | Tenant identifier. |
| psid | text | Facebook/WhatsApp sender ID. |
| display\_name | text | From messaging profile. |
| locale | text | Language/locale. |
| channel | enum | messenger | whatsapp |
| first\_seen\_at | timestamptz | First message timestamp. |
| last\_contact\_at | timestamptz | Most recent message of any type. |
| conversation\_count | int | Total conversation threads. |
| total\_order\_value | numeric | Lifetime spend. |
| order\_count | int | Completed orders. |
| avg\_order\_value | numeric | total\_order\_value / order\_count. |
| preferred\_inbox\_hours[] | int[] | Top 3 hours (recency-weighted). Computed nightly. |
| preferred\_days[] | int[] | Top 2 days (recency-weighted). Computed nightly. |
| top\_products\_mentioned[] | text[] | Product IDs by mention frequency, last 60 days. |
| churn\_risk\_score | numeric 0–1 | Deterministic formula. See Section 6.1. |
| churn\_label | text | LLM-assigned label only: active | at\_risk | dormant | lost. |
| intent\_score\_latest | numeric 0–1 | Intent score from most recent conversation. |
| priority\_score | numeric | churn\_risk × expected\_value. See Section 6.3. |
| outreach\_sent\_this\_week | int | Rolling 7-day count. Fatigue controller. |
| last\_outreach\_at | timestamptz | Last outreach sent timestamp. |
| segment\_label | text | K-means label for dashboard display only. |
| notes | text | Owner free-text notes. |

## **3.6 user\_metrics (pre-computed analytics per customer)**

Written nightly by the scoring job. These are the pre-computed inputs the scoring formulas read. Storing them separately from the customers table keeps the customers table clean and allows historical snapshots.

| **Field** | **Type** | **Description** |
| --- | --- | --- |
| metric\_id | UUID PK | Primary key. |
| shop\_id | UUID FK·idx | Tenant identifier. |
| customer\_id | UUID FK | References customers. |
| computed\_at | timestamptz | When this row was written. |
| conversations\_last\_7d | int | Conversation thread count, last 7 days. Input to F feature. |
| conversations\_last\_30d | int | Conversation thread count, last 30 days. |
| avg\_intent\_strength\_7d | numeric | Mean intent\_strength of all signals, last 7 days. Input to Q feature. |
| purchase\_intent\_ratio\_7d | numeric | Fraction of signals with intent\_type=purchase\_intent, last 7 days. |
| price\_inquiry\_count\_7d | int | Count of price\_inquiry signals, last 7 days. |
| avg\_conversation\_depth\_7d | numeric | Mean conversation\_depth across threads, last 7 days. Input to E feature. |
| avg\_response\_speed\_7d | numeric | Mean customer reply speed in seconds, last 7 days. Input to E feature. |
| unique\_products\_mentioned\_7d | int | Count of distinct products mentioned, last 7 days. Input to P feature. |
| product\_repeat\_max\_7d | int | Max times any single product was mentioned, last 7 days. Input to P feature. |
| drop\_off\_count\_7d | int | Conversations with drop\_off\_flag=true, last 7 days. Input to D feature. |
| days\_since\_last\_contact | int | Input to churn formula. |
| interaction\_frequency\_30d | numeric | Messages per day, last 30 days. Input to churn formula. |
| activity\_trend | numeric | (freq last 14d) − (freq prior 14d). Negative = declining. |

## **3.7 orders**

| **Field** | **Type** | **Description** |
| --- | --- | --- |
| order\_id | UUID PK | Primary key. |
| shop\_id | UUID FK·idx | Tenant identifier. |
| customer\_id | UUID FK | References customers. |
| conversation\_id | UUID FK nullable | Conversation that led to this order. |
| channel | enum | paddle | shopify | woocommerce | payment\_link |
| status | enum | pending | paid | shipped | cancelled | refunded |
| line\_items | jsonb[] | Array of {product\_id, product\_name, qty, unit\_price}. |
| total | numeric | Final charged amount. |
| currency | text | ISO 4217 code. |
| payment\_ref | text | Paddle transaction ID or external order ID. |
| created\_at | timestamptz | Order creation. |
| paid\_at | timestamptz nullable | Payment confirmation. |

## **3.8 product\_embeddings (RAG vector store)**

| **Field** | **Type** | **Description** |
| --- | --- | --- |
| embedding\_id | UUID PK | Primary key. |
| shop\_id | UUID FK·idx | Tenant identifier. |
| source\_type | enum | product | policy | faq | conversation\_summary | campaign |
| source\_id | text | ID of source record. |
| chunk\_index | int | Position within source document. |
| chunk\_text | text | Raw text returned to agent on retrieval. |
| embedding | vector(768) | pgvector column. Index: ivfflat cosine. |
| conversion\_rate | numeric nullable | For source\_type=campaign: reply rate. Used for template reuse. |
| created\_at | timestamptz | Indexed. |

## **3.9 outreach\_campaigns**

| **Field** | **Type** | **Description** |
| --- | --- | --- |
| campaign\_id | UUID PK | Primary key. |
| shop\_id | UUID FK·idx | Tenant identifier. |
| customer\_id | UUID FK | Target customer. |
| trigger\_id | UUID FK | References decision\_log. |
| trigger\_reason | text | Human-readable: 'Price inquiry drop-off 3h ago — jacket-04'. |
| message\_draft | text | Gemini-generated. Editable by owner. |
| status | enum | pending\_approval | approved | rejected | sent | replied | ignored |
| scheduled\_send\_at | timestamptz | Set from preferred\_inbox\_hours. |
| sent\_at | timestamptz nullable | Actual send time. |
| created\_at | timestamptz | Draft generation time. |

## **3.10 decision\_log (full audit trail)**

| **Field** | **Type** | **Description** |
| --- | --- | --- |
| log\_id | UUID PK | Primary key. |
| shop\_id | UUID FK·idx | Tenant identifier. |
| customer\_id | UUID FK | Evaluated customer. |
| evaluated\_at | timestamptz | Evaluation timestamp. |
| trigger\_type | text | price\_inquiry\_dropoff | repeated\_product\_view | high\_intent\_score | drop\_off\_reengagement | promo\_match | post\_purchase |
| input\_snapshot | jsonb | All formula inputs at evaluation time. |
| threshold\_crossed | bool | Whether action threshold was met. |
| action\_taken | text | campaign\_drafted | skipped\_fatigue | skipped\_cooldown | skipped\_no\_trigger |
| campaign\_id | UUID nullable | Campaign created, if any. |

# **4 Conversation Intelligence Layer**

This is the new core of the platform. Every incoming customer message passes through this layer, which converts raw text into structured signals that analytics and the trigger engine can act on.

## **4.1 Fast extractor (rule-based, synchronous)**

Runs in Python within the webhook handler. Target latency: under 50ms. Uses keyword dictionaries and regex patterns stored per shop (with sensible defaults). No external API calls.

| **Signal** | **Extraction method** | **Example match** |
| --- | --- | --- |
| intent\_type = price\_inquiry | Keywords: price, cost, how much, bao nhiêu, giá | 'how much is the jacket?' |
| intent\_type = product\_inquiry | Keywords: tell me about, what is, specs, color, size | 'do you have this in blue?' |
| intent\_type = availability\_inquiry | Keywords: in stock, available, còn hàng, still have | 'is size M still available?' |
| intent\_type = purchase\_intent | Keywords: want to buy, order, I'll take, cho mình | 'I want to order 2 of these' |
| intent\_type = complaint | Keywords: wrong, broken, late, refund, complaint | 'my order arrived damaged' |
| product\_mentioned | Token match against catalog product name / SKU list | 'wool jacket' → SKU jacket-04 |
| price\_mentioned | Regex: currency symbols, number + unit patterns | '200k', '$49', '200,000 VND' |
| quantity\_mentioned | Regex: number + order-size words | '2 pieces', 'order 3' |

|  |
| --- |
| **Multilingual note**  Keyword dictionaries should include Vietnamese, English, and Thai by default, as these cover the primary Southeast Asian markets. Shop owners can add custom keywords via the dashboard. |

## **4.2 Drop-off detection**

Drop-off is computed deterministically from the messages table — not by an LLM. A drop-off is recorded when all three conditions are met:

* **1.** Customer's last message in the conversation had intent\_type in (price\_inquiry, purchase\_intent, availability\_inquiry) — a high-intent signal.
* **2.** No customer message has followed within drop\_off\_window\_hours (configurable per shop, default 4 hours).
* **3.** No order\_completed event exists for this conversation.

-- Run hourly by the trigger engine

UPDATE conversations SET drop\_off\_flag = true, drop\_off\_at = NOW()

WHERE status = 'idle'

AND resulted\_in\_order = false

AND last\_message\_at < NOW() - INTERVAL '4 hours'

AND conversation\_id IN (

SELECT DISTINCT conversation\_id FROM extracted\_signals

WHERE intent\_type IN ('price\_inquiry','purchase\_intent','availability\_inquiry')

AND message\_id = (SELECT message\_id FROM messages WHERE conversation\_id = extracted\_signals.conversation\_id

ORDER BY sent\_at DESC LIMIT 1)

);

## **4.3 Conversation stage**

The conversation\_stage field tracks where the customer is in the purchase journey. It is computed from the sequence of intent\_type values across all messages in the thread. This is a deterministic state machine — not an LLM judgment.

| **Stage** | **Transition condition** | **Typical outreach action** |
| --- | --- | --- |
| discovery | First message received. No strong intent signal yet. | None — let the agent engage. |
| interest | product\_inquiry or availability\_inquiry signal detected. | None — agent handles. |
| intent | price\_inquiry or purchase\_intent signal detected. | Monitor. Trigger if drop-off follows. |
| negotiation | Multiple price\_inquiry or quantity\_mentioned signals in same thread. | Conversion push if drop-off. |
| converted | order\_completed event exists for this conversation. | Post-purchase follow-up (7 days). |
| dormant | No customer message for > idle\_window\_hours AND stage was intent or negotiation. | Re-engagement outreach. |

## **4.4 AI enrichment pass (async, optional enhancement)**

Runs as a background job within 60 seconds of message receipt. Makes a single Gemini API call with a structured prompt requesting JSON output only. Updates extracted\_signals.intent\_type\_refined, sentiment, and product normalization.

SYSTEM: You are a signal extractor for a chat-commerce platform.

Return ONLY valid JSON. No prose. No markdown.

USER: Classify this customer message:

message: "{content}"

known\_products: {catalog\_product\_names}

Return: {

"intent\_type": "price\_inquiry|product\_inquiry|availability\_inquiry|purchase\_intent|complaint|general\_chat",

"confidence": 0.0-1.0,

"product\_normalized": "product\_id or null",

"sentiment": "positive|neutral|negative",

"sentiment\_score": -1.0 to 1.0

}

|  |
| --- |
| **Hard constraint**  If Gemini is unavailable or times out, the fast extractor result stands. The system must never block message delivery waiting for AI enrichment. All scoring formulas must work with rule-based signals alone. |

# **5 Chat-Based Purchase Intent Score**

|  |
| --- |
| **Replaces RFM**  The v2.0 RFM formula has been retired. It required web-event data (page views, checkout tracking) that does not exist in chat-commerce. The new formula is derived entirely from conversation signals. |

## **5.1 Formula**

intent\_score = 0.30×Q + 0.25×E + 0.20×P + 0.15×F + 0.10×D

All five features are normalized to [0, 1] before weighting. Normalization uses min-max scaling across all customers in the same shop, computed nightly. Weights are stored in the shops table (intent\_weights) and are configurable per shop.

## **5.2 Feature definitions**

| **Feature** | **Name** | **Raw input** | **Normalization** |
| --- | --- | --- | --- |
| Q | Query intent | avg\_intent\_strength\_7d × (1 + purchase\_intent\_ratio\_7d). High if customer is asking purchase-related questions with strong signals. | min-max across shop customers |
| E | Engagement | (avg\_conversation\_depth\_7d / max\_depth) × (1 − norm(avg\_response\_speed\_7d)). High if deep exchanges AND fast replies. | depth: /max\_depth; speed: inverted, min-max |
| P | Product interest | (unique\_products\_mentioned\_7d + product\_repeat\_max\_7d) / normalizer. High if customer keeps returning to specific products. | min-max across shop customers |
| F | Frequency | conversations\_last\_7d / max\_conversations\_7d. High if customer opened multiple threads recently. | min-max across shop customers |
| D | Drop-off adjustment | −(drop\_off\_count\_7d / conversations\_last\_7d). Penalty: if customer drops off after high-intent signals, reduce score. | proportion; clamp 0–1; applied as subtraction |

The D feature is a penalty, not a score component. It is subtracted last. Final intent\_score is clamped to [0, 1] after applying D.

## **5.3 Example computation**

| **Customer** | **Q** | **E** | **P** | **F** | **D penalty** | **intent\_score** | **Interpretation** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Active buyer, fast replies | 0.85 | 0.80 | 0.70 | 0.60 | 0.00 | 0.75 | High — conversion push |
| Browsing, slow replies | 0.40 | 0.30 | 0.50 | 0.20 | 0.00 | 0.34 | Low — nurture |
| High intent, then silent | 0.90 | 0.70 | 0.80 | 0.50 | 0.30 | 0.65 | Medium — drop-off triggered |
| One message, never replied | 0.20 | 0.05 | 0.10 | 0.10 | 0.00 | 0.12 | Very low — skip |

|  |
| --- |
| **Computation constraint**  All feature values are read from the user\_metrics table (pre-computed nightly). The scoring function is a pure Python function with no database reads or API calls. It receives a dict of pre-fetched metric values and returns a float. This makes it fast, testable, and auditable. |

# **6 Churn Risk & Priority Scoring**

## **6.1 Churn risk score (conversation-adapted)**

Churn in chat-commerce means a previously engaged customer has gone silent. The formula uses conversation-native signals.

churn\_score = w1 × norm(days\_since\_last\_contact)

+ w2 × norm(1 / interaction\_frequency\_30d)

+ w3 × norm(1 / max(activity\_trend\_positive, 0.01))

Default weights: w1=0.50, w2=0.30, w3=0.20 (stored in shops.intent\_weights)

All inputs normalized 0–1 across shop customer population before weighting.

| **Input** | **High value means** | **Effect on churn\_score** |
| --- | --- | --- |
| days\_since\_last\_contact | Customer has been silent for a long time | Increases score |
| 1 / interaction\_frequency | Customer rarely messages | Increases score |
| 1 / activity\_trend | Activity is declining | Increases score |

## **6.2 Purchase likelihood**

purchase\_likelihood = 1 / (1 + exp(−5 × (intent\_score\_latest − churn\_score)))

Uses the latest intent\_score from the most recent conversation. A customer who is high-intent and low-churn has the highest purchase likelihood.

## **6.3 Priority score (targeting optimization)**

priority\_score = churn\_risk\_score × expected\_value

expected\_value = avg\_order\_value × purchase\_likelihood

The trigger engine and outreach approval queue both sort by priority\_score descending. High-value customers with rising churn risk are always actioned first.

# **7 Trigger-Based Decision Engine**

The trigger engine runs hourly. It evaluates each active customer against the trigger catalogue, checks fatigue controls, and — if a trigger fires — calls Gemini to draft an outreach message using the trigger context.

## **7.1 Trigger catalogue**

| **Trigger** | **Condition** | **Priority modifier** | **Outreach type** |
| --- | --- | --- | --- |
| price\_inquiry\_dropoff | conversation\_stage=intent AND drop\_off\_flag=true AND hours since drop\_off\_at ≤ 24h | × 2.0 | Follow-up offer / urgency |
| repeated\_product\_view | product\_repeat\_max\_7d ≥ 3 AND no order in 7 days | × 1.5 | Recommendation / social proof |
| high\_intent\_score | intent\_score > 0.70 (threshold breach — edge trigger, not level) | × 1.3 | Conversion push |
| drop\_off\_reengagement | drop\_off\_flag=true AND hours since drop\_off\_at between 24h and 72h | × 1.0 | Re-engagement |
| promo\_match | product in top\_products\_mentioned AND product currently on promotion | × 1.3 | Promotional offer |
| post\_purchase\_followup | order\_completed 7 days ago, no subsequent message | × 0.8 | Upsell / review request |
| churn\_threshold | churn\_score crosses 0.70 (edge trigger) | × 1.0 | Win-back message |

## **7.2 Evaluation flow**

1. Hourly job fetches all customers with status=active or recent conversation activity.
2. For each customer: read pre-computed fields from user\_metrics and customers tables. No live computation.
3. Evaluate each trigger condition. Write a decision\_log row for every evaluation (fired or not).
4. If trigger fires: check fatigue (outreach\_sent\_this\_week < cap AND cooldown elapsed).
5. If fatigue passes: compute effective\_priority = priority\_score × trigger\_modifier. Select highest.
6. Call Gemini with trigger context: customer\_profile\_block + trigger\_reason + top-2 high-conversion campaign templates from product\_embeddings.
7. Write campaign to outreach\_campaigns with status=pending\_approval.
8. Owner reviews in dashboard approval queue. Approves / edits / rejects.
9. On approval: scheduler queues send at next preferred\_inbox\_hours window.
10. On send: outreach\_sent event written. outreach\_sent\_this\_week incremented. last\_outreach\_at updated.

## **7.3 Message fatigue controls**

| **Control** | **Default** | **Configurable** |
| --- | --- | --- |
| Weekly outreach cap | 2 messages / customer / week | Yes — shops.outreach\_weekly\_cap |
| Cooldown period | 48h between any two outreach sends | Yes — shops.outreach\_cooldown\_hours |
| Pending suppression | No new draft while one is pending\_approval | Always enforced |
| Post-send silence window | No trigger re-evaluates customer for 24h after a send | Always enforced |

# **8 RAG Pipeline**

## **8.1 What gets embedded**

| **Source** | **Chunking** | **Purpose** |
| --- | --- | --- |
| Product catalog | One chunk per product: name + description + price + variants + tags. | Semantic product matching — 'something warm for winter' → wool jacket. |
| Store policy / FAQ | 500-token chunks with 50-token overlap. | Policy Q&A — returns, shipping, promotions. |
| Conversation summaries | Nightly: Gemini generates 2–3 sentence summary per closed session. Embedded. | Relevant past context beyond the 20-message inline window. |
| High-performing campaigns | Full message text + conversion\_rate stored. | Style reference for outreach generation. Top-2 by conversion\_rate retrieved per draft. |

## **8.2 Context window assembly (every incoming message)**

| **Block** | **Source** | **Max tokens** |
| --- | --- | --- |
| System prompt | shops.ai\_system\_prompt | ~300 |
| Customer profile block | Pre-computed fields from customers + user\_metrics | ~200 |
| Retrieved product chunks | pgvector top-3 cosine similarity to current message | ~400 |
| Conversation history | Last 20 messages from messages table | ~600 |
| Current message | Incoming message text | ~100 |

|  |
| --- |
| **Customer profile block format**  CUSTOMER PROFILE (pre-computed — do not modify): Name: {display\_name} | Active hours: {preferred\_inbox\_hours} | Interested in: {top\_products\_mentioned} | Orders: {order\_count} · avg {avg\_order\_value} | Intent score: {intent\_score\_latest} | Churn risk: {churn\_risk\_score} ({churn\_label}) | Segment: {segment\_label} |

# **9 AI Agent**

## **9.1 Live response path**

Gemini 2.0 Flash handles all natural language tasks in the live chat path. It reads the assembled context window and generates a reply. It does not query the database. It does not compute scores. All numeric facts it references were computed by the deterministic Python backend and injected as pre-computed values.

## **9.2 Checkout flows**

* **Paddle native:** Agent collects item + quantity. Backend generates Paddle payment link. Agent sends link in Messenger / WhatsApp. Paddle webhook writes order row and updates conversation.resulted\_in\_order.
* **Payment link:** Pre-built URL sent in chat. No Paddle account required from customer.
* **Shopify / WooCommerce:** Deep link with UTM parameters. Order sync via store webhook (optional, Phase 2).

## **9.3 Onboarding flow**

| **Step** | **Action** |
| --- | --- |
| 1. Channel connect | Owner connects Messenger Page or WhatsApp Business account via OAuth. Webhooks registered. |
| 2. Auto-learn | Platform fetches page About section. Gemini drafts initial ai\_system\_prompt. |
| 3. Catalog import | Owner uploads CSV or manually adds products. Each product chunked and embedded immediately. |
| 4. Keyword config | Owner reviews default keyword dictionaries. Can add custom keywords for their product category and language. |
| 5. Policy override | Owner edits ai\_system\_prompt: return policy, shipping, active promotions, tone. |
| 6. Go live | Webhooks activated. Fast extractor begins on first message. |

# **10 Customer Segmentation (Dashboard Insight Only)**

|  |
| --- |
| **Scope constraint**  K-means clustering is for dashboard display only. Cluster labels are stored in customers.segment\_label. The trigger engine, scoring formulas, and outreach decision path never read segment\_label. All real-time decisions use deterministic numeric scores. |

The nightly job runs K-means (k=4–6) across all shop customers using: intent\_score\_latest, churn\_risk\_score, avg\_order\_value, conversation\_count, days\_since\_last\_contact. Resulting labels are displayed in the CRM dashboard as human-readable segment badges.

# **11 Billing Model**

| **Tier** | **Price** | **Channels** | **Included** |
| --- | --- | --- | --- |
| Starter | $29/month | 1 Messenger page | AI agent · checkout · CRM list · 500 outreach/mo · basic triggers · rule-based extraction |
| Growth | $79/month | Up to 3 (Messenger + WhatsApp) | All Starter + full analytics · intent scoring · all triggers · AI enrichment · decision log |
| Pro | $179/month | Up to 10 | All Growth + custom intent weights · tunable keyword dicts · Qwen3 embedding migration · priority support |

|  |
| --- |
| **Infrastructure cost note**  Single Supabase project ~$25/month. Railway backend ~$25/month. Vercel ~$20/month. Gemini Flash at 50 shops × 200 msg/day ≈ $20–30/month. Total: ~$90–100/month. Revenue at 50 × $29 = $1,450/month. Margin: >93%. |

# **12 Recommended Build Phases**

## **Phase 1 — Foundation (weeks 1–4)**

* Supabase: create all tables with RLS policies. shops, messages, conversations, customers, extracted\_signals, user\_metrics, orders.
* FastAPI: webhook receiver for Messenger. Write messages. Return 200 immediately.
* Fast extractor: rule-based Python. Intent type + product match from keyword dict. Writes extracted\_signals.
* Conversation tracker: create/update conversations on each message.
* Basic Gemini agent: RAG retrieval from product catalog. Reply generation.
* Product catalog import + text-embedding-004 indexing into pgvector.

## **Phase 2 — Commerce (weeks 5–8)**

* WhatsApp Business API integration.
* Conversation stage state machine.
* Drop-off detection (hourly SQL job).
* Checkout: Paddle payment link generation. Order tracking.
* AI enrichment pass: async Gemini classification. Updates extracted\_signals.
* Session summarization: nightly Gemini job. Embeds conversation summaries.

## **Phase 3 — Scoring & Triggers (weeks 9–12)**

* Nightly scoring job: user\_metrics population. Intent score formula (Q, E, P, F, D). Churn formula. Priority score. Recency-weighted inbox-time.
* Trigger engine (hourly): full trigger catalogue. Decision log writes. Fatigue controls.
* Outreach campaign drafting: Gemini + high-conversion template retrieval.
* CRM dashboard: customer list with intent score, churn badge, segment, priority. Customer detail panel.

## **Phase 4 — Outreach & Intelligence (weeks 13–16)**

* Outreach approval queue and scheduler (preferred\_inbox\_hours targeting).
* Campaign performance tracking: conversion\_rate on product\_embeddings.
* K-means segmentation job + segment\_label display.
* Decision log view for intent\_weights tuning.
* Paddle SaaS subscription billing.

## **Phase 5 — Scale (weeks 17–20)**


* Celery + Redis for background job queue (replaces cron at scale).
* Custom keyword dictionary editor per shop (UI in dashboard).
* Intent weight tuning UI: sliders for w\_Q, w\_E, w\_P, w\_F, w\_D.
* Composite DB indexes: (shop\_id, customer\_id, sent\_at) on messages; (shop\_id, conversation\_id) on extracted\_signals.

— End of specification v3.0 —