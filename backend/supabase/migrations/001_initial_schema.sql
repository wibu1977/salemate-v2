-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";

-- Create ENUM types
CREATE TYPE sender_type AS ENUM ('customer', 'business');
CREATE TYPE channel_type AS ENUM ('messenger', 'whatsapp');
CREATE TYPE content_type AS ENUM ('text', 'image', 'audio', 'video', 'file', 'template');
CREATE TYPE conversation_stage AS ENUM (
    'discovery', 'interest', 'intent', 'negotiation', 'converted', 'dormant'
);
CREATE TYPE intent_type AS ENUM (
    'price_inquiry', 'product_inquiry', 'availability_inquiry',
    'purchase_intent', 'complaint', 'general_chat', 'unknown'
);
CREATE TYPE sentiment_type AS ENUM ('positive', 'neutral', 'negative');
CREATE TYPE plan_tier AS ENUM ('starter', 'growth', 'pro');
CREATE TYPE order_status AS ENUM ('pending', 'paid', 'shipped', 'cancelled', 'refunded');
CREATE TYPE order_channel AS ENUM ('paddle', 'shopify', 'woocommerce', 'payment_link');
CREATE TYPE campaign_status AS ENUM (
    'pending_approval', 'approved', 'rejected', 'sent', 'replied', 'ignored'
);

-- shops table (tenant configuration)
CREATE TABLE shops (
    shop_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    owner_email TEXT NOT NULL UNIQUE,
    channels JSONB DEFAULT '[]'::jsonb,
    plan_tier plan_tier NOT NULL DEFAULT 'starter',
    paddle_subscription_id TEXT,
    ai_system_prompt TEXT,
    outreach_weekly_cap INTEGER DEFAULT 2,
    outreach_cooldown_hours INTEGER DEFAULT 48,
    drop_off_window_hours INTEGER DEFAULT 4,
    intent_weights JSONB DEFAULT '{"w_Q": 0.30, "w_E": 0.25, "w_P": 0.20, "w_F": 0.15, "w_D": 0.10}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- customers table
CREATE TABLE customers (
    customer_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    shop_id UUID NOT NULL REFERENCES shops(shop_id) ON DELETE CASCADE,
    psid TEXT NOT NULL,
    display_name TEXT,
    locale TEXT,
    channel channel_type NOT NULL,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_contact_at TIMESTAMPTZ,
    conversation_count INTEGER DEFAULT 0,
    total_order_value NUMERIC DEFAULT 0,
    order_count INTEGER DEFAULT 0,
    avg_order_value NUMERIC,
    preferred_inbox_hours INTEGER[] DEFAULT ARRAY[]::INTEGER[],
    preferred_days INTEGER[] DEFAULT ARRAY[]::INTEGER[],
    top_products_mentioned TEXT[] DEFAULT ARRAY[]::TEXT[],
    churn_risk_score NUMERIC,
    churn_label TEXT,
    intent_score_latest NUMERIC,
    priority_score NUMERIC,
    outreach_sent_this_week INTEGER DEFAULT 0,
    last_outreach_at TIMESTAMPTZ,
    segment_label TEXT,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(shop_id, psid)
);

-- messages table (raw chat log - append-only)
CREATE TABLE messages (
    message_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    shop_id UUID NOT NULL REFERENCES shops(shop_id) ON DELETE CASCADE,
    conversation_id UUID NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    customer_id UUID NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    sender_type sender_type NOT NULL,
    channel channel_type NOT NULL,
    content TEXT NOT NULL,
    content_type content_type DEFAULT 'text',
    sent_at TIMESTAMPTZ NOT NULL,
    day_of_week INTEGER GENERATED ALWAYS AS (EXTRACT(DOW FROM sent_at)) STORED,
    hour_of_day INTEGER GENERATED ALWAYS AS (EXTRACT(HOUR FROM sent_at)) STORED,
    platform_msg_id TEXT,
    read_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- conversations table (thread-level aggregation)
CREATE TABLE conversations (
    conversation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    shop_id UUID NOT NULL REFERENCES shops(shop_id) ON DELETE CASCADE,
    customer_id UUID NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    channel channel_type NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_message_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    message_count INTEGER DEFAULT 1,
    customer_message_count INTEGER DEFAULT 0,
    business_message_count INTEGER DEFAULT 0,
    conversation_depth INTEGER DEFAULT 0,
    avg_response_speed_s NUMERIC,
    conversation_stage conversation_stage DEFAULT 'discovery',
    intent_score NUMERIC,
    drop_off_flag BOOLEAN DEFAULT FALSE,
    drop_off_at TIMESTAMPTZ,
    resulted_in_order BOOLEAN DEFAULT FALSE,
    products_mentioned TEXT[] DEFAULT ARRAY[]::TEXT[],
    status VARCHAR(20) DEFAULT 'active', -- active, idle, closed
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(shop_id, customer_id, started_at)
);

-- Fix forward reference in messages
ALTER TABLE messages DROP CONSTRAINT IF EXISTS messages_conversation_id_fkey;
ALTER TABLE messages
ADD CONSTRAINT messages_conversation_id_fkey
FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE;

-- extracted_signals table (conversation intelligence layer)
CREATE TABLE extracted_signals (
    signal_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    shop_id UUID NOT NULL REFERENCES shops(shop_id) ON DELETE CASCADE,
    message_id UUID NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
    conversation_id UUID NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    customer_id UUID NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    enriched_at TIMESTAMPTZ,
    intent_type intent_type NOT NULL,
    intent_type_refined intent_type,
    intent_strength NUMERIC NOT NULL CHECK (intent_strength >= 0 AND intent_strength <= 1),
    product_mentioned TEXT,
    product_raw TEXT,
    variant_mentioned TEXT,
    price_mentioned BOOLEAN DEFAULT FALSE,
    quantity_mentioned BOOLEAN DEFAULT FALSE,
    sentiment sentiment_type,
    sentiment_score NUMERIC,
    extraction_method VARCHAR(20) DEFAULT 'rule_based',
    UNIQUE(message_id)
);

-- user_metrics table (pre-computed analytics)
CREATE TABLE user_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    shop_id UUID NOT NULL REFERENCES shops(shop_id) ON DELETE CASCADE,
    customer_id UUID NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    conversations_last_7d INTEGER DEFAULT 0,
    conversations_last_30d INTEGER DEFAULT 0,
    avg_intent_strength_7d NUMERIC,
    purchase_intent_ratio_7d NUMERIC,
    price_inquiry_count_7d INTEGER DEFAULT 0,
    avg_conversation_depth_7d NUMERIC,
    avg_response_speed_7d NUMERIC,
    unique_products_mentioned_7d INTEGER DEFAULT 0,
    product_repeat_max_7d INTEGER DEFAULT 0,
    drop_off_count_7d INTEGER DEFAULT 0,
    days_since_last_contact INTEGER,
    interaction_frequency_30d NUMERIC,
    activity_trend NUMERIC
);

-- orders table
CREATE TABLE orders (
    order_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    shop_id UUID NOT NULL REFERENCES shops(shop_id) ON DELETE CASCADE,
    customer_id UUID NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    conversation_id UUID REFERENCES conversations(conversation_id),
    channel order_channel NOT NULL DEFAULT 'payment_link',
    status order_status DEFAULT 'pending',
    line_items JSONB DEFAULT '[]'::jsonb,
    total NUMERIC NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    payment_ref TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    paid_at TIMESTAMPTZ
);

-- product_embeddings table (RAG vector store)
CREATE TABLE product_embeddings (
    embedding_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    shop_id UUID NOT NULL REFERENCES shops(shop_id) ON DELETE CASCADE,
    source_type VARCHAR(20) NOT NULL, -- product, policy, faq, conversation_summary, campaign
    source_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(768),
    conversion_rate NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- outreach_campaigns table
CREATE TABLE outreach_campaigns (
    campaign_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    shop_id UUID NOT NULL REFERENCES shops(shop_id) ON DELETE CASCADE,
    customer_id UUID NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    trigger_id UUID REFERENCES decision_log(log_id),
    trigger_reason TEXT NOT NULL,
    message_draft TEXT NOT NULL,
    status campaign_status DEFAULT 'pending_approval',
    scheduled_send_at TIMESTAMPTZ,
    sent_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- decision_log table (audit trail)
CREATE TABLE decision_log (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    shop_id UUID NOT NULL REFERENCES shops(shop_id) ON DELETE CASCADE,
    customer_id UUID NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    trigger_type TEXT NOT NULL,
    input_snapshot JSONB,
    threshold_crossed BOOLEAN DEFAULT FALSE,
    action_taken TEXT,
    campaign_id UUID REFERENCES outreach_campaigns(campaign_id)
);

-- Create indexes for performance
CREATE INDEX idx_messages_shop_id ON messages(shop_id);
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_customer_id ON messages(customer_id);
CREATE INDEX idx_messages_sent_at ON messages(sent_at DESC);
CREATE INDEX idx_messages_day_of_week ON messages(day_of_week);
CREATE INDEX idx_messages_hour_of_day ON messages(hour_of_day);

CREATE INDEX idx_conversations_shop_id ON conversations(shop_id);
CREATE INDEX idx_conversations_customer_id ON conversations(customer_id);
CREATE INDEX idx_conversations_last_message_at ON conversations(last_message_at DESC);

CREATE INDEX idx_customers_shop_id ON customers(shop_id);
CREATE INDEX idx_customers_psid ON customers(psid);

CREATE INDEX idx_extracted_signals_shop_id ON extracted_signals(shop_id);
CREATE INDEX idx_extracted_signals_message_id ON extracted_signals(message_id);
CREATE INDEX idx_extracted_signals_conversation_id ON extracted_signals(conversation_id);

CREATE INDEX idx_user_metrics_shop_id ON user_metrics(shop_id);
CREATE INDEX idx_user_metrics_customer_id ON user_metrics(customer_id);

CREATE INDEX idx_product_embeddings_shop_id ON product_embeddings(shop_id);
CREATE INDEX idx_product_embeddings_embedding ON product_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_product_embeddings_source ON product_embeddings(source_type, source_id);

CREATE INDEX idx_outreach_campaigns_shop_id ON outreach_campaigns(shop_id);
CREATE INDEX idx_outreach_campaigns_customer_id ON outreach_campaigns(customer_id);
CREATE INDEX idx_outreach_campaigns_status ON outreach_campaigns(status);

CREATE INDEX idx_decision_log_shop_id ON decision_log(shop_id);
CREATE INDEX idx_decision_log_customer_id ON decision_log(customer_id);
CREATE INDEX idx_decision_log_evaluated_at ON decision_log(evaluated_at DESC);
