-- Enable RLS on all tables
ALTER TABLE shops ENABLE ROW LEVEL SECURITY;
ALTER TABLE customers ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE extracted_signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE product_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE outreach_campaigns ENABLE ROW LEVEL SECURITY;
ALTER TABLE decision_log ENABLE ROW LEVEL SECURITY;

-- Service role bypasses RLS (for background jobs)
CREATE POLICY service_role_bypass_shops ON shops
    USING (true) TO service_role WITH CHECK (true);

CREATE POLICY service_role_bypass_customers ON customers
    USING (true) TO service_role WITH CHECK (true);

CREATE POLICY service_role_bypass_messages ON messages
    USING (true) TO service_role WITH CHECK (true);

CREATE POLICY service_role_bypass_conversations ON conversations
    USING (true) TO service_role WITH CHECK (true);

CREATE POLICY service_role_bypass_extracted_signals ON extracted_signals
    USING (true) TO service_role WITH CHECK (true);

CREATE POLICY service_role_bypass_user_metrics ON user_metrics
    USING (true) TO service_role WITH CHECK (true);

CREATE POLICY service_role_bypass_orders ON orders
    USING (true) TO service_role WITH CHECK (true);

CREATE POLICY service_role_bypass_product_embeddings ON product_embeddings
    USING (true) TO service_role WITH CHECK (true);

CREATE POLICY service_role_bypass_outreach_campaigns ON outreach_campaigns
    USING (true) TO service_role WITH CHECK (true);

CREATE POLICY service_role_bypass_decision_log ON decision_log
    USING (true) TO service_role WITH CHECK (true);

-- Application policies (for authenticated users with shop_id claim)
-- In production, these would use auth.uid() mapped to shop_id
CREATE POLICY users_own_shops ON shops
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY users_own_customers ON customers
    FOR ALL USING (shop_id IN (SELECT shop_id FROM shops));

CREATE POLICY users_own_messages ON messages
    FOR ALL USING (shop_id IN (SELECT shop_id FROM shops));

CREATE POLICY users_own_conversations ON conversations
    FOR ALL USING (shop_id IN (SELECT shop_id FROM shops));

CREATE POLICY users_own_extracted_signals ON extracted_signals
    FOR ALL USING (shop_id IN (SELECT shop_id FROM shops));

CREATE POLICY users_own_user_metrics ON user_metrics
    FOR ALL USING (shop_id IN (SELECT shop_id FROM shops));

CREATE POLICY users_own_orders ON orders
    FOR ALL USING (shop_id IN (SELECT shop_id FROM shops));

CREATE POLICY users_own_product_embeddings ON product_embeddings
    FOR ALL USING (shop_id IN (SELECT shop_id FROM shops));

CREATE POLICY users_own_outreach_campaigns ON outreach_campaigns
    FOR ALL USING (shop_id IN (SELECT shop_id FROM shops));

CREATE POLICY users_own_decision_log ON decision_log
    FOR ALL USING (shop_id IN (SELECT shop_id FROM shops));
