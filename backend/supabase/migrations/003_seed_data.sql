-- Insert a test shop for development
INSERT INTO shops (shop_id, owner_email, plan_tier, ai_system_prompt, channels) VALUES
('00000000-0000-0000-0000-000000000001', 'test@example.com', 'starter',
 'You are a helpful sales assistant for a fashion store. Be friendly, concise, and helpful. Focus on product recommendations and answering questions naturally.',
 '[{"channel_type": "messenger", "page_id": "test_page_id", "access_token": "test_token"}]'::jsonb)
ON CONFLICT (owner_email) DO NOTHING;
