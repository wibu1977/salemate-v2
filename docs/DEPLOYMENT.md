# Deployment Guide

This guide covers deploying the Sellora Phase 1 Foundation backend to production.

## Prerequisites

- Supabase account and project
- Railway/Render account (for FastAPI backend)
- Vercel account (for Next.js frontend - Phase 2+)
- Google AI API key (Gemini 2.0 Flash)
- Meta Developer account (for Messenger app)
- Python 3.11+ installed locally

## Quick Start

1. Clone repository and navigate to backend
2. Set up Supabase
3. Deploy backend to Railway
4. Configure environment variables
5. Set up Meta Messenger webhook
6. Run smoke tests to verify

---

## Supabase Setup

### Create Project

1. Go to [supabase.com](https://supabase.com) and create a new project
2. Choose a region close to your users
3. Wait for project initialization (~2 minutes)

### Run Migrations

In Supabase SQL Editor, run these files in order:

```bash
# From your local backend directory
cat supabase/migrations/001_initial_schema.sql
cat supabase/migrations/002_rls_policies.sql
cat supabase/migrations/003_seed_data.sql
```

Copy and paste each file's content into the SQL Editor and execute.

### Enable Extensions

The initial schema automatically enables:
- `uuid-ossp` - for UUID generation
- `pgvector` - for vector embeddings and similarity search

### Get Connection Details

Navigate to Settings > Database and note:
- Connection string (for local development)
- Project URL
- Anon/public API key
- Service role key (for background jobs)

---

## Local Development Setup

### Install Dependencies

```bash
cd backend
py -m pip install -e ".[dev]"
```

### Environment Configuration

Copy `.env.example` to `.env` and fill in values:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Supabase
SUPABASE_URL=your-project-url.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# Google AI
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.0-flash-exp

# Messenger
MESSENGER_VERIFY_TOKEN=your-verify-token
MESSENGER_APP_SECRET=your-app-secret

# Environment
ENVIRONMENT=development
LOG_LEVEL=info
```

### Run Locally

```bash
# Development server with hot reload
py -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or using the run command
py -m uvicorn app.main:app --reload
```

Visit http://localhost:8000 to see API. http://localhost:8000/docs for Swagger UI.

---

## Railway Deployment

### Install Railway CLI

```bash
npm install -g @railway/cli
```

### Login

```bash
railway login
```

### Initialize Project

```bash
cd backend
railway init
```

This creates a `railway.json` configuration file.

### Add PostgreSQL Service (Optional)

If using Railway's managed PostgreSQL:

```bash
railway add postgresql
```

Update `SUPABASE_URL` and keys in Railway environment variables to use Railway's database instead of Supabase.

### Set Environment Variables

```bash
railway variables set SUPABASE_URL="your-supabase-url"
railway variables set SUPABASE_KEY="your-supabase-key"
railway variables set SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"
railway variables set GEMINI_API_KEY="your-gemini-key"
railway variables set GEMINI_MODEL="gemini-2.0-flash-exp"
railway variables set MESSENGER_VERIFY_TOKEN="your-verify-token"
railway variables set MESSENGER_APP_SECRET="your-app-secret"
railway variables set ENVIRONMENT="production"
railway variables set LOG_LEVEL="info"
```

### Deploy

```bash
railway up
```

Railway will:
- Detect the Python project
- Install dependencies from `pyproject.toml`
- Start the FastAPI server
- Assign a public URL

### Get Deployment URL

```bash
railway domain
```

Your backend will be accessible at: `https://your-project.railway.app`

---

## Render Deployment (Alternative)

### Create Account

Go to [render.com](https://render.com) and sign up.

### Create Web Service

1. Click "New +"
2. Select "Web Service"
3. Connect your GitHub repository
4. Select the `backend` folder root
5. Configure:

```
Build Command: pip install -e .
Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### Environment Variables

Add the same variables as Railway in the Render dashboard.

---

## Meta Messenger Webhook Setup

### Create Facebook App

1. Go to [developers.facebook.com](https://developers.facebook.com)
2. Create a new app or select existing
3. Add "Messenger" product

### Configure Webhook

1. In Messenger settings, click "Webhooks"
2. Enter callback URL: `https://your-railway-url.railway.app/webhooks/messenger`
3. Click "Verify and Save"
4. Enter the verify token (must match `MESSENGER_VERIFY_TOKEN`)
5. Subscribe to these events:
   - `messages`
   - `messaging_postbacks`

### Generate Access Token

1. In Messenger settings, generate a page access token
2. Store securely (add to `shops.channels` table or environment)
3. Configure your shop's page ID in the database

### App Secret

1. Go to your app's Settings > Basic
2. Copy the App Secret
3. Set as `MESSENGER_APP_SECRET` environment variable

---

## Testing Deployment

### Run Smoke Tests

The smoke test verifies basic functionality without external services:

```bash
# Test local deployment
py tests/smoke_test.py --host localhost --port 8000

# Test production deployment
py tests/smoke_test.py --host your-project.railway.app --port 443
```

Expected output:
```
============================================================
Sellora Backend Smoke Test
Target: http://localhost:8000

============================================================
Testing Configuration
✓ All required configuration fields present
⚠ SUPABASE_SERVICE_ROLE_KEY not set (service operations limited)

============================================================
Testing Module Imports
✓ app.main
✓ app.config
✓ app.db.connection
...

============================================================
Testing Server Connectivity
✓ Server is running at http://localhost:8000

============================================================
Testing Health Endpoint
✓ Health endpoint returned healthy status

============================================================
Test Summary
Passed: 10
Failed: 0
Warnings: 1
✓ All tests passed! (90.9%)
```

### Manual Tests

```bash
# Health check
curl https://your-project.railway.app/health

# Root endpoint
curl https://your-project.railway.app/

# API documentation
curl https://your-project.railway.app/docs
```

### Test Webhook Verification

```bash
curl -G "https://your-project.railway.app/webhooks/messenger" \
  --data-urlencode "hub.mode=subscribe" \
  --data-urlencode "hub.verify_token=YOUR_TOKEN" \
  --data-urlencode "hub.challenge=test_challenge"
```

Expected: Returns the challenge string if token matches.

---

## Monitoring

### Railway Monitoring

- Go to your project dashboard
- View logs in real-time
- Monitor CPU, memory, and disk usage
- Set up alerts for failures

### Health Check Endpoint

The `/health` endpoint returns:

```json
{
  "status": "healthy",
  "service": "sellora-api",
  "version": "0.1.0",
  "timestamp": "2024-03-30T12:00:00Z"
}
```

Use this for external monitoring (UptimeRobot, Pingdom, etc.).

---

## Troubleshooting

### Database Connection Errors

**Symptom**: `psycopg.OperationalError: could not connect to server`

**Solution**:
- Verify `SUPABASE_URL` is correct
- Check Supabase project is active
- Ensure IP is not blocked (Supabase has IP allowlists)

### Webhook Verification Fails

**Symptom**: Webhook returns 403 during verification

**Solution**:
- Check `MESSENGER_VERIFY_TOKEN` matches exactly
- Verify webhook URL is publicly accessible
- Check for TLS certificate issues

### AI API Timeouts

**Symptom**: `google.api_core.exceptions.GoogleAPIError: 503 Service Unavailable`

**Solution**:
- Verify `GEMINI_API_KEY` is valid
- Check quota limits in Google Cloud Console
- The system continues with rule-based extraction if AI fails (graceful degradation)

### Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'app'`

**Solution**:
- Ensure running from `backend` directory
- Or run with `py -m uvicorn app.main:app` instead of direct command

---

## Production Checklist

- [ ] Supabase migrations run (001, 002, 003)
- [ ] pgvector extension verified
- [ ] Environment variables set in Railway/Render
- [ ] Railway/Render deployment successful
- [ ] Health check returns 200
- [ ] Meta Messenger webhook configured
- [ ] Webhook URL publicly accessible
- [ ] App secret configured
- [ ] Smoke tests passing
- [ ] Logs monitoring set up
- [ ] Error alerting configured
- [ ] Domain configured (if using custom domain)

---

## Security Notes

1. **Never commit secrets**: `.env` file should be in `.gitignore`
2. **Use service role sparingly**: Only use `SUPABASE_SERVICE_ROLE_KEY` for background jobs
3. **Rotate API keys**: Regularly rotate Gemini and Meta keys
4. **Enable HTTPS**: Always use HTTPS for production webhooks
5. **Rate limiting**: Configure limits for webhook endpoints (Railway/Render may provide this)

---

## Scaling Considerations

Phase 1 handles:
- ~1000 concurrent connections (single Railway instance)
- 50ms webhook response time
- Basic RAG with pgvector

For Phase 2+ when scaling:
- Add Railway autoscaling (multiple instances)
- Implement Redis for caching
- Consider CDN for static assets
- Database read replicas for analytics queries

---

## Support

- GitHub Issues: [github.com/your-org/sellora/issues](https://github.com/your-org/sellora/issues)
- Documentation: See `backend/README.md`
- Tech Spec: See `MessengerIQ_TechSpec_v3.md`
