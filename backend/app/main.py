from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.db.connection import DatabasePool

# Include routers
from app.routers import webhooks, catalog

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    logger.info("Starting Sellora API...")
    await DatabasePool.init()
    logger.info("Database connection pool initialized")
    yield
    # Shutdown
    logger.info("Shutting down Sellora API...")
    await DatabasePool.close()
    logger.info("Database connection pool closed")


# Create FastAPI app
app = FastAPI(
    title="Sellora API",
    description="AI-Powered Chat-Commerce Platform",
    version="0.1.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(webhooks.router, prefix="/webhooks", tags=["webhooks"])
app.include_router(catalog.router, prefix="/catalog", tags=["catalog"])


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "sellora-api"}


@app.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {
        "name": "Sellora API",
        "version": "0.1.0",
        "status": "running",
    }


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )
