import logging
import time

import fastapi
import uvicorn
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from agents.agent.mcp import mcp_sse
from agents.api import agent_router, api_router, file_router, tool_router, prompt_router, model_router, image_router, \
    category_router, open_router
from agents.api.ai_image_router import router as ai_image_router
from agents.api.api_router import register_startup_events
from agents.api.auth_router import router as auth_router
from agents.api.data_router import router as data_router
from agents.api.profile_router import router as profiles_router
from agents.api.mcp_router import router as mcp_router
from agents.api.vip_router import router as vip_router
from agents.common.config import SETTINGS
from agents.common.log import Log
from agents.common.otel import Otel, OtelFastAPI
from agents.middleware.auth_middleware import JWTAuthMiddleware
from agents.middleware.gobal import exception_handler
from agents.models.db import SessionLocal
from agents.models.db_monitor import start_db_monitor, stop_db_monitor

logger = logging.getLogger(__name__)


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to measure and log HTTP request execution time.
    
    This middleware calculates the time elapsed during request processing
    and logs it along with the request method, path, and status code.
    """
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        
    async def dispatch(self, request: Request, call_next):
        # Record start time
        start_time = time.time()
        
        # Process the request
        response = await call_next(request)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        elapsed_ms = elapsed_time * 1000
        
        # Log the timing information
        logger.info(
            f"HTTP {request.method} {request.url.path} -> {response.status_code} took {elapsed_ms:.2f}ms"
        )
        
        return response


def create_app() -> FastAPI:
    # Initialize logging and telemetry
    Log.init()
    Otel.init()
    logger.info("Server starting...")

    app = FastAPI()

    # Add database session to app state
    app.state.db = SessionLocal

    # Register startup and shutdown events
    register_startup_events(app)
    
    # Register event to start database monitoring
    @app.on_event("startup")
    async def start_monitoring():
        """Start database connection monitoring"""
        logger.info("Starting database connection monitoring...")
        await start_db_monitor(log_level=logging.INFO)
        logger.info("Database connection monitoring started")
    
    # Register event to stop database monitoring
    @app.on_event("shutdown")
    async def stop_monitoring():
        """Stop database connection monitoring"""
        logger.info("Stopping database connection monitoring...")
        await stop_db_monitor()
        logger.info("Database connection monitoring stopped")

    # Add HTTP request timing middleware
    app.add_middleware(TimingMiddleware)
    
    # Add JWT middleware
    app.add_middleware(JWTAuthMiddleware)

    @app.exception_handler(Exception)
    async def default_exception_handler(request: fastapi.Request, exc):
        return await exception_handler(request, exc)

    # Include routers
    app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
    app.include_router(api_router.router)
    app.include_router(agent_router.router, prefix="/api", tags=["agent"])
    app.include_router(model_router.router, prefix="/api", tags=["model"])
    app.include_router(file_router.router, prefix="/api", tags=["file"])
    app.include_router(tool_router.router, prefix="/api", tags=["tool"])
    app.include_router(prompt_router.router, prefix="/api", tags=["prompt"])
    app.include_router(image_router.router, prefix="/api", tags=["images"])
    app.include_router(category_router.router, prefix="/api", tags=["category"])
    app.include_router(open_router.router, prefix="/api/open", tags=["open"])
    app.include_router(data_router, prefix="/api/p", tags=["data"])
    app.include_router(mcp_router, prefix="/api", tags=["mcp"])
    app.include_router(ai_image_router, prefix="/api", tags=["ai_image"])
    app.include_router(vip_router, prefix="/api", tags=["vip"])
    app.include_router(profiles_router, prefix="/api", tags=["profiles"])

    # add mcp
    app.mount("/", mcp_sse.get_application())

    # Initialize OpenTelemetry
    OtelFastAPI.init(app)

    logger.info("Application initialization complete")
    return app


# app = create_app()

if __name__ == '__main__':
    uvicorn.run("api:create_app", host=SETTINGS.HOST, port=SETTINGS.PORT, workers=SETTINGS.WORKERS)
