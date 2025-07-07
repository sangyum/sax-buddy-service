"""HTTP request/response logging middleware."""

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from src.logging_config import get_logger, RequestContext

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log HTTP requests and responses with structured data."""
    
    def __init__(self, app, exclude_paths: list = []):
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/docs", "/redoc", "/openapi.json", "/health", "/favicon.ico"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip logging for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Extract user information if available
        user_id = None
        if hasattr(request.state, "user") and request.state.user:
            user_id = getattr(request.state.user, "uid", None)
        
        # Start timing
        start_time = time.time()
        
        # Log request start
        with RequestContext(request_id, user_id, request.url.path) as request_logger:
            request_logger.info(
                "HTTP request started",
                method=request.method,
                url=str(request.url),
                path=request.url.path,
                query_params=dict(request.query_params),
                headers=dict(request.headers),
                client_ip=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                content_type=request.headers.get("content-type")
            )
            
            # Process request
            response = None
            try:
                # Add request ID to request state for use in route handlers
                request.state.request_id = request_id
                
                response = await call_next(request)
                duration = time.time() - start_time
                
                # Log successful response
                request_logger.info(
                    "HTTP request completed",
                    status_code=response.status_code,
                    duration_seconds=round(duration, 3),
                    response_size=response.headers.get("content-length"),
                    content_type=response.headers.get("content-type")
                )
                
                return response
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Log error response
                request_logger.error(
                    "HTTP request failed",
                    duration_seconds=round(duration, 3),
                    error=str(e),
                    error_type=type(e).__name__,
                    status_code=getattr(response, "status_code", 500) if response else 500
                )
                
                raise