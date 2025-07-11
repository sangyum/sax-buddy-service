"""JWT authentication middleware"""

import os
from typing import Optional, List
from fastapi import Request
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
from src.auth.firebase_auth import verify_firebase_token, extract_token_from_header
from src.auth.models import AuthenticatedUser, AuthErrorResponse
from src.auth.exceptions import AuthenticationError, MissingTokenError
from src.auth.utils import is_development_mode
from src.logging_config import get_logger

logger = get_logger(__name__)


class JWTMiddleware(BaseHTTPMiddleware):
    """JWT authentication middleware for FastAPI"""
    
    def __init__(
        self,
        app,
        excluded_paths: Optional[List[str]] = None,
        require_auth_by_default: bool = True
    ):
        """Initialize JWT middleware
        
        Args:
            app: FastAPI application instance
            excluded_paths: List of paths to exclude from authentication
            require_auth_by_default: Whether to require auth by default
        """
        super().__init__(app)
        self.excluded_paths = excluded_paths or [
            "/docs",
            "/redoc", 
            "/openapi.json",
            "/health",
            "/favicon.ico"
        ]
        # Disable auth in development mode
        self.is_development = is_development_mode()
        self.require_auth_by_default = require_auth_by_default and not self.is_development
        
        logger.info(
            "JWTMiddleware initialized",
            is_development=self.is_development,
            require_auth_by_default=self.require_auth_by_default,
            environment=os.getenv("ENVIRONMENT", "production")
        )
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request through JWT authentication
        
        Args:
            request: FastAPI request object
            call_next: Next middleware in chain
            
        Returns:
            Response object
        """
        # Skip authentication entirely in development mode
        if self.is_development:
            logger.info(
                "Skipping authentication - development mode",
                path=request.url.path,
                method=request.method,
                is_development=self.is_development
            )
            return await call_next(request)
        
        # Skip authentication for excluded paths
        if self._is_excluded_path(request.url.path):
            logger.info(
                "Skipping authentication - excluded path",
                path=request.url.path,
                method=request.method
            )
            return await call_next(request)
        
        # Skip authentication for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            logger.info(
                "Skipping authentication - OPTIONS request",
                path=request.url.path,
                method=request.method
            )
            return await call_next(request)
        
        logger.info(
            "Processing authentication",
            path=request.url.path,
            method=request.method,
            is_development=self.is_development,
            require_auth=self.require_auth_by_default
        )
        
        try:
            # Extract and verify token
            token = self._extract_token_from_request(request)
            
            if token:
                # Verify token and get user data
                token_data = await verify_firebase_token(token)
                
                # Create authenticated user object
                authenticated_user = AuthenticatedUser.from_jwt_token_data(token_data)
                
                # Store user in request state
                request.state.user = authenticated_user
                request.state.token_data = token_data
            elif self.require_auth_by_default:
                # No token provided and auth is required
                raise MissingTokenError("Authentication token is required")
            
            # Continue to next middleware/endpoint
            response = await call_next(request)
            return response
            
        except AuthenticationError as e:
            # Return authentication error response
            error_response = AuthErrorResponse(
                error="authentication_failed",
                message=e.detail,
                details=None
            )
            return JSONResponse(
                status_code=e.status_code,
                content=error_response.model_dump(),
                headers=e.headers
            )
        except Exception as e:
            # Handle unexpected errors
            error_response = AuthErrorResponse(
                error="internal_error",
                message="Authentication processing failed",
                details=None
            )
            return JSONResponse(
                status_code=500,
                content=error_response.model_dump()
            )
    
    def _is_excluded_path(self, path: str) -> bool:
        """Check if path is excluded from authentication
        
        Args:
            path: Request path
            
        Returns:
            True if path should be excluded from auth
        """
        return path in self.excluded_paths;
        # for excluded_path in self.excluded_paths:
        #     if path.startswith(excluded_path):
        #         return True
        # return False
    
    def _extract_token_from_request(self, request: Request) -> Optional[str]:
        """Extract JWT token from request headers
        
        Args:
            request: FastAPI request object
            
        Returns:
            JWT token string or None if not found
        """
        authorization = request.headers.get("authorization")
        if not authorization:
            return None
        
        try:
            return extract_token_from_header(authorization)
        except Exception:
            return None


# Security scheme for OpenAPI documentation
security = HTTPBearer(
    bearerFormat="JWT",
    description="Firebase ID token required for authentication"
)