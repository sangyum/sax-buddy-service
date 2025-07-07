"""Authentication dependencies for FastAPI"""

import os
from typing import Optional
from fastapi import Request, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .models import AuthenticatedUser, JWTTokenData
from .firebase_auth import verify_firebase_token, get_firebase_user_custom_claims
from .exceptions import AuthenticationError, MissingTokenError
from .utils import is_development_mode
from src.logging_config import get_logger

logger = get_logger(__name__)

# Security scheme for dependency injection
security = HTTPBearer(bearerFormat="JWT", auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> AuthenticatedUser:
    """Get current authenticated user from request
    
    This dependency can be used in two ways:
    1. With JWT middleware - user is already authenticated and stored in request.state
    2. Without middleware - authenticate using the provided credentials
    
    Args:
        request: FastAPI request object
        credentials: HTTP bearer credentials from Authorization header
        
    Returns:
        AuthenticatedUser object
        
    Raises:
        HTTPException: If authentication fails
    """
    # Skip authentication in development mode
    if is_development_mode():
        logger.info(
            "Returning mock user for development mode",
            environment=os.getenv("ENVIRONMENT", "production"),
            user_id="dev-user"
        )
        # Return a mock user for development
        return AuthenticatedUser(
            user_id="dev-user",
            email="dev@example.com",
            email_verified=True,
            name="Development User",
            picture=None,
            custom_claims={}
        )
    
    # First try to get user from middleware (if JWT middleware is enabled)
    if hasattr(request.state, 'user') and request.state.user:
        return request.state.user
    
    # If no user in state, try to authenticate using credentials
    if not credentials:
        raise MissingTokenError("Authentication token is required")
    
    try:
        # Verify token and get user data
        token_data = await verify_firebase_token(credentials.credentials)
        
        # Get custom claims for the user
        custom_claims = await get_firebase_user_custom_claims(token_data.user_id)
        
        # Create authenticated user object
        authenticated_user = AuthenticatedUser.from_jwt_token_data(token_data, custom_claims)
        
        return authenticated_user
        
    except AuthenticationError:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication processing failed: {str(e)}"
        )


async def get_current_token_data(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> JWTTokenData:
    """Get current JWT token data from request
    
    Args:
        request: FastAPI request object  
        credentials: HTTP bearer credentials from Authorization header
        
    Returns:
        JWTTokenData object
        
    Raises:
        HTTPException: If authentication fails
    """
    # Skip authentication in development mode
    if is_development_mode():
        logger.info(
            "Returning mock token data for development mode",
            environment=os.getenv("ENVIRONMENT", "production"),
            user_id="dev-user"
        )
        # Return mock token data for development
        from datetime import datetime, timezone
        return JWTTokenData(
            user_id="dev-user",
            email="dev@example.com",
            email_verified=True,
            name="Development User",
            picture=None,
            issued_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc),
            audience="dev",
            issuer="dev"
        )
    
    # First try to get token data from middleware
    if hasattr(request.state, 'token_data') and request.state.token_data:
        return request.state.token_data
    
    # If no token data in state, try to authenticate using credentials
    if not credentials:
        raise MissingTokenError("Authentication token is required")
    
    try:
        # Verify token and return token data
        token_data = await verify_firebase_token(credentials.credentials)
        return token_data
        
    except AuthenticationError:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token processing failed: {str(e)}"
        )


async def get_optional_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[AuthenticatedUser]:
    """Get current authenticated user if available (optional authentication)
    
    Args:
        request: FastAPI request object
        credentials: HTTP bearer credentials from Authorization header
        
    Returns:
        AuthenticatedUser object or None if not authenticated
    """
    try:
        return await get_current_user(request, credentials)
    except (AuthenticationError, HTTPException):
        return None


def require_auth(
    current_user: AuthenticatedUser = Depends(get_current_user)
) -> AuthenticatedUser:
    """Dependency that requires authentication
    
    This is a convenience dependency that simply returns the current user.
    Use this when you want to make authentication explicit in your endpoint signature.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        AuthenticatedUser object
    """
    return current_user


def require_email_verified(
    current_user: AuthenticatedUser = Depends(get_current_user)
) -> AuthenticatedUser:
    """Dependency that requires email verification
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        AuthenticatedUser object with verified email
        
    Raises:
        HTTPException: If email is not verified
    """
    if not current_user.email_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required"
        )
    return current_user


def require_custom_claim(claim_name: str, required_value: Optional[str] = None):
    """Factory function to create dependency that requires specific custom claim
    
    Args:
        claim_name: Name of the custom claim to check
        required_value: Optional required value for the claim
        
    Returns:
        Dependency function
    """
    def _require_custom_claim(
        current_user: AuthenticatedUser = Depends(get_current_user)
    ) -> AuthenticatedUser:
        if claim_name not in current_user.custom_claims:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required claim '{claim_name}' not found"
            )
        
        if required_value is not None:
            claim_value = current_user.custom_claims.get(claim_name)
            if claim_value != required_value:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Invalid value for claim '{claim_name}'"
                )
        
        return current_user
    
    return _require_custom_claim