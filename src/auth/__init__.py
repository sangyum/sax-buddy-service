"""Authentication module for Sax Buddy Service"""

from .models import AuthenticatedUser, JWTTokenData, AuthErrorResponse
from .exceptions import (
    AuthenticationError,
    TokenValidationError,
    TokenExpiredError,
    TokenRevokedError,
    InvalidTokenError,
    MissingTokenError,
    InsufficientPermissionsError
)
from .dependencies import (
    get_current_user,
    get_current_token_data,
    get_optional_current_user,
    require_auth,
    require_email_verified,
    require_custom_claim
)
from .firebase_auth import (
    verify_firebase_token,
    initialize_firebase_app,
    get_firebase_user_custom_claims,
    set_firebase_user_custom_claims,
    extract_token_from_header
)
from .middleware import JWTMiddleware

__all__ = [
    # Models
    "AuthenticatedUser",
    "JWTTokenData",
    "AuthErrorResponse",
    # Exceptions
    "AuthenticationError",
    "TokenValidationError",
    "TokenExpiredError",
    "TokenRevokedError",
    "InvalidTokenError",
    "MissingTokenError",
    "InsufficientPermissionsError",
    # Dependencies
    "get_current_user",
    "get_current_token_data",
    "get_optional_current_user",
    "require_auth",
    "require_email_verified",
    "require_custom_claim",
    # Firebase utilities
    "verify_firebase_token",
    "initialize_firebase_app",
    "get_firebase_user_custom_claims",
    "set_firebase_user_custom_claims",
    "extract_token_from_header",
    # Middleware
    "JWTMiddleware",
]