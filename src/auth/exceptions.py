"""Authentication-related exceptions"""

from fastapi import HTTPException, status


class AuthenticationError(HTTPException):
    """Base authentication error"""
    
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"}
        )


class TokenValidationError(AuthenticationError):
    """Token validation specific error"""
    
    def __init__(self, detail: str = "Token validation failed"):
        super().__init__(detail=detail)


class TokenExpiredError(AuthenticationError):
    """Token expired error"""
    
    def __init__(self, detail: str = "Token has expired"):
        super().__init__(detail=detail)


class TokenRevokedError(AuthenticationError):
    """Token revoked error"""
    
    def __init__(self, detail: str = "Token has been revoked"):
        super().__init__(detail=detail)


class InvalidTokenError(AuthenticationError):
    """Invalid token error"""
    
    def __init__(self, detail: str = "Invalid token"):
        super().__init__(detail=detail)


class MissingTokenError(AuthenticationError):
    """Missing token error"""
    
    def __init__(self, detail: str = "Authentication token is required"):
        super().__init__(detail=detail)


class InsufficientPermissionsError(HTTPException):
    """Insufficient permissions error"""
    
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail
        )