"""Unit tests for authentication exceptions"""

import pytest
from fastapi import status
from src.auth.exceptions import (
    AuthenticationError,
    TokenValidationError,
    TokenExpiredError,
    TokenRevokedError,
    InvalidTokenError,
    MissingTokenError,
    InsufficientPermissionsError
)


class TestAuthenticationExceptions:
    """Test authentication exception hierarchy"""
    
    def test_authentication_error_default(self):
        """Test default authentication error"""
        error = AuthenticationError()
        
        assert error.status_code == status.HTTP_401_UNAUTHORIZED
        assert error.detail == "Authentication failed"
        assert error.headers == {"WWW-Authenticate": "Bearer"}
    
    def test_authentication_error_custom_message(self):
        """Test authentication error with custom message"""
        custom_message = "Custom authentication failure"
        error = AuthenticationError(custom_message)
        
        assert error.status_code == status.HTTP_401_UNAUTHORIZED
        assert error.detail == custom_message
        assert error.headers == {"WWW-Authenticate": "Bearer"}
    
    def test_token_validation_error_default(self):
        """Test default token validation error"""
        error = TokenValidationError()
        
        assert error.status_code == status.HTTP_401_UNAUTHORIZED
        assert error.detail == "Token validation failed"
        assert error.headers == {"WWW-Authenticate": "Bearer"}
    
    def test_token_validation_error_custom_message(self):
        """Test token validation error with custom message"""
        custom_message = "Custom token validation failure"
        error = TokenValidationError(custom_message)
        
        assert error.status_code == status.HTTP_401_UNAUTHORIZED
        assert error.detail == custom_message
        assert error.headers == {"WWW-Authenticate": "Bearer"}
    
    def test_token_expired_error_default(self):
        """Test default token expired error"""
        error = TokenExpiredError()
        
        assert error.status_code == status.HTTP_401_UNAUTHORIZED
        assert error.detail == "Token has expired"
        assert error.headers == {"WWW-Authenticate": "Bearer"}
    
    def test_token_revoked_error_default(self):
        """Test default token revoked error"""
        error = TokenRevokedError()
        
        assert error.status_code == status.HTTP_401_UNAUTHORIZED
        assert error.detail == "Token has been revoked"
        assert error.headers == {"WWW-Authenticate": "Bearer"}
    
    def test_invalid_token_error_default(self):
        """Test default invalid token error"""
        error = InvalidTokenError()
        
        assert error.status_code == status.HTTP_401_UNAUTHORIZED
        assert error.detail == "Invalid token"
        assert error.headers == {"WWW-Authenticate": "Bearer"}
    
    def test_missing_token_error_default(self):
        """Test default missing token error"""
        error = MissingTokenError()
        
        assert error.status_code == status.HTTP_401_UNAUTHORIZED
        assert error.detail == "Authentication token is required"
        assert error.headers == {"WWW-Authenticate": "Bearer"}
    
    def test_insufficient_permissions_error_default(self):
        """Test default insufficient permissions error"""
        error = InsufficientPermissionsError()
        
        assert error.status_code == status.HTTP_403_FORBIDDEN
        assert error.detail == "Insufficient permissions"
        # Should not have WWW-Authenticate header for 403 errors
        assert not hasattr(error, 'headers') or error.headers is None
    
    def test_insufficient_permissions_error_custom_message(self):
        """Test insufficient permissions error with custom message"""
        custom_message = "Admin access required"
        error = InsufficientPermissionsError(custom_message)
        
        assert error.status_code == status.HTTP_403_FORBIDDEN
        assert error.detail == custom_message
    
    def test_exception_inheritance(self):
        """Test that all token errors inherit from AuthenticationError"""
        # These should inherit from AuthenticationError
        assert issubclass(TokenValidationError, AuthenticationError)
        assert issubclass(TokenExpiredError, AuthenticationError)
        assert issubclass(TokenRevokedError, AuthenticationError)
        assert issubclass(InvalidTokenError, AuthenticationError)
        assert issubclass(MissingTokenError, AuthenticationError)
        
        # This should NOT inherit from AuthenticationError
        assert not issubclass(InsufficientPermissionsError, AuthenticationError)