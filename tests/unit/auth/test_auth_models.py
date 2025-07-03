"""Unit tests for authentication models"""

import pytest
from datetime import datetime, timedelta
from src.auth.models import JWTTokenData, AuthenticatedUser, AuthErrorResponse


class TestJWTTokenData:
    """Test JWT token data model"""
    
    def test_jwt_token_data_creation(self):
        """Test creating JWT token data with all fields"""
        issued_at = datetime.now()
        expires_at = issued_at + timedelta(hours=1)
        
        token_data = JWTTokenData(
            user_id="abc123",
            email="test@example.com",
            email_verified=True,
            name="Test User",
            picture="https://example.com/avatar.jpg",
            issued_at=issued_at,
            expires_at=expires_at,
            audience="test-project",
            issuer="https://securetoken.google.com/test-project"
        )
        
        assert token_data.user_id == "abc123"
        assert token_data.email == "test@example.com"
        assert token_data.email_verified is True
        assert token_data.name == "Test User"
        assert token_data.picture == "https://example.com/avatar.jpg"
        assert token_data.issued_at == issued_at
        assert token_data.expires_at == expires_at
        assert token_data.audience == "test-project"
        assert token_data.issuer == "https://securetoken.google.com/test-project"
    
    def test_jwt_token_data_minimal(self):
        """Test creating JWT token data with minimal fields"""
        issued_at = datetime.now()
        expires_at = issued_at + timedelta(hours=1)
        
        token_data = JWTTokenData(
            user_id="abc123",
            issued_at=issued_at,
            expires_at=expires_at,
            audience="test-project",
            issuer="https://securetoken.google.com/test-project"
        )
        
        assert token_data.user_id == "abc123"
        assert token_data.email is None
        assert token_data.email_verified is False
        assert token_data.name is None
        assert token_data.picture is None


class TestAuthenticatedUser:
    """Test authenticated user model"""
    
    def test_authenticated_user_creation(self):
        """Test creating authenticated user with all fields"""
        user = AuthenticatedUser(
            user_id="abc123",
            email="test@example.com",
            email_verified=True,
            name="Test User",
            picture="https://example.com/avatar.jpg",
            custom_claims={"role": "admin", "premium": True}
        )
        
        assert user.user_id == "abc123"
        assert user.email == "test@example.com"
        assert user.email_verified is True
        assert user.name == "Test User"
        assert user.picture == "https://example.com/avatar.jpg"
        assert user.custom_claims == {"role": "admin", "premium": True}
    
    def test_authenticated_user_from_jwt_token_data(self):
        """Test creating authenticated user from JWT token data"""
        issued_at = datetime.now()
        expires_at = issued_at + timedelta(hours=1)
        
        token_data = JWTTokenData(
            user_id="abc123",
            email="test@example.com",
            email_verified=True,
            name="Test User",
            picture="https://example.com/avatar.jpg",
            issued_at=issued_at,
            expires_at=expires_at,
            audience="test-project",
            issuer="https://securetoken.google.com/test-project"
        )
        
        custom_claims = {"role": "user", "premium": False}
        user = AuthenticatedUser.from_jwt_token_data(token_data, custom_claims)
        
        assert user.user_id == "abc123"
        assert user.email == "test@example.com"
        assert user.email_verified is True
        assert user.name == "Test User"
        assert user.picture == "https://example.com/avatar.jpg"
        assert user.custom_claims == custom_claims
    
    def test_authenticated_user_minimal(self):
        """Test creating authenticated user with minimal fields"""
        user = AuthenticatedUser(user_id="abc123")
        
        assert user.user_id == "abc123"
        assert user.email is None
        assert user.email_verified is False
        assert user.name is None
        assert user.picture is None
        assert user.custom_claims == {}


class TestAuthErrorResponse:
    """Test authentication error response model"""
    
    def test_auth_error_response_creation(self):
        """Test creating authentication error response"""
        error = AuthErrorResponse(
            error="token_expired",
            message="The authentication token has expired",
            details={"expired_at": "2023-12-01T10:00:00Z"}
        )
        
        assert error.error == "token_expired"
        assert error.message == "The authentication token has expired"
        assert error.details == {"expired_at": "2023-12-01T10:00:00Z"}
    
    def test_auth_error_response_minimal(self):
        """Test creating authentication error response with minimal fields"""
        error = AuthErrorResponse(
            error="invalid_token",
            message="Invalid authentication token"
        )
        
        assert error.error == "invalid_token"
        assert error.message == "Invalid authentication token"
        assert error.details is None