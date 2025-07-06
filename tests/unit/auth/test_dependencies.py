"""Tests for authentication dependencies"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone
from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials
from src.auth.dependencies import get_current_user, get_current_token_data, get_optional_current_user
from src.auth.models import AuthenticatedUser, JWTTokenData
from src.auth.exceptions import AuthenticationError, MissingTokenError


class TestAuthenticationDependencies:
    """Test authentication dependencies"""

    @pytest.mark.asyncio
    async def test_get_current_user_development_mode(self):
        """Test get_current_user returns mock user in development mode"""
        request = Mock(spec=Request)
        credentials = Mock(spec=HTTPAuthorizationCredentials)
        
        with patch.dict('os.environ', {'ENVIRONMENT': 'development'}):
            user = await get_current_user(request, credentials)
            
        assert isinstance(user, AuthenticatedUser)
        assert user.user_id == "dev-user"
        assert user.email == "dev@example.com"
        assert user.email_verified is True
        assert user.name == "Development User"
        assert user.custom_claims == {}

    @pytest.mark.asyncio
    async def test_get_current_user_production_mode_with_middleware(self):
        """Test get_current_user uses middleware user in production"""
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.user = AuthenticatedUser(
            user_id="test-user",
            email="test@example.com",
            email_verified=True,
            name="Test User",
            custom_claims={}
        )
        credentials = Mock(spec=HTTPAuthorizationCredentials)
        
        with patch.dict('os.environ', {'ENVIRONMENT': 'production'}):
            user = await get_current_user(request, credentials)
            
        assert user.user_id == "test-user"
        assert user.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_get_current_user_production_mode_no_middleware_no_credentials(self):
        """Test get_current_user fails without credentials in production"""
        request = Mock(spec=Request)
        request.state = Mock()
        # Ensure no user in state
        del request.state.user
        
        with patch.dict('os.environ', {'ENVIRONMENT': 'production'}):
            with pytest.raises(MissingTokenError):
                await get_current_user(request, None)

    @pytest.mark.asyncio
    async def test_get_current_token_data_development_mode(self):
        """Test get_current_token_data returns mock token in development mode"""
        request = Mock(spec=Request)
        credentials = Mock(spec=HTTPAuthorizationCredentials)
        
        with patch.dict('os.environ', {'ENVIRONMENT': 'development'}):
            token_data = await get_current_token_data(request, credentials)
            
        assert isinstance(token_data, JWTTokenData)
        assert token_data.user_id == "dev-user"
        assert token_data.email == "dev@example.com"
        assert token_data.email_verified is True
        assert token_data.name == "Development User"
        assert token_data.issuer == "dev"
        assert token_data.audience == "dev"

    @pytest.mark.asyncio
    async def test_get_current_token_data_production_mode_with_middleware(self):
        """Test get_current_token_data uses middleware token in production"""
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.token_data = JWTTokenData(
            user_id="test-user",
            email="test@example.com",
            email_verified=True,
            name="Test User",
            issued_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc),
            audience="test-project",
            issuer="firebase"
        )
        credentials = Mock(spec=HTTPAuthorizationCredentials)
        
        with patch.dict('os.environ', {'ENVIRONMENT': 'production'}):
            token_data = await get_current_token_data(request, credentials)
            
        assert token_data.user_id == "test-user"
        assert token_data.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_get_current_token_data_production_mode_no_middleware_no_credentials(self):
        """Test get_current_token_data fails without credentials in production"""
        request = Mock(spec=Request)
        request.state = Mock()
        # Ensure no token_data in state
        del request.state.token_data
        
        with patch.dict('os.environ', {'ENVIRONMENT': 'production'}):
            with pytest.raises(MissingTokenError):
                await get_current_token_data(request, None)

    @pytest.mark.asyncio
    async def test_get_optional_current_user_development_mode(self):
        """Test get_optional_current_user returns mock user in development mode"""
        request = Mock(spec=Request)
        credentials = Mock(spec=HTTPAuthorizationCredentials)
        
        with patch.dict('os.environ', {'ENVIRONMENT': 'development'}):
            user = await get_optional_current_user(request, credentials)
            
        assert isinstance(user, AuthenticatedUser)
        assert user.user_id == "dev-user"

    @pytest.mark.asyncio
    async def test_get_optional_current_user_production_mode_no_credentials(self):
        """Test get_optional_current_user returns None without credentials in production"""
        request = Mock(spec=Request)
        request.state = Mock()
        # Ensure no user in state
        del request.state.user
        
        with patch.dict('os.environ', {'ENVIRONMENT': 'production'}):
            user = await get_optional_current_user(request, None)
            
        assert user is None