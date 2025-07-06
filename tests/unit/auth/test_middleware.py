"""Tests for JWT authentication middleware"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse
from src.auth.middleware import JWTMiddleware
from src.auth.models import AuthenticatedUser, JWTTokenData
from src.auth.exceptions import AuthenticationError, MissingTokenError


class TestJWTMiddleware:
    """Test JWT authentication middleware"""

    def test_middleware_initialization_production_mode(self):
        """Test middleware initialization in production mode"""
        app = FastAPI()
        with patch.dict('os.environ', {'ENVIRONMENT': 'production'}):
            middleware = JWTMiddleware(app)
            assert middleware.require_auth_by_default is True
            assert middleware.is_development is False

    def test_middleware_initialization_development_mode(self):
        """Test middleware initialization in development mode"""
        app = FastAPI()
        with patch.dict('os.environ', {'ENVIRONMENT': 'development'}):
            middleware = JWTMiddleware(app)
            assert middleware.require_auth_by_default is False
            assert middleware.is_development is True

    def test_middleware_initialization_no_environment(self):
        """Test middleware initialization with no environment variable"""
        app = FastAPI()
        with patch.dict('os.environ', {}, clear=True):
            middleware = JWTMiddleware(app)
            assert middleware.require_auth_by_default is True
            assert middleware.is_development is False

    @pytest.mark.asyncio
    async def test_development_mode_skips_authentication(self):
        """Test that development mode skips authentication entirely"""
        app = FastAPI()
        
        # Mock the call_next function
        mock_response = Mock()
        call_next = AsyncMock(return_value=mock_response)
        
        # Create request mock
        request = Mock(spec=Request)
        request.url.path = "/v1/users"
        request.method = "GET"
        
        with patch.dict('os.environ', {'ENVIRONMENT': 'development'}):
            middleware = JWTMiddleware(app)
            response = await middleware.dispatch(request, call_next)
            
        # Should call next middleware without any authentication
        call_next.assert_called_once_with(request)
        assert response == mock_response

    @pytest.mark.asyncio
    async def test_production_mode_requires_authentication(self):
        """Test that production mode requires authentication"""
        app = FastAPI()
        
        # Mock the call_next function
        call_next = AsyncMock()
        
        # Create request mock without authorization header
        request = Mock(spec=Request)
        request.url.path = "/v1/users"
        request.method = "GET"
        request.headers = {}
        
        with patch.dict('os.environ', {'ENVIRONMENT': 'production'}):
            middleware = JWTMiddleware(app)
            response = await middleware.dispatch(request, call_next)
            
        # Should return authentication error
        assert isinstance(response, JSONResponse)
        assert response.status_code == 401
        
        # Should not call next middleware
        call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_excluded_paths_skip_authentication(self):
        """Test that excluded paths skip authentication in production"""
        app = FastAPI()
        
        # Mock the call_next function
        mock_response = Mock()
        call_next = AsyncMock(return_value=mock_response)
        
        # Create request mock for excluded path
        request = Mock(spec=Request)
        request.url.path = "/docs"
        request.method = "GET"
        
        with patch.dict('os.environ', {'ENVIRONMENT': 'production'}):
            middleware = JWTMiddleware(app)
            response = await middleware.dispatch(request, call_next)
            
        # Should call next middleware without authentication
        call_next.assert_called_once_with(request)
        assert response == mock_response

    @pytest.mark.asyncio
    async def test_options_requests_skip_authentication(self):
        """Test that OPTIONS requests skip authentication"""
        app = FastAPI()
        
        # Mock the call_next function
        mock_response = Mock()
        call_next = AsyncMock(return_value=mock_response)
        
        # Create request mock for OPTIONS request
        request = Mock(spec=Request)
        request.url.path = "/v1/users"
        request.method = "OPTIONS"
        
        with patch.dict('os.environ', {'ENVIRONMENT': 'production'}):
            middleware = JWTMiddleware(app)
            response = await middleware.dispatch(request, call_next)
            
        # Should call next middleware without authentication
        call_next.assert_called_once_with(request)
        assert response == mock_response

    def test_is_excluded_path_method(self):
        """Test _is_excluded_path method"""
        app = FastAPI()
        middleware = JWTMiddleware(app)
        
        # Test excluded paths
        assert middleware._is_excluded_path("/docs") is True
        assert middleware._is_excluded_path("/docs/swagger") is True
        assert middleware._is_excluded_path("/redoc") is True
        assert middleware._is_excluded_path("/health") is True
        
        # Test non-excluded paths
        assert middleware._is_excluded_path("/v1/users") is False
        assert middleware._is_excluded_path("/api/test") is False

    def test_extract_token_from_request_method(self):
        """Test _extract_token_from_request method"""
        app = FastAPI()
        middleware = JWTMiddleware(app)
        
        # Test with valid authorization header
        request = Mock(spec=Request)
        request.headers = {"authorization": "Bearer valid-token"}
        
        with patch('src.auth.middleware.extract_token_from_header', return_value="valid-token"):
            token = middleware._extract_token_from_request(request)
            assert token == "valid-token"
        
        # Test with no authorization header
        request.headers = {}
        token = middleware._extract_token_from_request(request)
        assert token is None
        
        # Test with invalid authorization header
        request.headers = {"authorization": "Invalid header"}
        with patch('src.auth.middleware.extract_token_from_header', side_effect=Exception("Invalid")):
            token = middleware._extract_token_from_request(request)
            assert token is None