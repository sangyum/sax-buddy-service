"""Authentication-related Pydantic models"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class JWTTokenData(BaseModel):
    """JWT token payload data structure"""
    user_id: str = Field(..., description="Firebase user ID")
    email: Optional[str] = Field(None, description="User email address")
    email_verified: bool = Field(default=False, description="Whether email is verified")
    name: Optional[str] = Field(None, description="User display name")
    picture: Optional[str] = Field(None, description="User profile picture URL")
    issued_at: datetime = Field(..., description="Token issued at timestamp")
    expires_at: datetime = Field(..., description="Token expiration timestamp")
    audience: str = Field(..., description="Firebase project ID")
    issuer: str = Field(..., description="Token issuer")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "abc123def456",
                "email": "user@example.com",
                "email_verified": True,
                "name": "John Doe",
                "picture": "https://example.com/avatar.jpg",
                "issued_at": "2023-12-01T10:00:00Z",
                "expires_at": "2023-12-01T11:00:00Z",
                "audience": "sax-buddy-app",
                "issuer": "https://securetoken.google.com/sax-buddy-app"
            }
        }


class AuthenticatedUser(BaseModel):
    """Represents an authenticated user in the system"""
    user_id: str = Field(..., description="Firebase user ID")
    email: Optional[str] = Field(None, description="User email address") 
    email_verified: bool = Field(default=False, description="Whether email is verified")
    name: Optional[str] = Field(None, description="User display name")
    picture: Optional[str] = Field(None, description="User profile picture URL")
    custom_claims: Dict[str, Any] = Field(default_factory=dict, description="Custom Firebase claims")
    
    @classmethod
    def from_jwt_token_data(cls, token_data: JWTTokenData, custom_claims: Optional[Dict[str, Any]] = None) -> "AuthenticatedUser":
        """Create AuthenticatedUser from JWT token data"""
        return cls(
            user_id=token_data.user_id,
            email=token_data.email,
            email_verified=token_data.email_verified,
            name=token_data.name,
            picture=token_data.picture,
            custom_claims=custom_claims or {}
        )
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "abc123def456",
                "email": "user@example.com",
                "email_verified": True,
                "name": "John Doe",
                "picture": "https://example.com/avatar.jpg",
                "custom_claims": {
                    "role": "user",
                    "premium": True
                }
            }
        }


class AuthErrorResponse(BaseModel):
    """Authentication error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "token_expired",
                "message": "The provided authentication token has expired",
                "details": {
                    "expired_at": "2023-12-01T10:00:00Z",
                    "current_time": "2023-12-01T10:30:00Z"
                }
            }
        }