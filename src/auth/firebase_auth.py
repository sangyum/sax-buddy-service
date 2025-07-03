"""Firebase authentication utilities"""

from typing import Optional, Dict, Any
from datetime import datetime
import firebase_admin
from firebase_admin import auth, credentials
from fastapi import HTTPException, status
from .models import JWTTokenData


def initialize_firebase_app(
    credentials_path: Optional[str] = None,
    app_name: Optional[str] = None
) -> firebase_admin.App:
    """Initialize Firebase app with service account credentials
    
    Args:
        credentials_path: Path to service account JSON file
        app_name: Optional app name for multiple Firebase apps
        
    Returns:
        Firebase app instance
    """
    if credentials_path:
        cred = credentials.Certificate(credentials_path)
    else:
        # Use default credentials (useful for Cloud Run deployment)
        cred = credentials.ApplicationDefault()
    
    try:
        if app_name:
            return firebase_admin.initialize_app(cred, name=app_name)
        else:
            return firebase_admin.initialize_app(cred)
    except ValueError as e:
        # App may already be initialized
        if "already exists" in str(e):
            return firebase_admin.get_app(app_name) if app_name else firebase_admin.get_app()
        raise


async def verify_firebase_token(token: str) -> JWTTokenData:
    """Verify Firebase ID token and extract claims
    
    Args:
        token: Firebase ID token string
        
    Returns:
        JWTTokenData with verified token information
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        # Verify the token and get decoded claims
        decoded_token = auth.verify_id_token(token)
        
        # Extract standard claims
        user_id = decoded_token.get('uid')
        email = decoded_token.get('email')
        email_verified = decoded_token.get('email_verified', False)
        name = decoded_token.get('name')
        picture = decoded_token.get('picture')
        
        # Extract timing information
        issued_at = datetime.fromtimestamp(decoded_token.get('iat', 0))
        expires_at = datetime.fromtimestamp(decoded_token.get('exp', 0))
        
        # Extract Firebase-specific claims
        audience = decoded_token.get('aud', '')
        issuer = decoded_token.get('iss', '')
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user ID",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return JWTTokenData(
            user_id=user_id,
            email=email,
            email_verified=email_verified,
            name=name,
            picture=picture,
            issued_at=issued_at,
            expires_at=expires_at,
            audience=audience,
            issuer=issuer
        )
        
    except auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except auth.RevokedIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token verification failed: {str(e)}"
        )


async def get_firebase_user_custom_claims(user_id: str) -> Dict[str, Any]:
    """Get custom claims for a Firebase user
    
    Args:
        user_id: Firebase user ID
        
    Returns:
        Dictionary of custom claims
        
    Raises:
        HTTPException: If user not found or other Firebase error
    """
    try:
        user_record = auth.get_user(user_id)
        return user_record.custom_claims or {}
    except auth.UserNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user custom claims: {str(e)}"
        )


async def set_firebase_user_custom_claims(user_id: str, custom_claims: Dict[str, Any]) -> None:
    """Set custom claims for a Firebase user
    
    Args:
        user_id: Firebase user ID
        custom_claims: Dictionary of custom claims to set
        
    Raises:
        HTTPException: If user not found or other Firebase error
    """
    try:
        auth.set_custom_user_claims(user_id, custom_claims)
    except auth.UserNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set user custom claims: {str(e)}"
        )


def extract_token_from_header(authorization: str) -> str:
    """Extract token from Authorization header
    
    Args:
        authorization: Authorization header value
        
    Returns:
        Token string without 'Bearer ' prefix
        
    Raises:
        HTTPException: If header format is invalid
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    try:
        scheme, token = authorization.split(' ', 1)
        if scheme.lower() != 'bearer':
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme. Expected 'Bearer'",
                headers={"WWW-Authenticate": "Bearer"}
            )
        return token
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format",
            headers={"WWW-Authenticate": "Bearer"}
        )