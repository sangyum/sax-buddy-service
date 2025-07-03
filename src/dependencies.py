"""FastAPI dependency functions"""

from fastapi import Request, HTTPException, status
from google.cloud.firestore_v1.async_client import AsyncClient


def get_firestore_client(request: Request) -> AsyncClient:
    """Dependency to get Firestore async client from app state"""
    client = request.app.state.firestore_client
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firestore client is not available"
        )
    return client