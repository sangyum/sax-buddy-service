"""FastAPI dependency functions"""

from fastapi import Request, HTTPException, status
from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.storage.bucket import Bucket


def get_firestore_client(request: Request) -> AsyncClient:
    """Dependency to get Firestore async client from app state"""
    client = request.app.state.firestore_client
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firestore client is not available"
        )
    return client

def get_firestore_bucket(request: Request) -> Bucket:
    """Dependency to get Firestore Storage Bucket from app state"""
    bucket = request.app.state.bucket
    if bucket is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firestore Storage Bucket is not available"
        )
    return bucket