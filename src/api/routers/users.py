from fastapi import APIRouter, HTTPException, status, Depends
from google.cloud.firestore_v1.async_client import AsyncClient
from src.models.user import User, UserProfile, UserProgress
from src.api.schemas.requests import UserCreate, UserProfileUpdate
from src.services.user_service import UserService
from src.repositories.user_repository import UserRepository
from src.dependencies import get_firestore_client

router = APIRouter(prefix="/users", tags=["Users"])


def get_user_repository(
    firestore_client: AsyncClient = Depends(get_firestore_client)
) -> UserRepository:
    """Dependency to get UserRepository instance with Firestore client"""
    return UserRepository(firestore_client)


def get_user_service(
    user_repository: UserRepository = Depends(get_user_repository)
) -> UserService:
    """Dependency to get UserService instance with repository injected"""
    return UserService(user_repository)


@router.post("", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    user_service: UserService = Depends(get_user_service)
):
    """Create new user"""
    try:
        return await user_service.create_user(user_data)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="User creation not implemented yet"
        )


@router.get("/{user_id}", response_model=User)
async def get_user(
    user_id: str,
    user_service: UserService = Depends(get_user_service)
):
    """Get user by ID"""
    try:
        user = await user_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return user
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="User retrieval not implemented yet"
        )


@router.get("/{user_id}/profile", response_model=UserProfile)
async def get_user_profile(
    user_id: str,
    user_service: UserService = Depends(get_user_service)
):
    """Get user profile"""
    try:
        profile = await user_service.get_user_profile(user_id)
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )
        return profile
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="User profile retrieval not implemented yet"
        )


@router.put("/{user_id}/profile", response_model=UserProfile)
async def update_user_profile(
    user_id: str, 
    profile_data: UserProfileUpdate,
    user_service: UserService = Depends(get_user_service)
):
    """Update user profile"""
    try:
        profile = await user_service.update_user_profile(user_id, profile_data)
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )
        return profile
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="User profile update not implemented yet"
        )


@router.get("/{user_id}/progress", response_model=UserProgress)
async def get_user_progress(
    user_id: str,
    user_service: UserService = Depends(get_user_service)
):
    """Get user progress"""
    try:
        progress = await user_service.get_user_progress(user_id)
        if not progress:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User progress not found"
            )
        return progress
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="User progress retrieval not implemented yet"
        )