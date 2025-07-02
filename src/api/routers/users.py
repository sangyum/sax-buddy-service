from fastapi import APIRouter, HTTPException, status, Depends
from src.models.user import User, UserProfile, UserProgress
from src.api.schemas.requests import UserCreate, UserProfileUpdate
from src.services.user_service import UserService

router = APIRouter(prefix="/users", tags=["Users"])


def get_user_service() -> UserService:
    """Dependency to get UserService instance"""
    return UserService()


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