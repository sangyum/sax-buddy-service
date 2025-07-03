from typing import Optional
from src.models.user import User, UserProfile, UserProgress
from src.api.schemas.requests import UserCreate, UserProfileUpdate
from src.repositories.user_repository import UserRepository


class UserService:
    """Service class for user-related operations"""
    
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository
    
    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user"""
        # TODO: Convert UserCreate to User model and validate
        # For now, delegate to repository once user model is created
        raise NotImplementedError("User creation logic not implemented yet")
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return await self.user_repository.get_user_by_id(user_id)
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile"""
        return await self.user_repository.get_profile_by_user_id(user_id)
    
    async def update_user_profile(self, user_id: str, profile_data: UserProfileUpdate) -> Optional[UserProfile]:
        """Update user profile"""
        # TODO: Get existing profile, apply updates, and save
        # For now, delegate to repository once update logic is implemented
        raise NotImplementedError("User profile update logic not implemented yet")
    
    async def get_user_progress(self, user_id: str) -> Optional[UserProgress]:
        """Get user progress"""
        return await self.user_repository.get_progress_by_user_id(user_id)