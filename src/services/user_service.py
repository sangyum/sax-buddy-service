from typing import Optional

from src.exceptions.not_found import NotFoundException
from models.assessment import SkillLevel
from src.models.user import InitialAssessment, PracticeFrequency, User, UserProfile, UserProgress
from src.api.schemas.requests import InitialAssessmentCreate, UserCreate, UserProfileUpdate
from src.repositories.user_repository import UserRepository


class UserService:
    """Service class for user-related operations"""
    
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository
    
    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user"""
        user = User(
            id="",
            email=user_data.email,
            name=user_data.name,
            is_active=True,
            initial_assessment_completed=False,
            initial_assessment_completed_at=None,
        );
        user = await self.user_repository.create_user(user);
    
        # Create User Profile
        userProfile = UserProfile(
            id="",
            user_id=user.id,
            initial_assessment=None,
            current_skill_level=SkillLevel.BEGINNER,
            practice_frequency=PracticeFrequency.DAILY,
            preferred_practice_duration_minutes=30
        )
        await self.user_repository.create_user_profile(userProfile)
    
        return user
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return await self.user_repository.get_user_by_id(user_id)
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile"""
        return await self.user_repository.get_profile_by_user_id(user_id)
    
    async def update_user_profile(self, user_id: str, profile_data: UserProfileUpdate) -> Optional[UserProfile]:
        """Update user profile"""
        profile = await self.user_repository.get_profile_by_user_id(user_id)
        if not profile:
            raise NotFoundException("User profile not found")
        
        if profile_data.current_skill_level:
            profile.current_skill_level = profile_data.current_skill_level
        if profile_data.learning_goals:
            profile.learning_goals = profile_data.learning_goals
        if profile_data.practice_frequency:
            profile.practice_frequency = profile_data.practice_frequency
        profile.preferred_practice_duration_minutes = profile_data.preferred_practice_duration_minutes
        
        return await self.user_repository.update_user_profile(profile)
    
    async def get_user_progress(self, user_id: str) -> Optional[UserProgress]:
        """Get user progress"""
        return await self.user_repository.get_progress_by_user_id(user_id)
    
    async def add_intial_assessment(self, user_id: str, initial_assessment_data: InitialAssessmentCreate) -> Optional[UserProfile]:
        """Update user profile"""
        profile = await self.user_repository.get_profile_by_user_id(user_id)
        if not profile:
            raise NotFoundException("User profile not found")
        
        initial_assessment = InitialAssessment(
            experience_level=initial_assessment_data.experience_level,
            has_formal_instruction=initial_assessment_data.has_formal_instruction,
            instruction_duration=initial_assessment_data.instruction_duration,
            musical_goals=initial_assessment_data.musical_goals,
            music_reading_level=initial_assessment_data.music_reading_level,
            preferred_learning_style=initial_assessment_data.preferred_learning_style,
            identified_challenges=initial_assessment_data.identified_challenges,
            practice_frequence=initial_assessment_data.practice_frequence,
            preferred_practice_duration_minutes=initial_assessment_data.preferred_practice_duration_minutes
        )
        
        profile.initial_assessment = initial_assessment;

        return await self.user_repository.update_user_profile(profile)
