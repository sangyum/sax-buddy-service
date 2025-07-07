"""Unit tests for UserService"""
from unittest.mock import Mock
from api.schemas.requests import UserProfileUpdate
from models.assessment import SkillLevel
from models.user import PracticeFrequency
import pytest
from services.user_service import UserService
from src.repositories.user_repository import UserRepository

class TestUserService:
    """Test UserService methods"""

    @pytest.fixture
    def mock_user_repository(self):
        """Create mock UserRepository"""
        repository = Mock(spec=UserRepository)
        return repository
    
    @pytest.fixture
    def user_service(self, mock_user_repository):
        """Create UserService with mock repository"""
        return UserService(mock_user_repository)

    @pytest.mark.asyncio
    async def test_update_user_profile(self, user_service, mock_user_repository):
        """Test updating user profile"""
        user_id = "user-profile_id"
        user_profile_update = UserProfileUpdate(
            current_skill_level=SkillLevel.ADVANCED,
            learning_goals=["a", "b", "c"],
            practice_frequency=PracticeFrequency.DAILY,
            preferred_practice_duration_minutes=45
        )

        result = await user_service.update_user_profile(user_id, user_profile_update)

        assert result is not None

