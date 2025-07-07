import pytest
from datetime import datetime
from uuid import UUID
from src.models.user import User, UserProfile, UserProgress


class TestUser:
    def test_user_creation(self):
        user = User(
            id="550e8400-e29b-41d4-a716-446655440000",
            email="test@example.com",
            name="Test User"
        )
        
        assert user.id == "550e8400-e29b-41d4-a716-446655440000"
        assert user.email == "test@example.com"
        assert user.name == "Test User"
        assert isinstance(user.created_at, datetime)
        assert isinstance(user.updated_at, datetime)
        assert user.is_active is True
    
    def test_user_email_validation(self):
        with pytest.raises(ValueError):
            User(
                id="550e8400-e29b-41d4-a716-446655440000",
                email="invalid-email",
                name="Test User"
            )
    
    def test_user_optional_fields(self):
        user = User(
            id="550e8400-e29b-41d4-a716-446655440000",
            email="test@example.com",
            name="Test User",
            is_active=False
        )
        
        assert user.is_active is False
    
    def test_user_initial_assessment_defaults(self):
        """Test that initial assessment fields have correct default values"""
        user = User(
            id="550e8400-e29b-41d4-a716-446655440000",
            email="test@example.com",
            name="Test User"
        )
        
        assert user.initial_assessment_completed is False
        assert user.initial_assessment_completed_at is None
    
    def test_user_initial_assessment_completion(self):
        """Test setting initial assessment as completed"""
        completion_time = datetime(2023, 12, 1, 10, 0, 0)
        user = User(
            id="550e8400-e29b-41d4-a716-446655440000",
            email="test@example.com",
            name="Test User",
            initial_assessment_completed=True,
            initial_assessment_completed_at=completion_time
        )
        
        assert user.initial_assessment_completed is True
        assert user.initial_assessment_completed_at == completion_time


class TestUserProfile:
    def test_user_profile_creation(self):
        profile = UserProfile(
            id="550e8400-e29b-41d4-a716-446655440001",
            user_id="550e8400-e29b-41d4-a716-446655440000",
            current_skill_level="beginner",
            learning_goals=["improve intonation", "master scales"],
            practice_frequency="daily",
            formal_assessment_interval_days=30
        )
        
        assert profile.id == "550e8400-e29b-41d4-a716-446655440001"
        assert profile.user_id == "550e8400-e29b-41d4-a716-446655440000"
        assert profile.current_skill_level == "beginner"
        assert profile.learning_goals == ["improve intonation", "master scales"]
        assert profile.practice_frequency == "daily"
        assert profile.formal_assessment_interval_days == 30
    
    def test_user_profile_skill_level_validation(self):
        with pytest.raises(ValueError):
            UserProfile(
                id="550e8400-e29b-41d4-a716-446655440001",
                user_id="550e8400-e29b-41d4-a716-446655440000",
                current_skill_level="invalid_level",
                learning_goals=[],
                practice_frequency="daily"
            )
    
    def test_user_profile_practice_frequency_validation(self):
        with pytest.raises(ValueError):
            UserProfile(
                id="550e8400-e29b-41d4-a716-446655440001",
                user_id="550e8400-e29b-41d4-a716-446655440000",
                current_skill_level="beginner",
                learning_goals=[],
                practice_frequency="invalid_frequency"
            )


class TestUserProgress:
    def test_user_progress_creation(self):
        progress = UserProgress(
            id="550e8400-e29b-41d4-a716-446655440002",
            user_id="550e8400-e29b-41d4-a716-446655440000",
            intonation_score=85.5,
            rhythm_score=78.2,
            articulation_score=92.0,
            dynamics_score=80.5,
            overall_score=84.05,
            sessions_count=25,
            total_practice_minutes=750
        )
        
        assert progress.id == "550e8400-e29b-41d4-a716-446655440002"
        assert progress.user_id == "550e8400-e29b-41d4-a716-446655440000"
        assert progress.intonation_score == 85.5
        assert progress.rhythm_score == 78.2
        assert progress.articulation_score == 92.0
        assert progress.dynamics_score == 80.5
        assert progress.overall_score == 84.05
        assert progress.sessions_count == 25
        assert progress.total_practice_minutes == 750
    
    def test_user_progress_score_validation(self):
        with pytest.raises(ValueError):
            UserProgress(
                id="550e8400-e29b-41d4-a716-446655440002",
                user_id="550e8400-e29b-41d4-a716-446655440000",
                intonation_score=105.0,  # Invalid: > 100
                rhythm_score=78.2,
                articulation_score=92.0,
                dynamics_score=80.5,
                overall_score=84.05,
                sessions_count=25,
                total_practice_minutes=750
            )
    
    def test_user_progress_negative_values(self):
        with pytest.raises(ValueError):
            UserProgress(
                id="550e8400-e29b-41d4-a716-446655440002",
                user_id="550e8400-e29b-41d4-a716-446655440000",
                intonation_score=85.5,
                rhythm_score=78.2,
                articulation_score=92.0,
                dynamics_score=80.5,
                overall_score=84.05,
                sessions_count=-5,  # Invalid: negative
                total_practice_minutes=750
            )