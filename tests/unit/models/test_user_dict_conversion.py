"""Tests for User model dict conversion methods."""

import pytest
from datetime import datetime, timezone

from src.models.user import User, UserProfile, UserProgress, SkillLevel, PracticeFrequency


class TestUserDictConversion:
    """Test User model dict conversion methods."""
    
    def test_user_from_dict_with_datetime_strings(self):
        """Test creating User from dict with ISO datetime strings."""
        now = datetime.now(timezone.utc)
        data = {
            "id": "user-123",
            "email": "test@example.com",
            "name": "John Doe",
            "is_active": True,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat()
        }
        
        user = User.from_dict(data)
        
        assert user.id == "user-123"
        assert user.email == "test@example.com"
        assert isinstance(user.created_at, datetime)
        assert user.created_at == now
        assert isinstance(user.updated_at, datetime)
        assert user.updated_at == now
    
    def test_user_to_dict_converts_datetime_to_strings(self):
        """Test converting User to dict with datetime as ISO strings."""
        now = datetime.now(timezone.utc)
        user = User(
            id="user-123",
            email="test@example.com",
            name="John Doe",
            is_active=True,
            created_at=now,
            updated_at=now
        )
        
        data = user.to_dict()
        
        assert data["id"] == "user-123"
        assert data["email"] == "test@example.com"
        assert data["created_at"] == now.isoformat()
        assert data["updated_at"] == now.isoformat()
        assert isinstance(data["created_at"], str)
        assert isinstance(data["updated_at"], str)
    
    def test_user_profile_from_dict_with_datetime_strings(self):
        """Test creating UserProfile from dict with ISO datetime strings."""
        now = datetime.now(timezone.utc)
        data = {
            "id": "profile-123",
            "user_id": "user-123",
            "current_skill_level": "intermediate",
            "learning_goals": ["Improve intonation", "Learn jazz standards"],
            "practice_frequency": "daily",
            "formal_assessment_interval_days": 30,
            "preferred_practice_duration_minutes": 45,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat()
        }
        
        profile = UserProfile.from_dict(data)
        
        assert profile.id == "profile-123"
        assert profile.current_skill_level == SkillLevel.INTERMEDIATE
        assert profile.practice_frequency == PracticeFrequency.DAILY
        assert isinstance(profile.created_at, datetime)
        assert profile.created_at == now
    
    def test_user_progress_from_dict_with_optional_datetime(self):
        """Test creating UserProgress from dict with optional datetime field."""
        now = datetime.now(timezone.utc)
        assessment_date = now.replace(hour=12)
        
        data = {
            "id": "progress-123",
            "user_id": "user-123",
            "intonation_score": 85.5,
            "rhythm_score": 78.2,
            "articulation_score": 92.1,
            "dynamics_score": 80.0,
            "overall_score": 84.0,
            "sessions_count": 10,
            "total_practice_minutes": 300,
            "last_formal_assessment_date": assessment_date.isoformat(),
            "created_at": now.isoformat(),
            "updated_at": now.isoformat()
        }
        
        progress = UserProgress.from_dict(data)
        
        assert progress.id == "progress-123"
        assert isinstance(progress.last_formal_assessment_date, datetime)
        assert progress.last_formal_assessment_date == assessment_date
        assert isinstance(progress.created_at, datetime)
        assert progress.created_at == now
    
    def test_user_progress_from_dict_with_none_assessment_date(self):
        """Test creating UserProgress from dict with None assessment date."""
        now = datetime.now(timezone.utc)
        
        data = {
            "id": "progress-123", 
            "user_id": "user-123",
            "intonation_score": 85.5,
            "rhythm_score": 78.2,
            "articulation_score": 92.1,
            "dynamics_score": 80.0,
            "overall_score": 84.0,
            "sessions_count": 10,
            "total_practice_minutes": 300,
            "last_formal_assessment_date": None,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat()
        }
        
        progress = UserProgress.from_dict(data)
        
        assert progress.last_formal_assessment_date is None
        assert isinstance(progress.created_at, datetime)
    
    def test_user_round_trip_conversion(self):
        """Test that User dict → model → dict preserves data."""
        now = datetime.now(timezone.utc)
        original_data = {
            "id": "user-123",
            "email": "test@example.com",
            "name": "John Doe",
            "is_active": True,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat()
        }
        
        # Dict → Model → Dict
        user = User.from_dict(original_data)
        converted_data = user.to_dict()
        
        assert converted_data == original_data