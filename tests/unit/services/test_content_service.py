"""Unit tests for ContentService"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock
from src.services.content_service import ContentService
from src.repositories.content_repository import ContentRepository
from src.api.schemas.requests import LessonUpdate
from src.models.content import Lesson, DifficultyLevel


class TestContentService:
    """Test ContentService methods"""

    @pytest.fixture
    def mock_content_repository(self):
        """Create mock ContentRepository"""
        repository = Mock(spec=ContentRepository)
        # Make all repository methods async mocks
        repository.get_lesson_by_id = AsyncMock()
        repository.update_lesson = AsyncMock()
        return repository

    @pytest.fixture
    def content_service(self, mock_content_repository):
        """Create ContentService with mock repository"""
        return ContentService(mock_content_repository)

    @pytest.fixture
    def sample_lesson(self):
        """Create sample lesson for testing"""
        return Lesson(
            id="lesson123",
            lesson_plan_id="plan456",
            title="Test Lesson",
            description="A test lesson",
            order_in_plan=1,
            exercise_ids=["exercise1", "exercise2"],
            estimated_duration_minutes=30,
            learning_objectives=["Learn scales"],
            prerequisites=["Basic fingering"],
            is_completed=False,
            completed_at=None,
            created_at=datetime(2023, 1, 1, 12, 0, 0)
        )

    @pytest.mark.asyncio
    async def test_update_lesson_completion_status(self, content_service, mock_content_repository, sample_lesson):
        """Test updating lesson completion status"""
        # Setup
        lesson_id = "lesson123"
        lesson_update = LessonUpdate(
            is_completed=True,
            completed_at="2023-12-01T15:30:00"
        )
        
        # Mock repository to return existing lesson
        mock_content_repository.get_lesson_by_id.return_value = sample_lesson
        
        # Mock repository update to return updated lesson
        expected_updated_lesson = sample_lesson.model_copy(update={
            "is_completed": True,
            "completed_at": datetime.fromisoformat("2023-12-01T15:30:00")
        })
        mock_content_repository.update_lesson.return_value = expected_updated_lesson
        
        # Execute
        result = await content_service.update_lesson(lesson_id, lesson_update)
        
        # Verify
        assert result is not None
        assert result.is_completed is True
        assert result.completed_at == datetime(2023, 12, 1, 15, 30, 0)
        
        # Verify repository calls
        mock_content_repository.get_lesson_by_id.assert_called_once_with(lesson_id)
        mock_content_repository.update_lesson.assert_called_once()
        
        # Verify the lesson passed to update has correct values
        updated_lesson_arg = mock_content_repository.update_lesson.call_args[0][0]
        assert updated_lesson_arg.is_completed is True
        assert updated_lesson_arg.completed_at == datetime(2023, 12, 1, 15, 30, 0)

    @pytest.mark.asyncio
    async def test_update_lesson_partial_update(self, content_service, mock_content_repository, sample_lesson):
        """Test updating only completion status without date"""
        # Setup
        lesson_id = "lesson123"
        lesson_update = LessonUpdate(is_completed=True)
        
        # Mock repository to return existing lesson
        mock_content_repository.get_lesson_by_id.return_value = sample_lesson
        
        # Mock repository update to return updated lesson
        expected_updated_lesson = sample_lesson.model_copy(update={"is_completed": True})
        mock_content_repository.update_lesson.return_value = expected_updated_lesson
        
        # Execute
        result = await content_service.update_lesson(lesson_id, lesson_update)
        
        # Verify
        assert result is not None
        assert result.is_completed is True
        assert result.completed_at is None  # Should remain unchanged
        
        # Verify repository calls
        mock_content_repository.get_lesson_by_id.assert_called_once_with(lesson_id)
        mock_content_repository.update_lesson.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_lesson_not_found(self, content_service, mock_content_repository):
        """Test updating non-existent lesson"""
        # Setup
        lesson_id = "nonexistent"
        lesson_update = LessonUpdate(is_completed=True)
        
        # Mock repository to return None (lesson not found)
        mock_content_repository.get_lesson_by_id.return_value = None
        
        # Execute
        result = await content_service.update_lesson(lesson_id, lesson_update)
        
        # Verify
        assert result is None
        
        # Verify repository calls
        mock_content_repository.get_lesson_by_id.assert_called_once_with(lesson_id)
        mock_content_repository.update_lesson.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_lesson_empty_update(self, content_service, mock_content_repository, sample_lesson):
        """Test updating lesson with empty update (no changes)"""
        # Setup
        lesson_id = "lesson123"
        lesson_update = LessonUpdate()  # No fields set
        
        # Mock repository to return existing lesson
        mock_content_repository.get_lesson_by_id.return_value = sample_lesson
        
        # Mock repository update to return same lesson
        mock_content_repository.update_lesson.return_value = sample_lesson
        
        # Execute
        result = await content_service.update_lesson(lesson_id, lesson_update)
        
        # Verify
        assert result is not None
        assert result.is_completed == sample_lesson.is_completed
        assert result.completed_at == sample_lesson.completed_at
        
        # Verify repository calls
        mock_content_repository.get_lesson_by_id.assert_called_once_with(lesson_id)
        mock_content_repository.update_lesson.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_lesson_with_none_completed_at(self, content_service, mock_content_repository, sample_lesson):
        """Test updating lesson with explicit None completed_at"""
        # Setup
        lesson_id = "lesson123"
        lesson_update = LessonUpdate(
            is_completed=False,
            completed_at=None
        )
        
        # Mock repository to return existing lesson
        mock_content_repository.get_lesson_by_id.return_value = sample_lesson
        
        # Mock repository update to return updated lesson
        expected_updated_lesson = sample_lesson.model_copy(update={
            "is_completed": False,
            "completed_at": None
        })
        mock_content_repository.update_lesson.return_value = expected_updated_lesson
        
        # Execute
        result = await content_service.update_lesson(lesson_id, lesson_update)
        
        # Verify
        assert result is not None
        assert result.is_completed is False
        assert result.completed_at is None
        
        # Verify repository calls
        mock_content_repository.get_lesson_by_id.assert_called_once_with(lesson_id)
        mock_content_repository.update_lesson.assert_called_once()