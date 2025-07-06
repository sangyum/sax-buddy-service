import pytest
import pytest_asyncio
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator
from google.cloud.firestore_v1.async_client import AsyncClient

from src.repositories.assessment_repository import AssessmentRepository
from src.models.assessment import (
    FormalAssessment,
    Feedback,
    SkillMetrics,
    AssessmentType,
    TriggerReason,
    SkillLevel,
    FeedbackType,
)




@pytest_asyncio.fixture
async def assessment_repository(firestore_client: AsyncClient) -> AsyncGenerator[AssessmentRepository, None]:
    """Create AssessmentRepository instance for testing."""
    repo = AssessmentRepository(firestore_client)
    
    # Clean up any existing data before test
    try:
        collection = firestore_client.collection("assessments")
        docs = collection.stream()
        async for doc in docs:
            await doc.reference.delete()
    except Exception:
        pass
    
    yield repo
    
    # Clean up after test
    try:
        collection = firestore_client.collection("assessments")
        docs = collection.stream()
        async for doc in docs:
            await doc.reference.delete()
    except Exception:
        pass




@pytest.fixture
def sample_formal_assessment() -> FormalAssessment:
    """Create a sample FormalAssessment for testing."""
    return FormalAssessment(
        id="test-assessment-1",
        user_id="user-123",
        assessment_type=AssessmentType.PERIODIC,
        trigger_reason=TriggerReason.SCHEDULED_INTERVAL,
        skill_metrics={
            "intonation": 85.5,
            "rhythm": 78.2,
            "articulation": 92.1,
            "dynamics": 80.0
        },
        overall_score=83.95,
        skill_level_recommendation=SkillLevel.INTERMEDIATE,
        improvement_areas=["rhythm", "dynamics"],
        strengths=["articulation", "intonation"],
        next_lesson_plan_id="lesson-plan-456",
        assessed_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_feedback() -> Feedback:
    """Create a sample Feedback for testing."""
    return Feedback(
        id="test-feedback-1",
        user_id="user-123",
        session_id="session-789",
        feedback_type=FeedbackType.POST_SESSION,
        message="Great work on your practice session!",
        encouragement="You're making excellent progress!",
        specific_suggestions=[
            "Focus on maintaining steady rhythm in faster passages",
            "Work on dynamic contrasts"
        ],
        areas_of_improvement=["rhythm", "dynamics"],
        strengths_highlighted=["articulation", "intonation"],
        confidence_score=0.89,
        created_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_skill_metrics() -> SkillMetrics:
    """Create a sample SkillMetrics for testing."""
    return SkillMetrics(
        id="test-metrics-1",
        user_id="user-123",
        intonation_score=85.5,
        rhythm_score=78.2,
        articulation_score=92.1,
        dynamics_score=80.0,
        overall_score=83.95,
        measurement_period_days=30,
        sessions_analyzed=15,
        confidence_level=0.92,
        trend_direction="improving",
        calculated_at=datetime.now(timezone.utc)
    )


class TestFormalAssessmentOperations:
    """Test CRUD operations for FormalAssessment."""

    @pytest.mark.asyncio
    async def test_create_assessment(
        self, 
        assessment_repository: AssessmentRepository,
        sample_formal_assessment: FormalAssessment
    ):
        """Test creating a new formal assessment."""
        # Create assessment
        created_assessment = await assessment_repository.create_assessment(sample_formal_assessment)
        
        # Verify assessment was created with ID
        assert created_assessment.id is not None
        assert created_assessment.user_id == sample_formal_assessment.user_id
        assert created_assessment.assessment_type == sample_formal_assessment.assessment_type
        assert created_assessment.overall_score == sample_formal_assessment.overall_score
        assert created_assessment.skill_level_recommendation == sample_formal_assessment.skill_level_recommendation

    @pytest.mark.asyncio
    async def test_get_assessment_by_id(
        self, 
        assessment_repository: AssessmentRepository,
        sample_formal_assessment: FormalAssessment
    ):
        """Test retrieving assessment by ID."""
        # Create assessment first
        created_assessment = await assessment_repository.create_assessment(sample_formal_assessment)
        
        # Retrieve assessment by ID
        retrieved_assessment = await assessment_repository.get_assessment_by_id(created_assessment.id)
        
        # Verify retrieved assessment matches created assessment
        assert retrieved_assessment is not None
        assert retrieved_assessment.id == created_assessment.id
        assert retrieved_assessment.user_id == created_assessment.user_id
        assert retrieved_assessment.overall_score == created_assessment.overall_score

    @pytest.mark.asyncio
    async def test_get_assessment_by_id_not_found(
        self, 
        assessment_repository: AssessmentRepository
    ):
        """Test retrieving non-existent assessment returns None."""
        retrieved_assessment = await assessment_repository.get_assessment_by_id("non-existent-id")
        assert retrieved_assessment is None

    @pytest.mark.asyncio
    async def test_get_assessments_by_user_id(
        self, 
        assessment_repository: AssessmentRepository,
        sample_formal_assessment: FormalAssessment
    ):
        """Test retrieving assessments by user ID with pagination."""
        # Create multiple assessments for the same user
        assessment1 = sample_formal_assessment.model_copy()
        assessment1.overall_score = 80.0
        assessment1.assessed_at = datetime.now(timezone.utc) - timedelta(days=1)
        
        assessment2 = sample_formal_assessment.model_copy()
        assessment2.overall_score = 85.0
        assessment2.assessed_at = datetime.now(timezone.utc)
        
        await assessment_repository.create_assessment(assessment1)
        await assessment_repository.create_assessment(assessment2)
        
        # Retrieve assessments for user
        assessments = await assessment_repository.get_assessments_by_user_id("user-123", limit=5)
        
        # Verify results (should be ordered by assessed_at DESC)
        assert len(assessments) == 2
        assert assessments[0].overall_score == 85.0  # Most recent first
        assert assessments[1].overall_score == 80.0

    @pytest.mark.asyncio
    async def test_get_latest_assessment_by_user_id(
        self, 
        assessment_repository: AssessmentRepository,
        sample_formal_assessment: FormalAssessment
    ):
        """Test retrieving the latest assessment for a user."""
        # Create multiple assessments
        assessment1 = sample_formal_assessment.model_copy()
        assessment1.overall_score = 80.0
        assessment1.assessed_at = datetime.now(timezone.utc) - timedelta(days=1)
        
        assessment2 = sample_formal_assessment.model_copy()
        assessment2.overall_score = 85.0
        assessment2.assessed_at = datetime.now(timezone.utc)
        
        await assessment_repository.create_assessment(assessment1)
        await assessment_repository.create_assessment(assessment2)
        
        # Get latest assessment
        latest_assessment = await assessment_repository.get_latest_assessment_by_user_id("user-123")
        
        # Verify it's the most recent one
        assert latest_assessment is not None
        assert latest_assessment.overall_score == 85.0

    @pytest.mark.asyncio
    async def test_update_assessment(
        self, 
        assessment_repository: AssessmentRepository,
        sample_formal_assessment: FormalAssessment
    ):
        """Test updating an existing assessment."""
        # Create assessment
        created_assessment = await assessment_repository.create_assessment(sample_formal_assessment)
        
        # Update assessment
        created_assessment.overall_score = 90.0
        created_assessment.skill_level_recommendation = SkillLevel.ADVANCED
        
        updated_assessment = await assessment_repository.update_assessment(created_assessment)
        
        # Verify update
        assert updated_assessment.overall_score == 90.0
        assert updated_assessment.skill_level_recommendation == SkillLevel.ADVANCED
        
        # Verify in database
        retrieved_assessment = await assessment_repository.get_assessment_by_id(created_assessment.id)
        assert retrieved_assessment is not None
        assert retrieved_assessment.overall_score == 90.0
        assert retrieved_assessment.skill_level_recommendation == SkillLevel.ADVANCED

    @pytest.mark.asyncio
    async def test_delete_assessment(
        self, 
        assessment_repository: AssessmentRepository,
        sample_formal_assessment: FormalAssessment
    ):
        """Test deleting an assessment."""
        # Create assessment
        created_assessment = await assessment_repository.create_assessment(sample_formal_assessment)
        
        # Delete assessment
        result = await assessment_repository.delete_assessment(created_assessment.id)
        assert result is True
        
        # Verify deletion
        retrieved_assessment = await assessment_repository.get_assessment_by_id(created_assessment.id)
        assert retrieved_assessment is None


class TestFeedbackOperations:
    """Test CRUD operations for Feedback."""

    @pytest.mark.asyncio
    async def test_create_feedback(
        self, 
        assessment_repository: AssessmentRepository,
        sample_feedback: Feedback
    ):
        """Test creating new feedback."""
        created_feedback = await assessment_repository.create_feedback(sample_feedback)
        
        assert created_feedback.id is not None
        assert created_feedback.user_id == sample_feedback.user_id
        assert created_feedback.session_id == sample_feedback.session_id
        assert created_feedback.message == sample_feedback.message

    @pytest.mark.asyncio
    async def test_get_feedback_by_id(
        self, 
        assessment_repository: AssessmentRepository,
        sample_feedback: Feedback
    ):
        """Test retrieving feedback by ID."""
        created_feedback = await assessment_repository.create_feedback(sample_feedback)
        
        retrieved_feedback = await assessment_repository.get_feedback_by_id(created_feedback.id)
        
        assert retrieved_feedback is not None
        assert retrieved_feedback.id == created_feedback.id
        assert retrieved_feedback.message == created_feedback.message

    @pytest.mark.asyncio
    async def test_get_feedback_by_session_id(
        self, 
        assessment_repository: AssessmentRepository,
        sample_feedback: Feedback
    ):
        """Test retrieving feedback by session ID."""
        created_feedback = await assessment_repository.create_feedback(sample_feedback)
        
        retrieved_feedback = await assessment_repository.get_feedback_by_session_id(sample_feedback.session_id)
        
        assert retrieved_feedback is not None
        assert retrieved_feedback.session_id == sample_feedback.session_id

    @pytest.mark.asyncio
    async def test_get_feedback_by_user_id(
        self, 
        assessment_repository: AssessmentRepository,
        sample_feedback: Feedback
    ):
        """Test retrieving feedback by user ID."""
        # Create multiple feedback entries
        feedback1 = sample_feedback.model_copy()
        feedback1.session_id = "session-1"
        feedback1.created_at = datetime.now(timezone.utc) - timedelta(hours=1)
        
        feedback2 = sample_feedback.model_copy()
        feedback2.session_id = "session-2"
        feedback2.created_at = datetime.now(timezone.utc)
        
        await assessment_repository.create_feedback(feedback1)
        await assessment_repository.create_feedback(feedback2)
        
        # Retrieve feedback for user
        feedback_list = await assessment_repository.get_feedback_by_user_id("user-123", limit=5)
        
        assert len(feedback_list) == 2
        assert feedback_list[0].session_id == "session-2"  # Most recent first

    @pytest.mark.asyncio
    async def test_update_feedback(
        self, 
        assessment_repository: AssessmentRepository,
        sample_feedback: Feedback
    ):
        """Test updating existing feedback."""
        created_feedback = await assessment_repository.create_feedback(sample_feedback)
        
        created_feedback.message = "Updated feedback message"
        created_feedback.confidence_score = 0.95
        
        updated_feedback = await assessment_repository.update_feedback(created_feedback)
        
        assert updated_feedback.message == "Updated feedback message"
        assert updated_feedback.confidence_score == 0.95

    @pytest.mark.asyncio
    async def test_delete_feedback(
        self, 
        assessment_repository: AssessmentRepository,
        sample_feedback: Feedback
    ):
        """Test deleting feedback."""
        created_feedback = await assessment_repository.create_feedback(sample_feedback)
        
        result = await assessment_repository.delete_feedback(created_feedback.id)
        assert result is True
        
        retrieved_feedback = await assessment_repository.get_feedback_by_id(created_feedback.id)
        assert retrieved_feedback is None


class TestSkillMetricsOperations:
    """Test CRUD operations for SkillMetrics."""

    @pytest.mark.asyncio
    async def test_create_skill_metrics(
        self, 
        assessment_repository: AssessmentRepository,
        sample_skill_metrics: SkillMetrics
    ):
        """Test creating new skill metrics."""
        created_metrics = await assessment_repository.create_skill_metrics(sample_skill_metrics)
        
        assert created_metrics.id is not None
        assert created_metrics.user_id == sample_skill_metrics.user_id
        assert created_metrics.overall_score == sample_skill_metrics.overall_score
        assert created_metrics.intonation_score == sample_skill_metrics.intonation_score

    @pytest.mark.asyncio
    async def test_get_skill_metrics_by_id(
        self, 
        assessment_repository: AssessmentRepository,
        sample_skill_metrics: SkillMetrics
    ):
        """Test retrieving skill metrics by ID."""
        created_metrics = await assessment_repository.create_skill_metrics(sample_skill_metrics)
        
        retrieved_metrics = await assessment_repository.get_skill_metrics_by_id(created_metrics.id)
        
        assert retrieved_metrics is not None
        assert retrieved_metrics.id == created_metrics.id
        assert retrieved_metrics.overall_score == created_metrics.overall_score

    @pytest.mark.asyncio
    async def test_get_latest_skill_metrics_by_user_id(
        self, 
        assessment_repository: AssessmentRepository,
        sample_skill_metrics: SkillMetrics
    ):
        """Test retrieving latest skill metrics for a user."""
        # Create multiple metrics
        metrics1 = sample_skill_metrics.model_copy()
        metrics1.overall_score = 80.0
        metrics1.calculated_at = datetime.now(timezone.utc) - timedelta(days=1)
        
        metrics2 = sample_skill_metrics.model_copy()
        metrics2.overall_score = 85.0
        metrics2.calculated_at = datetime.now(timezone.utc)
        
        await assessment_repository.create_skill_metrics(metrics1)
        await assessment_repository.create_skill_metrics(metrics2)
        
        # Get latest metrics
        latest_metrics = await assessment_repository.get_latest_skill_metrics_by_user_id("user-123")
        
        assert latest_metrics is not None
        assert latest_metrics.overall_score == 85.0

    @pytest.mark.asyncio
    async def test_get_skill_metrics_by_user_id_and_period(
        self, 
        assessment_repository: AssessmentRepository,
        sample_skill_metrics: SkillMetrics
    ):
        """Test retrieving skill metrics within a date range."""
        now = datetime.now(timezone.utc)
        
        # Create metrics within and outside the date range
        metrics_in_range = sample_skill_metrics.model_copy()
        metrics_in_range.overall_score = 80.0
        metrics_in_range.calculated_at = now - timedelta(days=5)
        
        metrics_outside_range = sample_skill_metrics.model_copy()
        metrics_outside_range.overall_score = 85.0
        metrics_outside_range.calculated_at = now - timedelta(days=15)
        
        await assessment_repository.create_skill_metrics(metrics_in_range)
        await assessment_repository.create_skill_metrics(metrics_outside_range)
        
        # Query for metrics in the last 7 days
        start_date = now - timedelta(days=7)
        end_date = now
        
        metrics_list = await assessment_repository.get_skill_metrics_by_user_id_and_period(
            "user-123", start_date, end_date
        )
        
        assert len(metrics_list) == 1
        assert metrics_list[0].overall_score == 80.0

    @pytest.mark.asyncio
    async def test_update_skill_metrics(
        self, 
        assessment_repository: AssessmentRepository,
        sample_skill_metrics: SkillMetrics
    ):
        """Test updating existing skill metrics."""
        created_metrics = await assessment_repository.create_skill_metrics(sample_skill_metrics)
        
        created_metrics.overall_score = 90.0
        created_metrics.trend_direction = "improving"
        
        updated_metrics = await assessment_repository.update_skill_metrics(created_metrics)
        
        assert updated_metrics.overall_score == 90.0
        assert updated_metrics.trend_direction == "improving"

    @pytest.mark.asyncio
    async def test_delete_skill_metrics(
        self, 
        assessment_repository: AssessmentRepository,
        sample_skill_metrics: SkillMetrics
    ):
        """Test deleting skill metrics."""
        created_metrics = await assessment_repository.create_skill_metrics(sample_skill_metrics)
        
        result = await assessment_repository.delete_skill_metrics(created_metrics.id)
        assert result is True
        
        retrieved_metrics = await assessment_repository.get_skill_metrics_by_id(created_metrics.id)
        assert retrieved_metrics is None


class TestBusinessLogicQueries:
    """Test business logic query methods."""

    @pytest.mark.asyncio
    async def test_count_assessments_by_user_id(
        self, 
        assessment_repository: AssessmentRepository,
        sample_formal_assessment: FormalAssessment
    ):
        """Test counting assessments for a user."""
        # Create multiple assessments
        assessment1 = sample_formal_assessment.model_copy()
        assessment1.overall_score = 80.0
        
        assessment2 = sample_formal_assessment.model_copy()
        assessment2.overall_score = 85.0
        
        await assessment_repository.create_assessment(assessment1)
        await assessment_repository.create_assessment(assessment2)
        
        # Count assessments
        count = await assessment_repository.count_assessments_by_user_id("user-123")
        assert count == 2

    @pytest.mark.asyncio
    async def test_get_assessment_trend_by_user_id(
        self, 
        assessment_repository: AssessmentRepository,
        sample_formal_assessment: FormalAssessment
    ):
        """Test getting assessment trend for a user."""
        now = datetime.now(timezone.utc)
        
        # Create assessments within and outside the trend period
        assessment_in_period = sample_formal_assessment.model_copy()
        assessment_in_period.overall_score = 80.0
        assessment_in_period.assessed_at = now - timedelta(days=5)
        
        assessment_outside_period = sample_formal_assessment.model_copy()
        assessment_outside_period.overall_score = 85.0
        assessment_outside_period.assessed_at = now - timedelta(days=45)
        
        await assessment_repository.create_assessment(assessment_in_period)
        await assessment_repository.create_assessment(assessment_outside_period)
        
        # Get trend for last 30 days
        trend_assessments = await assessment_repository.get_assessment_trend_by_user_id("user-123", days=30)
        
        assert len(trend_assessments) == 1
        assert trend_assessments[0].overall_score == 80.0

    @pytest.mark.asyncio
    async def test_get_skill_progression_by_user_id(
        self, 
        assessment_repository: AssessmentRepository,
        sample_skill_metrics: SkillMetrics
    ):
        """Test getting skill progression for a user."""
        now = datetime.now(timezone.utc)
        
        # Create metrics within progression period
        metrics1 = sample_skill_metrics.model_copy()
        metrics1.intonation_score = 80.0
        metrics1.calculated_at = now - timedelta(days=10)
        
        metrics2 = sample_skill_metrics.model_copy()
        metrics2.intonation_score = 85.0
        metrics2.calculated_at = now - timedelta(days=5)
        
        await assessment_repository.create_skill_metrics(metrics1)
        await assessment_repository.create_skill_metrics(metrics2)
        
        # Get skill progression
        progression = await assessment_repository.get_skill_progression_by_user_id(
            "user-123", "intonation", days=30
        )
        
        assert len(progression) == 2
        assert progression[0].intonation_score == 80.0  # Ordered by calculated_at ASC
        assert progression[1].intonation_score == 85.0


class TestDataConsistency:
    """Test data consistency and edge cases."""

    @pytest.mark.asyncio
    async def test_datetime_serialization_deserialization(
        self, 
        assessment_repository: AssessmentRepository,
        sample_formal_assessment: FormalAssessment
    ):
        """Test that datetime fields are properly serialized and deserialized."""
        original_datetime = sample_formal_assessment.assessed_at
        
        # Create and retrieve assessment
        created_assessment = await assessment_repository.create_assessment(sample_formal_assessment)
        retrieved_assessment = await assessment_repository.get_assessment_by_id(created_assessment.id)
        
        # Verify datetime precision (within 1 second tolerance due to serialization)
        assert retrieved_assessment is not None
        assert abs((retrieved_assessment.assessed_at - original_datetime).total_seconds()) < 1

    @pytest.mark.asyncio
    async def test_empty_collections(
        self, 
        assessment_repository: AssessmentRepository
    ):
        """Test operations on empty collections."""
        # Test retrieving from empty collection
        assessments = await assessment_repository.get_assessments_by_user_id("non-existent-user")
        assert len(assessments) == 0
        
        latest_assessment = await assessment_repository.get_latest_assessment_by_user_id("non-existent-user")
        assert latest_assessment is None
        
        count = await assessment_repository.count_assessments_by_user_id("non-existent-user")
        assert count == 0

    @pytest.mark.asyncio
    async def test_pagination_limits(
        self, 
        assessment_repository: AssessmentRepository,
        sample_formal_assessment: FormalAssessment
    ):
        """Test pagination with limits and offsets."""
        # Create 5 assessments
        for i in range(5):
            assessment = sample_formal_assessment.model_copy()
            assessment.overall_score = 70.0 + i * 5
            assessment.assessed_at = datetime.now(timezone.utc) - timedelta(days=i)
            await assessment_repository.create_assessment(assessment)
        
        # Test pagination
        first_page = await assessment_repository.get_assessments_by_user_id("user-123", limit=3, offset=0)
        assert len(first_page) == 3
        
        second_page = await assessment_repository.get_assessments_by_user_id("user-123", limit=3, offset=3)
        assert len(second_page) == 2
        
        # Verify no overlap
        first_page_scores = [a.overall_score for a in first_page]
        second_page_scores = [a.overall_score for a in second_page]
        assert len(set(first_page_scores) & set(second_page_scores)) == 0