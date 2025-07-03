import pytest
import pytest_asyncio
import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator
from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1 import AsyncClient as FirestoreAsyncClient

from src.repositories.performance_repository import PerformanceRepository
from src.models.performance import (
    PerformanceSession,
    PerformanceMetrics,
    SessionStatus,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def firestore_client() -> AsyncGenerator[AsyncClient, None]:
    """Initialize Firestore client for emulator testing."""
    # Set environment variables for Firestore emulator
    os.environ["FIRESTORE_EMULATOR_HOST"] = "127.0.0.1:8080"
    os.environ["GOOGLE_CLOUD_PROJECT"] = "demo-test"
    
    # Initialize Firestore client
    client = FirestoreAsyncClient(project="demo-test")
    
    try:
        yield client
    finally:
        # Clean up client resources properly
        try:
            if hasattr(client, '_firestore_api') and client._firestore_api:
                if hasattr(client._firestore_api, 'transport'):
                    transport = client._firestore_api.transport
                    if hasattr(transport, 'close') and asyncio.iscoroutinefunction(transport.close):
                        await transport.close()
                    elif hasattr(transport, 'close'):
                        transport.close()
        except Exception:
            pass


@pytest_asyncio.fixture
async def performance_repository(firestore_client: AsyncClient) -> AsyncGenerator[PerformanceRepository, None]:
    """Create PerformanceRepository instance for testing."""
    repo = PerformanceRepository(firestore_client)
    
    # Clean up any existing data before test
    try:
        collection = firestore_client.collection("performance")
        docs = collection.stream()
        async for doc in docs:
            await doc.reference.delete()
    except Exception:
        pass
    
    yield repo
    
    # Clean up after test
    try:
        collection = firestore_client.collection("performance")
        docs = collection.stream()
        async for doc in docs:
            await doc.reference.delete()
    except Exception:
        pass


@pytest.fixture
def sample_performance_session() -> PerformanceSession:
    """Create a sample PerformanceSession for testing."""
    return PerformanceSession(
        id="test-session-1",
        user_id="user-123",
        exercise_id="exercise-456",
        duration_minutes=30,
        status=SessionStatus.COMPLETED,
        started_at=datetime.now(timezone.utc) - timedelta(minutes=30),
        ended_at=datetime.now(timezone.utc),
        notes="Great practice session with scales",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_performance_metrics() -> PerformanceMetrics:
    """Create a sample PerformanceMetrics for testing."""
    return PerformanceMetrics(
        id="test-metrics-1",
        session_id="session-789",
        timestamp=120.5,
        intonation_score=85.5,
        rhythm_score=78.2,
        articulation_score=92.1,
        dynamics_score=80.0,
        raw_metrics={
            "frequency_data": [440.0, 442.1, 439.8],
            "amplitude_data": [0.8, 0.9, 0.7],
            "detected_pitch": "A4"
        },
        created_at=datetime.now(timezone.utc)
    )


class TestPerformanceSessionOperations:
    """Test CRUD operations for PerformanceSession."""

    @pytest.mark.asyncio
    async def test_create_session(
        self, 
        performance_repository: PerformanceRepository,
        sample_performance_session: PerformanceSession
    ):
        """Test creating a new performance session."""
        created_session = await performance_repository.create_session(sample_performance_session)
        
        assert created_session.id is not None
        assert created_session.user_id == sample_performance_session.user_id
        assert created_session.exercise_id == sample_performance_session.exercise_id
        assert created_session.duration_minutes == sample_performance_session.duration_minutes
        assert created_session.status == sample_performance_session.status

    @pytest.mark.asyncio
    async def test_get_session_by_id(
        self, 
        performance_repository: PerformanceRepository,
        sample_performance_session: PerformanceSession
    ):
        """Test retrieving session by ID."""
        created_session = await performance_repository.create_session(sample_performance_session)
        
        retrieved_session = await performance_repository.get_session_by_id(created_session.id)
        
        assert retrieved_session is not None
        assert retrieved_session.id == created_session.id
        assert retrieved_session.user_id == created_session.user_id
        assert retrieved_session.exercise_id == created_session.exercise_id

    @pytest.mark.asyncio
    async def test_get_session_by_id_not_found(
        self, 
        performance_repository: PerformanceRepository
    ):
        """Test retrieving non-existent session returns None."""
        retrieved_session = await performance_repository.get_session_by_id("non-existent-id")
        assert retrieved_session is None

    @pytest.mark.asyncio
    async def test_get_sessions_by_user_id(
        self, 
        performance_repository: PerformanceRepository,
        sample_performance_session: PerformanceSession
    ):
        """Test retrieving sessions by user ID with pagination."""
        # Create multiple sessions for the same user
        session1 = sample_performance_session.model_copy()
        session1.duration_minutes = 20
        session1.started_at = datetime.now(timezone.utc) - timedelta(hours=2)
        
        session2 = sample_performance_session.model_copy()
        session2.duration_minutes = 40
        session2.started_at = datetime.now(timezone.utc) - timedelta(hours=1)
        
        await performance_repository.create_session(session1)
        await performance_repository.create_session(session2)
        
        # Retrieve sessions for user
        sessions = await performance_repository.get_sessions_by_user_id("user-123", limit=10)
        
        # Verify results (should be ordered by started_at DESC)
        assert len(sessions) == 2
        assert sessions[0].duration_minutes == 40  # Most recent first
        assert sessions[1].duration_minutes == 20

    @pytest.mark.asyncio
    async def test_get_sessions_by_exercise_id(
        self, 
        performance_repository: PerformanceRepository,
        sample_performance_session: PerformanceSession
    ):
        """Test retrieving sessions by exercise ID."""
        # Create sessions for different exercises
        session1 = sample_performance_session.model_copy()
        session1.exercise_id = "scales-exercise"
        session1.duration_minutes = 25
        
        session2 = sample_performance_session.model_copy()
        session2.exercise_id = "scales-exercise"
        session2.duration_minutes = 35
        
        session3 = sample_performance_session.model_copy()
        session3.exercise_id = "different-exercise"
        session3.duration_minutes = 15
        
        await performance_repository.create_session(session1)
        await performance_repository.create_session(session2)
        await performance_repository.create_session(session3)
        
        # Get sessions for specific exercise
        exercise_sessions = await performance_repository.get_sessions_by_exercise_id("scales-exercise")
        
        assert len(exercise_sessions) == 2
        for session in exercise_sessions:
            assert session.exercise_id == "scales-exercise"

    @pytest.mark.asyncio
    async def test_update_session(
        self, 
        performance_repository: PerformanceRepository,
        sample_performance_session: PerformanceSession
    ):
        """Test updating an existing session."""
        created_session = await performance_repository.create_session(sample_performance_session)
        
        created_session.status = SessionStatus.PAUSED
        created_session.notes = "Session paused - need to take a break"
        
        updated_session = await performance_repository.update_session(created_session)
        
        assert updated_session.status == SessionStatus.PAUSED
        assert updated_session.notes == "Session paused - need to take a break"
        
        # Verify in database
        retrieved_session = await performance_repository.get_session_by_id(created_session.id)
        assert retrieved_session is not None
        assert retrieved_session.status == SessionStatus.PAUSED

    @pytest.mark.asyncio
    async def test_delete_session(
        self, 
        performance_repository: PerformanceRepository,
        sample_performance_session: PerformanceSession
    ):
        """Test deleting a session."""
        created_session = await performance_repository.create_session(sample_performance_session)
        
        result = await performance_repository.delete_session(created_session.id)
        assert result is True
        
        retrieved_session = await performance_repository.get_session_by_id(created_session.id)
        assert retrieved_session is None

    @pytest.mark.asyncio
    async def test_get_active_sessions_by_user_id(
        self, 
        performance_repository: PerformanceRepository,
        sample_performance_session: PerformanceSession
    ):
        """Test retrieving active sessions for a user."""
        # Create active and completed sessions
        active_session = sample_performance_session.model_copy()
        active_session.status = SessionStatus.IN_PROGRESS
        active_session.ended_at = None
        
        completed_session = sample_performance_session.model_copy()
        completed_session.status = SessionStatus.COMPLETED
        
        await performance_repository.create_session(active_session)
        await performance_repository.create_session(completed_session)
        
        # Get active sessions
        active_sessions = await performance_repository.get_active_sessions_by_user_id("user-123")
        
        assert len(active_sessions) == 1
        assert active_sessions[0].status == SessionStatus.IN_PROGRESS


class TestPerformanceMetricsOperations:
    """Test CRUD operations for PerformanceMetrics."""

    @pytest.mark.asyncio
    async def test_create_metrics(
        self, 
        performance_repository: PerformanceRepository,
        sample_performance_metrics: PerformanceMetrics
    ):
        """Test creating new performance metrics."""
        created_metrics = await performance_repository.create_metrics(sample_performance_metrics)
        
        assert created_metrics.id is not None
        assert created_metrics.session_id == sample_performance_metrics.session_id
        assert created_metrics.timestamp == sample_performance_metrics.timestamp
        assert created_metrics.intonation_score == sample_performance_metrics.intonation_score
        assert created_metrics.raw_metrics == sample_performance_metrics.raw_metrics

    @pytest.mark.asyncio
    async def test_get_metrics_by_id(
        self, 
        performance_repository: PerformanceRepository,
        sample_performance_metrics: PerformanceMetrics
    ):
        """Test retrieving metrics by ID."""
        created_metrics = await performance_repository.create_metrics(sample_performance_metrics)
        
        retrieved_metrics = await performance_repository.get_metrics_by_id(created_metrics.id)
        
        assert retrieved_metrics is not None
        assert retrieved_metrics.id == created_metrics.id
        assert retrieved_metrics.session_id == created_metrics.session_id

    @pytest.mark.asyncio
    async def test_get_metrics_by_session_id(
        self, 
        performance_repository: PerformanceRepository,
        sample_performance_metrics: PerformanceMetrics
    ):
        """Test retrieving metrics by session ID."""
        # Create multiple metrics for the same session
        metrics1 = sample_performance_metrics.model_copy()
        metrics1.timestamp = 60.0
        metrics1.intonation_score = 80.0
        
        metrics2 = sample_performance_metrics.model_copy()
        metrics2.timestamp = 120.0
        metrics2.intonation_score = 85.0
        
        await performance_repository.create_metrics(metrics1)
        await performance_repository.create_metrics(metrics2)
        
        # Get all metrics for session
        session_metrics = await performance_repository.get_metrics_by_session_id("session-789")
        
        assert len(session_metrics) == 2
        # Should be ordered by timestamp
        assert session_metrics[0].timestamp <= session_metrics[1].timestamp

    @pytest.mark.asyncio
    async def test_update_metrics(
        self, 
        performance_repository: PerformanceRepository,
        sample_performance_metrics: PerformanceMetrics
    ):
        """Test updating existing metrics."""
        created_metrics = await performance_repository.create_metrics(sample_performance_metrics)
        
        created_metrics.intonation_score = 90.0
        created_metrics.raw_metrics["corrected_pitch"] = "A4+"
        
        updated_metrics = await performance_repository.update_metrics(created_metrics)
        
        assert updated_metrics.intonation_score == 90.0
        assert updated_metrics.raw_metrics["corrected_pitch"] == "A4+"

    @pytest.mark.asyncio
    async def test_delete_metrics(
        self, 
        performance_repository: PerformanceRepository,
        sample_performance_metrics: PerformanceMetrics
    ):
        """Test deleting metrics."""
        created_metrics = await performance_repository.create_metrics(sample_performance_metrics)
        
        result = await performance_repository.delete_metrics(created_metrics.id)
        assert result is True
        
        retrieved_metrics = await performance_repository.get_metrics_by_id(created_metrics.id)
        assert retrieved_metrics is None


class TestBusinessLogicQueries:
    """Test business logic query methods."""

    @pytest.mark.asyncio
    async def test_get_session_summary_by_user_id(
        self, 
        performance_repository: PerformanceRepository,
        sample_performance_session: PerformanceSession
    ):
        """Test getting session summary for a user."""
        # Create sessions with different durations
        session1 = sample_performance_session.model_copy()
        session1.duration_minutes = 30
        session1.status = SessionStatus.COMPLETED
        
        session2 = sample_performance_session.model_copy()
        session2.duration_minutes = 45
        session2.status = SessionStatus.COMPLETED
        
        session3 = sample_performance_session.model_copy()
        session3.duration_minutes = 20
        session3.status = SessionStatus.CANCELLED  # Should not be counted
        
        await performance_repository.create_session(session1)
        await performance_repository.create_session(session2)
        await performance_repository.create_session(session3)
        
        # Get summary
        summary = await performance_repository.get_session_summary_by_user_id("user-123")
        
        assert summary["total_sessions"] == 2  # Only completed sessions
        assert summary["total_minutes"] == 75  # 30 + 45

    @pytest.mark.asyncio
    async def test_get_performance_trend_by_user_id(
        self, 
        performance_repository: PerformanceRepository,
        sample_performance_session: PerformanceSession,
        sample_performance_metrics: PerformanceMetrics
    ):
        """Test getting performance trend over time."""
        now = datetime.now(timezone.utc)
        
        # Create sessions and metrics from different time periods
        old_session = sample_performance_session.model_copy()
        old_session.started_at = now - timedelta(days=10)
        old_session = await performance_repository.create_session(old_session)
        
        recent_session = sample_performance_session.model_copy()
        recent_session.started_at = now - timedelta(days=2)
        recent_session = await performance_repository.create_session(recent_session)
        
        # Create metrics for sessions
        old_metrics = sample_performance_metrics.model_copy()
        old_metrics.session_id = old_session.id
        old_metrics.intonation_score = 70.0
        
        recent_metrics = sample_performance_metrics.model_copy()
        recent_metrics.session_id = recent_session.id
        recent_metrics.intonation_score = 85.0
        
        await performance_repository.create_metrics(old_metrics)
        await performance_repository.create_metrics(recent_metrics)
        
        # Get trend for last 7 days
        trend = await performance_repository.get_performance_trend_by_user_id("user-123", days=7)
        
        # Should only include recent session (within 7 days)
        assert len(trend) == 1
        assert trend[0]["avg_intonation"] == 85.0


class TestDataConsistency:
    """Test data consistency and edge cases."""

    @pytest.mark.asyncio
    async def test_datetime_serialization_with_optional_fields(
        self, 
        performance_repository: PerformanceRepository,
        sample_performance_session: PerformanceSession
    ):
        """Test datetime serialization with optional ended_at field."""
        # Test with ended_at = None
        session_in_progress = sample_performance_session.model_copy()
        session_in_progress.status = SessionStatus.IN_PROGRESS
        session_in_progress.ended_at = None
        
        created_session = await performance_repository.create_session(session_in_progress)
        retrieved_session = await performance_repository.get_session_by_id(created_session.id)
        
        assert retrieved_session is not None
        assert retrieved_session.ended_at is None
        assert retrieved_session.status == SessionStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_raw_metrics_json_handling(
        self, 
        performance_repository: PerformanceRepository,
        sample_performance_metrics: PerformanceMetrics
    ):
        """Test that complex JSON data in raw_metrics is preserved."""
        complex_metrics = sample_performance_metrics.model_copy()
        complex_metrics.raw_metrics = {
            "frequency_spectrum": [440.0, 880.0, 1320.0],
            "volume_envelope": {"attack": 0.1, "decay": 0.3, "sustain": 0.7, "release": 0.2},
            "detected_notes": [
                {"note": "A4", "start": 0.0, "duration": 1.5},
                {"note": "C5", "start": 1.5, "duration": 2.0}
            ],
            "confidence_scores": {"overall": 0.85, "pitch": 0.90, "timing": 0.80}
        }
        
        created_metrics = await performance_repository.create_metrics(complex_metrics)
        retrieved_metrics = await performance_repository.get_metrics_by_id(created_metrics.id)
        
        assert retrieved_metrics is not None
        assert retrieved_metrics.raw_metrics == complex_metrics.raw_metrics
        assert retrieved_metrics.raw_metrics["confidence_scores"]["overall"] == 0.85

    @pytest.mark.asyncio
    async def test_pagination_with_large_dataset(
        self, 
        performance_repository: PerformanceRepository,
        sample_performance_session: PerformanceSession
    ):
        """Test pagination behavior with larger datasets."""
        # Create 15 sessions
        for i in range(15):
            session = sample_performance_session.model_copy()
            session.duration_minutes = 10 + i
            session.started_at = datetime.now(timezone.utc) - timedelta(hours=i)
            await performance_repository.create_session(session)
        
        # Test pagination
        first_page = await performance_repository.get_sessions_by_user_id("user-123", limit=10, offset=0)
        assert len(first_page) == 10
        
        second_page = await performance_repository.get_sessions_by_user_id("user-123", limit=10, offset=10)
        assert len(second_page) == 5
        
        # Verify chronological ordering (most recent first)
        assert first_page[0].duration_minutes == 10  # Most recent session (i=0)
        assert first_page[9].duration_minutes == 19  # 10th session (i=9)
        assert second_page[0].duration_minutes == 20  # 11th session (i=10)