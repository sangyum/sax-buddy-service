import pytest
import pytest_asyncio
import asyncio
import os
from datetime import datetime, timezone
from typing import AsyncGenerator
from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1 import AsyncClient as FirestoreAsyncClient

from src.repositories.content_repository import ContentRepository
from src.models.content import (
    Exercise,
    LessonPlan,
    Lesson,
    ExerciseType,
    DifficultyLevel,
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
    os.environ["FIRESTORE_EMULATOR_HOST"] = "127.0.0.1:8080"
    os.environ["GOOGLE_CLOUD_PROJECT"] = "demo-test"
    
    client = FirestoreAsyncClient(project="demo-test")
    
    try:
        yield client
    finally:
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
async def content_repository(firestore_client: AsyncClient) -> AsyncGenerator[ContentRepository, None]:
    """Create ContentRepository instance for testing."""
    repo = ContentRepository(firestore_client)
    
    # Clean up before test
    try:
        collection = firestore_client.collection("content")
        docs = collection.stream()
        async for doc in docs:
            await doc.reference.delete()
    except Exception:
        pass
    
    yield repo
    
    # Clean up after test
    try:
        collection = firestore_client.collection("content")
        docs = collection.stream()
        async for doc in docs:
            await doc.reference.delete()
    except Exception:
        pass


@pytest.fixture
def sample_exercise() -> Exercise:
    """Create a sample Exercise for testing."""
    return Exercise(
        id="test-exercise-1",
        title="Major Scales Practice",
        description="Practice all major scales in ascending and descending order",
        exercise_type=ExerciseType.SCALES,
        difficulty_level=DifficultyLevel.INTERMEDIATE,
        estimated_duration_minutes=30,
        instructions=[
            "Start with C major scale",
            "Play slowly and focus on intonation",
            "Gradually increase tempo"
        ],
        reference_audio_url="https://example.com/scales.mp3",
        sheet_music_url="https://example.com/scales.pdf",
        is_active=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )


class TestExerciseOperations:
    """Test CRUD operations for Exercise."""

    @pytest.mark.asyncio
    async def test_create_exercise(
        self, 
        content_repository: ContentRepository,
        sample_exercise: Exercise
    ):
        """Test creating a new exercise."""
        created_exercise = await content_repository.create_exercise(sample_exercise)
        
        assert created_exercise.id is not None
        assert created_exercise.title == sample_exercise.title
        assert created_exercise.exercise_type == sample_exercise.exercise_type
        assert created_exercise.difficulty_level == sample_exercise.difficulty_level

    @pytest.mark.asyncio
    async def test_get_exercise_by_id(
        self, 
        content_repository: ContentRepository,
        sample_exercise: Exercise
    ):
        """Test retrieving exercise by ID."""
        created_exercise = await content_repository.create_exercise(sample_exercise)
        
        retrieved_exercise = await content_repository.get_exercise_by_id(created_exercise.id)
        
        assert retrieved_exercise is not None
        assert retrieved_exercise.id == created_exercise.id
        assert retrieved_exercise.title == created_exercise.title

    @pytest.mark.asyncio
    async def test_get_exercises_by_type(
        self, 
        content_repository: ContentRepository,
        sample_exercise: Exercise
    ):
        """Test retrieving exercises by type."""
        # Create exercises of different types
        scales_exercise = sample_exercise.model_copy()
        scales_exercise.exercise_type = ExerciseType.SCALES
        
        arpeggios_exercise = sample_exercise.model_copy()
        arpeggios_exercise.title = "Arpeggios Practice"
        arpeggios_exercise.exercise_type = ExerciseType.ARPEGGIOS
        
        await content_repository.create_exercise(scales_exercise)
        await content_repository.create_exercise(arpeggios_exercise)
        
        # Get scales exercises
        scales_exercises = await content_repository.get_exercises_by_type(ExerciseType.SCALES)
        
        assert len(scales_exercises) == 1
        assert scales_exercises[0].exercise_type == ExerciseType.SCALES

    @pytest.mark.asyncio
    async def test_get_exercises_by_difficulty(
        self, 
        content_repository: ContentRepository,
        sample_exercise: Exercise
    ):
        """Test retrieving exercises by difficulty level."""
        # Create exercises with different difficulties
        beginner_exercise = sample_exercise.model_copy()
        beginner_exercise.difficulty_level = DifficultyLevel.BEGINNER
        
        advanced_exercise = sample_exercise.model_copy()
        advanced_exercise.title = "Advanced Scales"
        advanced_exercise.difficulty_level = DifficultyLevel.ADVANCED
        
        await content_repository.create_exercise(beginner_exercise)
        await content_repository.create_exercise(advanced_exercise)
        
        # Get beginner exercises
        beginner_exercises = await content_repository.get_exercises_by_difficulty(DifficultyLevel.BEGINNER)
        
        assert len(beginner_exercises) == 1
        assert beginner_exercises[0].difficulty_level == DifficultyLevel.BEGINNER

    @pytest.mark.asyncio
    async def test_update_exercise(
        self, 
        content_repository: ContentRepository,
        sample_exercise: Exercise
    ):
        """Test updating an existing exercise."""
        created_exercise = await content_repository.create_exercise(sample_exercise)
        
        created_exercise.title = "Updated Scales Practice"
        created_exercise.is_active = False
        
        updated_exercise = await content_repository.update_exercise(created_exercise)
        
        assert updated_exercise.title == "Updated Scales Practice"
        assert updated_exercise.is_active == False

    @pytest.mark.asyncio
    async def test_delete_exercise(
        self, 
        content_repository: ContentRepository,
        sample_exercise: Exercise
    ):
        """Test deleting an exercise."""
        created_exercise = await content_repository.create_exercise(sample_exercise)
        
        result = await content_repository.delete_exercise(created_exercise.id)
        assert result is True
        
        retrieved_exercise = await content_repository.get_exercise_by_id(created_exercise.id)
        assert retrieved_exercise is None


class TestSearchAndFilter:
    """Test search and filtering capabilities."""

    @pytest.mark.asyncio
    async def test_search_exercises_by_keyword(
        self, 
        content_repository: ContentRepository,
        sample_exercise: Exercise
    ):
        """Test searching exercises by keyword."""
        # Create exercises with different titles
        scales_exercise = sample_exercise.model_copy()
        scales_exercise.title = "Major Scales Practice"
        
        chords_exercise = sample_exercise.model_copy()
        chords_exercise.title = "Chord Progressions"
        chords_exercise.exercise_type = ExerciseType.TECHNICAL
        
        await content_repository.create_exercise(scales_exercise)
        await content_repository.create_exercise(chords_exercise)
        
        # Search for "scales"
        search_results = await content_repository.search_exercises("scales")
        
        assert len(search_results) >= 1
        # Should find the scales exercise
        scales_found = any("scales" in ex.title.lower() for ex in search_results)
        assert scales_found

    @pytest.mark.asyncio
    async def test_get_active_exercises(
        self, 
        content_repository: ContentRepository,
        sample_exercise: Exercise
    ):
        """Test retrieving only active exercises."""
        # Create active and inactive exercises
        active_exercise = sample_exercise.model_copy()
        active_exercise.is_active = True
        
        inactive_exercise = sample_exercise.model_copy()
        inactive_exercise.title = "Deprecated Exercise"
        inactive_exercise.is_active = False
        
        await content_repository.create_exercise(active_exercise)
        await content_repository.create_exercise(inactive_exercise)
        
        # Get only active exercises
        active_exercises = await content_repository.get_active_exercises()
        
        assert len(active_exercises) == 1
        assert active_exercises[0].is_active == True