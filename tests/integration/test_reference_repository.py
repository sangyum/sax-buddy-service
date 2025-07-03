import pytest
import pytest_asyncio
import asyncio
import os
from datetime import datetime, timezone
from typing import AsyncGenerator
from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1 import AsyncClient as FirestoreAsyncClient

from src.repositories.reference_repository import ReferenceRepository
from src.models.reference import (
    ReferencePerformance,
    SkillLevelDefinition,
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
async def reference_repository(firestore_client: AsyncClient) -> AsyncGenerator[ReferenceRepository, None]:
    """Create ReferenceRepository instance for testing."""
    repo = ReferenceRepository(firestore_client)
    
    # Clean up before test
    try:
        collection = firestore_client.collection("reference")
        docs = collection.stream()
        async for doc in docs:
            await doc.reference.delete()
    except Exception:
        pass
    
    yield repo
    
    # Clean up after test
    try:
        collection = firestore_client.collection("reference")
        docs = collection.stream()
        async for doc in docs:
            await doc.reference.delete()
    except Exception:
        pass


@pytest.fixture
def sample_reference_performance() -> ReferencePerformance:
    """Create a sample ReferencePerformance for testing."""
    return ReferencePerformance(
        id="test-ref-1",
        exercise_id="exercise-123",
        skill_level="intermediate",
        audio_url="https://example.com/ref-performance.mp3",
        target_metrics={
            "intonation_score": 85.0,
            "rhythm_score": 80.0,
            "articulation_score": 90.0,
            "dynamics_score": 75.0
        },
        description="Reference performance for intermediate-level major scales",
        created_by="instructor-456",
        is_active=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_skill_level_definition() -> SkillLevelDefinition:
    """Create a sample SkillLevelDefinition for testing."""
    return SkillLevelDefinition(
        id="test-skill-def-1",
        skill_level="intermediate",
        title="Intermediate Level",
        description="Students at intermediate level can play scales and simple songs",
        requirements=[
            "Master all major scales",
            "Demonstrate consistent intonation",
            "Play simple melodies from memory"
        ],
        skill_ranges={
            "intonation_min": 70.0,
            "intonation_max": 90.0,
            "rhythm_min": 65.0,
            "rhythm_max": 85.0
        },
        progression_criteria=[
            "Complete 10 scale exercises",
            "Achieve 80% average score",
            "Pass formal assessment"
        ],
        estimated_duration_weeks=12,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )


class TestReferencePerformanceOperations:
    """Test CRUD operations for ReferencePerformance."""

    @pytest.mark.asyncio
    async def test_create_reference_performance(
        self, 
        reference_repository: ReferenceRepository,
        sample_reference_performance: ReferencePerformance
    ):
        """Test creating a new reference performance."""
        created_ref = await reference_repository.create_reference_performance(sample_reference_performance)
        
        assert created_ref.id is not None
        assert created_ref.exercise_id == sample_reference_performance.exercise_id
        assert created_ref.skill_level == sample_reference_performance.skill_level
        assert created_ref.target_metrics == sample_reference_performance.target_metrics

    @pytest.mark.asyncio
    async def test_get_reference_performance_by_id(
        self, 
        reference_repository: ReferenceRepository,
        sample_reference_performance: ReferencePerformance
    ):
        """Test retrieving reference performance by ID."""
        created_ref = await reference_repository.create_reference_performance(sample_reference_performance)
        
        retrieved_ref = await reference_repository.get_reference_performance_by_id(created_ref.id)
        
        assert retrieved_ref is not None
        assert retrieved_ref.id == created_ref.id
        assert retrieved_ref.exercise_id == created_ref.exercise_id

    @pytest.mark.asyncio
    async def test_get_reference_performances_by_exercise_id(
        self, 
        reference_repository: ReferenceRepository,
        sample_reference_performance: ReferencePerformance
    ):
        """Test retrieving reference performances by exercise ID."""
        # Create references for the same exercise but different skill levels
        beginner_ref = sample_reference_performance.model_copy()
        beginner_ref.skill_level = "beginner"
        beginner_ref.target_metrics["intonation_score"] = 70.0
        
        advanced_ref = sample_reference_performance.model_copy()
        advanced_ref.skill_level = "advanced"
        advanced_ref.target_metrics["intonation_score"] = 95.0
        
        await reference_repository.create_reference_performance(beginner_ref)
        await reference_repository.create_reference_performance(advanced_ref)
        
        # Get references for the exercise
        exercise_refs = await reference_repository.get_reference_performances_by_exercise_id("exercise-123")
        
        assert len(exercise_refs) == 2
        skill_levels = [ref.skill_level for ref in exercise_refs]
        assert "beginner" in skill_levels
        assert "advanced" in skill_levels

    @pytest.mark.asyncio
    async def test_get_reference_performance_by_exercise_and_skill(
        self, 
        reference_repository: ReferenceRepository,
        sample_reference_performance: ReferencePerformance
    ):
        """Test retrieving reference performance by exercise and skill level."""
        created_ref = await reference_repository.create_reference_performance(sample_reference_performance)
        
        retrieved_ref = await reference_repository.get_reference_performance_by_exercise_and_skill(
            "exercise-123", "intermediate"
        )
        
        assert retrieved_ref is not None
        assert retrieved_ref.id == created_ref.id
        assert retrieved_ref.skill_level == "intermediate"

    @pytest.mark.asyncio
    async def test_update_reference_performance(
        self, 
        reference_repository: ReferenceRepository,
        sample_reference_performance: ReferencePerformance
    ):
        """Test updating an existing reference performance."""
        created_ref = await reference_repository.create_reference_performance(sample_reference_performance)
        
        created_ref.target_metrics["intonation_score"] = 88.0
        created_ref.description = "Updated reference performance"
        
        updated_ref = await reference_repository.update_reference_performance(created_ref)
        
        assert updated_ref.target_metrics["intonation_score"] == 88.0
        assert updated_ref.description == "Updated reference performance"

    @pytest.mark.asyncio
    async def test_delete_reference_performance(
        self, 
        reference_repository: ReferenceRepository,
        sample_reference_performance: ReferencePerformance
    ):
        """Test deleting a reference performance."""
        created_ref = await reference_repository.create_reference_performance(sample_reference_performance)
        
        result = await reference_repository.delete_reference_performance(created_ref.id)
        assert result is True
        
        retrieved_ref = await reference_repository.get_reference_performance_by_id(created_ref.id)
        assert retrieved_ref is None


class TestSkillLevelDefinitionOperations:
    """Test CRUD operations for SkillLevelDefinition."""

    @pytest.mark.asyncio
    async def test_create_skill_level_definition(
        self, 
        reference_repository: ReferenceRepository,
        sample_skill_level_definition: SkillLevelDefinition
    ):
        """Test creating a new skill level definition."""
        created_def = await reference_repository.create_skill_level_definition(sample_skill_level_definition)
        
        assert created_def.id is not None
        assert created_def.skill_level == sample_skill_level_definition.skill_level
        assert created_def.title == sample_skill_level_definition.title
        assert created_def.requirements == sample_skill_level_definition.requirements

    @pytest.mark.asyncio
    async def test_get_skill_level_definition_by_id(
        self, 
        reference_repository: ReferenceRepository,
        sample_skill_level_definition: SkillLevelDefinition
    ):
        """Test retrieving skill level definition by ID."""
        created_def = await reference_repository.create_skill_level_definition(sample_skill_level_definition)
        
        retrieved_def = await reference_repository.get_skill_level_definition_by_id(created_def.id)
        
        assert retrieved_def is not None
        assert retrieved_def.id == created_def.id
        assert retrieved_def.skill_level == created_def.skill_level

    @pytest.mark.asyncio
    async def test_get_skill_level_definition_by_level(
        self, 
        reference_repository: ReferenceRepository,
        sample_skill_level_definition: SkillLevelDefinition
    ):
        """Test retrieving skill level definition by skill level."""
        created_def = await reference_repository.create_skill_level_definition(sample_skill_level_definition)
        
        retrieved_def = await reference_repository.get_skill_level_definition_by_level("intermediate")
        
        assert retrieved_def is not None
        assert retrieved_def.id == created_def.id
        assert retrieved_def.skill_level == "intermediate"

    @pytest.mark.asyncio
    async def test_list_skill_level_definitions(
        self, 
        reference_repository: ReferenceRepository,
        sample_skill_level_definition: SkillLevelDefinition
    ):
        """Test listing all skill level definitions."""
        # Create definitions for different levels
        beginner_def = sample_skill_level_definition.model_copy()
        beginner_def.skill_level = "beginner"
        beginner_def.title = "Beginner Level"
        
        advanced_def = sample_skill_level_definition.model_copy()
        advanced_def.skill_level = "advanced"
        advanced_def.title = "Advanced Level"
        
        await reference_repository.create_skill_level_definition(beginner_def)
        await reference_repository.create_skill_level_definition(advanced_def)
        
        # List all definitions
        all_definitions = await reference_repository.list_skill_level_definitions()
        
        assert len(all_definitions) == 2
        skill_levels = [def_.skill_level for def_ in all_definitions]
        assert "beginner" in skill_levels
        assert "advanced" in skill_levels

    @pytest.mark.asyncio
    async def test_update_skill_level_definition(
        self, 
        reference_repository: ReferenceRepository,
        sample_skill_level_definition: SkillLevelDefinition
    ):
        """Test updating an existing skill level definition."""
        created_def = await reference_repository.create_skill_level_definition(sample_skill_level_definition)
        
        created_def.estimated_duration_weeks = 16
        created_def.title = "Updated Intermediate Level"
        
        updated_def = await reference_repository.update_skill_level_definition(created_def)
        
        assert updated_def.estimated_duration_weeks == 16
        assert updated_def.title == "Updated Intermediate Level"

    @pytest.mark.asyncio
    async def test_delete_skill_level_definition(
        self, 
        reference_repository: ReferenceRepository,
        sample_skill_level_definition: SkillLevelDefinition
    ):
        """Test deleting a skill level definition."""
        created_def = await reference_repository.create_skill_level_definition(sample_skill_level_definition)
        
        result = await reference_repository.delete_skill_level_definition(created_def.id)
        assert result is True
        
        retrieved_def = await reference_repository.get_skill_level_definition_by_id(created_def.id)
        assert retrieved_def is None


class TestQueryOperations:
    """Test query and search operations."""

    @pytest.mark.asyncio
    async def test_get_active_reference_performances(
        self, 
        reference_repository: ReferenceRepository,
        sample_reference_performance: ReferencePerformance
    ):
        """Test retrieving only active reference performances."""
        # Create active and inactive references
        active_ref = sample_reference_performance.model_copy()
        active_ref.is_active = True
        
        inactive_ref = sample_reference_performance.model_copy()
        inactive_ref.exercise_id = "exercise-999"
        inactive_ref.is_active = False
        
        await reference_repository.create_reference_performance(active_ref)
        await reference_repository.create_reference_performance(inactive_ref)
        
        # Get only active references
        active_refs = await reference_repository.get_active_reference_performances()
        
        assert len(active_refs) == 1
        assert active_refs[0].is_active == True

    @pytest.mark.asyncio
    async def test_get_references_by_skill_level(
        self, 
        reference_repository: ReferenceRepository,
        sample_reference_performance: ReferencePerformance
    ):
        """Test retrieving references by skill level."""
        # Create references for different skill levels
        intermediate_ref = sample_reference_performance.model_copy()
        intermediate_ref.skill_level = "intermediate"
        
        beginner_ref = sample_reference_performance.model_copy()
        beginner_ref.skill_level = "beginner"
        beginner_ref.exercise_id = "exercise-456"
        
        await reference_repository.create_reference_performance(intermediate_ref)
        await reference_repository.create_reference_performance(beginner_ref)
        
        # Get intermediate references
        intermediate_refs = await reference_repository.get_references_by_skill_level("intermediate")
        
        assert len(intermediate_refs) == 1
        assert intermediate_refs[0].skill_level == "intermediate"