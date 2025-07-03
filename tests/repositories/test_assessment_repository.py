"""Tests for AssessmentRepository"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from src.repositories.assessment_repository import AssessmentRepository
from src.models.assessment import FormalAssessment, AssessmentType, TriggerReason, SkillLevel


class TestAssessmentRepository:
    """Test cases for AssessmentRepository"""
    
    @pytest.fixture
    def mock_firestore_client(self):
        """Create a mock Firestore client"""
        client = MagicMock()
        collection = AsyncMock()
        client.collection.return_value = collection
        return client, collection
    
    @pytest.fixture
    def assessment_repository(self, mock_firestore_client):
        """Create AssessmentRepository with mock client"""
        client, _ = mock_firestore_client
        return AssessmentRepository(client)
    
    @pytest.fixture
    def sample_assessment(self):
        """Create a sample FormalAssessment for testing"""
        return FormalAssessment(
            id="test-assessment-id",
            user_id="user-123",
            assessment_type=AssessmentType.PERIODIC,
            trigger_reason=TriggerReason.SCHEDULED_INTERVAL,
            skill_metrics={
                "intonation": 85.0,
                "rhythm": 90.0,
                "articulation": 75.0,
                "dynamics": 88.0
            },
            overall_score=84.5,
            skill_level_recommendation=SkillLevel.INTERMEDIATE,
            improvement_areas=["articulation", "breath_control"],
            strengths=["rhythm", "dynamics"],
            next_lesson_plan_id="lesson-plan-456"
        )
    
    @pytest.mark.asyncio
    async def test_create_assessment_success(self, assessment_repository, mock_firestore_client, sample_assessment):
        """Test successful assessment creation"""
        client, collection = mock_firestore_client
        
        # Mock the Firestore add method to return a document reference
        mock_doc_ref = MagicMock()
        mock_doc_ref.id = "generated-firestore-id"
        collection.add.return_value = (None, mock_doc_ref)
        
        # Call create_assessment
        result = await assessment_repository.create_assessment(sample_assessment)
        
        # Verify the result
        assert isinstance(result, FormalAssessment)
        assert result.id == "generated-firestore-id"
        assert result.user_id == sample_assessment.user_id
        assert result.assessment_type == sample_assessment.assessment_type
        assert result.overall_score == sample_assessment.overall_score
        
        # Verify Firestore collection.add was called
        collection.add.assert_called_once()
        
        # Verify the data passed to Firestore
        call_args = collection.add.call_args[0][0]
        assert call_args["user_id"] == "user-123"
        assert call_args["assessment_type"] == "periodic"
        assert call_args["overall_score"] == 84.5
        assert "assessed_at" in call_args
        assert "created_at" in call_args
        # Verify datetime conversion to ISO string
        assert isinstance(call_args["assessed_at"], str)
        assert isinstance(call_args["created_at"], str)
    
    def test_assessment_repository_initialization(self, mock_firestore_client):
        """Test that AssessmentRepository initializes correctly"""
        client, collection = mock_firestore_client
        
        repo = AssessmentRepository(client)
        
        # Verify collection was set up correctly
        client.collection.assert_called_once_with("assessments")
        assert repo._collection == collection
    
    def test_assessment_model_dump_structure(self, sample_assessment):
        """Test that FormalAssessment model dumps correctly for Firestore"""
        assessment_data = sample_assessment.model_dump()
        
        # Verify all required fields are present
        required_fields = [
            "id", "user_id", "assessment_type", "trigger_reason",
            "skill_metrics", "overall_score", "skill_level_recommendation",
            "improvement_areas", "strengths", "assessed_at", "created_at"
        ]
        
        for field in required_fields:
            assert field in assessment_data
        
        # Verify data types
        assert isinstance(assessment_data["skill_metrics"], dict)
        assert isinstance(assessment_data["improvement_areas"], list)
        assert isinstance(assessment_data["strengths"], list)
        assert isinstance(assessment_data["overall_score"], float)