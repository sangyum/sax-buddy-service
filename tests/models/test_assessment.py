import pytest
from datetime import datetime
from typing import Dict, Any, List
from src.models.assessment import FormalAssessment, Feedback, SkillMetrics


class TestFormalAssessment:
    def test_formal_assessment_creation(self):
        skill_metrics = {
            "intonation": 85.5,
            "rhythm": 92.0,
            "articulation": 78.5,
            "dynamics": 88.0
        }
        
        assessment = FormalAssessment(
            id="550e8400-e29b-41d4-a716-446655440400",
            user_id="550e8400-e29b-41d4-a716-446655440000",
            assessment_type="periodic",
            trigger_reason="scheduled_interval",
            skill_metrics=skill_metrics,
            overall_score=86.0,
            skill_level_recommendation="intermediate",
            improvement_areas=["articulation", "dynamics"],
            strengths=["intonation", "rhythm"]
        )
        
        assert assessment.id == "550e8400-e29b-41d4-a716-446655440400"
        assert assessment.user_id == "550e8400-e29b-41d4-a716-446655440000"
        assert assessment.assessment_type == "periodic"
        assert assessment.trigger_reason == "scheduled_interval"
        assert assessment.skill_metrics == skill_metrics
        assert assessment.overall_score == 86.0
        assert assessment.skill_level_recommendation == "intermediate"
        assert assessment.improvement_areas == ["articulation", "dynamics"]
        assert assessment.strengths == ["intonation", "rhythm"]
        assert isinstance(assessment.assessed_at, datetime)
    
    def test_formal_assessment_type_validation(self):
        with pytest.raises(ValueError):
            FormalAssessment(
                id="550e8400-e29b-41d4-a716-446655440400",
                user_id="550e8400-e29b-41d4-a716-446655440000",
                assessment_type="invalid_type",
                trigger_reason="user_requested",
                skill_metrics={},
                overall_score=86.0,
                skill_level_recommendation="intermediate"
            )
    
    def test_formal_assessment_trigger_validation(self):
        with pytest.raises(ValueError):
            FormalAssessment(
                id="550e8400-e29b-41d4-a716-446655440400",
                user_id="550e8400-e29b-41d4-a716-446655440000",
                assessment_type="periodic",
                trigger_reason="invalid_reason",
                skill_metrics={},
                overall_score=86.0,
                skill_level_recommendation="intermediate"
            )
    
    def test_formal_assessment_skill_level_validation(self):
        with pytest.raises(ValueError):
            FormalAssessment(
                id="550e8400-e29b-41d4-a716-446655440400",
                user_id="550e8400-e29b-41d4-a716-446655440000",
                assessment_type="periodic",
                trigger_reason="user_requested",
                skill_metrics={},
                overall_score=86.0,
                skill_level_recommendation="invalid_level"
            )
    
    def test_formal_assessment_score_validation(self):
        with pytest.raises(ValueError):
            FormalAssessment(
                id="550e8400-e29b-41d4-a716-446655440400",
                user_id="550e8400-e29b-41d4-a716-446655440000",
                assessment_type="periodic",
                trigger_reason="user_requested",
                skill_metrics={},
                overall_score=105.0,  # Invalid: > 100
                skill_level_recommendation="intermediate"
            )


class TestFeedback:
    def test_feedback_creation(self):
        feedback = Feedback(
            id="550e8400-e29b-41d4-a716-446655440500",
            session_id="550e8400-e29b-41d4-a716-446655440010",
            feedback_type="post_session",
            message="Great improvement in your intonation! Keep focusing on breath support.",
            encouragement="You're making excellent progress!",
            specific_suggestions=["Practice long tones daily", "Work on scale exercises"],
            areas_of_improvement=["articulation", "dynamics"],
            strengths_highlighted=["intonation", "rhythm"]
        )
        
        assert feedback.id == "550e8400-e29b-41d4-a716-446655440500"
        assert feedback.session_id == "550e8400-e29b-41d4-a716-446655440010"
        assert feedback.feedback_type == "post_session"
        assert feedback.message == "Great improvement in your intonation! Keep focusing on breath support."
        assert feedback.encouragement == "You're making excellent progress!"
        assert feedback.specific_suggestions == ["Practice long tones daily", "Work on scale exercises"]
        assert feedback.areas_of_improvement == ["articulation", "dynamics"]
        assert feedback.strengths_highlighted == ["intonation", "rhythm"]
        assert isinstance(feedback.created_at, datetime)
    
    def test_feedback_type_validation(self):
        with pytest.raises(ValueError):
            Feedback(
                id="550e8400-e29b-41d4-a716-446655440500",
                session_id="550e8400-e29b-41d4-a716-446655440010",
                feedback_type="invalid_type",
                message="Test message"
            )


class TestSkillMetrics:
    def test_skill_metrics_creation(self):
        metrics = SkillMetrics(
            id="550e8400-e29b-41d4-a716-446655440600",
            user_id="550e8400-e29b-41d4-a716-446655440000",
            intonation_score=85.5,
            rhythm_score=92.0,
            articulation_score=78.5,
            dynamics_score=88.0,
            overall_score=86.0,
            measurement_period_days=7,
            sessions_analyzed=15,
            confidence_level=0.85
        )
        
        assert metrics.id == "550e8400-e29b-41d4-a716-446655440600"
        assert metrics.user_id == "550e8400-e29b-41d4-a716-446655440000"
        assert metrics.intonation_score == 85.5
        assert metrics.rhythm_score == 92.0
        assert metrics.articulation_score == 78.5
        assert metrics.dynamics_score == 88.0
        assert metrics.overall_score == 86.0
        assert metrics.measurement_period_days == 7
        assert metrics.sessions_analyzed == 15
        assert metrics.confidence_level == 0.85
        assert isinstance(metrics.calculated_at, datetime)
    
    def test_skill_metrics_score_validation(self):
        with pytest.raises(ValueError):
            SkillMetrics(
                id="550e8400-e29b-41d4-a716-446655440600",
                user_id="550e8400-e29b-41d4-a716-446655440000",
                intonation_score=105.0,  # Invalid: > 100
                rhythm_score=92.0,
                articulation_score=78.5,
                dynamics_score=88.0,
                overall_score=86.0,
                measurement_period_days=7,
                sessions_analyzed=15
            )
    
    def test_skill_metrics_confidence_validation(self):
        with pytest.raises(ValueError):
            SkillMetrics(
                id="550e8400-e29b-41d4-a716-446655440600",
                user_id="550e8400-e29b-41d4-a716-446655440000",
                intonation_score=85.5,
                rhythm_score=92.0,
                articulation_score=78.5,
                dynamics_score=88.0,
                overall_score=86.0,
                measurement_period_days=7,
                sessions_analyzed=15,
                confidence_level=1.5  # Invalid: > 1.0
            )
    
    def test_skill_metrics_negative_values(self):
        with pytest.raises(ValueError):
            SkillMetrics(
                id="550e8400-e29b-41d4-a716-446655440600",
                user_id="550e8400-e29b-41d4-a716-446655440000",
                intonation_score=85.5,
                rhythm_score=92.0,
                articulation_score=78.5,
                dynamics_score=88.0,
                overall_score=86.0,
                measurement_period_days=-1,  # Invalid: negative
                sessions_analyzed=15
            )