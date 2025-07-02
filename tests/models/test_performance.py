import pytest
from datetime import datetime
from typing import Dict, Any, List
from src.models.performance import PerformanceSession, PerformanceMetrics


class TestPerformanceSession:
    def test_performance_session_creation(self):
        session = PerformanceSession(
            id="550e8400-e29b-41d4-a716-446655440010",
            user_id="550e8400-e29b-41d4-a716-446655440000",
            exercise_id="550e8400-e29b-41d4-a716-446655440100",
            duration_minutes=25,
            status="completed"
        )
        
        assert session.id == "550e8400-e29b-41d4-a716-446655440010"
        assert session.user_id == "550e8400-e29b-41d4-a716-446655440000"
        assert session.exercise_id == "550e8400-e29b-41d4-a716-446655440100"
        assert session.duration_minutes == 25
        assert session.status == "completed"
        assert isinstance(session.started_at, datetime)
        assert session.ended_at is None
        assert session.notes == ""
    
    def test_performance_session_with_optional_fields(self):
        session = PerformanceSession(
            id="550e8400-e29b-41d4-a716-446655440010",
            user_id="550e8400-e29b-41d4-a716-446655440000",
            exercise_id="550e8400-e29b-41d4-a716-446655440100",
            duration_minutes=25,
            status="completed",
            ended_at=datetime.now(),
            notes="Great practice session"
        )
        
        assert session.ended_at is not None
        assert session.notes == "Great practice session"
    
    def test_performance_session_status_validation(self):
        with pytest.raises(ValueError):
            PerformanceSession(
                id="550e8400-e29b-41d4-a716-446655440010",
                user_id="550e8400-e29b-41d4-a716-446655440000",
                exercise_id="550e8400-e29b-41d4-a716-446655440100",
                duration_minutes=25,
                status="invalid_status"
            )
    
    def test_performance_session_negative_duration(self):
        with pytest.raises(ValueError):
            PerformanceSession(
                id="550e8400-e29b-41d4-a716-446655440010",
                user_id="550e8400-e29b-41d4-a716-446655440000",
                exercise_id="550e8400-e29b-41d4-a716-446655440100",
                duration_minutes=-5,
                status="completed"
            )


class TestPerformanceMetrics:
    def test_performance_metrics_creation(self):
        metrics_data = {
            "pitch_accuracy": 85.5,
            "rhythm_precision": 92.0,
            "note_onsets": [0.0, 0.5, 1.0, 1.5],
            "amplitude_profile": [0.8, 0.9, 0.7, 0.85]
        }
        
        metrics = PerformanceMetrics(
            id="550e8400-e29b-41d4-a716-446655440011",
            session_id="550e8400-e29b-41d4-a716-446655440010",
            timestamp=1.5,
            intonation_score=85.5,
            rhythm_score=92.0,
            articulation_score=78.5,
            dynamics_score=88.0,
            raw_metrics=metrics_data
        )
        
        assert metrics.id == "550e8400-e29b-41d4-a716-446655440011"
        assert metrics.session_id == "550e8400-e29b-41d4-a716-446655440010"
        assert metrics.timestamp == 1.5
        assert metrics.intonation_score == 85.5
        assert metrics.rhythm_score == 92.0
        assert metrics.articulation_score == 78.5
        assert metrics.dynamics_score == 88.0
        assert metrics.raw_metrics == metrics_data
        assert isinstance(metrics.created_at, datetime)
    
    def test_performance_metrics_score_validation(self):
        with pytest.raises(ValueError):
            PerformanceMetrics(
                id="550e8400-e29b-41d4-a716-446655440011",
                session_id="550e8400-e29b-41d4-a716-446655440010",
                timestamp=1.5,
                intonation_score=105.0,  # Invalid: > 100
                rhythm_score=92.0,
                articulation_score=78.5,
                dynamics_score=88.0,
                raw_metrics={}
            )
    
    def test_performance_metrics_negative_timestamp(self):
        with pytest.raises(ValueError):
            PerformanceMetrics(
                id="550e8400-e29b-41d4-a716-446655440011",
                session_id="550e8400-e29b-41d4-a716-446655440010",
                timestamp=-1.0,  # Invalid: negative
                intonation_score=85.5,
                rhythm_score=92.0,
                articulation_score=78.5,
                dynamics_score=88.0,
                raw_metrics={}
            )
    
    def test_performance_metrics_empty_raw_metrics(self):
        metrics = PerformanceMetrics(
            id="550e8400-e29b-41d4-a716-446655440011",
            session_id="550e8400-e29b-41d4-a716-446655440010",
            timestamp=1.5,
            intonation_score=85.5,
            rhythm_score=92.0,
            articulation_score=78.5,
            dynamics_score=88.0,
            raw_metrics={}
        )
        
        assert metrics.raw_metrics == {}