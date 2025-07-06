"""Tests for performance domain models"""

import pytest
from datetime import datetime, timezone
from src.models.performance import SessionStatistics, SessionSummary, PerformanceTrendPoint


class TestSessionStatistics:
    """Test SessionStatistics model"""

    def test_session_statistics_creation(self):
        """Test creating a SessionStatistics instance"""
        stats = SessionStatistics(
            total_sessions=10,
            completed_sessions=8,
            in_progress_sessions=1,
            cancelled_sessions=1,
            completion_rate=0.8,
            total_duration_seconds=3600.0,  # 1 hour
            average_duration_seconds=450.0,  # 7.5 minutes
            longest_session_seconds=900.0,  # 15 minutes
            total_practice_days=5
        )
        
        assert stats.total_sessions == 10
        assert stats.completed_sessions == 8
        assert stats.in_progress_sessions == 1
        assert stats.cancelled_sessions == 1
        assert stats.completion_rate == 0.8
        assert stats.total_duration_seconds == 3600.0
        assert stats.average_duration_seconds == 450.0
        assert stats.longest_session_seconds == 900.0
        assert stats.total_practice_days == 5

    def test_session_statistics_convenience_properties(self):
        """Test convenience properties for minute conversions"""
        stats = SessionStatistics(
            total_sessions=5,
            completed_sessions=5,
            in_progress_sessions=0,
            cancelled_sessions=0,
            completion_rate=1.0,
            total_duration_seconds=3600.0,  # 1 hour = 60 minutes
            average_duration_seconds=720.0,  # 12 minutes
            longest_session_seconds=1800.0,  # 30 minutes
            total_practice_days=3
        )
        
        # Test minute conversion properties
        assert stats.total_duration_minutes == 60.0
        assert stats.average_duration_minutes == 12.0
        assert stats.longest_session_minutes == 30.0

    def test_session_statistics_validation(self):
        """Test SessionStatistics field validation"""
        # Test valid completion rate
        stats = SessionStatistics(
            total_sessions=4,
            completed_sessions=2,
            in_progress_sessions=1,
            cancelled_sessions=1,
            completion_rate=0.5,
            total_duration_seconds=1800.0,
            average_duration_seconds=900.0,
            longest_session_seconds=1200.0,
            total_practice_days=2
        )
        assert stats.completion_rate == 0.5

        # Test invalid completion rate (should raise validation error)
        with pytest.raises(ValueError):
            SessionStatistics(
                total_sessions=4,
                completed_sessions=2,
                in_progress_sessions=1,
                cancelled_sessions=1,
                completion_rate=1.5,  # Invalid: > 1.0
                total_duration_seconds=1800.0,
                average_duration_seconds=900.0,
                longest_session_seconds=1200.0,
                total_practice_days=2
            )

        # Test negative values (should raise validation error)
        with pytest.raises(ValueError):
            SessionStatistics(
                total_sessions=-1,  # Invalid: negative
                completed_sessions=2,
                in_progress_sessions=1,
                cancelled_sessions=1,
                completion_rate=0.5,
                total_duration_seconds=1800.0,
                average_duration_seconds=900.0,
                longest_session_seconds=1200.0,
                total_practice_days=2
            )


class TestSessionSummary:
    """Test SessionSummary model"""

    def test_session_summary_creation(self):
        """Test creating a SessionSummary instance"""
        summary = SessionSummary(
            total_sessions=15,
            total_minutes=450
        )
        
        assert summary.total_sessions == 15
        assert summary.total_minutes == 450

    def test_session_summary_validation(self):
        """Test SessionSummary field validation"""
        # Test negative values (should raise validation error)
        with pytest.raises(ValueError):
            SessionSummary(
                total_sessions=-1,  # Invalid: negative
                total_minutes=450
            )
        
        with pytest.raises(ValueError):
            SessionSummary(
                total_sessions=15,
                total_minutes=-1  # Invalid: negative
            )


class TestPerformanceTrendPoint:
    """Test PerformanceTrendPoint model"""

    def test_performance_trend_point_creation(self):
        """Test creating a PerformanceTrendPoint instance"""
        now = datetime.now(timezone.utc)
        trend_point = PerformanceTrendPoint(
            avg_intonation=85.5,
            avg_rhythm=78.2,
            avg_articulation=92.1,
            avg_dynamics=80.0,
            session_count=10,
            period_start=now,
            period_end=now
        )
        
        assert trend_point.avg_intonation == 85.5
        assert trend_point.avg_rhythm == 78.2
        assert trend_point.avg_articulation == 92.1
        assert trend_point.avg_dynamics == 80.0
        assert trend_point.session_count == 10

    def test_performance_trend_point_validation(self):
        """Test PerformanceTrendPoint field validation"""
        now = datetime.now(timezone.utc)
        
        # Test invalid score values (should raise validation error)
        with pytest.raises(ValueError):
            PerformanceTrendPoint(
                avg_intonation=150.0,  # Invalid: > 100.0
                avg_rhythm=78.2,
                avg_articulation=92.1,
                avg_dynamics=80.0,
                session_count=10,
                period_start=now,
                period_end=now
            )
        
        with pytest.raises(ValueError):
            PerformanceTrendPoint(
                avg_intonation=85.5,
                avg_rhythm=-10.0,  # Invalid: < 0.0
                avg_articulation=92.1,
                avg_dynamics=80.0,
                session_count=10,
                period_start=now,
                period_end=now
            )
        
        with pytest.raises(ValueError):
            PerformanceTrendPoint(
                avg_intonation=85.5,
                avg_rhythm=78.2,
                avg_articulation=92.1,
                avg_dynamics=80.0,
                session_count=-1,  # Invalid: negative
                period_start=now,
                period_end=now
            )