from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from pydantic import Field, field_validator, ConfigDict

from src.models.base import BaseModel


class SessionStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class PerformanceSession(BaseModel):
    id: str = Field(..., description="Unique session identifier")
    user_id: str = Field(..., description="Reference to User")
    exercise_id: str = Field(..., description="Reference to Exercise")
    duration_minutes: int = Field(..., ge=0, description="Session duration in minutes")
    status: SessionStatus = Field(..., description="Current session status")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Session start timestamp")
    ended_at: Optional[datetime] = Field(None, description="Session end timestamp")
    notes: str = Field(default="", description="Optional session notes")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Record creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")



class PerformanceMetrics(BaseModel):
    id: str = Field(..., description="Unique metrics identifier")
    session_id: str = Field(..., description="Reference to PerformanceSession")
    timestamp: float = Field(..., ge=0.0, description="Time offset within session (seconds)")
    intonation_score: float = Field(..., ge=0.0, le=100.0, description="Intonation score (0-100)")
    rhythm_score: float = Field(..., ge=0.0, le=100.0, description="Rhythm score (0-100)")
    articulation_score: float = Field(..., ge=0.0, le=100.0, description="Articulation score (0-100)")
    dynamics_score: float = Field(..., ge=0.0, le=100.0, description="Dynamics score (0-100)")
    raw_metrics: Dict[str, Any] = Field(default_factory=dict, description="Raw DSP analysis data from mobile")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Metrics creation timestamp")



class SessionSummary(BaseModel):
    """Summary statistics for user sessions"""
    total_sessions: int = Field(..., ge=0, description="Total number of completed sessions")
    total_minutes: int = Field(..., ge=0, description="Total duration in minutes")


class PerformanceTrendPoint(BaseModel):
    """Performance trend data point"""
    avg_intonation: float = Field(..., ge=0.0, le=100.0, description="Average intonation score")
    avg_rhythm: float = Field(..., ge=0.0, le=100.0, description="Average rhythm score")
    avg_articulation: float = Field(..., ge=0.0, le=100.0, description="Average articulation score")
    avg_dynamics: float = Field(..., ge=0.0, le=100.0, description="Average dynamics score")
    session_count: int = Field(..., ge=0, description="Number of sessions in this period")
    period_start: datetime = Field(..., description="Start of the measurement period")
    period_end: datetime = Field(..., description="End of the measurement period")


class SessionStatistics(BaseModel):
    """Comprehensive session statistics for a user"""
    total_sessions: int = Field(..., ge=0, description="Total number of sessions")
    completed_sessions: int = Field(..., ge=0, description="Number of completed sessions")
    in_progress_sessions: int = Field(..., ge=0, description="Number of in-progress sessions")
    cancelled_sessions: int = Field(..., ge=0, description="Number of cancelled sessions")
    completion_rate: float = Field(..., ge=0.0, le=1.0, description="Completion rate (0.0 to 1.0)")
    total_duration_seconds: float = Field(..., ge=0.0, description="Total practice time in seconds")
    average_duration_seconds: float = Field(..., ge=0.0, description="Average session duration in seconds")
    longest_session_seconds: float = Field(..., ge=0.0, description="Longest session duration in seconds")
    total_practice_days: int = Field(..., ge=0, description="Number of unique practice days")
    
    @property
    def total_duration_minutes(self) -> float:
        """Total duration in minutes"""
        return self.total_duration_seconds / 60.0
    
    @property
    def average_duration_minutes(self) -> float:
        """Average duration in minutes"""
        return self.average_duration_seconds / 60.0
    
    @property
    def longest_session_minutes(self) -> float:
        """Longest session in minutes"""
        return self.longest_session_seconds / 60.0