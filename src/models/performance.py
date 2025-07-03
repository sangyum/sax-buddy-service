from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, field_serializer, ConfigDict


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

    @field_serializer('started_at', 'ended_at', 'created_at', 'updated_at')
    def serialize_dt(self, dt: datetime) -> str:
        return dt.isoformat()


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

    @field_serializer('created_at')
    def serialize_dt(self, dt: datetime) -> str:
        return dt.isoformat()