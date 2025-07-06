from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import EmailStr, Field, field_validator, ConfigDict

from src.models.base import BaseModel


class SkillLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class PracticeFrequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    OCCASIONAL = "occasional"


class User(BaseModel):
    id: str = Field(..., description="Unique user identifier")
    email: EmailStr = Field(..., description="User email address")
    name: str = Field(..., description="User display name")
    is_active: bool = Field(default=True, description="Whether user account is active")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Account creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")



class UserProfile(BaseModel):
    id: str = Field(..., description="Unique profile identifier")
    user_id: str = Field(..., description="Reference to User")
    current_skill_level: SkillLevel = Field(..., description="User's current skill level")
    learning_goals: List[str] = Field(default_factory=list, description="User's learning objectives")
    practice_frequency: PracticeFrequency = Field(..., description="How often user practices")
    formal_assessment_interval_days: int = Field(default=30, description="Days between formal assessments")
    preferred_practice_duration_minutes: Optional[int] = Field(None, description="Preferred practice session length")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Profile creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

    @field_validator('formal_assessment_interval_days')
    @classmethod
    def validate_assessment_interval(cls, v):
        if v < 1:
            raise ValueError('Assessment interval must be at least 1 day')
        return v

    @field_validator('preferred_practice_duration_minutes')
    @classmethod
    def validate_practice_duration(cls, v):
        if v is not None and v < 1:
            raise ValueError('Practice duration must be at least 1 minute')
        return v



class UserProgress(BaseModel):
    id: str = Field(..., description="Unique progress identifier")
    user_id: str = Field(..., description="Reference to User")
    intonation_score: float = Field(..., ge=0.0, le=100.0, description="Intonation skill score (0-100)")
    rhythm_score: float = Field(..., ge=0.0, le=100.0, description="Rhythm skill score (0-100)")
    articulation_score: float = Field(..., ge=0.0, le=100.0, description="Articulation skill score (0-100)")
    dynamics_score: float = Field(..., ge=0.0, le=100.0, description="Dynamics skill score (0-100)")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall weighted score (0-100)")
    sessions_count: int = Field(..., ge=0, description="Total number of practice sessions")
    total_practice_minutes: int = Field(..., ge=0, description="Total practice time in minutes")
    last_formal_assessment_date: Optional[datetime] = Field(None, description="Date of last formal assessment")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Progress record creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

