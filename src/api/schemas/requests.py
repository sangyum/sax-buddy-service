from typing import List, Optional
from pydantic import BaseModel, EmailStr
from src.models.user import SkillLevel, PracticeFrequency
from src.models.performance import SessionStatus
from src.models.assessment import TriggerReason


class UserCreate(BaseModel):
    email: EmailStr
    name: str


class UserProfileUpdate(BaseModel):
    current_skill_level: Optional[SkillLevel] = None
    learning_goals: Optional[List[str]] = None
    practice_frequency: Optional[PracticeFrequency] = None
    formal_assessment_interval_days: Optional[int] = None
    preferred_practice_duration_minutes: Optional[int] = None


class PerformanceSessionCreate(BaseModel):
    user_id: str
    exercise_id: str


class PerformanceSessionUpdate(BaseModel):
    duration_minutes: Optional[int] = None
    status: Optional[SessionStatus] = None
    ended_at: Optional[str] = None  # ISO datetime string
    notes: Optional[str] = None


class PerformanceMetricsCreate(BaseModel):
    timestamp: float
    intonation_score: float
    rhythm_score: float
    articulation_score: float
    dynamics_score: float
    raw_metrics: dict = {}


class LessonUpdate(BaseModel):
    is_completed: Optional[bool] = None
    completed_at: Optional[str] = None  # ISO datetime string


class AssessmentTrigger(BaseModel):
    trigger_reason: TriggerReason