from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import Field, field_validator, ConfigDict

from src.models.base import BaseModel


class AssessmentType(str, Enum):
    PERIODIC = "periodic"
    ON_DEMAND = "on_demand"
    MILESTONE = "milestone"


class TriggerReason(str, Enum):
    SCHEDULED_INTERVAL = "scheduled_interval"
    USER_REQUESTED = "user_requested"
    SKILL_THRESHOLD = "skill_threshold"
    LESSON_COMPLETION = "lesson_completion"


class SkillLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class FeedbackType(str, Enum):
    POST_SESSION = "post_session"
    MOTIVATIONAL = "motivational"
    CORRECTIVE = "corrective"


class FormalAssessment(BaseModel):
    id: str = Field(..., description="Unique assessment identifier")
    user_id: str = Field(..., description="Reference to User")
    assessment_type: AssessmentType = Field(..., description="Type of assessment")
    trigger_reason: TriggerReason = Field(..., description="What triggered this assessment")
    skill_metrics: Dict[str, float] = Field(..., description="Detailed skill scores")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Weighted overall score")
    skill_level_recommendation: SkillLevel = Field(..., description="Recommended skill level")
    improvement_areas: List[str] = Field(default_factory=list, description="Areas needing improvement")
    strengths: List[str] = Field(default_factory=list, description="User's strongest areas")
    next_lesson_plan_id: Optional[str] = Field(None, description="Generated lesson plan based on assessment")
    assessed_at: datetime = Field(default_factory=datetime.utcnow, description="Assessment timestamp")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Record creation timestamp")



class Feedback(BaseModel):
    id: str = Field(..., description="Unique feedback identifier")
    user_id: str = Field(..., description="Reference to User")
    session_id: str = Field(..., description="Reference to PerformanceSession")
    feedback_type: FeedbackType = Field(..., description="Type of feedback")
    message: str = Field(..., description="Main feedback message")
    encouragement: str = Field(default="", description="Motivational message")
    specific_suggestions: List[str] = Field(default_factory=list, description="Actionable improvement suggestions")
    areas_of_improvement: List[str] = Field(default_factory=list, description="Skills to focus on")
    strengths_highlighted: List[str] = Field(default_factory=list, description="User's demonstrated strengths")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="AI confidence in feedback accuracy")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Feedback creation timestamp")



class SkillMetrics(BaseModel):
    id: str = Field(..., description="Unique metrics identifier")
    user_id: str = Field(..., description="Reference to User")
    intonation_score: float = Field(..., ge=0.0, le=100.0, description="Weighted intonation score")
    rhythm_score: float = Field(..., ge=0.0, le=100.0, description="Weighted rhythm score")
    articulation_score: float = Field(..., ge=0.0, le=100.0, description="Weighted articulation score")
    dynamics_score: float = Field(..., ge=0.0, le=100.0, description="Weighted dynamics score")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Composite weighted score")
    measurement_period_days: int = Field(..., ge=1, description="Period over which metrics were calculated")
    sessions_analyzed: int = Field(..., ge=0, description="Number of sessions used in calculation")
    confidence_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Statistical confidence in metrics")
    trend_direction: Optional[str] = Field(None, description="Improving, declining, or stable")
    calculated_at: datetime = Field(default_factory=datetime.utcnow, description="Metrics calculation timestamp")

