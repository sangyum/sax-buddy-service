from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, field_serializer, ConfigDict


class ExerciseType(str, Enum):
    SCALES = "scales"
    ARPEGGIOS = "arpeggios"
    TECHNICAL = "technical"
    ETUDES = "etudes"
    SONGS = "songs"
    LONG_TONE = "long_tone"


class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class Exercise(BaseModel):
    id: str = Field(..., description="Unique exercise identifier")
    title: str = Field(..., description="Exercise title")
    description: str = Field(..., description="Exercise description")
    exercise_type: ExerciseType = Field(..., description="Type of exercise")
    difficulty_level: DifficultyLevel = Field(..., description="Exercise difficulty")
    estimated_duration_minutes: int = Field(..., ge=1, description="Estimated time to complete")
    instructions: List[str] = Field(default_factory=list, description="Step-by-step instructions")
    reference_audio_url: Optional[str] = Field(None, description="URL to reference audio file")
    sheet_music_url: Optional[str] = Field(None, description="URL to sheet music")
    is_active: bool = Field(default=True, description="Whether exercise is available")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Exercise creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

    @field_serializer('created_at', 'updated_at')
    def serialize_dt(self, dt: datetime) -> str:
        return dt.isoformat()


class LessonPlan(BaseModel):
    id: str = Field(..., description="Unique lesson plan identifier")
    user_id: str = Field(..., description="Reference to User")
    title: str = Field(..., description="Lesson plan title")
    description: str = Field(..., description="Plan description and objectives")
    target_skill_level: DifficultyLevel = Field(..., description="Target skill level for this plan")
    estimated_completion_days: int = Field(..., ge=1, description="Expected days to complete")
    lesson_ids: List[str] = Field(default_factory=list, description="Ordered list of lesson references")
    is_active: bool = Field(default=True, description="Whether plan is currently active")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Plan generation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Plan completion timestamp")
    next_formal_assessment_due: Optional[datetime] = Field(None, description="When formal assessment is due")

    @field_serializer('generated_at', 'completed_at', 'next_formal_assessment_due')
    def serialize_dt(self, dt: datetime) -> str:
        return dt.isoformat()


class Lesson(BaseModel):
    id: str = Field(..., description="Unique lesson identifier")
    lesson_plan_id: str = Field(..., description="Reference to LessonPlan")
    title: str = Field(..., description="Lesson title")
    description: str = Field(..., description="Lesson description")
    order_in_plan: int = Field(..., ge=1, description="Sequence order within lesson plan")
    exercise_ids: List[str] = Field(default_factory=list, description="Exercises included in this lesson")
    estimated_duration_minutes: int = Field(..., ge=1, description="Estimated lesson duration")
    learning_objectives: List[str] = Field(default_factory=list, description="What student should learn")
    prerequisites: List[str] = Field(default_factory=list, description="Skills needed before this lesson")
    is_completed: bool = Field(default=False, description="Whether user has completed this lesson")
    completed_at: Optional[datetime] = Field(None, description="Lesson completion timestamp")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Lesson creation timestamp")

    @field_serializer('completed_at', 'created_at')
    def serialize_dt(self, dt: datetime) -> str:
        return dt.isoformat()