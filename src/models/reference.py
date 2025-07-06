from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import Field, field_validator, ConfigDict

from src.models.base import BaseModel


class SkillLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class ReferencePerformance(BaseModel):
    id: str = Field(..., description="Unique reference performance identifier")
    exercise_id: str = Field(..., description="Reference to Exercise")
    skill_level: SkillLevel = Field(..., description="Skill level this reference applies to")
    target_metrics: Dict[str, Any] = Field(..., description="Expected performance metrics for this skill level")
    difficulty_weight: float = Field(..., ge=0.0, le=1.0, description="Difficulty weighting factor")
    description: str = Field(default="", description="Description of this reference performance")
    audio_reference_url: Optional[str] = Field(None, description="URL to reference audio file")
    is_active: bool = Field(default=True, description="Whether this reference is currently active")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Reference creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")



class SkillLevelDefinition(BaseModel):
    id: str = Field(..., description="Unique skill level definition identifier")
    skill_level: SkillLevel = Field(..., description="The skill level being defined")
    display_name: str = Field(..., description="Human-readable name for this skill level")
    description: str = Field(..., description="Detailed description of this skill level")
    score_thresholds: Dict[str, float] = Field(..., description="Minimum scores required for this level")
    characteristics: List[str] = Field(..., description="Key characteristics of players at this level")
    typical_exercises: List[str] = Field(..., description="Exercise types appropriate for this level")
    progression_criteria: str = Field(..., description="How a player advances from this level")
    estimated_hours_to_achieve: Optional[int] = Field(None, description="Typical practice hours to reach this level")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Definition creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")