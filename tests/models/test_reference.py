import pytest
from datetime import datetime
from typing import Dict, Any
from src.models.reference import ReferencePerformance, SkillLevelDefinition


class TestReferencePerformance:
    def test_reference_performance_creation(self):
        target_metrics = {
            "intonation_threshold": 85.0,
            "rhythm_precision": 0.95,
            "articulation_clarity": 88.0,
            "dynamics_range": 75.0
        }
        
        reference = ReferencePerformance(
            id="550e8400-e29b-41d4-a716-446655440700",
            exercise_id="550e8400-e29b-41d4-a716-446655440100",
            skill_level="intermediate",
            target_metrics=target_metrics,
            difficulty_weight=0.7,
            description="Reference performance for intermediate level major scales"
        )
        
        assert reference.id == "550e8400-e29b-41d4-a716-446655440700"
        assert reference.exercise_id == "550e8400-e29b-41d4-a716-446655440100"
        assert reference.skill_level == "intermediate"
        assert reference.target_metrics == target_metrics
        assert reference.difficulty_weight == 0.7
        assert reference.description == "Reference performance for intermediate level major scales"
        assert reference.is_active is True
        assert isinstance(reference.created_at, datetime)
    
    def test_reference_performance_skill_level_validation(self):
        with pytest.raises(ValueError):
            ReferencePerformance(
                id="550e8400-e29b-41d4-a716-446655440700",
                exercise_id="550e8400-e29b-41d4-a716-446655440100",
                skill_level="invalid_level",
                target_metrics={},
                difficulty_weight=0.7
            )
    
    def test_reference_performance_weight_validation(self):
        with pytest.raises(ValueError):
            ReferencePerformance(
                id="550e8400-e29b-41d4-a716-446655440700",
                exercise_id="550e8400-e29b-41d4-a716-446655440100",
                skill_level="intermediate",
                target_metrics={},
                difficulty_weight=1.5  # Invalid: > 1.0
            )
    
    def test_reference_performance_negative_weight(self):
        with pytest.raises(ValueError):
            ReferencePerformance(
                id="550e8400-e29b-41d4-a716-446655440700",
                exercise_id="550e8400-e29b-41d4-a716-446655440100",
                skill_level="intermediate",
                target_metrics={},
                difficulty_weight=-0.1  # Invalid: < 0.0
            )


class TestSkillLevelDefinition:
    def test_skill_level_definition_creation(self):
        score_thresholds = {
            "intonation_min": 80.0,
            "rhythm_min": 75.0,
            "articulation_min": 70.0,
            "dynamics_min": 65.0,
            "overall_min": 72.5
        }
        
        characteristics = [
            "Can play major scales accurately",
            "Demonstrates good breath control",
            "Shows consistent tone quality"
        ]
        
        definition = SkillLevelDefinition(
            id="550e8400-e29b-41d4-a716-446655440800",
            skill_level="intermediate",
            display_name="Intermediate Player",
            description="Solid fundamental skills with room for advanced techniques",
            score_thresholds=score_thresholds,
            characteristics=characteristics,
            typical_exercises=["scales", "arpeggios", "etudes"],
            progression_criteria="Consistent performance above threshold scores"
        )
        
        assert definition.id == "550e8400-e29b-41d4-a716-446655440800"
        assert definition.skill_level == "intermediate"
        assert definition.display_name == "Intermediate Player"
        assert definition.description == "Solid fundamental skills with room for advanced techniques"
        assert definition.score_thresholds == score_thresholds
        assert definition.characteristics == characteristics
        assert definition.typical_exercises == ["scales", "arpeggios", "etudes"]
        assert definition.progression_criteria == "Consistent performance above threshold scores"
        assert isinstance(definition.created_at, datetime)
    
    def test_skill_level_definition_skill_level_validation(self):
        with pytest.raises(ValueError):
            SkillLevelDefinition(
                id="550e8400-e29b-41d4-a716-446655440800",
                skill_level="invalid_level",
                display_name="Test Level",
                description="Test description",
                score_thresholds={},
                characteristics=[],
                typical_exercises=[],
                progression_criteria="Test criteria"
            )
    
    def test_skill_level_definition_with_optional_fields(self):
        definition = SkillLevelDefinition(
            id="550e8400-e29b-41d4-a716-446655440800",
            skill_level="beginner",
            display_name="Beginner Player",
            description="Starting their saxophone journey",
            score_thresholds={"overall_min": 50.0},
            characteristics=["Learning basic fingerings"],
            typical_exercises=["long_tone"],
            progression_criteria="Complete beginner exercises"
        )
        
        assert definition.skill_level == "beginner"
        assert definition.score_thresholds == {"overall_min": 50.0}
        assert definition.characteristics == ["Learning basic fingerings"]