import pytest
from datetime import datetime
from typing import List
from src.models.content import Exercise, LessonPlan, Lesson


class TestExercise:
    def test_exercise_creation(self):
        exercise = Exercise(
            id="550e8400-e29b-41d4-a716-446655440100",
            title="Major Scale Practice",
            description="Practice C major scale ascending and descending",
            exercise_type="scales",
            difficulty_level="beginner",
            estimated_duration_minutes=10,
            instructions=["Start with long tones", "Play slowly", "Focus on intonation"]
        )
        
        assert exercise.id == "550e8400-e29b-41d4-a716-446655440100"
        assert exercise.title == "Major Scale Practice"
        assert exercise.description == "Practice C major scale ascending and descending"
        assert exercise.exercise_type == "scales"
        assert exercise.difficulty_level == "beginner"
        assert exercise.estimated_duration_minutes == 10
        assert exercise.instructions == ["Start with long tones", "Play slowly", "Focus on intonation"]
        assert exercise.is_active is True
        assert isinstance(exercise.created_at, datetime)
    
    def test_exercise_type_validation(self):
        with pytest.raises(ValueError):
            Exercise(
                id="550e8400-e29b-41d4-a716-446655440100",
                title="Test Exercise",
                description="Test description",
                exercise_type="invalid_type",
                difficulty_level="beginner",
                estimated_duration_minutes=10
            )
    
    def test_exercise_difficulty_validation(self):
        with pytest.raises(ValueError):
            Exercise(
                id="550e8400-e29b-41d4-a716-446655440100",
                title="Test Exercise",
                description="Test description",
                exercise_type="scales",
                difficulty_level="invalid_level",
                estimated_duration_minutes=10
            )
    
    def test_exercise_negative_duration(self):
        with pytest.raises(ValueError):
            Exercise(
                id="550e8400-e29b-41d4-a716-446655440100",
                title="Test Exercise",
                description="Test description",
                exercise_type="scales",
                difficulty_level="beginner",
                estimated_duration_minutes=-5
            )


class TestLessonPlan:
    def test_lesson_plan_creation(self):
        lesson_plan = LessonPlan(
            id="550e8400-e29b-41d4-a716-446655440200",
            user_id="550e8400-e29b-41d4-a716-446655440000",
            title="Beginner Fundamentals Week 1",
            description="Focus on basic techniques and scales",
            target_skill_level="beginner",
            estimated_completion_days=7,
            lesson_ids=["lesson1", "lesson2", "lesson3"]
        )
        
        assert lesson_plan.id == "550e8400-e29b-41d4-a716-446655440200"
        assert lesson_plan.user_id == "550e8400-e29b-41d4-a716-446655440000"
        assert lesson_plan.title == "Beginner Fundamentals Week 1"
        assert lesson_plan.description == "Focus on basic techniques and scales"
        assert lesson_plan.target_skill_level == "beginner"
        assert lesson_plan.estimated_completion_days == 7
        assert lesson_plan.lesson_ids == ["lesson1", "lesson2", "lesson3"]
        assert lesson_plan.is_active is True
        assert isinstance(lesson_plan.generated_at, datetime)
    
    def test_lesson_plan_skill_level_validation(self):
        with pytest.raises(ValueError):
            LessonPlan(
                id="550e8400-e29b-41d4-a716-446655440200",
                user_id="550e8400-e29b-41d4-a716-446655440000",
                title="Test Plan",
                description="Test description",
                target_skill_level="invalid_level",
                estimated_completion_days=7,
                lesson_ids=[]
            )
    
    def test_lesson_plan_negative_completion_days(self):
        with pytest.raises(ValueError):
            LessonPlan(
                id="550e8400-e29b-41d4-a716-446655440200",
                user_id="550e8400-e29b-41d4-a716-446655440000",
                title="Test Plan",
                description="Test description",
                target_skill_level="beginner",
                estimated_completion_days=-1,
                lesson_ids=[]
            )


class TestLesson:
    def test_lesson_creation(self):
        lesson = Lesson(
            id="550e8400-e29b-41d4-a716-446655440300",
            lesson_plan_id="550e8400-e29b-41d4-a716-446655440200",
            title="Basic Breathing Techniques",
            description="Learn proper saxophone breathing",
            order_in_plan=1,
            exercise_ids=["ex1", "ex2", "ex3"],
            estimated_duration_minutes=30,
            learning_objectives=["Master diaphragmatic breathing", "Understand air flow control"]
        )
        
        assert lesson.id == "550e8400-e29b-41d4-a716-446655440300"
        assert lesson.lesson_plan_id == "550e8400-e29b-41d4-a716-446655440200"
        assert lesson.title == "Basic Breathing Techniques"
        assert lesson.description == "Learn proper saxophone breathing"
        assert lesson.order_in_plan == 1
        assert lesson.exercise_ids == ["ex1", "ex2", "ex3"]
        assert lesson.estimated_duration_minutes == 30
        assert lesson.learning_objectives == ["Master diaphragmatic breathing", "Understand air flow control"]
        assert lesson.is_completed is False
        assert isinstance(lesson.created_at, datetime)
    
    def test_lesson_negative_order(self):
        with pytest.raises(ValueError):
            Lesson(
                id="550e8400-e29b-41d4-a716-446655440300",
                lesson_plan_id="550e8400-e29b-41d4-a716-446655440200",
                title="Test Lesson",
                description="Test description",
                order_in_plan=-1,
                exercise_ids=[],
                estimated_duration_minutes=30
            )
    
    def test_lesson_negative_duration(self):
        with pytest.raises(ValueError):
            Lesson(
                id="550e8400-e29b-41d4-a716-446655440300",
                lesson_plan_id="550e8400-e29b-41d4-a716-446655440200",
                title="Test Lesson",
                description="Test description",
                order_in_plan=1,
                exercise_ids=[],
                estimated_duration_minutes=-5
            )