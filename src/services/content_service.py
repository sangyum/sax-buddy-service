from typing import List, Optional
from src.models.content import Exercise, LessonPlan, Lesson, ExerciseType, DifficultyLevel
from src.api.schemas.requests import LessonUpdate
from src.repositories.content_repository import ContentRepository


class ContentService:
    """Service class for content-related operations"""
    
    def __init__(self, content_repository: ContentRepository):
        self.content_repository = content_repository
    
    async def list_exercises(
        self, 
        exercise_type: Optional[ExerciseType] = None,
        difficulty_level: Optional[DifficultyLevel] = None,
        limit: int = 20
    ) -> List[Exercise]:
        """List exercises with optional filtering"""
        return await self.content_repository.list_exercises(exercise_type, difficulty_level, limit)
    
    async def get_exercise_by_id(self, exercise_id: str) -> Optional[Exercise]:
        """Get exercise details"""
        return await self.content_repository.get_exercise_by_id(exercise_id)
    
    async def get_user_lesson_plans(self, user_id: str, is_active: bool = True) -> List[LessonPlan]:
        """Get user's lesson plans"""
        return await self.content_repository.get_lesson_plans_by_user_id(user_id, is_active)
    
    async def generate_lesson_plan(self, user_id: str) -> LessonPlan:
        """Generate new lesson plan for user"""
        # TODO: Implement AI lesson plan generation logic
        raise NotImplementedError("Lesson plan generation not implemented yet")
    
    async def get_lessons_in_plan(self, plan_id: str) -> List[Lesson]:
        """Get lessons in plan"""
        return await self.content_repository.get_lessons_by_plan_id(plan_id)
    
    async def get_lesson_by_id(self, lesson_id: str) -> Optional[Lesson]:
        """Get lesson details"""
        return await self.content_repository.get_lesson_by_id(lesson_id)
    
    async def update_lesson(self, lesson_id: str, lesson_update: LessonUpdate) -> Optional[Lesson]:
        """Update lesson completion status"""
        # Get existing lesson
        existing_lesson = await self.content_repository.get_lesson_by_id(lesson_id)
        if not existing_lesson:
            return None
        
        # Apply updates
        update_data = lesson_update.model_dump(exclude_unset=True)
        
        # Handle completed_at datetime string conversion
        if "completed_at" in update_data and update_data["completed_at"]:
            from datetime import datetime
            update_data["completed_at"] = datetime.fromisoformat(update_data["completed_at"])
        
        # Create updated lesson with new values
        updated_lesson = existing_lesson.model_copy(update=update_data)
        
        # Save to repository
        return await self.content_repository.update_lesson(updated_lesson)