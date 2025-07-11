from typing import List, Optional
from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1.async_collection import AsyncCollectionReference
from src.models.content import Exercise, LessonPlan, Lesson, ExerciseType, DifficultyLevel


class ContentRepository:
    """Repository for content-related data operations"""
    
    _collection: AsyncCollectionReference
    
    def __init__(self, firestore_client: AsyncClient):
        """Initialize ContentRepository with Firestore client"""
        self._collection = firestore_client.collection("content")
    
    # Exercise CRUD operations
    async def create_exercise(self, exercise: Exercise) -> Exercise:
        """Create a new exercise"""
        # Convert Pydantic model to dict for Firestore
        exercise_data = exercise.to_dict()
        
        # Create document in Firestore
        doc_ref = await self._collection.add(exercise_data)
        
        # Update the exercise with the generated document ID
        exercise_data["id"] = doc_ref[1].id
        
        # Return the created exercise as a Pydantic model
        return Exercise(**exercise_data)
    
    async def get_exercise_by_id(self, exercise_id: str) -> Optional[Exercise]:
        """Get exercise by ID"""
        doc = await self._collection.document(exercise_id).get()
        if not doc.exists:
            return None
        
        data = doc.to_dict()
        if data is None:
            return None
            
        data["id"] = doc.id
        
        
        return Exercise.from_dict(data)
    
    async def list_exercises(
        self, 
        exercise_type: Optional[ExerciseType] = None,
        difficulty_level: Optional[DifficultyLevel] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Exercise]:
        """List exercises with optional filtering and pagination"""
        query = self._collection.where("is_active", "==", True)
        
        if exercise_type:
            query = query.where("exercise_type", "==", exercise_type.value)
        if difficulty_level:
            query = query.where("difficulty_level", "==", difficulty_level.value)
        
        query = query.order_by("created_at", direction="DESCENDING")
        query = query.limit(limit).offset(offset)
        
        docs = query.stream()
        exercises = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            
            exercises.append(Exercise.from_dict(data))
        
        return exercises
    
    async def get_exercises_by_type(self, exercise_type: ExerciseType) -> List[Exercise]:
        """Get exercises by type"""
        query = self._collection.where("exercise_type", "==", exercise_type.value)
        query = query.where("is_active", "==", True)
        query = query.order_by("created_at", direction="DESCENDING")
        
        docs = query.stream()
        exercises = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            
            exercises.append(Exercise.from_dict(data))
        
        return exercises
    
    async def get_exercises_by_difficulty(self, difficulty_level: DifficultyLevel) -> List[Exercise]:
        """Get exercises by difficulty level"""
        query = self._collection.where("difficulty_level", "==", difficulty_level.value)
        query = query.where("is_active", "==", True)
        query = query.order_by("created_at", direction="DESCENDING")
        
        docs = query.stream()
        exercises = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            
            exercises.append(Exercise.from_dict(data))
        
        return exercises
    
    async def update_exercise(self, exercise: Exercise) -> Exercise:
        """Update an existing exercise"""
        exercise_data = exercise.to_dict()
        
        # Remove the ID from data since it's used as document ID
        doc_id = exercise_data.pop("id")
        
        # Update document in Firestore
        await self._collection.document(doc_id).update(exercise_data)
        
        return exercise
    
    async def delete_exercise(self, exercise_id: str) -> bool:
        """Delete an exercise"""
        try:
            await self._collection.document(exercise_id).delete()
            return True
        except Exception:
            return False
    
    # LessonPlan CRUD operations
    async def create_lesson_plan(self, lesson_plan: LessonPlan) -> LessonPlan:
        """Create a new lesson plan"""
        # Convert Pydantic model to dict for Firestore
        lesson_plan_data = lesson_plan.to_dict()
        
        # Create document in Firestore
        doc_ref = await self._collection.add(lesson_plan_data)
        
        # Update the lesson plan with the generated document ID
        lesson_plan_data["id"] = doc_ref[1].id
        
        # Return the created lesson plan as a Pydantic model
        return LessonPlan(**lesson_plan_data)
    
    async def get_lesson_plan_by_id(self, plan_id: str) -> Optional[LessonPlan]:
        """Get lesson plan by ID"""
        doc = await self._collection.document(plan_id).get()
        if not doc.exists:
            return None
        
        data = doc.to_dict()
        if data is None:
            return None
            
        data["id"] = doc.id
        
        
        return LessonPlan.from_dict(data)
    
    async def get_lesson_plans_by_user_id(
        self, 
        user_id: str, 
        is_active: bool = True,
        limit: int = 50,
        offset: int = 0
    ) -> List[LessonPlan]:
        """Get user's lesson plans with optional active filter"""
        query = self._collection.where("user_id", "==", user_id)
        query = query.where("is_active", "==", is_active)
        query = query.order_by("generated_at", direction="DESCENDING")
        query = query.limit(limit).offset(offset)
        
        docs = query.stream()
        plans = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            
            plans.append(LessonPlan.from_dict(data))
        
        return plans
    
    async def get_active_lesson_plan_by_user_id(self, user_id: str) -> Optional[LessonPlan]:
        """Get user's currently active lesson plan"""
        query = self._collection.where("user_id", "==", user_id)
        query = query.where("is_active", "==", True)
        query = query.order_by("generated_at", direction="DESCENDING")
        query = query.limit(1)
        
        docs = query.stream()
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            
            return LessonPlan.from_dict(data)
        
        return None
    
    async def update_lesson_plan(self, lesson_plan: LessonPlan) -> LessonPlan:
        """Update an existing lesson plan"""
        lesson_plan_data = lesson_plan.to_dict()
        
        # Remove the ID from data since it's used as document ID
        doc_id = lesson_plan_data.pop("id")
        
        # Update document in Firestore
        await self._collection.document(doc_id).update(lesson_plan_data)
        
        return lesson_plan
    
    async def delete_lesson_plan(self, plan_id: str) -> bool:
        """Delete a lesson plan"""
        try:
            await self._collection.document(plan_id).delete()
            return True
        except Exception:
            return False
    
    # Lesson CRUD operations
    async def create_lesson(self, lesson: Lesson) -> Lesson:
        """Create a new lesson"""
        # Convert Pydantic model to dict for Firestore
        lesson_data = lesson.to_dict()
        
        # Create document in Firestore
        doc_ref = await self._collection.add(lesson_data)
        
        # Update the lesson with the generated document ID
        lesson_data["id"] = doc_ref[1].id
        
        # Return the created lesson as a Pydantic model
        return Lesson(**lesson_data)
    
    async def get_lesson_by_id(self, lesson_id: str) -> Optional[Lesson]:
        """Get lesson by ID"""
        doc = await self._collection.document(lesson_id).get()
        if not doc.exists:
            return None
        
        data = doc.to_dict()
        if data is None:
            return None
            
        data["id"] = doc.id
        
        
        return Lesson.from_dict(data)
    
    async def get_lessons_by_plan_id(self, plan_id: str) -> List[Lesson]:
        """Get all lessons in a lesson plan"""
        query = self._collection.where("lesson_plan_id", "==", plan_id)
        query = query.order_by("order_in_plan", direction="ASCENDING")
        
        docs = query.stream()
        lessons = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            
            lessons.append(Lesson.from_dict(data))
        
        return lessons
    
    async def get_lessons_by_user_id(
        self, 
        user_id: str,
        completed_only: bool = False,
        limit: int = 50,
        offset: int = 0
    ) -> List[Lesson]:
        """Get user's lessons with optional completion filter"""
        # First get user's lesson plans to find lessons
        plan_query = self._collection.where("user_id", "==", user_id)
        plan_docs = plan_query.stream()
        
        lesson_plan_ids = []
        async for doc in plan_docs:
            lesson_plan_ids.append(doc.id)
        
        if not lesson_plan_ids:
            return []
        
        # Query lessons by lesson plan IDs
        query = self._collection.where("lesson_plan_id", "in", lesson_plan_ids)
        
        if completed_only:
            query = query.where("is_completed", "==", True)
        
        query = query.order_by("created_at", direction="DESCENDING")
        query = query.limit(limit).offset(offset)
        
        docs = query.stream()
        lessons = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            
            lessons.append(Lesson.from_dict(data))
        
        return lessons
    
    async def get_next_lesson_in_plan(self, plan_id: str, current_order: int) -> Optional[Lesson]:
        """Get the next lesson in a plan based on order"""
        query = self._collection.where("lesson_plan_id", "==", plan_id)
        query = query.where("order_in_plan", ">", current_order)
        query = query.order_by("order_in_plan", direction="ASCENDING")
        query = query.limit(1)
        
        docs = query.stream()
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            
            return Lesson.from_dict(data)
        
        return None
    
    async def update_lesson(self, lesson: Lesson) -> Lesson:
        """Update an existing lesson"""
        lesson_data = lesson.to_dict()
        
        # Remove the ID from data since it's used as document ID
        doc_id = lesson_data.pop("id")
        
        # Update document in Firestore
        await self._collection.document(doc_id).update(lesson_data)
        
        return lesson
    
    async def delete_lesson(self, lesson_id: str) -> bool:
        """Delete a lesson"""
        try:
            await self._collection.document(lesson_id).delete()
            return True
        except Exception:
            return False
    
    # Query methods for business logic
    async def count_exercises_by_type(self, exercise_type: ExerciseType) -> int:
        """Count exercises by type"""
        query = self._collection.where("exercise_type", "==", exercise_type.value)
        query = query.where("is_active", "==", True)
        docs = query.stream()
        
        count = 0
        async for _ in docs:
            count += 1
        
        return count
    
    async def get_recommended_exercises_for_skill_level(
        self, 
        skill_level: DifficultyLevel,
        limit: int = 10
    ) -> List[Exercise]:
        """Get recommended exercises for skill level"""
        query = self._collection.where("difficulty_level", "==", skill_level.value)
        query = query.where("is_active", "==", True)
        query = query.order_by("created_at", direction="DESCENDING")
        query = query.limit(limit)
        
        docs = query.stream()
        exercises = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            
            exercises.append(Exercise.from_dict(data))
        
        return exercises
    
    async def get_lesson_completion_rate_by_user_id(self, user_id: str) -> float:
        """Get lesson completion rate for user"""
        # Get user's lesson plans
        plan_query = self._collection.where("user_id", "==", user_id)
        plan_docs = plan_query.stream()
        
        lesson_plan_ids = []
        async for doc in plan_docs:
            lesson_plan_ids.append(doc.id)
        
        if not lesson_plan_ids:
            return 0.0
        
        # Get all lessons for user's plans
        lesson_query = self._collection.where("lesson_plan_id", "in", lesson_plan_ids)
        lesson_docs = lesson_query.stream()
        
        total_lessons = 0
        completed_lessons = 0
        
        async for doc in lesson_docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            total_lessons += 1
            if data.get("is_completed", False):
                completed_lessons += 1
        
        if total_lessons == 0:
            return 0.0
        
        return completed_lessons / total_lessons
    
    async def count_completed_lessons_by_plan_id(self, plan_id: str) -> int:
        """Count completed lessons in a plan"""
        query = self._collection.where("lesson_plan_id", "==", plan_id)
        query = query.where("is_completed", "==", True)
        docs = query.stream()
        
        count = 0
        async for _ in docs:
            count += 1
        
        return count
    
    async def search_exercises(self, keyword: str) -> List[Exercise]:
        """Search exercises by keyword in title and description"""
        if not keyword:
            return []
        
        # Since Firestore doesn't support full-text search, we'll get all active exercises
        # and filter them in memory. For production, consider using Algolia or Elasticsearch.
        query = self._collection.where("is_active", "==", True)
        docs = query.stream()
        exercises = []
        
        keyword_lower = keyword.lower()
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            # Check if keyword is in title or description
            title = data.get("title", "").lower()
            description = data.get("description", "").lower()
            
            if keyword_lower in title or keyword_lower in description:
                data["id"] = doc.id
                
                
                exercises.append(Exercise.from_dict(data))
        
        # Sort by relevance (title matches first, then by creation date)
        def sort_key(exercise):
            title_match = keyword_lower in exercise.title.lower()
            return (not title_match, -exercise.created_at.timestamp())
        
        exercises.sort(key=sort_key)
        return exercises
    
    async def get_active_exercises(self, limit: int = 50, offset: int = 0) -> List[Exercise]:
        """Get all active exercises"""
        query = self._collection.where("is_active", "==", True)
        query = query.order_by("created_at", direction="DESCENDING")
        query = query.limit(limit).offset(offset)
        
        docs = query.stream()
        exercises = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            
            exercises.append(Exercise.from_dict(data))
        
        return exercises