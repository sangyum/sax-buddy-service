from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Query, Depends
from google.cloud.firestore_v1.async_client import AsyncClient
from src.models.content import Exercise, LessonPlan, Lesson, ExerciseType, DifficultyLevel
from src.api.schemas.requests import LessonUpdate
from src.services.content_service import ContentService
from src.repositories.content_repository import ContentRepository
from src.dependencies import get_firestore_client
from src.auth import AuthenticatedUser, require_auth

router = APIRouter(tags=["Content"])


def get_content_repository(
    firestore_client: AsyncClient = Depends(get_firestore_client)
) -> ContentRepository:
    """Dependency to get ContentRepository instance with Firestore client"""
    return ContentRepository(firestore_client)


def get_content_service(
    content_repository: ContentRepository = Depends(get_content_repository)
) -> ContentService:
    """Dependency to get ContentService instance with repository injected"""
    return ContentService(content_repository)


@router.get("/exercises", response_model=List[Exercise])
async def list_exercises(
    exercise_type: Optional[ExerciseType] = Query(None),
    difficulty_level: Optional[DifficultyLevel] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    content_service: ContentService = Depends(get_content_service),
    current_user: AuthenticatedUser = Depends(require_auth)
):
    """List exercises"""
    try:
        return await content_service.list_exercises(exercise_type, difficulty_level, limit)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Exercise listing not implemented yet"
        )


@router.get("/exercises/{exercise_id}", response_model=Exercise)
async def get_exercise(
    exercise_id: str,
    content_service: ContentService = Depends(get_content_service),
    current_user: AuthenticatedUser = Depends(require_auth)
):
    """Get exercise details"""
    try:
        exercise = await content_service.get_exercise_by_id(exercise_id)
        if not exercise:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Exercise not found"
            )
        return exercise
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Exercise retrieval not implemented yet"
        )


@router.get("/users/{user_id}/lesson-plans", response_model=List[LessonPlan])
async def get_user_lesson_plans(
    user_id: str,
    is_active: bool = Query(True),
    content_service: ContentService = Depends(get_content_service),
    current_user: AuthenticatedUser = Depends(require_auth)
):
    """Get user's lesson plans"""
    try:
        return await content_service.get_user_lesson_plans(user_id, is_active)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Lesson plan retrieval not implemented yet"
        )


@router.post("/users/{user_id}/lesson-plans", response_model=LessonPlan, status_code=status.HTTP_201_CREATED)
async def generate_lesson_plan(
    user_id: str,
    content_service: ContentService = Depends(get_content_service),
    current_user: AuthenticatedUser = Depends(require_auth)
):
    """Generate new lesson plan for user"""
    try:
        return await content_service.generate_lesson_plan(user_id)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Lesson plan generation not implemented yet"
        )


@router.get("/lesson-plans/{plan_id}/lessons", response_model=List[Lesson])
async def get_lessons_in_plan(
    plan_id: str,
    content_service: ContentService = Depends(get_content_service),
    current_user: AuthenticatedUser = Depends(require_auth)
):
    """Get lessons in plan"""
    try:
        return await content_service.get_lessons_in_plan(plan_id)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Lesson retrieval not implemented yet"
        )


@router.get("/lessons/{lesson_id}", response_model=Lesson)
async def get_lesson(
    lesson_id: str,
    content_service: ContentService = Depends(get_content_service),
    current_user: AuthenticatedUser = Depends(require_auth)
):
    """Get lesson details"""
    try:
        lesson = await content_service.get_lesson_by_id(lesson_id)
        if not lesson:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Lesson not found"
            )
        return lesson
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Lesson retrieval not implemented yet"
        )


@router.patch("/lessons/{lesson_id}", response_model=Lesson)
async def update_lesson(
    lesson_id: str, 
    lesson_update: LessonUpdate,
    content_service: ContentService = Depends(get_content_service),
    current_user: AuthenticatedUser = Depends(require_auth)
):
    """Update lesson completion status"""
    try:
        lesson = await content_service.update_lesson(lesson_id, lesson_update)
        if not lesson:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Lesson not found"
            )
        return lesson
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Lesson update not implemented yet"
        )