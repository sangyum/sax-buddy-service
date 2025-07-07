from typing import List
from fastapi import APIRouter, HTTPException, status, Query, Depends
from google.cloud.firestore_v1.async_client import AsyncClient
from src.models.assessment import FormalAssessment, Feedback, SkillMetrics
from src.api.schemas.requests import AssessmentTrigger
from src.services.assessment_service import AssessmentService
from src.repositories.assessment_repository import AssessmentRepository
from src.dependencies import get_firestore_client
from src.auth import AuthenticatedUser, require_auth

router = APIRouter(tags=["Assessment"])


def get_assessment_repository(
    firestore_client: AsyncClient = Depends(get_firestore_client)
) -> AssessmentRepository:
    """Dependency to get AssessmentRepository instance with Firestore client"""
    return AssessmentRepository(firestore_client)


def get_assessment_service(
    assessment_repository: AssessmentRepository = Depends(get_assessment_repository)
) -> AssessmentService:
    """Dependency to get AssessmentService instance with repository injected"""
    return AssessmentService(assessment_repository)


@router.get("/users/{user_id}/assessments", response_model=List[FormalAssessment])
async def get_user_assessments(
    user_id: str,
    limit: int = Query(10, ge=1, le=100),
    assessment_service: AssessmentService = Depends(get_assessment_service),
    current_user: AuthenticatedUser = Depends(require_auth)
):
    """Get user's formal assessments"""
    try:
        return await assessment_service.get_user_assessments(user_id, limit)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Assessment retrieval not implemented yet"
        )


@router.post("/users/{user_id}/assessments", response_model=FormalAssessment, status_code=status.HTTP_201_CREATED)
async def trigger_assessment(
    user_id: str, 
    trigger_data: AssessmentTrigger,
    assessment_service: AssessmentService = Depends(get_assessment_service),
    current_user: AuthenticatedUser = Depends(require_auth)
):
    """Trigger formal assessment"""
    try:
        return await assessment_service.trigger_assessment(trigger_data)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Assessment trigger not implemented yet"
        )


@router.get("/performance/sessions/{session_id}/feedback", response_model=Feedback)
async def get_session_feedback(
    session_id: str,
    assessment_service: AssessmentService = Depends(get_assessment_service),
    current_user: AuthenticatedUser = Depends(require_auth)
):
    """Get session feedback"""
    try:
        feedback = await assessment_service.get_session_feedback(session_id)
        if not feedback:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session feedback not found"
            )
        return feedback
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Feedback retrieval not implemented yet"
        )


@router.get("/users/{user_id}/skill-metrics", response_model=SkillMetrics)
async def get_user_skill_metrics(
    user_id: str,
    period_days: int = Query(30, ge=1),
    assessment_service: AssessmentService = Depends(get_assessment_service),
    current_user: AuthenticatedUser = Depends(require_auth)
):
    """Get user's current skill metrics"""
    try:
        metrics = await assessment_service.get_user_skill_metrics(user_id, period_days)
        if not metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User skill metrics not found"
            )
        return metrics
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Skill metrics calculation not implemented yet"
        )