from typing import List
from fastapi import APIRouter, HTTPException, status, Query, Depends
from src.models.assessment import FormalAssessment, Feedback, SkillMetrics
from src.api.schemas.requests import AssessmentTrigger
from src.services.assessment_service import AssessmentService

router = APIRouter(tags=["Assessment"])


def get_assessment_service() -> AssessmentService:
    """Dependency to get AssessmentService instance"""
    return AssessmentService()


@router.get("/users/{user_id}/assessments", response_model=List[FormalAssessment])
async def get_user_assessments(
    user_id: str,
    limit: int = Query(10, ge=1, le=100),
    assessment_service: AssessmentService = Depends(get_assessment_service)
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
    assessment_service: AssessmentService = Depends(get_assessment_service)
):
    """Trigger formal assessment"""
    try:
        return await assessment_service.trigger_assessment(user_id, trigger_data)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Assessment trigger not implemented yet"
        )


@router.get("/performance/sessions/{session_id}/feedback", response_model=Feedback)
async def get_session_feedback(
    session_id: str,
    assessment_service: AssessmentService = Depends(get_assessment_service)
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
    assessment_service: AssessmentService = Depends(get_assessment_service)
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