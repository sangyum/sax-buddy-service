from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Query, Depends
from src.models.reference import ReferencePerformance, SkillLevelDefinition, SkillLevel
from src.services.reference_service import ReferenceService

router = APIRouter(prefix="/reference", tags=["Reference"])


def get_reference_service() -> ReferenceService:
    """Dependency to get ReferenceService instance"""
    return ReferenceService()


@router.get("/skill-levels", response_model=List[SkillLevelDefinition])
async def get_skill_level_definitions(
    reference_service: ReferenceService = Depends(get_reference_service)
):
    """Get skill level definitions"""
    try:
        return await reference_service.get_skill_level_definitions()
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Skill level definitions retrieval not implemented yet"
        )


@router.get("/performances", response_model=List[ReferencePerformance])
async def get_reference_performances(
    exercise_id: Optional[str] = Query(None),
    skill_level: Optional[SkillLevel] = Query(None),
    reference_service: ReferenceService = Depends(get_reference_service)
):
    """Get reference performances"""
    try:
        return await reference_service.get_reference_performances(exercise_id, skill_level)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Reference performance retrieval not implemented yet"
        )