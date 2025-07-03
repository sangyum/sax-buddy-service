from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Query, Depends
from google.cloud.firestore_v1.async_client import AsyncClient
from src.models.reference import ReferencePerformance, SkillLevelDefinition, SkillLevel
from src.services.reference_service import ReferenceService
from src.repositories.reference_repository import ReferenceRepository
from src.dependencies import get_firestore_client
from src.auth import AuthenticatedUser, require_auth

router = APIRouter(prefix="/reference", tags=["Reference"])


def get_reference_repository(
    firestore_client: AsyncClient = Depends(get_firestore_client)
) -> ReferenceRepository:
    """Dependency to get ReferenceRepository instance with Firestore client"""
    return ReferenceRepository(firestore_client)


def get_reference_service(
    reference_repository: ReferenceRepository = Depends(get_reference_repository)
) -> ReferenceService:
    """Dependency to get ReferenceService instance with repository injected"""
    return ReferenceService(reference_repository)


@router.get("/skill-levels", response_model=List[SkillLevelDefinition])
async def get_skill_level_definitions(
    reference_service: ReferenceService = Depends(get_reference_service),
    current_user: AuthenticatedUser = Depends(require_auth)
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
    reference_service: ReferenceService = Depends(get_reference_service),
    current_user: AuthenticatedUser = Depends(require_auth)
):
    """Get reference performances"""
    try:
        return await reference_service.get_reference_performances(exercise_id, skill_level)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Reference performance retrieval not implemented yet"
        )