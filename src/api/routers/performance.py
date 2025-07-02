from typing import List
from fastapi import APIRouter, HTTPException, status, Depends
from src.models.performance import PerformanceSession, PerformanceMetrics
from src.api.schemas.requests import PerformanceSessionCreate, PerformanceSessionUpdate, PerformanceMetricsCreate
from src.services.performance_service import PerformanceService

router = APIRouter(prefix="/performance", tags=["Performance"])


def get_performance_service() -> PerformanceService:
    """Dependency to get PerformanceService instance"""
    return PerformanceService()


@router.post("/sessions", response_model=PerformanceSession, status_code=status.HTTP_201_CREATED)
async def create_session(
    session_data: PerformanceSessionCreate,
    performance_service: PerformanceService = Depends(get_performance_service)
):
    """Start new practice session"""
    try:
        return await performance_service.create_session(session_data)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Session creation not implemented yet"
        )


@router.get("/sessions/{session_id}", response_model=PerformanceSession)
async def get_session(
    session_id: str,
    performance_service: PerformanceService = Depends(get_performance_service)
):
    """Get session details"""
    try:
        session = await performance_service.get_session_by_id(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        return session
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Session retrieval not implemented yet"
        )


@router.patch("/sessions/{session_id}", response_model=PerformanceSession)
async def update_session(
    session_id: str, 
    session_update: PerformanceSessionUpdate,
    performance_service: PerformanceService = Depends(get_performance_service)
):
    """Update session (e.g., end session)"""
    try:
        session = await performance_service.update_session(session_id, session_update)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        return session
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Session update not implemented yet"
        )


@router.post("/sessions/{session_id}/metrics", response_model=List[PerformanceMetrics], status_code=status.HTTP_201_CREATED)
async def submit_metrics(
    session_id: str, 
    metrics_list: List[PerformanceMetricsCreate],
    performance_service: PerformanceService = Depends(get_performance_service)
):
    """Submit performance metrics from mobile DSP"""
    try:
        return await performance_service.submit_metrics(session_id, metrics_list)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Metrics submission not implemented yet"
        )


@router.get("/sessions/{session_id}/metrics", response_model=List[PerformanceMetrics])
async def get_session_metrics(
    session_id: str,
    performance_service: PerformanceService = Depends(get_performance_service)
):
    """Get session metrics"""
    try:
        return await performance_service.get_session_metrics(session_id)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Session metrics retrieval not implemented yet"
        )