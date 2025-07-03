from typing import List, Optional
from src.models.performance import PerformanceSession, PerformanceMetrics
from src.api.schemas.requests import PerformanceSessionCreate, PerformanceSessionUpdate, PerformanceMetricsCreate
from src.repositories.performance_repository import PerformanceRepository


class PerformanceService:
    """Service class for performance-related operations"""
    
    def __init__(self, performance_repository: PerformanceRepository):
        self.performance_repository = performance_repository
    
    async def create_session(self, session_data: PerformanceSessionCreate) -> PerformanceSession:
        """Start new practice session"""
        # TODO: Convert PerformanceSessionCreate to PerformanceSession model
        # For now, delegate to repository once session model is created
        raise NotImplementedError("Session creation logic not implemented yet")
    
    async def get_session_by_id(self, session_id: str) -> Optional[PerformanceSession]:
        """Get session details"""
        return await self.performance_repository.get_session_by_id(session_id)
    
    async def update_session(self, session_id: str, session_update: PerformanceSessionUpdate) -> Optional[PerformanceSession]:
        """Update session (e.g., end session)"""
        # TODO: Get existing session, apply updates, and save
        # For now, delegate to repository once update logic is implemented
        raise NotImplementedError("Session update logic not implemented yet")
    
    async def submit_metrics(self, session_id: str, metrics_list: List[PerformanceMetricsCreate]) -> List[PerformanceMetrics]:
        """Submit performance metrics from mobile DSP"""
        # TODO: Convert PerformanceMetricsCreate to PerformanceMetrics models
        # For now, delegate to repository once metrics models are created
        raise NotImplementedError("Metrics submission logic not implemented yet")
    
    async def get_session_metrics(self, session_id: str) -> List[PerformanceMetrics]:
        """Get session metrics"""
        return await self.performance_repository.get_metrics_by_session_id(session_id)