from typing import List, Optional
from src.models.assessment import FormalAssessment, Feedback, SkillMetrics
from src.api.schemas.requests import AssessmentTrigger
from src.repositories.assessment_repository import AssessmentRepository


class AssessmentService:
    """Service class for assessment-related operations"""
    
    def __init__(self, assessment_repository: AssessmentRepository):
        self.assessment_repository = assessment_repository
    
    async def get_user_assessments(self, user_id: str, limit: int = 10) -> List[FormalAssessment]:
        """Get user's formal assessments"""
        return await self.assessment_repository.get_assessments_by_user_id(user_id, limit)
    
    async def trigger_assessment(self, trigger_data: AssessmentTrigger) -> FormalAssessment:
        """Trigger formal assessment"""
        # TODO: Implement assessment generation logic using AI/ML
        # This would create a FormalAssessment based on recent performance data
        # For now, delegate to repository once assessment is created
        raise NotImplementedError("Assessment trigger logic not implemented yet")
    
    
    
    async def get_session_feedback(self, session_id: str) -> Optional[Feedback]:
        """Get session feedback"""
        return await self.assessment_repository.get_feedback_by_session_id(session_id)
    
    async def get_user_skill_metrics(self, user_id: str, period_days: int = 30) -> Optional[SkillMetrics]:
        """Get user's current skill metrics"""
        return await self.assessment_repository.get_latest_skill_metrics_by_user_id(user_id)