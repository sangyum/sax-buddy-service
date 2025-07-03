from typing import List, Optional
from src.models.reference import ReferencePerformance, SkillLevelDefinition, SkillLevel
from src.repositories.reference_repository import ReferenceRepository


class ReferenceService:
    """Service class for reference data operations"""
    
    def __init__(self, reference_repository: ReferenceRepository):
        self.reference_repository = reference_repository
    
    async def get_skill_level_definitions(self) -> List[SkillLevelDefinition]:
        """Get skill level definitions"""
        return await self.reference_repository.get_skill_level_definitions()
    
    async def get_reference_performances(
        self, 
        exercise_id: Optional[str] = None,
        skill_level: Optional[SkillLevel] = None
    ) -> List[ReferencePerformance]:
        """Get reference performances"""
        return await self.reference_repository.get_reference_performances(exercise_id, skill_level)