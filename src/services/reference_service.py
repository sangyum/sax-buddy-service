from typing import List, Optional
from src.models.reference import ReferencePerformance, SkillLevelDefinition, SkillLevel


class ReferenceService:
    """Service class for reference data operations"""
    
    async def get_skill_level_definitions(self) -> List[SkillLevelDefinition]:
        """Get skill level definitions"""
        # TODO: Implement skill level definitions retrieval logic
        raise NotImplementedError("Skill level definitions retrieval not implemented yet")
    
    async def get_reference_performances(
        self, 
        exercise_id: Optional[str] = None,
        skill_level: Optional[SkillLevel] = None
    ) -> List[ReferencePerformance]:
        """Get reference performances"""
        # TODO: Implement reference performance retrieval logic
        raise NotImplementedError("Reference performance retrieval not implemented yet")