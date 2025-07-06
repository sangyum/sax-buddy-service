from typing import List, Optional
from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1.async_collection import AsyncCollectionReference
from src.models.reference import ReferencePerformance, SkillLevelDefinition, SkillLevel


class ReferenceRepository:
    """Repository for reference data operations"""
    
    _collection: AsyncCollectionReference
    
    def __init__(self, firestore_client: AsyncClient):
        """Initialize ReferenceRepository with Firestore client"""
        self._collection = firestore_client.collection("reference")
    
    # SkillLevelDefinition CRUD operations
    async def create_skill_level_definition(self, definition: SkillLevelDefinition) -> SkillLevelDefinition:
        """Create a new skill level definition"""
        # Convert Pydantic model to dict for Firestore
        definition_data = definition.to_dict()
        
        # Create document in Firestore
        doc_ref = await self._collection.add(definition_data)
        
        # Update the definition with the generated document ID
        definition_data["id"] = doc_ref[1].id
        
        # Return the created definition as a Pydantic model
        return SkillLevelDefinition.from_dict(definition_data)
    
    async def get_skill_level_definition_by_id(self, definition_id: str) -> Optional[SkillLevelDefinition]:
        """Get skill level definition by ID"""
        doc = await self._collection.document(definition_id).get()
        if not doc.exists:
            return None
        
        data = doc.to_dict()
        if data is None:
            return None
            
        data["id"] = doc.id
        
        return SkillLevelDefinition.from_dict(data)
    
    async def get_skill_level_definition_by_level(self, skill_level: SkillLevel) -> Optional[SkillLevelDefinition]:
        """Get skill level definition by skill level"""
        query = self._collection.where("skill_level", "==", skill_level.value)
        query = query.limit(1)
        
        docs = query.stream()
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            return SkillLevelDefinition.from_dict(data)
        
        return None
    
    async def list_skill_level_definitions(self) -> List[SkillLevelDefinition]:
        """Get all skill level definitions"""
        query = self._collection.order_by("created_at", direction="ASCENDING")
        
        docs = query.stream()
        definitions = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            definitions.append(SkillLevelDefinition.from_dict(data))
        
        return definitions
    
    async def update_skill_level_definition(self, definition: SkillLevelDefinition) -> SkillLevelDefinition:
        """Update an existing skill level definition"""
        definition_data = definition.to_dict()
        
        # Remove the ID from data since it's used as document ID
        doc_id = definition_data.pop("id")
        
        # Update document in Firestore
        await self._collection.document(doc_id).update(definition_data)
        
        return definition
    
    async def delete_skill_level_definition(self, definition_id: str) -> bool:
        """Delete a skill level definition"""
        try:
            await self._collection.document(definition_id).delete()
            return True
        except Exception:
            return False
    
    # ReferencePerformance CRUD operations
    async def create_reference_performance(self, performance: ReferencePerformance) -> ReferencePerformance:
        """Create a new reference performance"""
        # Convert Pydantic model to dict for Firestore
        performance_data = performance.to_dict()
        
        # Create document in Firestore
        doc_ref = await self._collection.add(performance_data)
        
        # Update the performance with the generated document ID
        performance_data["id"] = doc_ref[1].id
        
        # Return the created performance as a Pydantic model
        return ReferencePerformance.from_dict(performance_data)
    
    async def get_reference_performance_by_id(self, performance_id: str) -> Optional[ReferencePerformance]:
        """Get reference performance by ID"""
        doc = await self._collection.document(performance_id).get()
        if not doc.exists:
            return None
        
        data = doc.to_dict()
        if data is None:
            return None
            
        data["id"] = doc.id
        
        return ReferencePerformance.from_dict(data)
    
    async def get_reference_performances_by_exercise_id(self, exercise_id: str) -> List[ReferencePerformance]:
        """Get reference performances for specific exercise"""
        query = self._collection.where("exercise_id", "==", exercise_id)
        query = query.where("is_active", "==", True)
        query = query.order_by("created_at", direction="DESCENDING")
        
        docs = query.stream()
        performances = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            performances.append(ReferencePerformance.from_dict(data))
        
        return performances
    
    async def get_reference_performances_by_skill_level(self, skill_level: SkillLevel) -> List[ReferencePerformance]:
        """Get reference performances for specific skill level"""
        query = self._collection.where("skill_level", "==", skill_level.value)
        query = query.where("is_active", "==", True)
        query = query.order_by("created_at", direction="DESCENDING")
        
        docs = query.stream()
        performances = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            performances.append(ReferencePerformance.from_dict(data))
        
        return performances
    
    async def get_reference_performances_by_exercise_and_skill_level(
        self, 
        exercise_id: str, 
        skill_level: SkillLevel
    ) -> List[ReferencePerformance]:
        """Get reference performances for specific exercise and skill level"""
        query = self._collection.where("exercise_id", "==", exercise_id)
        query = query.where("skill_level", "==", skill_level.value)
        query = query.where("is_active", "==", True)
        query = query.order_by("created_at", direction="DESCENDING")
        
        docs = query.stream()
        performances = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            performances.append(ReferencePerformance.from_dict(data))
        
        return performances
    
    async def list_reference_performances(
        self, 
        exercise_id: Optional[str] = None,
        skill_level: Optional[SkillLevel] = None,
        is_active: bool = True,
        limit: int = 50,
        offset: int = 0
    ) -> List[ReferencePerformance]:
        """List reference performances with optional filtering"""
        query = self._collection.where("is_active", "==", is_active)
        
        if exercise_id:
            query = query.where("exercise_id", "==", exercise_id)
        if skill_level:
            query = query.where("skill_level", "==", skill_level.value)
        
        query = query.order_by("created_at", direction="DESCENDING")
        query = query.limit(limit).offset(offset)
        
        docs = query.stream()
        performances = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            performances.append(ReferencePerformance.from_dict(data))
        
        return performances
    
    async def update_reference_performance(self, performance: ReferencePerformance) -> ReferencePerformance:
        """Update an existing reference performance"""
        performance_data = performance.to_dict()
        
        # Remove the ID from data since it's used as document ID
        doc_id = performance_data.pop("id")
        
        # Update document in Firestore
        await self._collection.document(doc_id).update(performance_data)
        
        return performance
    
    async def delete_reference_performance(self, performance_id: str) -> bool:
        """Delete a reference performance"""
        try:
            await self._collection.document(performance_id).delete()
            return True
        except Exception:
            return False
    
    # Query methods for business logic
    async def get_skill_thresholds_by_level(self, skill_level: SkillLevel) -> dict:
        """Get score thresholds for specific skill level"""
        definition = await self.get_skill_level_definition_by_level(skill_level)
        if definition:
            return definition.score_thresholds
        return {}
    
    async def get_next_skill_level(self, current_level: SkillLevel) -> Optional[SkillLevel]:
        """Get the next skill level in progression"""
        # Define skill level progression order
        level_progression = {
            SkillLevel.BEGINNER: SkillLevel.INTERMEDIATE,
            SkillLevel.INTERMEDIATE: SkillLevel.ADVANCED,
            SkillLevel.ADVANCED: None  # No next level after advanced
        }
        
        return level_progression.get(current_level)
    
    async def get_recommended_exercises_for_skill_level(self, skill_level: SkillLevel) -> List[str]:
        """Get recommended exercise types for skill level"""
        definition = await self.get_skill_level_definition_by_level(skill_level)
        if definition:
            return definition.typical_exercises
        return []
    
    async def count_reference_performances_by_exercise(self, exercise_id: str) -> int:
        """Count reference performances for an exercise"""
        query = self._collection.where("exercise_id", "==", exercise_id)
        query = query.where("is_active", "==", True)
        docs = query.stream()
        
        count = 0
        async for _ in docs:
            count += 1
        
        return count