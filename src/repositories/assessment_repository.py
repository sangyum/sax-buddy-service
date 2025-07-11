from typing import List, Optional
from datetime import datetime
from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1.async_collection import AsyncCollectionReference
from src.models.assessment import FormalAssessment, Feedback, SkillMetrics


class AssessmentRepository:
    """Repository for assessment-related data operations"""
    
    _collection: AsyncCollectionReference
    
    def __init__(self, firestore_client: AsyncClient):
        """Initialize AssessmentRepository with Firestore client"""
        self._collection = firestore_client.collection("assessments")
    
    # FormalAssessment CRUD operations
    async def create_assessment(self, assessment: FormalAssessment) -> FormalAssessment:
        """Create a new formal assessment"""
        # Convert Pydantic model to dict for Firestore
        assessment_data = assessment.to_dict()
        
        # Create document in Firestore
        doc_ref = await self._collection.add(assessment_data)
        
        # Update the assessment with the generated document ID
        assessment_data["id"] = doc_ref[1].id
        
        # Return the created assessment as a Pydantic model
        return FormalAssessment.from_dict(assessment_data)
    
    async def get_assessment_by_id(self, assessment_id: str) -> Optional[FormalAssessment]:
        """Get formal assessment by ID"""
        doc = await self._collection.document(assessment_id).get()
        if not doc.exists:
            return None
        
        data = doc.to_dict()
        if data is None:
            return None
            
        data["id"] = doc.id
        
        return FormalAssessment.from_dict(data)
    
    async def get_assessments_by_user_id(
        self, 
        user_id: str, 
        limit: int = 10,
        offset: int = 0
    ) -> List[FormalAssessment]:
        """Get user's formal assessments with pagination"""
        query = self._collection.where(field_path="user_id", op_string="==", value=user_id)
        query = query.order_by("assessed_at", direction="DESCENDING")
        query = query.limit(limit).offset(offset)
        
        docs = query.stream()
        assessments = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            assessments.append(FormalAssessment.from_dict(data))
        
        return assessments
    
    async def get_latest_assessment_by_user_id(self, user_id: str) -> Optional[FormalAssessment]:
        """Get user's most recent formal assessment"""
        query = self._collection.where(field_path="user_id", op_string="==", value=user_id)
        query = query.order_by("assessed_at", direction="DESCENDING")
        query = query.limit(1)
        
        docs = query.stream()
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            return FormalAssessment.from_dict(data)
        
        return None
    
    async def update_assessment(self, assessment: FormalAssessment) -> FormalAssessment:
        """Update an existing formal assessment"""
        assessment_data = assessment.to_dict()
        
        # Remove the ID from data since it's used as document ID
        doc_id = assessment_data.pop("id")
        
        # Update document in Firestore
        await self._collection.document(doc_id).update(assessment_data)
        
        return assessment
    
    async def delete_assessment(self, assessment_id: str) -> bool:
        """Delete a formal assessment"""
        try:
            await self._collection.document(assessment_id).delete()
            return True
        except Exception:
            return False
    
    # Feedback CRUD operations
    async def create_feedback(self, feedback: Feedback) -> Feedback:
        """Create new session feedback"""
        # Convert Pydantic model to dict for Firestore
        feedback_data = feedback.to_dict()
        
        # Create document in Firestore
        doc_ref = await self._collection.add(feedback_data)
        
        # Update the feedback with the generated document ID
        feedback_data["id"] = doc_ref[1].id
        
        # Return the created feedback as a Pydantic model
        return Feedback.from_dict(feedback_data)
    
    async def get_feedback_by_id(self, feedback_id: str) -> Optional[Feedback]:
        """Get feedback by ID"""
        doc = await self._collection.document(feedback_id).get()
        if not doc.exists:
            return None
        
        data = doc.to_dict()
        if data is None:
            return None
            
        data["id"] = doc.id
        
        return Feedback.from_dict(data)
    
    
    
    async def get_feedback_by_session_id(self, session_id: str) -> Optional[Feedback]:
        """Get feedback for a specific session"""
        query = self._collection.where(field_path="session_id", op_string="==", value=session_id)
        query = query.limit(1)
        
        docs = query.stream()
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            return Feedback.from_dict(data)
        
        return None
    
    async def get_feedback_by_user_id(
        self, 
        user_id: str, 
        limit: int = 10,
        offset: int = 0
    ) -> List[Feedback]:
        """Get user's feedback history with pagination"""
        query = self._collection.where(field_path="user_id", op_string="==", value=user_id)
        query = query.order_by("created_at", direction="DESCENDING")
        query = query.limit(limit).offset(offset)
        
        docs = query.stream()
        feedback_list = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            feedback_list.append(Feedback.from_dict(data))
        
        return feedback_list
    
    async def update_feedback(self, feedback: Feedback) -> Feedback:
        """Update existing feedback"""
        feedback_data = feedback.to_dict()
        
        # Remove the ID from data since it's used as document ID
        doc_id = feedback_data.pop("id")
        
        # Update document in Firestore
        await self._collection.document(doc_id).update(feedback_data)
        
        return feedback
    
    async def delete_feedback(self, feedback_id: str) -> bool:
        """Delete feedback"""
        try:
            await self._collection.document(feedback_id).delete()
            return True
        except Exception:
            return False
    
    # SkillMetrics CRUD operations
    async def create_skill_metrics(self, metrics: SkillMetrics) -> SkillMetrics:
        """Create new skill metrics record"""
        # Convert Pydantic model to dict for Firestore
        metrics_data = metrics.to_dict()
        
        # Create document in Firestore
        doc_ref = await self._collection.add(metrics_data)
        
        # Update the metrics with the generated document ID
        metrics_data["id"] = doc_ref[1].id
        
        # Return the created metrics as a Pydantic model
        return SkillMetrics.from_dict(metrics_data)
    
    async def get_skill_metrics_by_id(self, metrics_id: str) -> Optional[SkillMetrics]:
        """Get skill metrics by ID"""
        doc = await self._collection.document(metrics_id).get()
        if not doc.exists:
            return None
        
        data = doc.to_dict()
        if data is None:
            return None
            
        data["id"] = doc.id
        
        return SkillMetrics.from_dict(data)
    
    async def get_latest_skill_metrics_by_user_id(self, user_id: str) -> Optional[SkillMetrics]:
        """Get user's most recent skill metrics"""
        query = self._collection.where(field_path="user_id", op_string="==", value=user_id)
        query = query.order_by("calculated_at", direction="DESCENDING")
        query = query.limit(1)
        
        docs = query.stream()
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            return SkillMetrics.from_dict(data)
        
        return None
    
    async def get_skill_metrics_by_user_id_and_period(
        self, 
        user_id: str, 
        start_date: datetime,
        end_date: datetime
    ) -> List[SkillMetrics]:
        """Get user's skill metrics within a date range"""
        query = self._collection.where(field_path="user_id", op_string="==", value=user_id)
        query = query.where(field_path="calculated_at", op_string=">=", value=start_date.isoformat())
        query = query.where(field_path="calculated_at", op_string="<=", value=end_date.isoformat())
        query = query.order_by("calculated_at", direction="ASCENDING")
        
        docs = query.stream()
        metrics_list = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            metrics_list.append(SkillMetrics.from_dict(data))
        
        return metrics_list
    
    async def get_skill_metrics_by_user_id(
        self, 
        user_id: str, 
        limit: int = 10,
        offset: int = 0
    ) -> List[SkillMetrics]:
        """Get user's skill metrics history with pagination"""
        query = self._collection.where(field_path="user_id", op_string="==", value=user_id)
        query = query.order_by("calculated_at", direction="DESCENDING")
        query = query.limit(limit).offset(offset)
        
        docs = query.stream()
        metrics_list = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            metrics_list.append(SkillMetrics.from_dict(data))
        
        return metrics_list
    
    async def update_skill_metrics(self, metrics: SkillMetrics) -> SkillMetrics:
        """Update existing skill metrics"""
        metrics_data = metrics.to_dict()
        
        # Remove the ID from data since it's used as document ID
        doc_id = metrics_data.pop("id")
        
        # Update document in Firestore
        await self._collection.document(doc_id).update(metrics_data)
        
        return metrics
    
    async def delete_skill_metrics(self, metrics_id: str) -> bool:
        """Delete skill metrics record"""
        try:
            await self._collection.document(metrics_id).delete()
            return True
        except Exception:
            return False
    
    # Query methods for business logic
    async def count_assessments_by_user_id(self, user_id: str) -> int:
        """Count total assessments for a user"""
        query = self._collection.where(field_path="user_id", op_string="==", value=user_id)
        docs = query.stream()
        
        count = 0
        async for _ in docs:
            count += 1
        
        return count
    
    async def get_assessment_trend_by_user_id(
        self, 
        user_id: str, 
        days: int = 30
    ) -> List[FormalAssessment]:
        """Get assessment trend over specified days"""
        from datetime import timedelta, timezone
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        query = self._collection.where(field_path="user_id", op_string="==", value=user_id)
        query = query.where(field_path="assessed_at", op_string=">=", value=start_date.isoformat())
        query = query.where(field_path="assessed_at", op_string="<=", value=end_date.isoformat())
        query = query.order_by("assessed_at", direction="ASCENDING")
        
        docs = query.stream()
        assessments = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            assessments.append(FormalAssessment.from_dict(data))
        
        return assessments
    
    async def get_skill_progression_by_user_id(
        self, 
        user_id: str, 
        skill_dimension: str,
        days: int = 90
    ) -> List[SkillMetrics]:
        """Get skill progression for specific dimension over time"""
        from datetime import timedelta, timezone
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        query = self._collection.where(field_path="user_id", op_string="==", value=user_id)
        query = query.where(field_path="calculated_at", op_string=">=", value=start_date.isoformat())
        query = query.where(field_path="calculated_at", op_string="<=", value=end_date.isoformat())
        query = query.order_by("calculated_at", direction="ASCENDING")
        
        docs = query.stream()
        metrics_list = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            # Filter by skill dimension if the metrics contain it
            skill_metrics = SkillMetrics.from_dict(data)
            # Check if the skill dimension exists as a field in the metrics
            if hasattr(skill_metrics, f'{skill_dimension}_score'):
                metrics_list.append(skill_metrics)
            else:
                # Include all metrics if dimension not found (for backward compatibility)
                metrics_list.append(skill_metrics)
        
        return metrics_list