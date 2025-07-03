from typing import List, Optional
from datetime import datetime
from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1.async_collection import AsyncCollectionReference
from src.models.performance import PerformanceSession, PerformanceMetrics


class PerformanceRepository:
    """Repository for performance-related data operations"""
    
    _collection: AsyncCollectionReference
    
    def __init__(self, firestore_client: AsyncClient):
        """Initialize PerformanceRepository with Firestore client"""
        self._collection = firestore_client.collection("performance")
    
    # PerformanceSession CRUD operations
    async def create_session(self, session: PerformanceSession) -> PerformanceSession:
        """Create a new performance session"""
        # Convert Pydantic model to dict for Firestore
        session_data = session.model_dump()
        
        # Convert datetime objects to ISO strings for Firestore
        session_data["started_at"] = session.started_at.isoformat()
        session_data["created_at"] = session.created_at.isoformat()
        session_data["updated_at"] = session.updated_at.isoformat()
        if session.ended_at:
            session_data["ended_at"] = session.ended_at.isoformat()
        
        # Create document in Firestore
        doc_ref = await self._collection.add(session_data)
        
        # Update the session with the generated document ID
        session_data["id"] = doc_ref[1].id
        
        # Return the created session as a Pydantic model
        return PerformanceSession(**session_data)
    
    async def get_session_by_id(self, session_id: str) -> Optional[PerformanceSession]:
        """Get session by ID"""
        doc = await self._collection.document(session_id).get()
        if not doc.exists:
            return None
        
        data = doc.to_dict()
        if data is None:
            return None
            
        data["id"] = doc.id
        
        # Convert ISO strings back to datetime objects
        if "started_at" in data and data["started_at"]:
            data["started_at"] = datetime.fromisoformat(data["started_at"])
        if "created_at" in data and data["created_at"]:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and data["updated_at"]:
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        if "ended_at" in data and data["ended_at"]:
            data["ended_at"] = datetime.fromisoformat(data["ended_at"])
        
        return PerformanceSession(**data)
    
    async def get_sessions_by_user_id(
        self, 
        user_id: str, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[PerformanceSession]:
        """Get user's sessions with pagination"""
        query = self._collection.where("user_id", "==", user_id)
        query = query.order_by("started_at", direction="DESCENDING")
        query = query.limit(limit).offset(offset)
        
        docs = query.stream()
        sessions = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            # Convert ISO strings back to datetime objects
            if "started_at" in data and data["started_at"]:
                data["started_at"] = datetime.fromisoformat(data["started_at"])
            if "created_at" in data and data["created_at"]:
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            if "updated_at" in data and data["updated_at"]:
                data["updated_at"] = datetime.fromisoformat(data["updated_at"])
            if "ended_at" in data and data["ended_at"]:
                data["ended_at"] = datetime.fromisoformat(data["ended_at"])
            
            sessions.append(PerformanceSession(**data))
        
        return sessions
    
    async def get_sessions_by_exercise_id(
        self, 
        exercise_id: str, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[PerformanceSession]:
        """Get sessions for specific exercise with pagination"""
        query = self._collection.where("exercise_id", "==", exercise_id)
        query = query.order_by("started_at", direction="DESCENDING")
        query = query.limit(limit).offset(offset)
        
        docs = query.stream()
        sessions = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            # Convert ISO strings back to datetime objects
            if "started_at" in data and data["started_at"]:
                data["started_at"] = datetime.fromisoformat(data["started_at"])
            if "created_at" in data and data["created_at"]:
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            if "updated_at" in data and data["updated_at"]:
                data["updated_at"] = datetime.fromisoformat(data["updated_at"])
            if "ended_at" in data and data["ended_at"]:
                data["ended_at"] = datetime.fromisoformat(data["ended_at"])
            
            sessions.append(PerformanceSession(**data))
        
        return sessions
    
    async def get_active_sessions_by_user_id(self, user_id: str) -> List[PerformanceSession]:
        """Get user's active/in-progress sessions"""
        query = self._collection.where("user_id", "==", user_id)
        query = query.where("is_completed", "==", False)
        query = query.order_by("started_at", direction="DESCENDING")
        
        docs = query.stream()
        sessions = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            # Convert ISO strings back to datetime objects
            if "started_at" in data and data["started_at"]:
                data["started_at"] = datetime.fromisoformat(data["started_at"])
            if "created_at" in data and data["created_at"]:
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            if "updated_at" in data and data["updated_at"]:
                data["updated_at"] = datetime.fromisoformat(data["updated_at"])
            if "ended_at" in data and data["ended_at"]:
                data["ended_at"] = datetime.fromisoformat(data["ended_at"])
            
            sessions.append(PerformanceSession(**data))
        
        return sessions
    
    async def update_session(self, session: PerformanceSession) -> PerformanceSession:
        """Update an existing session"""
        session_data = session.model_dump()
        
        # Convert datetime objects to ISO strings for Firestore
        session_data["started_at"] = session.started_at.isoformat()
        session_data["created_at"] = session.created_at.isoformat()
        session_data["updated_at"] = session.updated_at.isoformat()
        if session.ended_at:
            session_data["ended_at"] = session.ended_at.isoformat()
        
        # Remove the ID from data since it's used as document ID
        doc_id = session_data.pop("id")
        
        # Update document in Firestore
        await self._collection.document(doc_id).update(session_data)
        
        return session
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        try:
            await self._collection.document(session_id).delete()
            return True
        except Exception:
            return False
    
    # PerformanceMetrics CRUD operations
    async def create_metrics(self, metrics: PerformanceMetrics) -> PerformanceMetrics:
        """Create new performance metrics"""
        # Convert Pydantic model to dict for Firestore
        metrics_data = metrics.model_dump()
        
        # Convert datetime objects to ISO strings for Firestore
        metrics_data["created_at"] = metrics.created_at.isoformat()
        
        # Create document in Firestore
        doc_ref = await self._collection.add(metrics_data)
        
        # Update the metrics with the generated document ID
        metrics_data["id"] = doc_ref[1].id
        
        # Return the created metrics as a Pydantic model
        return PerformanceMetrics(**metrics_data)
    
    async def create_metrics_batch(self, metrics_list: List[PerformanceMetrics]) -> List[PerformanceMetrics]:
        """Create multiple performance metrics in batch"""
        created_metrics = []
        
        # Use Firestore batch write for efficient bulk operations
        batch = self._collection._client.batch()
        
        for metrics in metrics_list:
            # Convert Pydantic model to dict for Firestore
            metrics_data = metrics.model_dump()
            
            # Convert datetime objects to ISO strings for Firestore
            metrics_data["created_at"] = metrics.created_at.isoformat()
            
            # Create new document reference
            doc_ref = self._collection.document()
            metrics_data["id"] = doc_ref.id
            
            # Add to batch
            batch.set(doc_ref, metrics_data)
            
            # Create return object
            created_metrics.append(PerformanceMetrics(**metrics_data))
        
        # Commit batch write
        await batch.commit()
        
        return created_metrics
    
    async def get_metrics_by_id(self, metrics_id: str) -> Optional[PerformanceMetrics]:
        """Get metrics by ID"""
        doc = await self._collection.document(metrics_id).get()
        if not doc.exists:
            return None
        
        data = doc.to_dict()
        if data is None:
            return None
            
        data["id"] = doc.id
        
        # Convert ISO strings back to datetime objects
        if "created_at" in data and data["created_at"]:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        
        return PerformanceMetrics(**data)
    
    async def get_metrics_by_session_id(self, session_id: str) -> List[PerformanceMetrics]:
        """Get all metrics for a specific session"""
        query = self._collection.where("session_id", "==", session_id)
        query = query.order_by("created_at", direction="ASCENDING")
        
        docs = query.stream()
        metrics_list = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            # Convert ISO strings back to datetime objects
            if "created_at" in data and data["created_at"]:
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            
            metrics_list.append(PerformanceMetrics(**data))
        
        return metrics_list
    
    async def get_metrics_by_user_id(
        self, 
        user_id: str, 
        limit: int = 100, 
        offset: int = 0
    ) -> List[PerformanceMetrics]:
        """Get user's metrics with pagination"""
        query = self._collection.where("user_id", "==", user_id)
        query = query.order_by("created_at", direction="DESCENDING")
        query = query.limit(limit).offset(offset)
        
        docs = query.stream()
        metrics_list = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            # Convert ISO strings back to datetime objects
            if "created_at" in data and data["created_at"]:
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            
            metrics_list.append(PerformanceMetrics(**data))
        
        return metrics_list
    
    async def get_metrics_by_date_range(
        self, 
        user_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[PerformanceMetrics]:
        """Get user's metrics within date range"""
        query = self._collection.where("user_id", "==", user_id)
        query = query.where("created_at", ">=", start_date.isoformat())
        query = query.where("created_at", "<=", end_date.isoformat())
        query = query.order_by("created_at", direction="ASCENDING")
        
        docs = query.stream()
        metrics_list = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            # Convert ISO strings back to datetime objects
            if "created_at" in data and data["created_at"]:
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            
            metrics_list.append(PerformanceMetrics(**data))
        
        return metrics_list
    
    async def update_metrics(self, metrics: PerformanceMetrics) -> PerformanceMetrics:
        """Update existing metrics"""
        metrics_data = metrics.model_dump()
        
        # Convert datetime objects to ISO strings for Firestore
        metrics_data["created_at"] = metrics.created_at.isoformat()
        
        # Remove the ID from data since it's used as document ID
        doc_id = metrics_data.pop("id")
        
        # Update document in Firestore
        await self._collection.document(doc_id).update(metrics_data)
        
        return metrics
    
    async def delete_metrics(self, metrics_id: str) -> bool:
        """Delete metrics record"""
        try:
            await self._collection.document(metrics_id).delete()
            return True
        except Exception:
            return False
    
    async def delete_metrics_by_session_id(self, session_id: str) -> bool:
        """Delete all metrics for a session"""
        try:
            query = self._collection.where("session_id", "==", session_id)
            docs = query.stream()
            
            # Use batch delete for efficiency
            batch = self._collection._client.batch()
            delete_count = 0
            
            async for doc in docs:
                batch.delete(doc.reference)
                delete_count += 1
            
            if delete_count > 0:
                await batch.commit()
            
            return True
        except Exception:
            return False
    
    # Query methods for business logic
    async def get_recent_sessions_by_user_id(
        self, 
        user_id: str, 
        days: int = 30
    ) -> List[PerformanceSession]:
        """Get user's recent sessions within specified days"""
        from datetime import timedelta, timezone
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        query = self._collection.where("user_id", "==", user_id)
        query = query.where("started_at", ">=", start_date.isoformat())
        query = query.where("started_at", "<=", end_date.isoformat())
        query = query.order_by("started_at", direction="DESCENDING")
        
        docs = query.stream()
        sessions = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            # Convert ISO strings back to datetime objects
            if "started_at" in data and data["started_at"]:
                data["started_at"] = datetime.fromisoformat(data["started_at"])
            if "created_at" in data and data["created_at"]:
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            if "updated_at" in data and data["updated_at"]:
                data["updated_at"] = datetime.fromisoformat(data["updated_at"])
            if "ended_at" in data and data["ended_at"]:
                data["ended_at"] = datetime.fromisoformat(data["ended_at"])
            
            sessions.append(PerformanceSession(**data))
        
        return sessions
    
    async def get_performance_trend_by_user_id(
        self, 
        user_id: str, 
        skill_dimension: str,
        days: int = 90
    ) -> List[PerformanceMetrics]:
        """Get performance trend for specific skill dimension"""
        from datetime import timedelta, timezone
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        query = self._collection.where("user_id", "==", user_id)
        query = query.where("created_at", ">=", start_date.isoformat())
        query = query.where("created_at", "<=", end_date.isoformat())
        query = query.order_by("created_at", direction="ASCENDING")
        
        docs = query.stream()
        metrics_list = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            # Convert ISO strings back to datetime objects
            if "created_at" in data and data["created_at"]:
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            
            # Filter by skill dimension if available in metrics
            metrics = PerformanceMetrics(**data)
            if hasattr(metrics, f'{skill_dimension}_score'):
                metrics_list.append(metrics)
            else:
                # Include all metrics if dimension not found (for backward compatibility)
                metrics_list.append(metrics)
        
        return metrics_list
    
    async def get_session_statistics_by_user_id(self, user_id: str) -> dict:
        """Get session statistics for user (count, total time, etc.)"""
        query = self._collection.where("user_id", "==", user_id)
        docs = query.stream()
        
        total_sessions = 0
        completed_sessions = 0
        total_duration_seconds = 0
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            total_sessions += 1
            
            if data.get("is_completed", False):
                completed_sessions += 1
            
            # Calculate duration if both started_at and ended_at exist
            if "started_at" in data and "ended_at" in data and data["ended_at"]:
                try:
                    started = datetime.fromisoformat(data["started_at"])
                    ended = datetime.fromisoformat(data["ended_at"])
                    duration = (ended - started).total_seconds()
                    total_duration_seconds += duration
                except (ValueError, TypeError):
                    # Skip if datetime parsing fails
                    pass
        
        avg_duration_seconds = total_duration_seconds / completed_sessions if completed_sessions > 0 else 0
        
        return {
            "total_sessions": total_sessions,
            "completed_sessions": completed_sessions,
            "completion_rate": completed_sessions / total_sessions if total_sessions > 0 else 0,
            "total_duration_seconds": total_duration_seconds,
            "average_duration_seconds": avg_duration_seconds
        }
    
    async def count_sessions_by_user_id(self, user_id: str) -> int:
        """Count total sessions for a user"""
        query = self._collection.where("user_id", "==", user_id)
        docs = query.stream()
        
        count = 0
        async for _ in docs:
            count += 1
        
        return count