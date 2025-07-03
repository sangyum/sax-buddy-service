from typing import List, Optional
from datetime import datetime
from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1.async_collection import AsyncCollectionReference
from src.models.user import User, UserProfile, UserProgress


class UserRepository:
    """Repository for user-related data operations"""
    
    _collection: AsyncCollectionReference
    
    def __init__(self, firestore_client: AsyncClient):
        """Initialize UserRepository with Firestore client"""
        self._collection = firestore_client.collection("users")
    
    # User CRUD operations
    async def create_user(self, user: User) -> User:
        """Create a new user"""
        # Convert Pydantic model to dict for Firestore
        user_data = user.model_dump()
        
        # Convert datetime objects to ISO strings for Firestore
        user_data["created_at"] = user.created_at.isoformat()
        user_data["updated_at"] = user.updated_at.isoformat()
        
        # Create document in Firestore
        doc_ref = await self._collection.add(user_data)
        
        # Update the user with the generated document ID
        user_data["id"] = doc_ref[1].id
        
        # Return the created user as a Pydantic model
        return User(**user_data)
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        doc = await self._collection.document(user_id).get()
        if not doc.exists:
            return None
        
        data = doc.to_dict()
        if data is None:
            return None
            
        data["id"] = doc.id
        
        # Convert ISO strings back to datetime objects
        if "created_at" in data and data["created_at"]:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and data["updated_at"]:
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        
        return User(**data)
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email address"""
        query = self._collection.where("email", "==", email)
        query = query.limit(1)
        
        docs = query.stream()
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            # Convert ISO strings back to datetime objects
            if "created_at" in data and data["created_at"]:
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            if "updated_at" in data and data["updated_at"]:
                data["updated_at"] = datetime.fromisoformat(data["updated_at"])
            
            return User(**data)
        
        return None
    
    async def update_user(self, user: User) -> User:
        """Update an existing user"""
        user_data = user.model_dump()
        
        # Convert datetime objects to ISO strings for Firestore
        user_data["created_at"] = user.created_at.isoformat()
        user_data["updated_at"] = user.updated_at.isoformat()
        
        # Remove the ID from data since it's used as document ID
        doc_id = user_data.pop("id")
        
        # Update document in Firestore
        await self._collection.document(doc_id).update(user_data)
        
        return user
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete a user"""
        try:
            await self._collection.document(user_id).delete()
            return True
        except Exception:
            return False
    
    async def list_users(self, limit: int = 50, offset: int = 0) -> List[User]:
        """List users with pagination"""
        query = self._collection.order_by("created_at", direction="DESCENDING")
        query = query.limit(limit).offset(offset)
        
        docs = query.stream()
        users = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            # Convert ISO strings back to datetime objects
            if "created_at" in data and data["created_at"]:
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            if "updated_at" in data and data["updated_at"]:
                data["updated_at"] = datetime.fromisoformat(data["updated_at"])
            
            users.append(User(**data))
        
        return users
    
    # UserProfile CRUD operations
    async def create_user_profile(self, profile: UserProfile) -> UserProfile:
        """Create a new user profile"""
        # Convert Pydantic model to dict for Firestore
        profile_data = profile.model_dump()
        
        # Convert datetime objects to ISO strings for Firestore
        profile_data["created_at"] = profile.created_at.isoformat()
        profile_data["updated_at"] = profile.updated_at.isoformat()
        
        # Create document in Firestore
        doc_ref = await self._collection.add(profile_data)
        
        # Update the profile with the generated document ID
        profile_data["id"] = doc_ref[1].id
        
        # Return the created profile as a Pydantic model
        return UserProfile(**profile_data)
    
    async def get_profile_by_user_id(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by user ID"""
        query = self._collection.where("user_id", "==", user_id)
        query = query.limit(1)
        
        docs = query.stream()
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            # Convert ISO strings back to datetime objects
            if "created_at" in data and data["created_at"]:
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            if "updated_at" in data and data["updated_at"]:
                data["updated_at"] = datetime.fromisoformat(data["updated_at"])
            
            return UserProfile(**data)
        
        return None
    
    async def get_profile_by_id(self, profile_id: str) -> Optional[UserProfile]:
        """Get user profile by profile ID"""
        doc = await self._collection.document(profile_id).get()
        if not doc.exists:
            return None
        
        data = doc.to_dict()
        if data is None:
            return None
            
        data["id"] = doc.id
        
        # Convert ISO strings back to datetime objects
        if "created_at" in data and data["created_at"]:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and data["updated_at"]:
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        
        return UserProfile(**data)
    
    async def update_user_profile(self, profile: UserProfile) -> UserProfile:
        """Update an existing user profile"""
        profile_data = profile.model_dump()
        
        # Convert datetime objects to ISO strings for Firestore
        profile_data["created_at"] = profile.created_at.isoformat()
        profile_data["updated_at"] = profile.updated_at.isoformat()
        
        # Remove the ID from data since it's used as document ID
        doc_id = profile_data.pop("id")
        
        # Update document in Firestore
        await self._collection.document(doc_id).update(profile_data)
        
        return profile
    
    async def delete_user_profile(self, profile_id: str) -> bool:
        """Delete a user profile"""
        try:
            await self._collection.document(profile_id).delete()
            return True
        except Exception:
            return False
    
    # UserProgress CRUD operations
    async def create_user_progress(self, progress: UserProgress) -> UserProgress:
        """Create a new user progress record"""
        # Convert Pydantic model to dict for Firestore
        progress_data = progress.model_dump()
        
        # Convert datetime objects to ISO strings for Firestore
        progress_data["created_at"] = progress.created_at.isoformat()
        progress_data["updated_at"] = progress.updated_at.isoformat()
        if progress.last_formal_assessment_date:
            progress_data["last_formal_assessment_date"] = progress.last_formal_assessment_date.isoformat()
        
        # Create document in Firestore
        doc_ref = await self._collection.add(progress_data)
        
        # Update the progress with the generated document ID
        progress_data["id"] = doc_ref[1].id
        
        # Return the created progress as a Pydantic model
        return UserProgress(**progress_data)
    
    async def get_progress_by_user_id(self, user_id: str) -> Optional[UserProgress]:
        """Get user progress by user ID"""
        query = self._collection.where("user_id", "==", user_id)
        query = query.limit(1)
        
        docs = query.stream()
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            # Convert ISO strings back to datetime objects
            if "created_at" in data and data["created_at"]:
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            if "updated_at" in data and data["updated_at"]:
                data["updated_at"] = datetime.fromisoformat(data["updated_at"])
            if "last_formal_assessment_date" in data and data["last_formal_assessment_date"]:
                data["last_formal_assessment_date"] = datetime.fromisoformat(data["last_formal_assessment_date"])
            
            return UserProgress(**data)
        
        return None
    
    async def get_progress_by_id(self, progress_id: str) -> Optional[UserProgress]:
        """Get user progress by progress ID"""
        doc = await self._collection.document(progress_id).get()
        if not doc.exists:
            return None
        
        data = doc.to_dict()
        if data is None:
            return None
            
        data["id"] = doc.id
        
        # Convert ISO strings back to datetime objects
        if "created_at" in data and data["created_at"]:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and data["updated_at"]:
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        if "last_formal_assessment_date" in data and data["last_formal_assessment_date"]:
            data["last_formal_assessment_date"] = datetime.fromisoformat(data["last_formal_assessment_date"])
        
        return UserProgress(**data)
    
    async def update_user_progress(self, progress: UserProgress) -> UserProgress:
        """Update an existing user progress record"""
        progress_data = progress.model_dump()
        
        # Convert datetime objects to ISO strings for Firestore
        progress_data["created_at"] = progress.created_at.isoformat()
        progress_data["updated_at"] = progress.updated_at.isoformat()
        if progress.last_formal_assessment_date:
            progress_data["last_formal_assessment_date"] = progress.last_formal_assessment_date.isoformat()
        
        # Remove the ID from data since it's used as document ID
        doc_id = progress_data.pop("id")
        
        # Update document in Firestore
        await self._collection.document(doc_id).update(progress_data)
        
        return progress
    
    async def delete_user_progress(self, progress_id: str) -> bool:
        """Delete a user progress record"""
        try:
            await self._collection.document(progress_id).delete()
            return True
        except Exception:
            return False
    
    # Query methods for business logic
    async def get_users_by_skill_level(self, skill_level: str) -> List[User]:
        """Get users by skill level"""
        query = self._collection.where("skill_level", "==", skill_level)
        
        docs = query.stream()
        users = []
        
        async for doc in docs:
            data = doc.to_dict()
            if data is None:
                continue
                
            data["id"] = doc.id
            
            # Convert ISO strings back to datetime objects
            if "created_at" in data and data["created_at"]:
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            if "updated_at" in data and data["updated_at"]:
                data["updated_at"] = datetime.fromisoformat(data["updated_at"])
            
            users.append(User(**data))
        
        return users
    
    async def count_active_users(self) -> int:
        """Count total active users"""
        query = self._collection.where("is_active", "==", True)
        docs = query.stream()
        
        count = 0
        async for _ in docs:
            count += 1
        
        return count