"""Tests for base model dict conversion methods."""

import pytest
from datetime import datetime, timezone
from typing import Optional
from pydantic import Field

from src.models.base import BaseModel


class SampleModel(BaseModel):
    """Test model for validating BaseModel functionality."""
    id: str = Field(..., description="Test ID")
    name: str = Field(..., description="Test name")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Update timestamp")
    count: int = Field(default=0, description="Test counter")


class TestBaseModel:
    """Test BaseModel dict conversion methods."""
    
    def test_from_dict_with_datetime_strings(self):
        """Test creating model from dict with ISO datetime strings."""
        now = datetime.now(timezone.utc)
        data = {
            "id": "test-123",
            "name": "Test Model",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "count": 42
        }
        
        model = SampleModel.from_dict(data)
        
        assert model.id == "test-123"
        assert model.name == "Test Model"
        assert isinstance(model.created_at, datetime)
        assert model.created_at == now
        assert isinstance(model.updated_at, datetime)
        assert model.updated_at == now
        assert model.count == 42
    
    def test_from_dict_with_none_optional_datetime(self):
        """Test creating model from dict with None optional datetime."""
        now = datetime.now(timezone.utc)
        data = {
            "id": "test-123",
            "name": "Test Model",
            "created_at": now.isoformat(),
            "updated_at": None,
            "count": 42
        }
        
        model = SampleModel.from_dict(data)
        
        assert model.id == "test-123"
        assert model.updated_at is None
        assert isinstance(model.created_at, datetime)
    
    def test_from_dict_with_datetime_objects(self):
        """Test creating model from dict with datetime objects (no conversion needed)."""
        now = datetime.now(timezone.utc)
        data = {
            "id": "test-123",
            "name": "Test Model",
            "created_at": now,
            "updated_at": now,
            "count": 42
        }
        
        model = SampleModel.from_dict(data)
        
        assert model.created_at == now
        assert model.updated_at == now
    
    def test_from_dict_empty_data_raises_error(self):
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="Data dictionary cannot be empty"):
            SampleModel.from_dict({})
    
    def test_to_dict_converts_datetime_to_iso_strings(self):
        """Test converting model to dict with datetime as ISO strings."""
        now = datetime.now(timezone.utc)
        model = SampleModel(
            id="test-123",
            name="Test Model",
            created_at=now,
            updated_at=now,
            count=42
        )
        
        data = model.to_dict()
        
        assert data["id"] == "test-123"
        assert data["name"] == "Test Model"
        assert data["created_at"] == now.isoformat()
        assert data["updated_at"] == now.isoformat()
        assert data["count"] == 42
        assert isinstance(data["created_at"], str)
        assert isinstance(data["updated_at"], str)
    
    def test_to_dict_with_none_optional_datetime(self):
        """Test converting model to dict with None optional datetime."""
        now = datetime.now(timezone.utc)
        model = SampleModel(
            id="test-123",
            name="Test Model", 
            created_at=now,
            updated_at=None,
            count=42
        )
        
        data = model.to_dict()
        
        assert data["created_at"] == now.isoformat()
        assert data["updated_at"] is None
    
    def test_round_trip_conversion(self):
        """Test that dict → model → dict preserves data."""
        now = datetime.now(timezone.utc)
        original_data = {
            "id": "test-123",
            "name": "Test Model",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "count": 42
        }
        
        # Dict → Model → Dict
        model = SampleModel.from_dict(original_data)
        converted_data = model.to_dict()
        
        assert converted_data == original_data