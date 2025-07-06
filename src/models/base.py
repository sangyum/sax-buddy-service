"""Base model with enhanced dict conversion capabilities."""

from datetime import datetime
from typing import Dict, Any, Optional, get_origin, get_args, TypeVar
from pydantic import BaseModel as PydanticBaseModel

T = TypeVar('T', bound='BaseModel')


class BaseModel(PydanticBaseModel):
    """Enhanced base model with infrastructure-agnostic dict conversion methods."""
    
    @classmethod
    def from_dict(cls: type[T], data: Dict[str, Any]) -> T:
        """
        Create model instance from dictionary with automatic type conversion.
        
        Handles common conversions like ISO datetime strings to datetime objects.
        """
        if not data:
            raise ValueError("Data dictionary cannot be empty")
            
        converted_data = data.copy()
        
        # Convert ISO datetime strings to datetime objects for datetime fields
        for field_name, field_info in cls.model_fields.items():
            if field_name in converted_data and converted_data[field_name] is not None:
                # Handle direct datetime fields
                if field_info.annotation == datetime:
                    if isinstance(converted_data[field_name], str):
                        converted_data[field_name] = datetime.fromisoformat(converted_data[field_name])
                
                # Handle Optional[datetime] fields
                elif (get_origin(field_info.annotation) is type(Optional[int]) and 
                      len(get_args(field_info.annotation)) == 2 and
                      get_args(field_info.annotation)[0] == datetime):
                    if isinstance(converted_data[field_name], str):
                        converted_data[field_name] = datetime.fromisoformat(converted_data[field_name])
        
        return cls(**converted_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert model to dictionary with automatic type conversion.
        
        Handles conversions like datetime objects to ISO strings.
        """
        data = self.model_dump()
        
        # Convert datetime objects to ISO strings
        for field_name, value in data.items():
            if isinstance(value, datetime):
                data[field_name] = value.isoformat()
        
        return data