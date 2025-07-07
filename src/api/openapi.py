from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI


def custom_openapi(app: FastAPI):
    """Generate custom OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Sax Buddy Service API",
        version="0.1.0",
        description="""
        REST backend service for a multi-agent AI application for saxophone learning.
        This system processes structured performance data from mobile devices to provide 
        adaptive feedback and generate personalized lesson plans based on multi-dimensional 
        skill tracking (intonation, rhythm, articulation, dynamics).
        """,
        routes=app.routes,
        servers=[
            {"url": "http://localhost:8000", "description": "Development server"},
            {"url": "https://api.saxbuddy.com", "description": "Production server"},
        ]
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    # Add global security requirement
    openapi_schema["security"] = [{"bearerAuth": []}]
    
    # Add contact and license info
    openapi_schema["info"]["contact"] = {
        "name": "Sax Buddy Team"
    }
    openapi_schema["info"]["license"] = {
        "name": "MIT"
    }
    
    # Add tags descriptions
    openapi_schema["tags"] = [
        {
            "name": "Users",
            "description": "User account and profile management"
        },
        {
            "name": "Performance", 
            "description": "Practice session and performance metrics tracking"
        },
        {
            "name": "Content",
            "description": "Learning content and lesson plan management"
        },
        {
            "name": "Assessment",
            "description": "Formal assessments and feedback"
        },
        {
            "name": "Reference",
            "description": "Reference data and skill level definitions"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema