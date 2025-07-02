from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.api.routers import users, performance, content, assessment, reference
from src.api.openapi import custom_openapi

app = FastAPI(
    title="Sax Buddy Service API",
    description="""
    REST backend service for a multi-agent AI application for saxophone learning.
    This system processes structured performance data from mobile devices to provide 
    adaptive feedback and generate personalized lesson plans based on multi-dimensional 
    skill tracking (intonation, rhythm, articulation, dynamics).
    """,
    version="0.1.0",
    contact={
        "name": "Sax Buddy Team",
    },
    license_info={
        "name": "MIT",
    },
    servers=[
        {"url": "http://localhost:8000", "description": "Development server"},
        {"url": "https://api.saxbuddy.com", "description": "Production server"},
    ],
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure specific origins for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(users.router, prefix="/v1")
app.include_router(performance.router, prefix="/v1")
app.include_router(content.router, prefix="/v1")
app.include_router(assessment.router, prefix="/v1")
app.include_router(reference.router, prefix="/v1")

# Set custom OpenAPI schema
app.openapi = lambda: custom_openapi(app)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Sax Buddy Service API",
        "version": "0.1.0",
        "docs": "/docs",
        "redoc": "/redoc", 
        "swagger": "/swagger"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/swagger")
async def get_swagger_ui():
    """Swagger UI endpoint - redirects to the main docs"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)