from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, firestore_async, storage
import os
from src.api.routers import users, performance, content, assessment, reference
from src.api.openapi import custom_openapi
from src.middleware import LoggingMiddleware, JWTMiddleware
from src.logging_config import setup_logging, get_logger
from dotenv import load_dotenv

load_dotenv()

# Setup structured logging
setup_logging()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Firebase Admin SDK on startup, cleanup on shutdown"""
    # Startup - Initialize Firebase Admin SDK
    logger.info("Starting application initialization")
    
    try:
        if not firebase_admin._apps:
            # Check if running on Google Cloud (uses Application Default Credentials)
            if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GAE_APPLICATION"):
                logger.info("Initializing Firebase with Application Default Credentials")
                firebase_admin.initialize_app()
            else:
                # Local development - use service account key
                service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY", "serviceAccountKey.json")
                if os.path.exists(service_account_path):
                    logger.info("Initializing Firebase with service account key", path=service_account_path)
                    cred = credentials.Certificate(service_account_path)
                    firebase_admin.initialize_app(cred)
                else:
                    logger.info("Initializing Firebase with default credentials")
                    firebase_admin.initialize_app()
        
        # Store Firestore async client in app state
        app.state.firestore_client = firestore_async.client()
        bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET", "sax-buddy")
        app.state.bucket = storage.bucket(bucket_name)
        
        logger.success(
            "Firebase Admin SDK initialized successfully",
            bucket=bucket_name,
            environment=os.getenv("ENVIRONMENT", "development")
        )
        
    except Exception as e:
        logger.error(
            "Failed to initialize Firebase Admin SDK",
            error=str(e),
            error_type=type(e).__name__
        )
        app.state.firestore_client = None
        app.state.bucket = None
    
    yield
    
    # Shutdown - Cleanup Firebase resources
    logger.info("Starting application shutdown")
    try:
        if firebase_admin._apps:
            firebase_admin.delete_app(firebase_admin.get_app())
            logger.success("Firebase Admin SDK cleaned up successfully")
    except Exception as e:
        logger.error(
            "Error during Firebase cleanup",
            error=str(e),
            error_type=type(e).__name__
        )


app = FastAPI(
    lifespan=lifespan,
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

# Request/Response logging middleware
app.add_middleware(
    LoggingMiddleware,
    exclude_paths=[
        "/docs",
        "/redoc", 
        "/openapi.json",
        "/health",
        "/favicon.ico",
        "/",
        "/swagger"
    ]
)

# JWT authentication middleware
app.add_middleware(
    JWTMiddleware,
    excluded_paths=[
        "/docs",
        "/redoc", 
        "/openapi.json",
        "/health",
        "/favicon.ico",
        "/",
        "/swagger"
    ],
    require_auth_by_default=True
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