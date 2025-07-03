# Sax Buddy Service

REST backend service to support Sax Buddy (mobile application for iOS/Android)

## Frameworks

* **Python 3.11+** - Modern Python with type hints and async support
* **FastAPI** - High-performance web framework with automatic API documentation
* **Firebase Admin SDK** - Authentication and Firestore database integration
* **Firestore** - NoSQL document database for scalable data storage
* **Pydantic** - Data validation and serialization with type safety
* **JWT Authentication** - Firebase ID token verification for secure API access
* **pytest** - Comprehensive testing framework with async support

## Project Structure

```
src/
├── auth/                    # JWT Authentication Module
│   ├── __init__.py         # Authentication exports
│   ├── models.py           # JWTTokenData, AuthenticatedUser, AuthErrorResponse
│   ├── exceptions.py       # Custom authentication exceptions
│   ├── firebase_auth.py    # Firebase token verification utilities
│   ├── middleware.py       # JWT middleware for request interception
│   └── dependencies.py     # FastAPI authentication dependencies
├── api/
│   ├── routers/            # API endpoint routers (all secured with JWT)
│   │   ├── users.py        # User management endpoints
│   │   ├── performance.py  # Performance session and metrics
│   │   ├── content.py      # Exercises and lesson plans
│   │   ├── assessment.py   # Formal assessments and feedback
│   │   └── reference.py    # Reference data and skill levels
│   ├── schemas/            # Request/response schemas
│   └── openapi.py          # Custom OpenAPI configuration
├── models/                 # Domain models (Pydantic BaseModel)
│   ├── user.py            # User, UserProfile, UserProgress
│   ├── performance.py     # PerformanceSession, PerformanceMetrics
│   ├── content.py         # Exercise, LessonPlan, Lesson
│   ├── assessment.py      # FormalAssessment, Feedback, SkillMetrics
│   └── reference.py       # ReferencePerformance, SkillLevelDefinition
├── repositories/          # Data access layer (Firestore integration)
├── services/              # Business logic layer
├── dependencies.py        # FastAPI dependency providers
└── main.py               # Application entry point with JWT middleware
tests/
├── unit/                  # Unit tests (63 tests)
│   ├── models/           # Domain model validation tests
│   └── auth/             # Authentication system tests
└── integration/          # Integration tests with Firestore emulator
    ├── test_assessment_repository.py
    ├── test_user_repository.py
    ├── test_performance_repository.py
    ├── test_content_repository.py
    └── test_reference_repository.py
```

## Domain Models

### User Domain
- **User**: Basic user account information with email validation
- **UserProfile**: Learning preferences, skill levels, assessment intervals
- **UserProgress**: Multi-dimensional skill tracking (intonation, rhythm, articulation, dynamics)

### Performance Domain
- **PerformanceSession**: Practice session containers with duration and status tracking
- **PerformanceMetrics**: Time-series performance data from mobile DSP processing

### Content Domain
- **Exercise**: Practice exercises (scales, arpeggios, technical, etudes, songs, long-tone)
- **LessonPlan**: Adaptive AI-generated lesson sequences
- **Lesson**: Individual lessons within plans with learning objectives

### Assessment Domain
- **FormalAssessment**: Discrete evaluations with skill level recommendations
- **Feedback**: Post-session summary with encouragement and specific suggestions
- **SkillMetrics**: Weighted skill calculations over measurement periods

### Reference Domain
- **ReferencePerformance**: Skill-level appropriate target performances
- **SkillLevelDefinition**: Standardized skill level criteria and progression requirements

### Authentication Domain
- **JWTTokenData**: Firebase ID token payload with user claims and metadata
- **AuthenticatedUser**: Verified user context with permissions and custom claims
- **AuthErrorResponse**: Standardized authentication error responses

## Security Architecture

### JWT Authentication Flow
1. **Mobile Client** → Authenticates with Firebase Auth → Receives ID token
2. **API Request** → Includes `Authorization: Bearer <token>` header
3. **JWT Middleware** → Intercepts requests → Verifies Firebase ID token
4. **Dependency Injection** → Provides `AuthenticatedUser` to endpoints
5. **Route Protection** → All API endpoints require valid authentication

### Excluded Paths (No Authentication Required)
- `/docs` - OpenAPI documentation
- `/redoc` - Alternative API documentation  
- `/openapi.json` - OpenAPI specification
- `/health` - Health check endpoint
- `/` - Root endpoint with API information
- `/swagger` - Swagger UI redirect

## API Endpoints (All Secured)

### Users API (`/v1/users`)
- `POST /users` - Create new user
- `GET /users/{user_id}` - Get user by ID
- `GET /users/{user_id}/profile` - Get user profile
- `PUT /users/{user_id}/profile` - Update user profile
- `GET /users/{user_id}/progress` - Get user progress

### Performance API (`/v1/performance`)
- `POST /performance/sessions` - Start practice session
- `GET /performance/sessions/{session_id}` - Get session details
- `PATCH /performance/sessions/{session_id}` - Update session
- `POST /performance/sessions/{session_id}/metrics` - Submit performance metrics
- `GET /performance/sessions/{session_id}/metrics` - Get session metrics

### Content API (`/v1`)
- `GET /exercises` - List exercises with filters
- `GET /exercises/{exercise_id}` - Get exercise details
- `GET /users/{user_id}/lesson-plans` - Get user's lesson plans
- `POST /users/{user_id}/lesson-plans` - Generate new lesson plan
- `GET /lesson-plans/{plan_id}/lessons` - Get lessons in plan
- `GET /lessons/{lesson_id}` - Get lesson details
- `PATCH /lessons/{lesson_id}` - Update lesson status

### Assessment API (`/v1`)
- `GET /users/{user_id}/assessments` - Get user's assessments
- `POST /users/{user_id}/assessments` - Trigger formal assessment
- `GET /performance/sessions/{session_id}/feedback` - Get session feedback
- `GET /users/{user_id}/skill-metrics` - Get user's skill metrics

### Reference API (`/v1/reference`)
- `GET /reference/skill-levels` - Get skill level definitions
- `GET /reference/performances` - Get reference performances

## Key Design Decisions

- **JWT Authentication**: Firebase ID token verification for all API endpoints
- **Middleware-based Security**: Automatic request interception and user context injection
- **Flexible Dependencies**: Optional and required authentication with role-based access
- **Database-agnostic models**: Pure Pydantic BaseModel for future flexibility
- **Mobile-first processing**: Structured data sent from mobile device DSP analysis
- **Multi-dimensional skills**: Separate tracking for intonation, rhythm, articulation, dynamics
- **Adaptive learning**: AI-generated lessons based on weighted recent performance
- **User-controlled assessments**: Formal assessments triggered by user request or intervals
- **Skill-level references**: Target performances appropriate for user's current level
- **Comprehensive testing**: 63 unit tests + integration tests with Firestore emulator