# Sax Buddy Service

REST backend service to support Sax Buddy (mobile application for iOS/Android)

## Development Practice

## Development Practice

1. Act as a pairing partner for the user. Do not commit anything until explicit consent is given
2. **STRICT TDD PRACTICE - RED/GREEN/REFACTOR CYCLE:**
  * **RED**: Write a failing test that describes the intended behavior FIRST
  * **GREEN**: Write minimal code to make that specific test pass
  * **REFACTOR**: Clean up code while keeping tests green
  * **NEVER** write implementation before tests
  * **ALWAYS** see the test fail first to validate it works
  * Run test after each step: `uv run pytest tests/ -v`
3. Update this document with up-to-date project structures and architecture.
4. **TDD Reminder**: If you find yourself writing implementation code before seeing a failing test, STOP and write the test first.

### Development Environment Setup

**Firebase Emulator for Development:**
- **Start Emulator**: `firebase emulators:start --only firestore`
- **Development Configuration**: `ENVIRONMENT=development` in `.env`
- **Authentication Bypass**: JWT auth bypassed in development mode (returns mock user)
- **Mock User**: `dev-user` with email `dev@example.com` for all authenticated endpoints
- **Emulator UI**: http://127.0.0.1:4000/firestore
- **Debug Endpoint**: `GET /debug/auth-status` for authentication flow verification

**Logging Configuration:**
- **Default Level**: ERROR (configurable via `LOG_LEVEL` environment variable)
- **Valid Levels**: TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL
- **Development**: Console logging with colors and structured output
- **Production**: Additional file logging with JSON format and log rotation
- **Usage**: `from src.logging_config import get_logger; logger = get_logger(__name__)`

**Test Categories:**
- **Unit Tests** (106 tests): Domain models, business logic, authentication, middleware
- **Integration Tests** (88+ tests): Repository operations with Firestore emulator
- **Test Command**: `uv run pytest tests/unit/auth/ -v` (auth-specific)
- **All Tests**: `uv run pytest tests/ -v`

## Frameworks

* **Python 3.11+** - Modern Python with type hints and async support
* **FastAPI** - High-performance web framework with automatic API documentation
* **Firebase Admin SDK** - Authentication and Firestore database integration
* **Firestore** - NoSQL document database for scalable data storage
* **Pydantic** - Data validation and serialization with type safety
* **JWT Authentication** - Firebase ID token verification for secure API access
* **Loguru** - Structured logging with JSON output and automatic rotation
* **pytest** - Comprehensive testing framework with async support

## Project Structure

```
src/
├── auth/                    # JWT Authentication Module
│   ├── __init__.py         # Authentication exports
│   ├── models.py           # JWTTokenData, AuthenticatedUser, AuthErrorResponse (Pydantic v2)
│   ├── exceptions.py       # Custom authentication exceptions
│   ├── firebase_auth.py    # Firebase token verification utilities
│   ├── utils.py            # Development mode detection utilities
│   └── dependencies.py     # FastAPI authentication dependencies (auto_error=False)
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
│   ├── base.py            # Enhanced BaseModel with dict conversion
│   ├── user.py            # User, UserProfile, UserProgress
│   ├── performance.py     # PerformanceSession, PerformanceMetrics
│   ├── content.py         # Exercise, LessonPlan, Lesson
│   ├── assessment.py      # FormalAssessment, Feedback, SkillMetrics
│   └── reference.py       # ReferencePerformance, SkillLevelDefinition
├── middleware/            # HTTP middleware components
│   ├── __init__.py        # Middleware exports
│   ├── auth_middleware.py # JWT authentication middleware with development bypass
│   └── logging_middleware.py # Request/response logging with structured data
├── repositories/          # Data access layer (Firestore integration)
│   ├── content_repository.py    # Exercise, LessonPlan, Lesson operations
│   ├── performance_repository.py # PerformanceSession, PerformanceMetrics, analytics
│   ├── assessment_repository.py  # FormalAssessment, Feedback, SkillMetrics
│   ├── user_repository.py       # User, UserProfile, UserProgress operations
│   └── reference_repository.py  # ReferencePerformance, SkillLevelDefinition
├── services/              # Business logic layer
├── dependencies.py        # FastAPI dependency providers
├── logging_config.py      # Structured logging configuration with loguru
└── main.py               # Application entry point with middleware stack
tests/
├── unit/                  # Unit tests (106 tests)
│   ├── models/           # Domain model validation tests
│   ├── auth/             # Authentication system tests
│   └── middleware/       # Middleware tests (JWT auth, logging)
└── integration/          # Integration tests with Firestore emulator
    ├── test_assessment_repository.py
    ├── test_user_repository.py
    ├── test_performance_repository.py
    ├── test_content_repository.py
    └── test_reference_repository.py
logs/                     # Application logs (production)
├── .gitkeep             # Keep directory in git
└── sax_buddy_*.log      # Daily rotated log files with compression
```

## Domain Models

### User Domain
- **User**: Basic user account information with email validation
- **UserProfile**: Learning preferences, skill levels, assessment intervals
- **UserProgress**: Multi-dimensional skill tracking (intonation, rhythm, articulation, dynamics)

### Performance Domain
- **PerformanceSession**: Practice session containers with duration and status tracking
- **PerformanceMetrics**: Time-series performance data from mobile DSP processing
- **SessionSummary**: Aggregated session data (total sessions, total minutes) for completed sessions
- **SessionStatistics**: Comprehensive session analytics with breakdowns by status, durations, and practice patterns
- **PerformanceTrendPoint**: Time-series performance averages across all skill dimensions

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
- `/debug/*` - Debug endpoints for development

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
- `GET /users/{user_id}/performance/statistics` - Get comprehensive session statistics
- `GET /users/{user_id}/performance/summary` - Get session summary (completed sessions)
- `GET /users/{user_id}/performance/trend` - Get performance trend over time

### Content API (`/v1`)
- `GET /exercises` - List exercises with filters
- `GET /exercises/search?q={keyword}` - Search exercises by keyword
- `GET /exercises/active` - Get only active exercises
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

### Debug API (`/debug`) - Development Only
- `GET /debug/auth-status` - Authentication flow verification and mock user status

## Key Design Decisions

- **Development-Friendly Authentication**: JWT authentication bypassed in development mode with mock user
- **Middleware-based Security**: Automatic request interception and user context injection
- **Pydantic v2 Migration**: Modern ConfigDict instead of deprecated class-based Config
- **Structured Logging**: Loguru-based logging with JSON output and environment-aware configuration
- **Auto-error False**: HTTPBearer security with `auto_error=False` for development bypass
- **Database-agnostic models**: Enhanced BaseModel with `from_dict()` and `to_dict()` methods
- **Mobile-first processing**: Structured data sent from mobile device DSP analysis
- **Multi-dimensional skills**: Separate tracking for intonation, rhythm, articulation, dynamics
- **Adaptive learning**: AI-generated lessons based on weighted recent performance
- **User-controlled assessments**: Formal assessments triggered by user request or intervals
- **Skill-level references**: Target performances appropriate for user's current level
- **Comprehensive testing**: 106 unit tests + integration tests with Firestore emulator