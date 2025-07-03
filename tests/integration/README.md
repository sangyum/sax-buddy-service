# Integration Tests

This directory contains integration tests for the Sax Buddy Service that use the Firestore emulator.

## Prerequisites

1. Install Firebase CLI:
   ```bash
   npm install -g firebase-tools
   ```

2. Install Java (required for Firestore emulator):
   ```bash
   # On macOS with Homebrew
   brew install openjdk
   
   # Or download from: https://adoptopenjdk.net/
   ```

3. Install test dependencies:
   ```bash
   pip install -r tests/integration/requirements.txt
   ```

## Running Integration Tests

### Method 1: Manual Emulator Management (Recommended)

Start the emulator manually for better control:

1. Start the Firestore emulator:
   ```bash
   export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"  # On macOS with Homebrew
   firebase emulators:start --only firestore --project demo-test
   ```

2. In another terminal, run the tests:
   ```bash
   uv run pytest tests/integration/ -v
   
   # Run specific test file
   uv run pytest tests/integration/test_assessment_repository.py -v
   
   # Run specific test method
   uv run pytest tests/integration/test_assessment_repository.py::TestFormalAssessmentOperations::test_create_assessment -v
   ```

### Method 2: Automatic Emulator Management

The tests can also automatically detect running emulators, but manual management provides better control.

## Test Structure

The integration tests are organized into several test classes:

- **TestFormalAssessmentOperations**: Tests CRUD operations for formal assessments
- **TestFeedbackOperations**: Tests CRUD operations for feedback
- **TestSkillMetricsOperations**: Tests CRUD operations for skill metrics
- **TestBusinessLogicQueries**: Tests complex queries and business logic
- **TestDataConsistency**: Tests data consistency and edge cases

## Test Coverage

The integration tests cover:

- Creating, reading, updating, and deleting assessments, feedback, and skill metrics
- Pagination and filtering operations
- Date range queries
- Business logic operations (counting, trends, progressions)
- Data consistency and datetime serialization
- Error handling and edge cases

## Configuration

The tests use the following configuration:

- Firestore emulator host: `localhost:8080`
- Test project ID: `test-project`
- All test data is automatically cleaned up after each test

## Troubleshooting

### Port Already in Use

If you get a "port already in use" error, either:

1. Stop any existing emulator processes:
   ```bash
   pkill -f "firebase.*emulator"
   ```

2. Or use a different port in the test configuration

### Java Not Found

If you get a Java-related error:

1. Ensure Java is installed and in your PATH
2. Set JAVA_HOME environment variable if needed:
   ```bash
   export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
   ```

### Firestore Emulator Not Starting

If the emulator fails to start:

1. Check if Firebase CLI is installed: `firebase --version`
2. Make sure you have the latest version: `npm update -g firebase-tools`
3. Try running the emulator manually first: `firebase emulators:start --only firestore`