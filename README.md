# Sax Buddy Service

REST backend service for a multi-agent AI application for saxophone learning that moves beyond traditional discrete assessments. This system continuously monitors user performance through real-time audio analysis, leveraging specialized digital signal processing (DSP) to extract granular data on intonation, rhythm, articulation, and dynamics. The backend processes structured performance data from mobile devices to provide adaptive feedback and generate personalized lesson plans.

## Quick Start

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Start development server
uv run python run_server.py
```

## Architecture Overview

The backend implements a comprehensive domain model supporting:

- **Mobile-first DSP processing**: Audio analysis happens on device, structured data sent to backend
- **Multi-dimensional skill tracking**: Separate scores for intonation, rhythm, articulation, dynamics  
- **Adaptive lesson generation**: AI-powered lesson plans based on weighted recent performance
- **User-controlled formal assessments**: Triggered by user request or configurable intervals
- **Database-agnostic design**: Pydantic models ready for any storage backend

## Domain Models

### 📱 User Management
- `User` - Account information with email validation
- `UserProfile` - Learning preferences and skill levels
- `UserProgress` - Historical progress with multi-dimensional scoring

### 🎵 Performance Tracking  
- `PerformanceSession` - Practice session containers
- `PerformanceMetrics` - Time-series data from mobile DSP analysis

### 📚 Learning Content
- `Exercise` - Practice exercises (scales, arpeggios, etudes, songs, long-tone)
- `LessonPlan` - Adaptive AI-generated learning sequences
- `Lesson` - Individual lessons with objectives and exercises

### 📊 Assessment & Feedback
- `FormalAssessment` - Discrete skill evaluations with recommendations
- `Feedback` - Post-session summaries with encouragement and suggestions
- `SkillMetrics` - Weighted calculations over time periods

### 🎯 Reference Standards
- `ReferencePerformance` - Skill-level appropriate targets
- `SkillLevelDefinition` - Standardized progression criteria

# Main Components for Adaptive Assessment

Our discussion has highlighted a core distinction between continuous, real-time monitoring/feedback and discrete, post-performance assessment. The continuous monitoring will be crucial for informing more appropriate and tailored formal assessments over time. To achieve this, several key architectural components and data models will be vital:

1. Audio Ingestion & Processing Module - UI

    This module will be responsible for capturing and cleaning raw audio input from the user's instrument. It'll handle real-time streaming, noise reduction, and potentially instrument isolation, outputting a clear audio stream ready for analysis.

2. Real-time Performance Analysis Module (The "Monitor") - UI
    
    This is the heart of your continuous monitoring. It will analyze the processed audio stream in real-time for various musical parameters like pitch, rhythm, dynamics, and note recognition. Its output will be streams of granular, time-series performance data, forming the "multi-faceted analysis" of the user's playing.

3. Performance Data Storage (The "History") - Backend
    
    Given the volume of data, this component will store the detailed, granular performance data from the real-time analysis module. A NoSQL or time-series database would likely be suitable. Key data model elements would include SessionID, UserID, Piece/ExerciseID, Timestamp, PerformanceMetrics (detailed granular data like pitch arrays, rhythm onsets), and ReferenceData for comparison.

4. Assessment Logic Module - Backend
  
    This module will compare the granular performance data (from storage or directly from the real-time module) against reference data and user goals. It will generate both immediate, post-segment feedback and formal assessment scores. This is where statistical analysis, deviation calculations, and rule-based grading would occur.

5. User Profile & Progress Tracking (The "Current State") - Backend

    Crucially, this component will store the user's current skill level, historical formal assessment scores, and personalized lesson plan progress. It represents the "output from the previous formal assessment." A relational database would likely manage UserID, CurrentSkillLevel, AssessmentHistory, LessonPlanHistory, and user Preferences.

6. Lesson Plan Generation & Adaptation Module - Backend

    This module will leverage the current skill level from the User Profile and insights from recent performance data and assessments to generate or adapt the personalized lesson plan. This could involve AI/ML models for recommendations or rule-based systems.

7. Feedback & Visualization Module - UI

    Finally, this component will be responsible for presenting the real-time monitoring visualizations and the more structured post-performance assessment feedback to the user, ensuring clear and actionable insights.

## How These Components Ensure "Appropriateness":

By continuously collecting detailed performance data through the Real-time Performance Analysis Module and storing it in Performance Data Storage, the Assessment Logic Module gains a rich, nuanced understanding of the user's tendencies over multiple practice sessions. This holistic view, combined with the historical data from User Profile & Progress Tracking, allows the system to make much more informed and "appropriate" judgments of skill level, directly feeding into the Lesson Plan Generation & Adaptation Module.
