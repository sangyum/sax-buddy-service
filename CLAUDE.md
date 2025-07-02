# Sax Buddy Service

REST backend service to support Sax Buddy (mobile application for iOS/Android)

## Frameworks

* Python
* FastAPI
* Time-series database (TBD)
* OpenAI for LLM
* Pydantic for data validation
* Authentication: Firebase (planned)

## Project Structure

```
src/
├── models/
│   ├── user.py              # User, UserProfile, UserProgress
│   ├── performance.py       # PerformanceSession, PerformanceMetrics
│   ├── content.py           # Exercise, LessonPlan, Lesson
│   ├── assessment.py        # FormalAssessment, Feedback, SkillMetrics
│   └── reference.py         # ReferencePerformance, SkillLevelDefinition
tests/
├── models/
│   ├── test_user.py
│   ├── test_performance.py
│   ├── test_content.py
│   ├── test_assessment.py
│   └── test_reference.py
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

## Key Design Decisions

- **Database-agnostic models**: Pure Pydantic BaseModel for future flexibility
- **Mobile-first processing**: Structured data sent from mobile device DSP analysis
- **Multi-dimensional skills**: Separate tracking for intonation, rhythm, articulation, dynamics
- **Adaptive learning**: AI-generated lessons based on weighted recent performance
- **User-controlled assessments**: Formal assessments triggered by user request or intervals
- **Skill-level references**: Target performances appropriate for user's current level