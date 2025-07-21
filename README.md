# Sax Buddy Service

Firebase Functions for comprehensive saxophone assessment audio analysis using Essentia.js DSP library.

## Overview

This service provides HTTP-triggered Firebase Functions for analyzing saxophone recordings and generating detailed performance assessments across multiple dimensions using real-time audio signal processing:

### 🎵 **Analysis Categories**

- **Pitch and Intonation**: 
  - Cent-level accuracy measurements using YinFFT algorithm
  - Intonation stability across sustained notes
  - Interval precision for melodic and harmonic intervals
  - Real-time tuning drift detection and analysis
  
- **Timing and Rhythm**: 
  - Metronome deviation tracking with millisecond precision
  - Rhythmic subdivision analysis (quarters, eighths, sixteenths, triplets)
  - Groove consistency and swing ratio detection
  - Rubato and expressive timing evaluation
  
- **Tone Quality and Timbre**: 
  - Harmonic content analysis using spectral processing
  - Dynamic range utilization and accent detection
  - Timbral consistency across registers and dynamics
  - Vibrato detection with rate, depth, and quality metrics
  
- **Technical Execution**: 
  - Articulation clarity and attack consistency
  - Finger technique smoothness analysis
  - Breath management and phrase timing
  - Extended technique detection (multiphonics, altissimo, growls)
  
- **Musical Expression**: 
  - Phrasing sophistication and melodic arch construction
  - Dynamic contour complexity and shaping control
  - Style-specific evaluation (jazz, classical, contemporary)
  - Improvisational coherence and motivic development
  
- **Performance Consistency**: 
  - Error pattern detection and classification
  - Recovery speed and effectiveness analysis
  - Endurance tracking throughout performance
  - Difficulty scaling and limitation identification

## Architecture

- **Framework**: Firebase Functions with TypeScript
- **Audio Analysis**: Essentia.js v0.1.3 (WebAssembly C++ DSP library)
- **Storage**: Firebase Storage for audio files
- **Database**: Firestore for assessment sessions and results
- **Runtime**: Node.js 18+ with 2GB memory allocation

## API Endpoints

### POST `/analyzeAssessmentAudio`

Analyzes all exercise recordings in an assessment session.

**Request Body:**
```json
{
  "userId": "string",
  "assessmentId": "string"
}
```

**Response:**
```json
{
  "success": boolean,
  "assessmentId": "string",
  "processedExercises": number,
  "failedExercises": number,
  "results": {
    "exerciseId": {
      "status": "completed" | "failed",
      "analysis": { /* detailed analysis */ },
      "error": "string (if failed)"
    }
  },
  "summary": {
    "overallPerformanceScore": number,
    "averageScores": { /* category scores */ },
    "strengthAreas": string[],
    "improvementAreas": string[],
    "processingTime": number
  }
}
```

### GET `/healthCheck`

Service health check endpoint.

## Data Models

### AssessmentSession
```typescript
{
  id: string;
  userId: string;
  exercises: Exercise[];
  overallStatus: "pending" | "processing" | "completed" | "failed";
  summary?: AssessmentSummary;
}
```

### Exercise
```typescript
{
  id: string;
  recordingUrl: string; // Firebase Storage URL
  analysisStatus: ProcessingStatus;
  analysis?: ExtendedAudioAnalysis;
  metadata?: ExerciseMetadata;
}
```

## Development

### Prerequisites

- Node.js 18+
- Firebase CLI
- Firebase project with Functions, Storage, and Firestore enabled

### Setup

1. Install dependencies:
```bash
npm install
```

2. Build the project:
```bash
npm run build
```

3. Start local emulator:
```bash
npm run serve
```

4. Deploy to Firebase:
```bash
npm run deploy
```

### Development Workflow

Follow Test-Driven Development (TDD) practices:

1. **Write failing test** for new functionality
2. **Implement minimal code** to make test pass
3. **Refactor** code while keeping tests green
4. **Run tests** after every change: `npm run test`
5. **Run linting** before committing: `npm run lint`

### Scripts

- `npm run build` - Compile TypeScript
- `npm run build:watch` - Watch mode compilation
- `npm run serve` - Start Firebase emulators
- `npm run deploy` - Deploy to Firebase
- `npm run lint` - Run ESLint
- `npm run lint:fix` - Auto-fix linting issues
- `npm run test` - Run unit tests
- `npm run test:watch` - Run tests in watch mode

## Configuration

### Firebase Configuration

The function is configured with:
- **Memory**: 2GB (for audio processing)
- **Timeout**: 540 seconds (9 minutes)
- **Runtime**: Node.js 18
- **Region**: us-central1

### Audio File Support

**Supported Formats:**
- WAV (16-bit PCM) - Primary support with real-time processing
- Future: MP3, AAC, FLAC (requires additional decoding)

**Audio Processing Pipeline:**
1. **Input Validation**: File size (max 100MB), format verification
2. **Signal Processing**: Frame-based analysis (2048 samples, 512 hop)
3. **Feature Extraction**: 
   - Pitch tracking with YinFFT algorithm
   - Spectral analysis (MFCC, centroid, rolloff)
   - Energy and onset detection
   - Rhythm and tempo extraction
4. **Advanced Analysis**: 
   - Vibrato detection using autocorrelation
   - Harmonic content analysis
   - Dynamic range and accent detection
5. **Performance Scoring**: Multi-dimensional evaluation with confidence metrics

### Performance Considerations

- Maximum audio file size: 100MB
- Concurrent executions: Limited to 10 instances
- Processing time varies by audio length and complexity
- Results cached in Firestore for subsequent access

## Error Handling

The service implements comprehensive error handling:

1. **Validation Errors**: Invalid request parameters
2. **Storage Errors**: File not found, access denied, format unsupported
3. **Analysis Errors**: Audio processing failures, insufficient data
4. **Database Errors**: Firestore write failures, query timeouts
5. **Timeout Errors**: Long-running analysis timeout protection

## Monitoring and Logging

- Structured JSON logging for all operations
- Request correlation IDs for tracing
- Performance metrics for analysis duration
- Error rates and failure categorization

## Security

- CORS enabled for web client access
- Input validation for all endpoints
- Rate limiting via Firebase Functions quotas
- Audio file size restrictions

## Testing

The service includes:
- Unit tests for core analysis algorithms
- Integration tests for Firebase services
- Mock audio data for consistent testing
- Performance benchmarks

## License

MIT License