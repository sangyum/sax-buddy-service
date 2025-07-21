# Development Practice

1. Act as a pairing partner for the user. Do not commit anything until explicit consent is given
2. Use test-driven development (TDD) practice:
   * Write a failing test
   * Implement by writing minimal amount of code to make the test pass
   * Refactor
3. Update this document with up-to-date project structures and architecture
4. Run unit tests after every change: `npm run test`
5. Do not mark task completed unless all unit tests are passing
6. Run linting after changes: `npm run lint`

# Architecture

## Framework
* **Firebase Functions** - Serverless HTTP-triggered functions for audio analysis
* **TypeScript** - Type-safe development with strict configuration
* **Essentia.js v0.1.3** - WebAssembly DSP library for real-time audio analysis
* **Firebase Admin SDK** - Database and storage operations
* **Jest** - Unit testing framework with TypeScript support

## Model Configuration
* **Runtime**: Node.js 18+ with 2GB memory allocation for audio processing
* **Timeout**: 540 seconds (9 minutes) for complex audio analysis
* **Region**: us-central1 for optimal performance
* **Concurrency**: Limited to 10 instances for resource management

## Project Structure
```
src/
├── index.ts                    # Firebase Functions entry point
├── services/
│   ├── AssessmentProcessor.ts   # Main assessment orchestration
│   ├── FirestoreService.ts     # Database operations
│   ├── SaxophoneAudioAnalyzer.ts # Core audio analysis engine
│   ├── StorageService.ts       # Firebase Storage operations
│   └── analysis/               # Specialized analyzers
│       ├── AudioUtils.ts       # Audio processing utilities
│       ├── EssentiaProcessor.ts # Essentia.js integration
│       ├── MusicalExpressionAnalyzer.ts
│       ├── PerformanceConsistencyAnalyzer.ts
│       ├── PitchIntonationAnalyzer.ts
│       ├── SaxophoneConstants.ts # Musical constants and thresholds
│       ├── TechnicalExecutionAnalyzer.ts
│       ├── TimingRhythmAnalyzer.ts
│       └── ToneQualityTimbreAnalyzer.ts
├── types/
│   ├── analysis.ts             # Analysis result types
│   ├── analyzer.ts             # Analyzer interface definitions
│   ├── api.ts                  # API request/response types
│   ├── assessment.ts           # Assessment session types
│   ├── essentia.d.ts           # Essentia.js type definitions
│   └── index.ts                # Type exports
├── utils/
│   └── logger.ts               # Structured logging utilities
└── tests/
    └── SaxophoneAudioAnalyzer.test.ts # Core analyzer tests
```

## Services
* **SaxophoneAudioAnalyzer**: Core audio analysis engine
  * Multi-dimensional analysis across 6 categories
  * Real-time DSP using Essentia.js algorithms
  * Comprehensive scoring with confidence metrics
  
* **AssessmentProcessor**: Assessment session orchestration
  * Batch processing of exercise recordings
  * Error handling and recovery
  * Result aggregation and summary generation

* **FirestoreService**: Database operations
  * Assessment session management
  * Analysis result persistence
  * Query optimization for large datasets

* **StorageService**: Firebase Storage integration
  * Audio file download and validation
  * Format support (WAV 16-bit PCM primary)
  * File size limits (max 100MB)

## Analysis Categories
1. **Pitch and Intonation**: YinFFT algorithm, cent-level accuracy
2. **Timing and Rhythm**: Metronome deviation, subdivision analysis
3. **Tone Quality and Timbre**: Spectral analysis, vibrato detection
4. **Technical Execution**: Articulation, finger technique, breath management
5. **Musical Expression**: Phrasing, dynamic contour, style evaluation
6. **Performance Consistency**: Error patterns, recovery analysis

## Testing Strategy
* **Unit Tests**: Core analysis algorithms with mock audio data
* **Integration Tests**: Firebase services with emulators
* **Performance Tests**: Processing time benchmarks
* **Coverage**: Minimum 80% code coverage requirement

## Development Commands
* `npm run build` - Compile TypeScript
* `npm run build:watch` - Watch mode compilation
* `npm run serve` - Start Firebase emulators
* `npm run test` - Run unit tests
* `npm run test:watch` - Watch mode testing
* `npm run lint` - Run ESLint
* `npm run lint:fix` - Auto-fix linting issues
* `npm run deploy` - Deploy to Firebase

## Error Handling
* Comprehensive validation for all inputs
* Graceful degradation for partial analysis failures
* Structured error logging with correlation IDs
* Timeout protection for long-running operations

## Performance Considerations
* Frame-based audio processing (2048 samples, 512 hop)
* Memory-efficient streaming for large files
* Concurrent execution limits
* Result caching in Firestore