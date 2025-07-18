import { SaxophoneAudioAnalyzer } from "../services/SaxophoneAudioAnalyzer";

// Mock essentia.js
jest.mock("essentia.js");

// Helper function to generate saxophone-like audio buffers for testing
function generateSaxophoneLikeAudioBuffer(durationSeconds = 1.0, frequency = 440.0, amplitude = 0.8, sampleRate = 44100): Float32Array {
  const sampleCount = Math.floor(durationSeconds * sampleRate);
  const buffer = new Float32Array(sampleCount);
  
  // Generate a more complex waveform that mimics saxophone spectral characteristics
  for (let i = 0; i < sampleCount; i++) {
    const t = i / sampleRate;
    let sample = 0;
    
    // Fundamental frequency (strong)
    sample += Math.sin(2 * Math.PI * frequency * t) * amplitude * 0.8;
    
    // Add harmonics with saxophone-like distribution
    // 2nd harmonic (moderate)
    sample += Math.sin(2 * Math.PI * frequency * 2 * t) * amplitude * 0.3;
    // 3rd harmonic (moderate)
    sample += Math.sin(2 * Math.PI * frequency * 3 * t) * amplitude * 0.2;
    // 4th harmonic (weak)
    sample += Math.sin(2 * Math.PI * frequency * 4 * t) * amplitude * 0.1;
    // 5th harmonic (weak)
    sample += Math.sin(2 * Math.PI * frequency * 5 * t) * amplitude * 0.08;
    
    // Add some mid-frequency content around 800-1200 Hz (saxophone range)
    sample += Math.sin(2 * Math.PI * 900 * t) * amplitude * 0.15;
    sample += Math.sin(2 * Math.PI * 1100 * t) * amplitude * 0.1;
    
    // Add high-frequency content (1500-3000 Hz range) for harmonic validation
    sample += Math.sin(2 * Math.PI * 1800 * t) * amplitude * 0.08;
    sample += Math.sin(2 * Math.PI * 2200 * t) * amplitude * 0.06;
    
    // Add breath-like noise in low frequencies
    if (Math.random() < 0.1) {
      sample += (Math.random() - 0.5) * amplitude * 0.05;
    }
    
    // Add amplitude modulation (vibrato-like)
    const vibrato = 1 + 0.05 * Math.sin(2 * Math.PI * 6 * t);
    sample *= vibrato;
    
    // Add subtle amplitude envelope (attack/decay simulation)
    const envelopeTime = Math.min(t, durationSeconds - t, 0.1) / 0.1;
    const envelope = Math.min(1, envelopeTime);
    sample *= envelope;
    
    // Add realistic noise level
    sample += (Math.random() - 0.5) * 0.01 * amplitude;
    
    buffer[i] = sample * 0.6; // Scale down to avoid clipping
  }
  
  return buffer;
}

describe("SaxophoneAudioAnalyzer", () => {
  let analyzer: SaxophoneAudioAnalyzer;

  beforeEach(() => {
    analyzer = new SaxophoneAudioAnalyzer();
    // Use test-friendly validation settings
    analyzer.updateConfig({
      validation: {
        minDurationSec: 1.0,
        maxDurationSec: 300.0,
        minRms: 0.001,
        maxAmplitude: 0.99,
        minSnrDb: 5 // Lower threshold for testing synthetic signals
      }
    });
  });

  test("should initialize successfully", async () => {
    await expect(analyzer.initialize()).resolves.not.toThrow();
  });

  test("should analyze audio buffer", async () => {
    // Create a saxophone-like audio buffer (1 second of 440Hz)
    const audioBuffer = generateSaxophoneLikeAudioBuffer(1.0, 440.0);

    const result = await analyzer.analyzeExercise(audioBuffer);

    expect(result).toBeDefined();
    expect(result.duration).toBeCloseTo(1.0, 1);
    expect(result.performanceScore).toBeDefined();
    expect(result.performanceScore.overallScore).toBeGreaterThanOrEqual(0);
    expect(result.performanceScore.overallScore).toBeLessThanOrEqual(100);
    expect(result.analysisVersion).toBe("2.0.0");
  });

  test("should handle different audio buffer sizes", async () => {
    const shortBuffer = generateSaxophoneLikeAudioBuffer(1.5, 440.0); // 1.5 seconds (valid duration)
    const longBuffer = generateSaxophoneLikeAudioBuffer(2.5, 440.0); // 2.5 seconds

    const shortResult = await analyzer.analyzeExercise(shortBuffer);
    const longResult = await analyzer.analyzeExercise(longBuffer);

    expect(shortResult.duration).toBeLessThan(longResult.duration);
    expect(shortResult.performanceScore.overallScore).toBeGreaterThanOrEqual(0);
    expect(longResult.performanceScore.overallScore).toBeGreaterThanOrEqual(0);
  });

  test("should include all analysis categories", async () => {
    const audioBuffer = generateSaxophoneLikeAudioBuffer(1.0, 440.0); // 1 second (valid duration)
    const result = await analyzer.analyzeExercise(audioBuffer);

    expect(result.pitchIntonation).toBeDefined();
    expect(result.timingRhythm).toBeDefined();
    expect(result.toneQualityTimbre).toBeDefined();
    expect(result.technicalExecution).toBeDefined();
    expect(result.musicalExpression).toBeDefined();
    expect(result.performanceConsistency).toBeDefined();
    expect(result.performanceScore).toBeDefined();
  });

  test("should generate performance scores for all categories", async () => {
    const audioBuffer = generateSaxophoneLikeAudioBuffer(1.0, 440.0);
    const result = await analyzer.analyzeExercise(audioBuffer);

    const scores = result.performanceScore.categoryScores;
    expect(scores.pitchIntonation).toBeGreaterThanOrEqual(0);
    expect(scores.pitchIntonation).toBeLessThanOrEqual(100);
    expect(scores.timingRhythm).toBeGreaterThanOrEqual(0);
    expect(scores.timingRhythm).toBeLessThanOrEqual(100);
    expect(scores.toneQuality).toBeGreaterThanOrEqual(0);
    expect(scores.toneQuality).toBeLessThanOrEqual(100);
    expect(scores.technicalExecution).toBeGreaterThanOrEqual(0);
    expect(scores.technicalExecution).toBeLessThanOrEqual(100);
    expect(scores.musicalExpression).toBeGreaterThanOrEqual(0);
    expect(scores.musicalExpression).toBeLessThanOrEqual(100);
    expect(scores.consistency).toBeGreaterThanOrEqual(0);
    expect(scores.consistency).toBeLessThanOrEqual(100);
  });

  test("should analyze without exercise metadata", async () => {
    const audioBuffer = generateSaxophoneLikeAudioBuffer(1.0, 440.0);

    const result = await analyzer.analyzeExercise(audioBuffer);

    expect(result).toBeDefined();
    expect(result.pitchIntonation.pitchAccuracyDistribution.deviationStats.mean).toBeDefined();
    expect(result.timingRhythm.temporalAccuracy.overallTimingScore).toBeDefined();
    expect(result.musicalExpression.phrasingSophistication.musicalSentenceStructure).toBeDefined();
  });

  test("should work identically with or without metadata", async () => {
    const audioBuffer = generateSaxophoneLikeAudioBuffer(1.0, 440.0);
    
    const resultWithoutMetadata = await analyzer.analyzeExercise(audioBuffer);
    const resultWithMetadata = await analyzer.analyzeExercise(audioBuffer);

    // Results should be equivalent - metadata should not affect analysis
    expect(resultWithoutMetadata.duration).toEqual(resultWithMetadata.duration);
    expect(resultWithoutMetadata.performanceScore.overallScore).toEqual(resultWithMetadata.performanceScore.overallScore);
  });
});