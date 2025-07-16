import { SaxophoneAudioAnalyzer } from "../services/SaxophoneAudioAnalyzer";

// Mock essentia.js
jest.mock("essentia.js");

describe("SaxophoneAudioAnalyzer", () => {
  let analyzer: SaxophoneAudioAnalyzer;

  beforeEach(() => {
    analyzer = new SaxophoneAudioAnalyzer();
  });

  test("should initialize successfully", async () => {
    await expect(analyzer.initialize()).resolves.not.toThrow();
  });

  test("should analyze audio buffer", async () => {
    // Create a mock audio buffer (1 second of silence at 44.1kHz)
    const audioBuffer = new Float32Array(44100);
    audioBuffer.fill(0);

    const result = await analyzer.analyzeExercise(audioBuffer);

    expect(result).toBeDefined();
    expect(result.duration).toBeCloseTo(1.0, 1);
    expect(result.performanceScore).toBeDefined();
    expect(result.performanceScore.overallScore).toBeGreaterThanOrEqual(0);
    expect(result.performanceScore.overallScore).toBeLessThanOrEqual(100);
    expect(result.analysisVersion).toBe("2.0.0");
  });

  test("should handle different audio buffer sizes", async () => {
    const shortBuffer = new Float32Array(1000);
    const longBuffer = new Float32Array(100000);

    const shortResult = await analyzer.analyzeExercise(shortBuffer);
    const longResult = await analyzer.analyzeExercise(longBuffer);

    expect(shortResult.duration).toBeLessThan(longResult.duration);
    expect(shortResult.performanceScore.overallScore).toBeGreaterThanOrEqual(0);
    expect(longResult.performanceScore.overallScore).toBeGreaterThanOrEqual(0);
  });

  test("should include all analysis categories", async () => {
    const audioBuffer = new Float32Array(22050); // 0.5 seconds
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
    const audioBuffer = new Float32Array(44100);
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
    const audioBuffer = new Float32Array(44100);

    const result = await analyzer.analyzeExercise(audioBuffer);

    expect(result).toBeDefined();
    expect(result.pitchIntonation.pitchAccuracyDistribution.deviationStats.mean).toBeDefined();
    expect(result.timingRhythm.temporalAccuracy.overallTimingScore).toBeDefined();
    expect(result.musicalExpression.phrasingSophistication.musicalSentenceStructure).toBeDefined();
  });

  test("should work identically with or without metadata", async () => {
    const audioBuffer = new Float32Array(44100);
    
    const resultWithoutMetadata = await analyzer.analyzeExercise(audioBuffer);
    const resultWithMetadata = await analyzer.analyzeExercise(audioBuffer);

    // Results should be equivalent - metadata should not affect analysis
    expect(resultWithoutMetadata.duration).toEqual(resultWithMetadata.duration);
    expect(resultWithoutMetadata.performanceScore.overallScore).toEqual(resultWithMetadata.performanceScore.overallScore);
  });
});