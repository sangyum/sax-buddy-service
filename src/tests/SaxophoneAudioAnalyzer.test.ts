import { SaxophoneAudioAnalyzer } from "../services/SaxophoneAudioAnalyzer";

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

  test("should handle exercise metadata", async () => {
    const audioBuffer = new Float32Array(44100);
    const metadata = {
      exerciseType: "improvisation" as const,
      targetKey: "Bb",
      targetTempo: 120,
      difficulty: "intermediate" as const,
      musicalStyle: "jazz" as const,
      hasBackingTrack: true,
      expectedDuration: 60,
      technicalFocus: ["scales", "arpeggios"]
    };

    const result = await analyzer.analyzeExercise(audioBuffer, metadata);

    expect(result).toBeDefined();
    expect(result.musicalExpression.improvisationalCoherence.overallImprovisationScore)
      .toBeGreaterThan(0.5); // Should be higher for improvisation exercises
  });
});