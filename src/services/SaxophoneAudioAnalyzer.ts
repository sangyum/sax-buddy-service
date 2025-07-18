import { 
  ExtendedAudioAnalysis, 
  PitchIntonationAnalysis,
  TimingRhythmAnalysis,
  ToneQualityTimbreAnalysis,
  TechnicalExecutionAnalysis,
  MusicalExpressionAnalysis,
  PerformanceConsistencyAnalysis,
  PerformanceScore,
  BasicAnalysis,
  EssentiaAnalysisResult,
} from "../types";
import { Logger } from "../utils/logger";

// Import Essentia.js
import * as EssentiaJS from "essentia.js";
const { Essentia, EssentiaWASM } = EssentiaJS;
import { AudioUtils } from "./analysis/AudioUtils";
import { EssentiaProcessor, EssentiaConfig } from "./analysis/EssentiaProcessor";
import { PitchIntonationAnalyzer, PitchAnalysisConfig } from "./analysis/PitchIntonationAnalyzer";
import { TimingRhythmAnalyzer } from "./analysis/TimingRhythmAnalyzer";
import { ToneQualityTimbreAnalyzer, VibratoConfig } from "./analysis/ToneQualityTimbreAnalyzer";
import { TechnicalExecutionAnalyzer  } from "./analysis/TechnicalExecutionAnalyzer";
import { MusicalExpressionAnalyzer, MusicalExpressionConfig } from "./analysis/MusicalExpressionAnalyzer";
import { PerformanceConsistencyAnalyzer } from "./analysis/PerformanceConsistencyAnalyzer";
import { SAXOPHONE_CONFIG, isValidSaxophoneFrequency, detectSaxophoneType, getSaxophoneRange, isAltissimoRange } from "./analysis/SaxophoneConstants";

interface AnalysisConfig {
  sampleRate: number;
  frameSize: number;
  hopSize: number;
  pitchTracking: PitchAnalysisConfig;
  vibrato: VibratoConfig;
  onset: {
    threshold: number;
    minSilenceLength: number;
  };
  validation: {
    minDurationSec: number;
    maxDurationSec: number;
    minRms: number;
    maxAmplitude: number;
    minSnrDb: number;
  };
}

export class SaxophoneAudioAnalyzer {
  private logger = new Logger("SaxophoneAudioAnalyzer");
  private isInitialized = false;
  private essentia: EssentiaJS.Essentia | null = null;
  private essentiaWasm: EssentiaJS.EssentiaWASM | null = null;
  private essentiaProcessor: EssentiaProcessor | null = null;
  private pitchIntonationAnalyzer: PitchIntonationAnalyzer | null = null;
  private timingRhythmAnalyzer: TimingRhythmAnalyzer | null = null;
  private toneQualityTimbreAnalyzer: ToneQualityTimbreAnalyzer | null = null;
  private technicalExecutionAnalyzer: TechnicalExecutionAnalyzer | null = null;
  private musicalExpressionAnalyzer: MusicalExpressionAnalyzer | null = null;
  private performanceConsistencyAnalyzer: PerformanceConsistencyAnalyzer | null = null;

  private config: AnalysisConfig = {
    sampleRate: SAXOPHONE_CONFIG.SAMPLE_RATE.PREFERRED,
    frameSize: SAXOPHONE_CONFIG.ANALYSIS_WINDOWS.PITCH_FRAME_SIZE,
    hopSize: SAXOPHONE_CONFIG.ANALYSIS_WINDOWS.PITCH_HOP_SIZE,
    pitchTracking: {
      smoothingFactor: SAXOPHONE_CONFIG.PITCH_ANALYSIS.SMOOTHING_FACTOR,
      medianFilterSize: SAXOPHONE_CONFIG.PITCH_ANALYSIS.MEDIAN_FILTER_SIZE,
      confidenceThreshold: SAXOPHONE_CONFIG.PITCH_ANALYSIS.CONFIDENCE_THRESHOLD,
      octaveErrorThreshold: SAXOPHONE_CONFIG.PITCH_ANALYSIS.OCTAVE_ERROR_THRESHOLD
    },
    vibrato: {
      minRate: SAXOPHONE_CONFIG.VIBRATO.MIN_RATE,
      maxRate: SAXOPHONE_CONFIG.VIBRATO.MAX_RATE,
      minDepth: SAXOPHONE_CONFIG.VIBRATO.MIN_DEPTH,
      qualityThreshold: SAXOPHONE_CONFIG.VIBRATO.QUALITY_THRESHOLD
    },
    onset: {
      threshold: SAXOPHONE_CONFIG.ONSET_DETECTION.THRESHOLD,
      minSilenceLength: SAXOPHONE_CONFIG.ONSET_DETECTION.MIN_SILENCE_LENGTH
    },
    validation: {
      minDurationSec: SAXOPHONE_CONFIG.VALIDATION.MIN_DURATION,
      maxDurationSec: SAXOPHONE_CONFIG.VALIDATION.MAX_DURATION,
      minRms: SAXOPHONE_CONFIG.VALIDATION.MIN_RMS,
      maxAmplitude: SAXOPHONE_CONFIG.VALIDATION.MAX_AMPLITUDE,
      minSnrDb: SAXOPHONE_CONFIG.VALIDATION.MIN_SNR_DB
    }
  };

  private get sampleRate(): number {
    return this.config.sampleRate;
  }

  updateConfig(partialConfig: Partial<AnalysisConfig>): void {
    this.config = {
      ...this.config,
      ...partialConfig,
      pitchTracking: {
        ...this.config.pitchTracking,
        ...partialConfig.pitchTracking
      },
      vibrato: {
        ...this.config.vibrato,
        ...partialConfig.vibrato
      },
      onset: {
        ...this.config.onset,
        ...partialConfig.onset
      },
      validation: {
        ...this.config.validation,
        ...partialConfig.validation
      }
    };
  }

  getConfig(): AnalysisConfig {
    return { ...this.config };
  }

  async initialize(): Promise<void> {
    if (!this.isInitialized) {
      this.logger.info("Initializing audio analyzer");
      
      try {
        // Initialize Essentia.js WASM backend
        this.essentiaWasm = new EssentiaWASM();
        await this.essentiaWasm.initialize();

        // Create Essentia.js instance
        this.essentia = new Essentia(this.essentiaWasm);
        this.essentiaProcessor = new EssentiaProcessor(this.getConfig() as EssentiaConfig);
        
        // Initialize the EssentiaProcessor with the Essentia instance
        await this.essentiaProcessor.initialize();
        this.pitchIntonationAnalyzer = new PitchIntonationAnalyzer(this.config.pitchTracking)
        this.timingRhythmAnalyzer = new TimingRhythmAnalyzer();
        this.toneQualityTimbreAnalyzer = new ToneQualityTimbreAnalyzer(this.config.vibrato);
        this.technicalExecutionAnalyzer = new TechnicalExecutionAnalyzer();
        this.musicalExpressionAnalyzer = new MusicalExpressionAnalyzer(this.getConfig() as MusicalExpressionConfig);
        this.performanceConsistencyAnalyzer = new PerformanceConsistencyAnalyzer();

        this.isInitialized = true;
        this.logger.info("Audio analyzer initialized", {
          version: this.essentia?.version,
          algorithms: this.essentia?.algorithmNames?.split(",").length || 0
        });
      } catch (error) {
        this.logger.error("Failed to initialize Essentia.js", {
          error: error instanceof Error ? error.message : "Unknown error"
        });
        throw new Error(`Failed to initialize audio analyzer: ${error}`);
      }
    }
  }

  async analyzeExercise(
    audioBuffer: Float32Array
  ): Promise<ExtendedAudioAnalysis> {
    await this.initialize();
    
    // Validate input audio buffer
    const validationResult = this.validateAudioBuffer(audioBuffer);
    if (!validationResult.isValid) {
      throw new Error(`Invalid audio buffer: ${validationResult.error}`);
    }
    
    this.logger.info("Starting comprehensive audio analysis", {
      bufferLength: audioBuffer.length,
      duration: audioBuffer.length / this.sampleRate,
      snr: validationResult.snr,
      rms: validationResult.rms
    });

    try {
      if (!this.essentiaProcessor) {
        throw new Error("EssentiaProcess not initialized");
      }

      if (!this.pitchIntonationAnalyzer) {
        throw new Error("PitchIntonationAnalyzer not initialized");
      }

      if (!this.timingRhythmAnalyzer) {
        throw new Error("TimingRhythmAnalyzer not initialized");
      }

      if (!this.toneQualityTimbreAnalyzer) {
        throw new Error("ToneQualityTimbreAnalyzer not initialized");
      }

      if (!this.technicalExecutionAnalyzer) {
        throw new Error("TechnicalExecutionAnalyzer not initialized");
      }

      if (!this.musicalExpressionAnalyzer) {
        throw new Error("MusicalExpressionAnalyzer not initialized");
      }

      if (!this.performanceConsistencyAnalyzer) {
        throw new Error("PerformanceConsistencyAnalyzer not initialized");
      }

      // Perform Essentia analysis once
      const analysisResult = await this.essentiaProcessor.performEssentiaAnalysis(audioBuffer);
      
      // Perform basic audio analysis using the result
      const basicAnalysis = this.performBasicAnalysisFromResult(analysisResult);

      const pitchIntonation = await this.pitchIntonationAnalyzer.analyze(analysisResult);
      const timingRhythm = await this.timingRhythmAnalyzer.analyze(analysisResult);
      const toneQualityTimbre = await this.toneQualityTimbreAnalyzer.analyze(analysisResult);
      const technicalExecution = await this.technicalExecutionAnalyzer.analyze(analysisResult);
      const musicalExpression = await this.musicalExpressionAnalyzer.analyze(analysisResult);
      const performanceConsistency = await this.performanceConsistencyAnalyzer.analyze(analysisResult);

      // Perform extended analysis using the same result
      const extendedAnalysis: ExtendedAudioAnalysis = {
        ...basicAnalysis,
        pitchIntonation,
        timingRhythm,
        toneQualityTimbre,
        technicalExecution,
        musicalExpression,
        performanceConsistency,
        performanceScore: {
          overallScore: 0,
          categoryScores: {
            pitchIntonation: 0,
            timingRhythm: 0,
            toneQuality: 0,
            technicalExecution: 0,
            musicalExpression: 0,
            consistency: 0,
          },
          strengthAreas: [],
          improvementAreas: [],
          specificFeedback: [],
          nextLevelRecommendations: [],
        },
        duration: audioBuffer.length / this.sampleRate,
        processedAt: new Date(),
        analysisVersion: "2.0.0",
        confidenceScores: {}
      };

      // Calculate performance score based on all analyses
      extendedAnalysis.performanceScore = this.calculatePerformanceScore(extendedAnalysis);
      extendedAnalysis.confidenceScores = this.calculateConfidenceScores(extendedAnalysis);

      this.logger.info("Audio analysis completed", {
        overallScore: extendedAnalysis.performanceScore.overallScore,
        duration: extendedAnalysis.duration
      });

      return extendedAnalysis;

    } catch (error) {
      this.logger.error("Audio analysis failed", {
        error: error instanceof Error ? error.message : "Unknown error",
        bufferLength: audioBuffer.length
      });
      throw error;
    }
  }

  private validateAudioBuffer(audioBuffer: Float32Array): {
    isValid: boolean;
    error?: string;
    snr?: number;
    rms?: number;
  } {
    // Check if buffer exists and is not empty
    if (!audioBuffer || audioBuffer.length === 0) {
      return { isValid: false, error: "Audio buffer is empty or null" };
    }

    // Check minimum duration
    const duration = audioBuffer.length / this.sampleRate;
    if (duration < this.config.validation.minDurationSec) {
      return { isValid: false, error: `Audio too short: ${duration.toFixed(2)}s (minimum ${this.config.validation.minDurationSec}s required)` };
    }

    // Check maximum duration
    if (duration > this.config.validation.maxDurationSec) {
      return { isValid: false, error: `Audio too long: ${duration.toFixed(2)}s (maximum ${this.config.validation.maxDurationSec}s)` };
    }

    // Check for NaN or infinite values
    for (let i = 0; i < audioBuffer.length; i++) {
      const sample = audioBuffer[i];
      if (sample === undefined || !isFinite(sample)) {
        return { isValid: false, error: `Invalid sample at index ${i}: ${sample}` };
      }
    }

    // Calculate RMS level
    const rms = Math.sqrt(
      audioBuffer.reduce((sum, sample) => sum + sample * sample, 0) / audioBuffer.length
    );

    // Check if signal is too quiet (saxophone-specific threshold)
    if (rms < this.config.validation.minRms) {
      return { isValid: false, error: `Signal too quiet for saxophone analysis: RMS = ${rms.toFixed(6)} (minimum ${this.config.validation.minRms})` };
    }

    // Check if signal is clipping (saxophone-specific threshold)
    let maxAmplitude = 0;
    for (let i = 0; i < audioBuffer.length; i++) {
      const sample = audioBuffer[i];
      if (sample !== undefined) {
        const absValue = Math.abs(sample);
        if (absValue > maxAmplitude) {
          maxAmplitude = absValue;
        }
      }
    }
    if (maxAmplitude > this.config.validation.maxAmplitude) {
      return { isValid: false, error: `Signal clipping detected (saxophone analysis): max amplitude = ${maxAmplitude.toFixed(3)} (maximum ${this.config.validation.maxAmplitude})` };
    }

    // Estimate SNR (saxophone-specific noise floor estimation accounting for breath patterns)
    const sortedSamples = [...audioBuffer].map(Math.abs).sort((a, b) => a - b);
    
    // Use breath noise threshold for more accurate saxophone SNR calculation
    const breathNoiseThreshold = SAXOPHONE_CONFIG.BREATH_MANAGEMENT.BREATH_NOISE_THRESHOLD;
    const breathNoiseSamples = sortedSamples.filter(sample => sample < breathNoiseThreshold);
    
    // Calculate noise floor using breath-noise-filtered samples or fallback to percentile method
    const noiseFloor = breathNoiseSamples.length > 0 
      ? breathNoiseSamples.reduce((sum, sample) => sum + sample, 0) / breathNoiseSamples.length
      : sortedSamples[Math.floor(sortedSamples.length * 0.15)] || 1e-10; // 15th percentile fallback
    
    const snr = 20 * Math.log10(rms / (noiseFloor + 1e-10));

    // Check SNR threshold (lower for saxophone due to breath noise)
    if (snr < this.config.validation.minSnrDb) {
      return { isValid: false, error: `Poor signal quality for saxophone analysis: SNR = ${snr.toFixed(1)}dB (minimum ${this.config.validation.minSnrDb}dB)` };
    }

    return { isValid: true, snr, rms };
  }

  private performBasicAnalysisFromResult(analysisResult: EssentiaAnalysisResult): BasicAnalysis {
    return {
      tempo: {
        bpm: analysisResult.tempo,
        confidence: analysisResult.confidence
      },
      rhythm: {
        beats: analysisResult.beats,
        onsets: analysisResult.onsets
      },
      pitch: {
        fundamentalFreq: AudioUtils.calculateMeanFrequency(analysisResult.pitchTrack),
        melody: analysisResult.pitchTrack
      },
      harmony: {
        key: this.estimateKey(analysisResult.pitchTrack),
        chords: this.estimateChords(analysisResult.pitchTrack),
        chroma: this.calculateChromaVector(analysisResult.pitchTrack)
      },
      spectral: {
        mfcc: analysisResult.mfcc,
        centroid: AudioUtils.calculateMean(analysisResult.spectralCentroid),
        rolloff: AudioUtils.calculateMean(analysisResult.spectralRolloff)
      },
      quality: {
        snr: 0.8, // Default value since we don't have audioBuffer
        clarity: this.calculateSaxophoneHarmonicClarity(analysisResult.harmonics, analysisResult.pitchTrack)
      }
    };
  }

  private calculatePerformanceScore(analysis: ExtendedAudioAnalysis): PerformanceScore {
    // Calculate category scores
    const categoryScores = {
      pitchIntonation: this.scorePitchIntonation(analysis.pitchIntonation),
      timingRhythm: this.scoreTimingRhythm(analysis.timingRhythm),
      toneQuality: this.scoreToneQuality(analysis.toneQualityTimbre),
      technicalExecution: this.scoreTechnicalExecution(analysis.technicalExecution),
      musicalExpression: this.scoreMusicalExpression(analysis.musicalExpression),
      consistency: this.scoreConsistency(analysis.performanceConsistency)
    };

    // Calculate overall score as weighted average (saxophone-specific weights)
    const weights = {
      pitchIntonation: 0.25,  // More important for saxophone
      timingRhythm: 0.20,
      toneQuality: 0.20,     // More important for saxophone
      technicalExecution: 0.15,
      musicalExpression: 0.15,
      consistency: 0.05      // Less critical for individual performance
    };

    const overallScore = Object.entries(categoryScores).reduce((sum, [category, score]) => {
      return sum + score * weights[category as keyof typeof weights];
    }, 0);

    return {
      overallScore: Math.round(overallScore),
      categoryScores,
      strengthAreas: this.identifyStrengthAreas(categoryScores),
      improvementAreas: this.identifyImprovementAreas(categoryScores),
      specificFeedback: this.generateSpecificFeedback(analysis),
      nextLevelRecommendations: this.generateRecommendations(analysis)
    };
  }

  private scorePitchIntonation(analysis: PitchIntonationAnalysis): number {
    const accuracy = analysis.intervalPrecision.overallIntervalScore;
    const stability = analysis.intonationStability.sustainedNoteConsistency;
    return Math.round((accuracy * 0.6 + stability * 0.4) * 100);
  }

  private scoreTimingRhythm(analysis: TimingRhythmAnalysis): number {
    const timing = analysis.temporalAccuracy.overallTimingScore;
    const rhythm = analysis.rhythmicSubdivisionPrecision.complexPatternSuccess;
    const groove = analysis.grooveConsistency.grooveStability;
    return Math.round((timing * 0.4 + rhythm * 0.3 + groove * 0.3) * 100);
  }

  private scoreToneQuality(analysis: ToneQualityTimbreAnalysis): number {
    const richness = analysis.harmonicContentAnalysis.harmonicRichness;
    const consistency = analysis.timbralConsistency.overallTimbralUniformity;
    const dynamics = analysis.dynamicRangeUtilization.dynamicVariationEffectiveness;
    return Math.round((richness * 0.4 + consistency * 0.4 + dynamics * 0.2) * 100);
  }

  private scoreTechnicalExecution(analysis: TechnicalExecutionAnalysis): number {
    const articulation = analysis.articulationClarity.tonguingPrecision;
    const fingers = analysis.fingerTechniqueEfficiency.keyTransitionSmoothness;
    const breath = analysis.breathManagementIndicators.breathSupportConsistency;
    return Math.round((articulation * 0.4 + fingers * 0.4 + breath * 0.2) * 100);
  }

  private scoreMusicalExpression(analysis: MusicalExpressionAnalysis): number {
    const phrasing = analysis.phrasingSophistication.phraseShaping;
    const dynamics = analysis.dynamicContourComplexity.dynamicShapingScore;
    const style = analysis.stylisticAuthenticity.styleConsistency;
    return Math.round((phrasing * 0.4 + dynamics * 0.3 + style * 0.3) * 100);
  }

  private scoreConsistency(analysis: PerformanceConsistencyAnalysis): number {
    const consistency = analysis.endurancePatterns.consistencyThroughoutPerformance;
    const recovery = analysis.recoverySpeed.recoveryEffectiveness;
    const errorImpact = 1 - analysis.recoverySpeed.errorImpactOnSubsequentPerformance;
    return Math.round((consistency * 0.5 + recovery * 0.3 + errorImpact * 0.2) * 100);
  }

  private identifyStrengthAreas(scores: Record<string, number>): string[] {
    const entries = Object.entries(scores);
    const topScores = entries
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      .filter(([_, score]) => score >= 80)
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      .sort(([_, a], [__, b]) => b - a)
      .slice(0, 3)
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      .map(([category, _]) => this.formatCategoryName(category))
      .filter((name): name is string => name !== undefined);
    
    return topScores;
  }

  private identifyImprovementAreas(scores: Record<string, number>): string[] {
    const entries = Object.entries(scores);
    const lowScores = entries
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      .filter(([_, score]) => score < 75)
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      .sort(([_, a], [__, b]) => a - b)
      .slice(0, 3)
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      .map(([category, _]) => this.formatCategoryName(category))
      .filter((name): name is string => name !== undefined);
    
    return lowScores;
  }

  private formatCategoryName(category: string): string {
    return category.replace(/([A-Z])/g, " $1").trim().toLowerCase()
      .split(" ")
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  }

  private generateSpecificFeedback(analysis: ExtendedAudioAnalysis): string[] {
    const feedback: string[] = [];
    
    if (analysis.pitchIntonation.tuningDriftPatterns.driftDirection === "flat") {
      feedback.push("Watch for pitch dropping throughout the performance");
    }
    
    if (analysis.timingRhythm.temporalAccuracy.rushTendency > 0.05) {
      feedback.push("Tendency to rush - focus on steady tempo");
    }
    
    if (analysis.toneQualityTimbre.vibratoCharacteristics.vibratoControl < 0.8) {
      feedback.push("Work on vibrato control and consistency");
    }
    
    return feedback.slice(0, 5);
  }

  private generateRecommendations(analysis: ExtendedAudioAnalysis): string[] {
    const recommendations: string[] = [];
    
    if (analysis.performanceScore.categoryScores.pitchIntonation < 75) {
      recommendations.push("Practice with tuner for intonation improvement");
    }
    
    if (analysis.performanceScore.categoryScores.technicalExecution < 80) {
      recommendations.push("Focus on technical exercises and scales");
    }
    
    return recommendations.slice(0, 3);
  }

  private calculateConfidenceScores(analysis: ExtendedAudioAnalysis): Record<string, number> {
    // Calculate confidence scores based on actual analysis data quality and reliability
    
    // Pitch confidence based on pitch track quality and consistency
    const pitchConfidence = this.calculatePitchConfidence(analysis.pitchIntonation, analysis.pitch);
    
    // Timing confidence based on onset detection quality and rhythm consistency
    const timingConfidence = this.calculateTimingConfidence(analysis.timingRhythm, analysis.rhythm);
    
    // Tone quality confidence based on spectral analysis stability
    const toneQualityConfidence = this.calculateToneQualityConfidence(analysis.toneQualityTimbre, analysis.spectral);
    
    // Technical execution confidence based on analysis consistency
    const technicalConfidence = this.calculateTechnicalConfidence(analysis.technicalExecution);
    
    // Musical expression confidence based on feature extraction quality
    const musicalConfidence = this.calculateMusicalConfidence(analysis.musicalExpression);
    
    // Consistency confidence based on performance stability
    const consistencyConfidence = this.calculateConsistencyConfidence(analysis.performanceConsistency);
    
    return {
      pitchIntonation: pitchConfidence,
      timingRhythm: timingConfidence,
      toneQuality: toneQualityConfidence,
      technicalExecution: technicalConfidence,
      musicalExpression: musicalConfidence,
      consistency: consistencyConfidence
    };
  }

  // Helper methods for confidence score calculation
  private calculatePitchConfidence(pitchAnalysis: PitchIntonationAnalysis, pitchData: { fundamentalFreq: number; melody: number[] }): number {
    // Base confidence from pitch tracking algorithm - use melody length as indicator
    const pitchDataQuality = pitchData.melody.filter(p => p > 0).length / pitchData.melody.length;
    
    // Factor in analysis quality indicators
    const stabilityFactor = 1 - (pitchAnalysis.pitchAccuracyDistribution.deviationStats.std / 100);
    const consistencyFactor = 1 - Math.abs(pitchAnalysis.tuningDriftPatterns.driftRate);
    
    // Add saxophone register detection for confidence adjustment
    const registerConfidence = this.calculateSaxophoneRegisterConfidence(pitchData.melody);
    
    // Combine factors with weights (including register confidence)
    const confidence = (pitchDataQuality * 0.4) + (stabilityFactor * 0.25) + (consistencyFactor * 0.15) + (registerConfidence * 0.2);
    
    return Math.min(0.95, Math.max(0.5, confidence));
  }
  
  private calculateSaxophoneRegisterConfidence(melody: number[]): number {
    const validPitches = melody.filter(p => p > 0 && isValidSaxophoneFrequency(p));
    if (validPitches.length === 0) return 0.5;
    
    // Detect saxophone type and get its range
    const saxType = detectSaxophoneType(validPitches);
    const saxRange = getSaxophoneRange(saxType);
    
    // Categorize pitches by register
    const registerCounts = {
      low: 0,
      middle: 0,
      high: 0,
      altissimo: 0
    };
    
    validPitches.forEach(pitch => {
      if (isAltissimoRange(pitch, saxType)) {
        registerCounts.altissimo++;
      } else if (pitch >= saxRange.COMFORTABLE_HIGH) {
        registerCounts.high++;
      } else if (pitch >= saxRange.COMFORTABLE_LOW) {
        registerCounts.middle++;
      } else {
        registerCounts.low++;
      }
    });
    
    // Calculate confidence based on register distribution
    const totalPitches = validPitches.length;
    const middleRegisterRatio = registerCounts.middle / totalPitches;
    const altissimoRatio = registerCounts.altissimo / totalPitches;
    
    // Higher confidence for middle register (easiest to track)
    // Lower confidence for altissimo (harder to track accurately)
    let registerConfidence = 0.5;
    
    if (middleRegisterRatio > 0.6) {
      registerConfidence = 0.9; // High confidence for middle register dominated performance
    } else if (altissimoRatio > 0.3) {
      registerConfidence = 0.6; // Lower confidence for altissimo heavy performance
    } else {
      registerConfidence = 0.75; // Medium confidence for balanced range
    }
    
    return registerConfidence;
  }
  
  private calculateTimingConfidence(timingAnalysis: TimingRhythmAnalysis, rhythmData: { beats: number[]; onsets: number[] }): number {
    // Base confidence from timing analysis consistency
    const timingAccuracy = timingAnalysis.temporalAccuracy.overallTimingScore / 100;
    const rhythmStability = 1 - Math.abs(timingAnalysis.temporalAccuracy.rushTendency);
    
    // Factor in onset detection quality
    const onsetQuality = rhythmData.onsets.length > 0 ? Math.min(1, rhythmData.onsets.length / (rhythmData.beats.length * 2)) : 0.5;
    
    const confidence = (timingAccuracy * 0.5) + (rhythmStability * 0.3) + (onsetQuality * 0.2);
    
    return Math.min(0.95, Math.max(0.5, confidence));
  }
  
  private calculateToneQualityConfidence(toneAnalysis: ToneQualityTimbreAnalysis, spectralData: { mfcc: number[]; centroid: number; rolloff: number }): number {
    // Base confidence from spectral analysis stability
    const timbralStability = 1 - (toneAnalysis.vibratoCharacteristics.vibratoRate / 10); // Use vibrato as timbral indicator
    const harmonicQuality = toneAnalysis.harmonicContentAnalysis.harmonicRichness;
    
    // Factor in spectral data quality
    const spectralQuality = spectralData.mfcc.length > 0 ? Math.min(1, spectralData.mfcc.filter(m => !isNaN(m)).length / spectralData.mfcc.length) : 0.5;
    
    const confidence = (timbralStability * 0.4) + (harmonicQuality * 0.4) + (spectralQuality * 0.2);
    
    return Math.min(0.95, Math.max(0.5, confidence));
  }
  
  private calculateTechnicalConfidence(technicalAnalysis: TechnicalExecutionAnalysis): number {
    // Base confidence from technical execution consistency
    const articulationConfidence = technicalAnalysis.articulationClarity.tonguingPrecision;
    const fingeringConfidence = technicalAnalysis.fingerTechniqueEfficiency.fingeringAccuracy;
    const breathingConfidence = technicalAnalysis.breathManagementIndicators.breathSupportConsistency;
    
    const confidence = (articulationConfidence + fingeringConfidence + breathingConfidence) / 3;
    
    return Math.min(0.95, Math.max(0.5, confidence));
  }
  
  private calculateMusicalConfidence(musicalAnalysis: MusicalExpressionAnalysis): number {
    // Base confidence from musical expression consistency
    const phrasingConfidence = (musicalAnalysis.phrasingSophistication.musicalSentenceStructure + 
                              musicalAnalysis.phrasingSophistication.breathingLogic) / 2;
    const dynamicConfidence = musicalAnalysis.dynamicContourComplexity.dynamicShapingScore;
    const styleConfidence = musicalAnalysis.stylisticAuthenticity.styleConsistency;
    
    const confidence = (phrasingConfidence + dynamicConfidence + styleConfidence) / 3;
    
    return Math.min(0.95, Math.max(0.5, confidence));
  }
  
  private calculateConsistencyConfidence(consistencyAnalysis: PerformanceConsistencyAnalysis): number {
    // Base confidence from performance consistency metrics
    const errorRate = (consistencyAnalysis.errorFrequencyAndType.missedNotes + 
                      consistencyAnalysis.errorFrequencyAndType.crackedNotes +
                      consistencyAnalysis.errorFrequencyAndType.timingSlips) / 3;
    const recoveryEffectiveness = consistencyAnalysis.recoverySpeed.recoveryEffectiveness;
    const enduranceStability = consistencyAnalysis.endurancePatterns.consistencyThroughoutPerformance;
    
    // Lower error rate = higher confidence
    const errorConfidence = Math.max(0.3, 1 - (errorRate / 10));
    
    const confidence = (errorConfidence * 0.4) + (recoveryEffectiveness * 0.3) + (enduranceStability * 0.3);
    
    return Math.min(0.95, Math.max(0.5, confidence));
  }

  // Utility methods

  private calculateSaxophoneHarmonicClarity(harmonics: number[], pitchTrack: number[]): number {
    // Filter out zero harmonics
    const validHarmonics = harmonics.filter(h => h > 0);
    if (validHarmonics.length === 0) return 0.5;
    
    // Calculate basic harmonic clarity
    const basicClarity = AudioUtils.calculateMean(validHarmonics);
    
    // Apply saxophone-specific harmonic weighting based on pitch range
    const validPitches = pitchTrack.filter(p => p > 0 && isValidSaxophoneFrequency(p));
    if (validPitches.length === 0) return basicClarity;
    
    const avgPitch = validPitches.reduce((sum, p) => sum + p, 0) / validPitches.length;
    
    // Saxophone harmonic weighting factors
    let harmonicWeight = 1.0;
    
    // Low register (below 200 Hz) - richer harmonics expected
    if (avgPitch < 200) {
      harmonicWeight = 1.2;
    }
    // High register (above 800 Hz) - fewer harmonics expected
    else if (avgPitch > 800) {
      harmonicWeight = 0.8;
    }
    // Altissimo register (above 1200 Hz) - very few harmonics
    else if (avgPitch > 1200) {
      harmonicWeight = 0.6;
    }
    
    // Apply fundamental prominence weighting (saxophone has strong fundamental)
    const fundamentalProminence = SAXOPHONE_CONFIG.TONE_QUALITY.FUNDAMENTAL_PROMINENCE;
    const weightedClarity = basicClarity * harmonicWeight * fundamentalProminence;
    
    return Math.max(0, Math.min(1, weightedClarity));
  }

  private estimateKey(pitchTrack: number[]): string {
    const keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    
    // Filter out invalid pitch values using saxophone-specific frequency ranges
    const validPitches = pitchTrack.filter(pitch => pitch > 0 && isValidSaxophoneFrequency(pitch));
    
    if (validPitches.length === 0) {
      return "C"; // Default fallback
    }
    
    // Convert frequencies to pitch classes (0-11)
    const pitchClasses = validPitches.map(freq => {
      // Convert frequency to MIDI note number
      const midiNote = 12 * Math.log2(freq / 440) + 69;
      // Get pitch class (0-11)
      return Math.round(midiNote) % 12;
    });
    
    // Count occurrences of each pitch class
    const pitchClassCounts = new Array(12).fill(0);
    pitchClasses.forEach(pc => {
      if (pc >= 0 && pc < 12) {
        pitchClassCounts[pc]++;
      }
    });
    
    // Find the most common pitch class as the key center
    const mostCommonPitchClass = pitchClassCounts.indexOf(Math.max(...pitchClassCounts));
    
    return keys[mostCommonPitchClass] || "C";
  }

  private estimateChords(pitchTrack: number[]): string[] {
    // Filter out invalid pitch values using saxophone-specific frequency ranges
    const validPitches = pitchTrack.filter(pitch => pitch > 0 && isValidSaxophoneFrequency(pitch));
    
    if (validPitches.length === 0) {
      return ["Cmaj7"]; // Default fallback
    }
    
    // Divide the pitch track into segments for chord analysis
    const segmentSize = Math.max(1, Math.floor(validPitches.length / 4));
    const chords: string[] = [];
    
    for (let i = 0; i < validPitches.length; i += segmentSize) {
      const segment = validPitches.slice(i, i + segmentSize);
      const chord = this.analyzeChordInSegment(segment);
      chords.push(chord);
      
      // Limit to reasonable number of chords
      if (chords.length >= 8) break;
    }
    
    return chords.length > 0 ? chords : ["Cmaj7"];
  }

  private calculateChromaVector(pitchTrack: number[]): number[] {
    // Filter out invalid pitch values using saxophone-specific frequency ranges
    const validPitches = pitchTrack.filter(pitch => pitch > 0 && isValidSaxophoneFrequency(pitch));
    
    if (validPitches.length === 0) {
      return new Array(12).fill(0.083); // Equal distribution fallback
    }
    
    // Initialize chroma vector (12 pitch classes)
    const chromaVector = new Array(12).fill(0);
    
    // Convert frequencies to pitch classes and accumulate
    validPitches.forEach(freq => {
      // Convert frequency to MIDI note number
      const midiNote = 12 * Math.log2(freq / 440) + 69;
      // Get pitch class (0-11)
      const pitchClass = Math.round(midiNote) % 12;
      
      if (pitchClass >= 0 && pitchClass < 12) {
        chromaVector[pitchClass]++;
      }
    });
    
    // Normalize to get probability distribution
    const total = chromaVector.reduce((sum, count) => sum + count, 0);
    if (total > 0) {
      return chromaVector.map(count => count / total);
    }
    
    return new Array(12).fill(0.083); // Equal distribution fallback
  }
  
  private analyzeChordInSegment(pitchSegment: number[]): string {
    if (pitchSegment.length === 0) return "Cmaj7";
    
    // Convert frequencies to pitch classes
    const pitchClasses = pitchSegment.map(freq => {
      const midiNote = 12 * Math.log2(freq / 440) + 69;
      return Math.round(midiNote) % 12;
    });
    
    // Count pitch class occurrences
    const pitchClassCounts = new Array(12).fill(0);
    pitchClasses.forEach(pc => {
      if (pc >= 0 && pc < 12) {
        pitchClassCounts[pc]++;
      }
    });
    
    // Find the most prominent pitch classes
    const sortedPitchClasses = pitchClassCounts
      .map((count, index) => ({ pitchClass: index, count }))
      .filter(item => item.count > 0)
      .sort((a, b) => b.count - a.count);
    
    if (sortedPitchClasses.length === 0) return "Cmaj7";
    
    // Get the root pitch class
    const rootPitchClass = sortedPitchClasses[0]?.pitchClass ?? 0;
    
    // Map pitch classes to note names
    const noteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    const rootNote = noteNames[rootPitchClass];
    
    // Simple chord estimation based on prominent pitch classes
    const prominentPitchClasses = sortedPitchClasses.slice(0, 3).map(item => item.pitchClass);
    
    // Check for common chord patterns
    if (prominentPitchClasses.length >= 3) {
      // Sort for pattern matching
      const sortedPCs = prominentPitchClasses.sort((a, b) => a - b);
      const intervals = [];
      
      for (let i = 1; i < sortedPCs.length; i++) {
        const current = sortedPCs[i];
        const root = sortedPCs[0];
        if (current !== undefined && root !== undefined) {
          intervals.push((current - root + 12) % 12);
        }
      }
      
      // Major chord pattern (root, major third, perfect fifth)
      if (intervals.includes(4) && intervals.includes(7)) {
        return `${rootNote}maj7`;
      }
      // Minor chord pattern (root, minor third, perfect fifth)
      if (intervals.includes(3) && intervals.includes(7)) {
        return `${rootNote}m7`;
      }
      // Dominant chord pattern
      if (intervals.includes(4) && intervals.includes(10)) {
        return `${rootNote}7`;
      }
    }
    
    // Default to major 7th chord (common in saxophone repertoire)
    return `${rootNote}maj7`;
  }

}

