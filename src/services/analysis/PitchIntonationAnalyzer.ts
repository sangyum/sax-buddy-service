import { PitchIntonationAnalysis, EssentiaAnalysisResult } from "../../types";
import { AudioUtils } from "./AudioUtils";
import { 
  SAXOPHONE_CONFIG, 
  detectSaxophoneType, 
  getSaxophoneRange, 
  isValidSaxophoneFrequency, 
  isAltissimoRange 
} from "./SaxophoneConstants";

export interface PitchAnalysisConfig {
  smoothingFactor: number;
  medianFilterSize: number;
  confidenceThreshold: number;
  octaveErrorThreshold: number;
  pitchStabilityThreshold?: number;
  intonationTolerance?: number;
  vibratoPitchDeviation?: number;
}

export class PitchIntonationAnalyzer {
  private config: PitchAnalysisConfig;
  private saxophoneType: keyof typeof SAXOPHONE_CONFIG.FREQUENCY_RANGES;

  constructor(config: PitchAnalysisConfig) {
    this.config = {
      ...config,
      pitchStabilityThreshold: config.pitchStabilityThreshold ?? SAXOPHONE_CONFIG.PITCH_ANALYSIS.PITCH_STABILITY_THRESHOLD,
      intonationTolerance: config.intonationTolerance ?? SAXOPHONE_CONFIG.PITCH_ANALYSIS.INTONATION_TOLERANCE,
      vibratoPitchDeviation: config.vibratoPitchDeviation ?? SAXOPHONE_CONFIG.PITCH_ANALYSIS.VIBRATO_PITCH_DEVIATION
    };
    this.saxophoneType = 'ALTO'; // Default, will be updated during analysis
  }

  async analyze(analysisResult: EssentiaAnalysisResult): Promise<PitchIntonationAnalysis> {
    // Detect saxophone type from pitch range
    this.saxophoneType = detectSaxophoneType(analysisResult.pitchTrack);
    
    // Filter for saxophone-valid pitches only
    const validPitches = analysisResult.pitchTrack.filter(p => 
      p > 0 && isValidSaxophoneFrequency(p, this.saxophoneType)
    );
    
    const pitchDeviations = this.calculatePitchDeviations(validPitches);
    const deviationStats = AudioUtils.calculateStatistics(pitchDeviations);

    const stabilityMetrics = this.analyzePitchStability(
      analysisResult.pitchTrack,
      analysisResult.pitchConfidence
    );

    const intervalMetrics = this.analyzeIntervalPrecision(validPitches);

    const driftMetrics = this.analyzeTuningDrift(validPitches);

    return {
      pitchAccuracyDistribution: {
        deviationStats: {
          mean: deviationStats.mean,
          std: deviationStats.std,
          range: [deviationStats.min, deviationStats.max]
        },
        frequencyRanges: this.categorizeByFrequencyRanges(validPitches, pitchDeviations),
        targetFrequencyDeviations: pitchDeviations
      },
      intonationStability: {
        sustainedNoteConsistency: stabilityMetrics.consistency,
        pitchVariationWithinNotes: stabilityMetrics.variations,
        stabilityByRegister: stabilityMetrics.registerStability
      },
      intervalPrecision: {
        melodicIntervalAccuracy: intervalMetrics.melodic,
        harmonicIntervalAccuracy: intervalMetrics.harmonic,
        overallIntervalScore: intervalMetrics.overall
      },
      tuningDriftPatterns: {
        driftOverTime: driftMetrics.driftOverTime,
        driftDirection: driftMetrics.direction,
        driftRate: driftMetrics.rate,
        performanceRegions: driftMetrics.regions
      }
    };
  }

  private calculatePitchDeviations(pitchTrack: number[]): number[] {
    if (pitchTrack.length === 0) return [];

    return pitchTrack.map(freq => {
      if (freq <= 0) return 0;

      const semitonesFromA4 = 12 * Math.log2(freq / 440);
      const closestSemitone = Math.round(semitonesFromA4);
      const expectedFreq = 440 * Math.pow(2, closestSemitone / 12);

      const centsDeviation = 1200 * Math.log2(freq / expectedFreq);
      return centsDeviation;
    });
  }

  private analyzePitchStability(
    pitchTrack: number[],
    confidences: number[]
  ): {
    consistency: number;
    variations: number[];
    registerStability: Record<string, number>;
  } {
    const validPitches = pitchTrack.filter(p => p > 0);

    const consistency = confidences.length > 0
      ? confidences.reduce((sum, conf) => sum + conf, 0) / confidences.length
      : 0.5;

    const windowSize = 10;
    const variations: number[] = [];

    for (let i = 0; i <= validPitches.length - windowSize; i++) {
      const window = validPitches.slice(i, i + windowSize);
      const stats = AudioUtils.calculateStatistics(window);
      variations.push(stats.std);
    }

    const saxRange = getSaxophoneRange(this.saxophoneType);
    const lowRegister = validPitches.filter(p => p >= saxRange.LOWEST && p < saxRange.COMFORTABLE_LOW);
    const midRegister = validPitches.filter(p => p >= saxRange.COMFORTABLE_LOW && p < saxRange.COMFORTABLE_HIGH);
    const highRegister = validPitches.filter(p => p >= saxRange.COMFORTABLE_HIGH && p < saxRange.ALTISSIMO_START);
    const altissimoRegister = validPitches.filter(p => p >= saxRange.ALTISSIMO_START);

    const registerStability = {
      low: this.calculateRegisterStability(lowRegister),
      middle: this.calculateRegisterStability(midRegister),
      high: this.calculateRegisterStability(highRegister),
      altissimo: this.calculateRegisterStability(altissimoRegister)
    };

    return { consistency, variations, registerStability };
  }

  private calculateRegisterStability(pitches: number[]): number {
    if (pitches.length < 3) return 0.5;

    const stats = AudioUtils.calculateStatistics(pitches);
    // Use saxophone-specific stability threshold
    const stabilityThreshold = this.config.pitchStabilityThreshold! * 100; // Convert to Hz
    const stability = Math.max(0, 1 - (stats.std / stabilityThreshold));
    return Math.min(1, stability);
  }

  private analyzeIntervalPrecision(pitchTrack: number[]): {
    melodic: Array<{ interval: string; accuracy: number }>;
    harmonic: Array<{ interval: string; accuracy: number }>;
    overall: number;
  } {
    if (pitchTrack.length < 2) {
      return {
        melodic: [],
        harmonic: [],
        overall: 0.5
      };
    }

    const intervals: number[] = [];
    for (let i = 1; i < pitchTrack.length; i++) {
      const current = pitchTrack[i];
      const previous = pitchTrack[i - 1];
      if (current && current > 0 && previous && previous > 0) {
        const semitones = 12 * Math.log2(current / previous);
        intervals.push(semitones);
      }
    }

    const intervalTypes = [
      { name: "m2", semitones: 1 },
      { name: "M2", semitones: 2 },
      { name: "m3", semitones: 3 },
      { name: "M3", semitones: 4 },
      { name: "P4", semitones: 5 },
      { name: "P5", semitones: 7 }
    ];

    const melodic = intervalTypes.map(intervalType => {
      const matchingIntervals = intervals.filter(interval =>
        Math.abs(interval - intervalType.semitones) < 0.5
      );

      const accuracy = matchingIntervals.length > 0
        ? this.calculateIntervalAccuracy(matchingIntervals, intervalType.semitones)
        : 0.5;

      return { interval: intervalType.name, accuracy };
    });

    const harmonic = melodic.map(item => ({ ...item }));

    const overall = [...melodic, ...harmonic]
      .reduce((sum, item) => sum + item.accuracy, 0) / (melodic.length + harmonic.length);

    return { melodic, harmonic, overall };
  }

  private calculateIntervalAccuracy(intervals: number[], targetSemitones: number): number {
    const deviations = intervals.map(interval => Math.abs(interval - targetSemitones));
    const avgDeviation = deviations.reduce((sum, dev) => sum + dev, 0) / deviations.length;

    // Use saxophone-specific interval tolerance (in semitones)
    const intervalTolerance = this.config.intonationTolerance! / 100; // Convert cents to semitones
    return Math.max(0, 1 - (avgDeviation / intervalTolerance));
  }

  private analyzeTuningDrift(pitchTrack: number[]): {
    driftOverTime: number[];
    direction: "flat" | "sharp" | "stable";
    rate: number;
    regions: Array<{ time: number; drift: number }>;
  } {
    if (pitchTrack.length < 10) {
      return {
        driftOverTime: [],
        direction: "stable",
        rate: 0,
        regions: []
      };
    }

    const referencePitch = pitchTrack[0];
    const driftOverTime = pitchTrack.map(pitch => {
      if (pitch <= 0 || !referencePitch || referencePitch <= 0) return 0;
      return 1200 * Math.log2(pitch / referencePitch);
    });

    const finalDrift = driftOverTime[driftOverTime.length - 1] || 0;
    const stabilityThreshold = this.config.pitchStabilityThreshold! * 1200; // Convert to cents
    const direction: "flat" | "sharp" | "stable" =
      Math.abs(finalDrift) < stabilityThreshold ? "stable" :
        finalDrift < 0 ? "flat" : "sharp";

    const durationSeconds = pitchTrack.length * 0.01;
    const rate = finalDrift / durationSeconds;

    const regions: Array<{ time: number; drift: number }> = [];
    const regionSize = Math.max(1, Math.floor(pitchTrack.length / 10));

    for (let i = 0; i < pitchTrack.length; i += regionSize) {
      const regionEnd = Math.min(i + regionSize, pitchTrack.length - 1);
      const time = (i / pitchTrack.length) * durationSeconds;
      const drift = driftOverTime[regionEnd] || 0;
      regions.push({ time, drift });
    }

    return { driftOverTime, direction, rate, regions };
  }

  private categorizeByFrequencyRanges(
    pitches: number[],
    deviations: number[]
  ): Array<{ range: string; accuracy: number }> {
    const saxRange = getSaxophoneRange(this.saxophoneType);
    const ranges = [
      { name: "low", min: saxRange.LOWEST, max: saxRange.COMFORTABLE_LOW },
      { name: "middle", min: saxRange.COMFORTABLE_LOW, max: saxRange.COMFORTABLE_HIGH },
      { name: "high", min: saxRange.COMFORTABLE_HIGH, max: saxRange.ALTISSIMO_START },
      { name: "altissimo", min: saxRange.ALTISSIMO_START, max: saxRange.ALTISSIMO_END }
    ];

    return ranges.map(range => {
      const rangeIndices = pitches
        .map((pitch, index) => ({ pitch, index }))
        .filter(({ pitch }) => pitch >= range.min && pitch < range.max)
        .map(({ index }) => index);

      if (rangeIndices.length === 0) {
        return { range: range.name, accuracy: 0.5 };
      }

      const rangeDeviations = rangeIndices.map(i => Math.abs(deviations[i] || 0));
      const avgDeviation = rangeDeviations.reduce((sum, dev) => sum + dev, 0) / rangeDeviations.length;

      // Use saxophone-specific intonation tolerance
      const accuracy = Math.max(0, 1 - (avgDeviation / this.config.intonationTolerance!));

      return { range: range.name, accuracy };
    });
  }

  public smoothPitchTrack(pitchTrack: number[], pitchConfidence: number[]): number[] {
    if (pitchTrack.length === 0) return [];
    
    // Step 1: Remove octave errors
    const octaveCorrectedTrack = this.correctOctaveErrors(pitchTrack, pitchConfidence);
    
    // Step 2: Apply median filter for outlier removal
    const medianFilteredTrack = this.applyMedianFilter(octaveCorrectedTrack, this.config.medianFilterSize);
    
    // Step 3: Apply confidence-weighted smoothing
    const smoothedTrack = this.applyConfidenceWeightedSmoothing(medianFilteredTrack, pitchConfidence);
    
    // Step 4: Fill gaps with interpolation
    const interpolatedTrack = this.interpolatePitchGaps(smoothedTrack);
    
    return interpolatedTrack;
  }

  private correctOctaveErrors(pitchTrack: number[], pitchConfidence: number[]): number[] {
    if (pitchTrack.length < 3) return pitchTrack;
    
    const correctedTrack = [...pitchTrack];
    const windowSize = 5;
    
    for (let i = windowSize; i < correctedTrack.length - windowSize; i++) {
      const current = correctedTrack[i];
      const currentConfidence = pitchConfidence[i] || 0;
      
      if (current === undefined || current <= 0) continue;
      
      if (currentConfidence > this.config.confidenceThreshold) {
        continue;
      }
      
      const surrounding = [];
      const surroundingConfidences = [];
      
      for (let j = Math.max(0, i - windowSize); j <= Math.min(correctedTrack.length - 1, i + windowSize); j++) {
        const value = correctedTrack[j];
        const confidence = pitchConfidence[j] || 0;
        
        if (j !== i && value !== undefined && value > 0 && confidence > 0) {
          surrounding.push(value);
          surroundingConfidences.push(confidence);
        }
      }
      
      if (surrounding.length < 3) continue;
      
      const weightedMedian = AudioUtils.calculateConfidenceWeightedMedian(surrounding, surroundingConfidences);
      
      const ratioUp = current / weightedMedian;
      const ratioDown = weightedMedian / current;
      const threshold = this.config.octaveErrorThreshold;
      
      const avgSurroundingConfidence = surroundingConfidences.reduce((sum, conf) => sum + conf, 0) / surroundingConfidences.length;
      
      const shouldCorrect = avgSurroundingConfidence > currentConfidence + 0.1;
      
      if (shouldCorrect) {
        if (ratioUp > (2.0 - threshold) && ratioUp < (2.0 + threshold)) {
          const correctionStrength = Math.min(1, avgSurroundingConfidence - currentConfidence);
          correctedTrack[i] = current / (1 + correctionStrength);
        } else if (ratioDown > (2.0 - threshold) && ratioDown < (2.0 + threshold)) {
          const correctionStrength = Math.min(1, avgSurroundingConfidence - currentConfidence);
          correctedTrack[i] = current * (1 + correctionStrength);
        }
      }
    }
    
    return correctedTrack;
  }

  private applyMedianFilter(pitchTrack: number[], windowSize: number): number[] {
    if (pitchTrack.length === 0) return [];
    
    const filtered = [...pitchTrack];
    const halfWindow = Math.floor(windowSize / 2);
    
    for (let i = halfWindow; i < pitchTrack.length - halfWindow; i++) {
      const window = [];
      for (let j = i - halfWindow; j <= i + halfWindow; j++) {
        const value = pitchTrack[j];
        if (value !== undefined && value > 0) {
          window.push(value);
        }
      }
      
      if (window.length >= 3) {
        filtered[i] = AudioUtils.calculateMedian(window);
      }
    }
    
    return filtered;
  }

  private applyConfidenceWeightedSmoothing(pitchTrack: number[], pitchConfidence: number[]): number[] {
    if (pitchTrack.length === 0) return [];
    
    const smoothed = [...pitchTrack];
    const alpha = this.config.smoothingFactor;
    
    for (let i = 1; i < pitchTrack.length; i++) {
      const currentPitch = pitchTrack[i];
      const previousPitch = smoothed[i - 1];
      
      if (currentPitch === undefined || currentPitch <= 0) continue;
      
      const currentConfidence = pitchConfidence[i] || 0;
      const previousConfidence = pitchConfidence[i - 1] || 0;
      
      const weight = Math.min(currentConfidence, previousConfidence);
      const effectiveAlpha = alpha * weight;
      
      if (previousPitch !== undefined && previousPitch > 0) {
        smoothed[i] = effectiveAlpha * previousPitch + (1 - effectiveAlpha) * currentPitch;
      }
    }
    
    return smoothed;
  }

  private interpolatePitchGaps(pitchTrack: number[]): number[] {
    if (pitchTrack.length === 0) return [];
    
    const interpolated = [...pitchTrack];
    
    for (let i = 1; i < pitchTrack.length - 1; i++) {
      const currentPitch = pitchTrack[i];
      if (currentPitch === undefined || currentPitch <= 0) {
        let beforeIndex = -1;
        let afterIndex = -1;
        
        for (let j = i - 1; j >= 0; j--) {
          const value = pitchTrack[j];
          if (value !== undefined && value > 0) {
            beforeIndex = j;
            break;
          }
        }
        
        for (let j = i + 1; j < pitchTrack.length; j++) {
          const value = pitchTrack[j];
          if (value !== undefined && value > 0) {
            afterIndex = j;
            break;
          }
        }
        
        if (beforeIndex !== -1 && afterIndex !== -1) {
          const beforePitch = pitchTrack[beforeIndex];
          const afterPitch = pitchTrack[afterIndex];
          
          if (beforePitch !== undefined && afterPitch !== undefined) {
            const ratio = (i - beforeIndex) / (afterIndex - beforeIndex);
            
            interpolated[i] = Math.exp(
              Math.log(beforePitch) + ratio * (Math.log(afterPitch) - Math.log(beforePitch))
            );
          }
        }
      }
    }
    
    return interpolated;
  }
}
