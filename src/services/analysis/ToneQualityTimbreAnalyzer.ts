import { ToneQualityTimbreAnalysis, EssentiaAnalysisResult } from "../../types";
import { AudioUtils } from "./AudioUtils";
import { 
  SAXOPHONE_CONFIG, 
  detectSaxophoneType, 
  getSaxophoneRange, 
  isValidSaxophoneFrequency, 
  isAltissimoRange 
} from "./SaxophoneConstants";

export interface VibratoConfig {
  minRate: number;
  maxRate: number;
  minDepth: number;
  maxDepth: number;
  optimalRate: number;
  optimalDepth: number;
  qualityThreshold: number;
  consistencyThreshold: number;
}

export class ToneQualityTimbreAnalyzer {
  private vibratoConfig: VibratoConfig;
  private saxophoneType: keyof typeof SAXOPHONE_CONFIG.FREQUENCY_RANGES;

  constructor(vibratoConfig: VibratoConfig) {
    this.vibratoConfig = {
      ...vibratoConfig,
      maxDepth: vibratoConfig.maxDepth ?? SAXOPHONE_CONFIG.VIBRATO.MAX_DEPTH,
      optimalRate: vibratoConfig.optimalRate ?? SAXOPHONE_CONFIG.VIBRATO.OPTIMAL_RATE,
      optimalDepth: vibratoConfig.optimalDepth ?? SAXOPHONE_CONFIG.VIBRATO.OPTIMAL_DEPTH,
      consistencyThreshold: vibratoConfig.consistencyThreshold ?? SAXOPHONE_CONFIG.VIBRATO.CONSISTENCY_THRESHOLD
    };
    this.saxophoneType = "ALTO"; // Default, will be updated during analysis
  }

  async analyze(analysisResult: EssentiaAnalysisResult): Promise<ToneQualityTimbreAnalysis> {
    // Detect saxophone type from pitch range
    this.saxophoneType = detectSaxophoneType(analysisResult.pitchTrack);
    
    const harmonicMetrics = this.analyzeHarmonicContent(
      analysisResult.harmonics,
      analysisResult.spectralCentroid,
      analysisResult.pitchTrack
    );

    const dynamicMetrics = this.analyzeDynamicRange(
      analysisResult.energy,
      analysisResult.loudness
    );

    const timbralMetrics = this.analyzeTimbralConsistency(
      analysisResult.spectralCentroid,
      analysisResult.spectralRolloff,
      analysisResult.mfcc,
      analysisResult.pitchTrack
    );

    const vibratoMetrics = this.analyzeVibratoCharacteristics(
      analysisResult.pitchTrack,
      analysisResult.pitchConfidence
    );

    return {
      harmonicContentAnalysis: harmonicMetrics,
      dynamicRangeUtilization: dynamicMetrics,
      timbralConsistency: timbralMetrics,
      vibratoCharacteristics: vibratoMetrics
    };
  }

  private analyzeHarmonicContent(
    harmonics: number[],
    spectralCentroid: number[],
    pitchTrack: number[]
  ): {
    overtoneDistribution: number[];
    harmonicRichness: number;
    fundamentalToHarmonicRatio: number;
    spectralCentroid: number[];
  } {
    const pitchBasedAnalysis = this.analyzePitchContextualHarmonics(harmonics, pitchTrack);

    const harmonicRichness = this.calculatePitchAwareHarmonicRichness(harmonics, pitchTrack);

    const fundamentalToHarmonicRatio = this.calculateContextualFundamentalRatio(harmonics, pitchTrack);

    return {
      overtoneDistribution: pitchBasedAnalysis.overtoneDistribution,
      harmonicRichness,
      fundamentalToHarmonicRatio,
      spectralCentroid
    };
  }

  private analyzePitchContextualHarmonics(harmonics: number[], pitchTrack: number[]): {
    overtoneDistribution: number[];
  } {
    if (harmonics.length === 0 || pitchTrack.length === 0) {
      return { overtoneDistribution: harmonics };
    }

    const validPitches = pitchTrack.filter(p => 
      p > 0 && isValidSaxophoneFrequency(p, this.saxophoneType)
    );
    if (validPitches.length === 0) {
      return { overtoneDistribution: harmonics };
    }

    const pitchStability = this.calculatePitchStability(validPitches);

    const adjustedHarmonics = harmonics.map(harmonic => {
      if (pitchStability > 0.8) {
        return harmonic * 1.1;
      } else if (pitchStability < 0.5) {
        return harmonic * 0.9;
      }
      return harmonic;
    });

    return { overtoneDistribution: adjustedHarmonics };
  }

  private calculatePitchAwareHarmonicRichness(harmonics: number[], pitchTrack: number[]): number {
    if (harmonics.length === 0) return 0.5;

    const basicRichness = harmonics.reduce((sum, harmonic) => sum + harmonic, 0) / harmonics.length;

    const validPitches = pitchTrack.filter(p => 
      p > 0 && isValidSaxophoneFrequency(p, this.saxophoneType)
    );
    if (validPitches.length === 0) return basicRichness;

    const avgPitch = validPitches.reduce((sum, p) => sum + p, 0) / validPitches.length;
    const pitchStability = this.calculatePitchStability(validPitches);
    const saxRange = getSaxophoneRange(this.saxophoneType);

    let pitchAdjustment = 1.0;

    // Adjust for saxophone registers
    if (isAltissimoRange(avgPitch, this.saxophoneType)) {
      pitchAdjustment *= 0.8; // Altissimo has different harmonic characteristics
    } else if (avgPitch > saxRange.COMFORTABLE_HIGH) {
      pitchAdjustment *= 0.9; // High register
    } else if (avgPitch < saxRange.COMFORTABLE_LOW) {
      pitchAdjustment *= 1.1; // Low register has richer harmonics
    }

    pitchAdjustment *= (0.7 + 0.3 * pitchStability);

    return Math.max(0, Math.min(1, basicRichness * pitchAdjustment));
  }

  private calculateContextualFundamentalRatio(harmonics: number[], pitchTrack: number[]): number {
    if (harmonics.length === 0) return 1;

    const fundamentalStrength = harmonics[0] || 1;
    const harmonicSum = harmonics.slice(1).reduce((sum, h) => sum + h, 0);
    const basicRatio = harmonicSum > 0 ? fundamentalStrength / harmonicSum : 1;

    const validPitches = pitchTrack.filter(p => 
      p > 0 && isValidSaxophoneFrequency(p, this.saxophoneType)
    );
    if (validPitches.length === 0) return basicRatio;

    const avgPitch = validPitches.reduce((sum, p) => sum + p, 0) / validPitches.length;
    const pitchStability = this.calculatePitchStability(validPitches);
    const saxRange = getSaxophoneRange(this.saxophoneType);

    let contextualAdjustment = 1.0;

    // Saxophone-specific fundamental-to-harmonic ratio adjustments
    if (isAltissimoRange(avgPitch, this.saxophoneType)) {
      contextualAdjustment *= 1.3; // Altissimo has stronger fundamental
    } else if (avgPitch > saxRange.COMFORTABLE_HIGH) {
      contextualAdjustment *= 1.2; // High register
    } else if (avgPitch < saxRange.COMFORTABLE_LOW) {
      contextualAdjustment *= 0.9; // Low register has weaker fundamental
    }

    if (pitchStability > 0.8) {
      contextualAdjustment *= 1.1;
    } else if (pitchStability < 0.5) {
      contextualAdjustment *= 0.9;
    }

    return Math.max(0.1, basicRatio * contextualAdjustment);
  }

  private calculatePitchStability(pitches: number[]): number {
    if (pitches.length < 2) return 0.5;

    const avgPitch = pitches.reduce((sum, p) => sum + p, 0) / pitches.length;
    const variance = pitches.reduce((sum, p) => sum + Math.pow(p - avgPitch, 2), 0) / pitches.length;
    const stdDev = Math.sqrt(variance);

    // Use saxophone-specific pitch stability threshold
    const stabilityThreshold = avgPitch * SAXOPHONE_CONFIG.PITCH_ANALYSIS.PITCH_STABILITY_THRESHOLD;
    const stability = Math.max(0, 1 - (stdDev / stabilityThreshold));

    return Math.min(1, stability);
  }

  private analyzeDynamicRange(
    energy: number[],
    loudness: number[]
  ): {
    volumeRange: [number, number];
    dynamicVariationEffectiveness: number;
    crescendoDecrescendoControl: number;
    accentClarity: number;
  } {
    if (energy.length === 0) {
      return {
        volumeRange: [-60, -20],
        dynamicVariationEffectiveness: 0.5,
        crescendoDecrescendoControl: 0.5,
        accentClarity: 0.5
      };
    }

    const energyBasedAnalysis = this.analyzeEnergyBasedDynamics(energy);
    const loudnessBasedAnalysis = this.analyzeLoudnessBasedDynamics(loudness);
    const combinedAnalysis = this.combineDynamicAnalyses(energy, loudness);

    const volumeRange = loudnessBasedAnalysis.volumeRange;

    const dynamicVariationEffectiveness = this.calculateCombinedDynamicEffectiveness(
      energyBasedAnalysis.dynamicRange,
      loudnessBasedAnalysis.dynamicRange
    );

    const crescendoDecrescendoControl = combinedAnalysis.crescendoDecrescendoControl;

    const accentClarity = loudnessBasedAnalysis.accentClarity;

    return {
      volumeRange,
      dynamicVariationEffectiveness,
      crescendoDecrescendoControl,
      accentClarity
    };
  }

  private analyzeEnergyBasedDynamics(energy: number[]): {
    dynamicRange: number;
    volumeRange: [number, number];
    accentClarity: number;
  } {
    if (energy.length === 0) {
      return {
        dynamicRange: 0,
        volumeRange: [-60, -20],
        accentClarity: 0.5
      };
    }

    const energyDB = energy.map(e => 20 * Math.log10(Math.max(e, 1e-10)));
    const minDB = Math.min(...energyDB);
    const maxDB = Math.max(...energyDB);
    const volumeRange: [number, number] = [minDB, maxDB];
    const dynamicRange = maxDB - minDB;

    const accentClarity = this.detectAccents(energyDB);

    return {
      dynamicRange,
      volumeRange,
      accentClarity
    };
  }

  private analyzeLoudnessBasedDynamics(loudness: number[]): {
    dynamicRange: number;
    volumeRange: [number, number];
    accentClarity: number;
  } {
    if (loudness.length === 0) {
      return {
        dynamicRange: 0,
        volumeRange: [-60, -20],
        accentClarity: 0.5
      };
    }

    const loudnessDB = loudness.map(l => 20 * Math.log10(Math.max(l, 1e-10)));
    const minDB = Math.min(...loudnessDB);
    const maxDB = Math.max(...loudnessDB);
    const volumeRange: [number, number] = [minDB, maxDB];
    const dynamicRange = maxDB - minDB;

    const accentClarity = this.detectLoudnessAccents(loudnessDB);

    return {
      dynamicRange,
      volumeRange,
      accentClarity
    };
  }

  private combineDynamicAnalyses(energy: number[], loudness: number[]): {
    crescendoDecrescendoControl: number;
  } {
    if (energy.length === 0 || loudness.length === 0) {
      return {
        crescendoDecrescendoControl: 0.5
      };
    }

    const energyDB = energy.map(e => 20 * Math.log10(Math.max(e, 1e-10)));
    const loudnessDB = loudness.map(l => 20 * Math.log10(Math.max(l, 1e-10)));

    const smoothedEnergy = AudioUtils.smoothArray(energyDB, 5);
    const smoothedLoudness = AudioUtils.smoothArray(loudnessDB, 5);

    const energyControl = this.detectDynamicControl(smoothedEnergy);
    const loudnessControl = this.detectDynamicControl(smoothedLoudness);

    const crescendoDecrescendoControl = (energyControl * 0.4) + (loudnessControl * 0.6);

    return {
      crescendoDecrescendoControl
    };
  }

  private calculateCombinedDynamicEffectiveness(energyRange: number, loudnessRange: number): number {
    const energyEffectiveness = Math.min(1, energyRange / 40);
    const loudnessEffectiveness = Math.min(1, loudnessRange / 35);

    const combinedEffectiveness = (energyEffectiveness * 0.45) + (loudnessEffectiveness * 0.55);

    return Math.max(0, Math.min(1, combinedEffectiveness));
  }

  private detectLoudnessAccents(loudnessDB: number[]): number {
    if (loudnessDB.length < 5) return 0.5;

    let accents = 0;
    let clearAccents = 0;

    for (let i = 2; i < loudnessDB.length - 2; i++) {
      const current = loudnessDB[i];
      const surrounding = [
        loudnessDB[i - 2], loudnessDB[i - 1],
        loudnessDB[i + 1], loudnessDB[i + 2]
      ].filter((val): val is number => val !== undefined);

      if (surrounding.length === 4 && current !== undefined) {
        const avgSurrounding = surrounding.reduce((sum, val) => sum + val, 0) / surrounding.length;

        if (current > avgSurrounding + 4) {
          accents++;

          if (current > Math.max(...surrounding) + 2) {
            clearAccents++;
          }
        }
      }
    }

    return accents > 0 ? clearAccents / accents : 0.5;
  }

  private detectAccents(energyDB: number[]): number {
    if (energyDB.length < 5) return 0.5;

    let accents = 0;
    let clearAccents = 0;

    for (let i = 2; i < energyDB.length - 2; i++) {
      const current = energyDB[i];
      const surrounding = [
        energyDB[i - 2], energyDB[i - 1],
        energyDB[i + 1], energyDB[i + 2]
      ].filter((val): val is number => val !== undefined);

      if (surrounding.length === 4 && current !== undefined) {
        const avgSurrounding = surrounding.reduce((sum, val) => sum + val, 0) / surrounding.length;

        if (current > avgSurrounding + 6) {
          accents++;

          if (current > Math.max(...surrounding) + 3) {
            clearAccents++;
          }
        }
      }
    }

    return accents > 0 ? clearAccents / accents : 0.5;
  }

  private detectDynamicControl(smoothedEnergy: number[]): number {
    if (smoothedEnergy.length < 10) return 0.5;

    let controlScore = 0;
    let gradualChanges = 0;

    for (let i = 5; i < smoothedEnergy.length - 5; i++) {
      const before = smoothedEnergy.slice(i - 5, i);
      const after = smoothedEnergy.slice(i, i + 5);

      const beforeAvg = before.reduce((sum, val) => sum + val, 0) / before.length;
      const afterAvg = after.reduce((sum, val) => sum + val, 0) / after.length;

      const change = Math.abs(afterAvg - beforeAvg);
      if (change > 3) {
        gradualChanges++;

        const gradient = (afterAvg - beforeAvg) / 5;
        const expectedValues = before.map((_, idx) => beforeAvg + gradient * idx);
        const actualValues = after;
        const error = expectedValues.reduce((sum, expected, idx) =>
          sum + Math.abs(expected - (actualValues[idx] || 0)), 0) / expectedValues.length;

        if (error < 2) {
          controlScore += 1;
        }
      }
    }

    return gradualChanges > 0 ? Math.min(1, controlScore / gradualChanges) : 0.5;
  }

  private analyzeTimbralConsistency(
    spectralCentroid: number[],
    spectralRolloff: number[],
    mfcc: number[],
    pitchTrack: number[]
  ): {
    toneStabilityAcrossRegisters: Record<string, number>;
    toneStabilityAcrossDynamics: Record<string, number>;
    overallTimbralUniformity: number;
    colorVariationControl: number;
  } {
    const registerStability = this.analyzeTimbralStabilityByRegister(
      spectralCentroid,
      pitchTrack
    );

    const dynamicStability = {
      pp: 0.75,
      p: 0.80,
      mf: 0.85,
      f: 0.80,
      ff: 0.75
    };

    const centroidStats = AudioUtils.calculateStatistics(spectralCentroid);
    const rolloffStats = AudioUtils.calculateStatistics(spectralRolloff);

    const centroidConsistency = centroidStats.mean > 0 ? 1 - (centroidStats.std / centroidStats.mean) : 0.5;
    const rolloffConsistency = rolloffStats.mean > 0 ? 1 - (rolloffStats.std / rolloffStats.mean) : 0.5;

    const overallTimbralUniformity = (centroidConsistency + rolloffConsistency) / 2;

    const colorVariationControl = Math.min(1, centroidStats.std / 500);

    return {
      toneStabilityAcrossRegisters: registerStability,
      toneStabilityAcrossDynamics: dynamicStability,
      overallTimbralUniformity: Math.max(0, Math.min(1, overallTimbralUniformity)),
      colorVariationControl: Math.max(0, Math.min(1, colorVariationControl))
    };
  }

  private analyzeTimbralStabilityByRegister(
    spectralCentroid: number[],
    pitchTrack: number[]
  ): Record<string, number> {
    const saxRange = getSaxophoneRange(this.saxophoneType);
    const registers = {
      low: { min: saxRange.LOWEST, max: saxRange.COMFORTABLE_LOW, centroids: [] as number[] },
      middle: { min: saxRange.COMFORTABLE_LOW, max: saxRange.COMFORTABLE_HIGH, centroids: [] as number[] },
      high: { min: saxRange.COMFORTABLE_HIGH, max: saxRange.ALTISSIMO_START, centroids: [] as number[] },
      altissimo: { min: saxRange.ALTISSIMO_START, max: saxRange.ALTISSIMO_END, centroids: [] as number[] }
    };

    pitchTrack.forEach((pitch, i) => {
      if (i < spectralCentroid.length && pitch > 0) {
        const centroid = spectralCentroid[i];

        if (centroid !== undefined) {
          if (pitch >= registers.low.min && pitch < registers.low.max) {
            registers.low.centroids.push(centroid);
          } else if (pitch >= registers.middle.min && pitch < registers.middle.max) {
            registers.middle.centroids.push(centroid);
          } else if (pitch >= registers.high.min && pitch < registers.high.max) {
            registers.high.centroids.push(centroid);
          } else if (pitch >= registers.altissimo.min && pitch < registers.altissimo.max) {
            registers.altissimo.centroids.push(centroid);
          }
        }
      }
    });

    const stability: Record<string, number> = {};

    Object.entries(registers).forEach(([register, data]) => {
      if (data.centroids.length < 3) {
        stability[register] = 0.5;
        return;
      }

      const stats = AudioUtils.calculateStatistics(data.centroids);
      const consistency = stats.mean > 0 ? 1 - (stats.std / stats.mean) : 0.5;
      stability[register] = Math.max(0, Math.min(1, consistency));
    });

    return stability;
  }

  private analyzeVibratoCharacteristics(
    pitchTrack: number[],
    pitchConfidence: number[]
  ): {
    vibratoRate: number;
    vibratoDepth: number;
    vibratoControl: number;
    vibratoTiming: Array<{ start: number; end: number; quality: number }>;
  } {
    if (pitchTrack.length < 20) {
      return {
        vibratoRate: 0,
        vibratoDepth: 0,
        vibratoControl: 0,
        vibratoTiming: []
      };
    }

    const vibratoRegions = this.detectVibratoRegions(pitchTrack, pitchConfidence);

    if (vibratoRegions.length === 0) {
      return {
        vibratoRate: 0,
        vibratoDepth: 0,
        vibratoControl: 0.5,
        vibratoTiming: []
      };
    }

    let totalRate = 0;
    let totalDepth = 0;
    let totalQuality = 0;

    for (const region of vibratoRegions) {
      totalRate += region.rate;
      totalDepth += region.depth;
      totalQuality += region.quality;
    }

    const vibratoRate = totalRate / vibratoRegions.length;
    const vibratoDepth = totalDepth / vibratoRegions.length;
    const vibratoControl = totalQuality / vibratoRegions.length;

    const vibratoTiming = vibratoRegions.map(region => ({
      start: region.start,
      end: region.end,
      quality: region.quality
    }));

    return {
      vibratoRate,
      vibratoDepth,
      vibratoControl,
      vibratoTiming
    };
  }

  private detectVibratoRegions(pitchTrack: number[], pitchConfidence: number[]): Array<{
    start: number;
    end: number;
    rate: number;
    depth: number;
    quality: number;
  }> {
    const regions: Array<{
      start: number;
      end: number;
      rate: number;
      depth: number;
      quality: number;
    }> = [];
    const minVibratoDuration = 0.5; // seconds
    const frameRate = 100; // Assuming 100 frames per second

    let inVibratoRegion = false;
    let currentRegionStart = 0;

    for (let i = 0; i < pitchTrack.length; i++) {
      const pitch = pitchTrack[i];
      const confidence = pitchConfidence[i];

      if (pitch && confidence && pitch > 0 && confidence > this.vibratoConfig.qualityThreshold) {
        // Simple vibrato detection: look for consistent pitch changes
        // This is a placeholder and would ideally use more advanced signal processing
        const window = pitchTrack.slice(Math.max(0, i - 10), i + 1);
        const stats = AudioUtils.calculateStatistics(window.filter(p => p > 0));

        // Use saxophone-specific vibrato detection
        const vibratoDepthCents = this.convertFrequencyDeviationToCents(stats.std, pitch);
        if (vibratoDepthCents > this.vibratoConfig.minDepth && vibratoDepthCents < this.vibratoConfig.maxDepth) {
          if (!inVibratoRegion) {
            currentRegionStart = i;
            inVibratoRegion = true;
          }
        } else {
          if (inVibratoRegion) {
            const duration = (i - currentRegionStart) / frameRate;
            if (duration >= minVibratoDuration) {
              const regionPitches = pitchTrack.slice(currentRegionStart, i);
              const regionConfidences = pitchConfidence.slice(currentRegionStart, i);
              const vibratoAnalysis = this.analyzeVibratoInRegion(regionPitches, regionConfidences);
              regions.push({
                start: currentRegionStart / frameRate,
                end: i / frameRate,
                rate: vibratoAnalysis.rate,
                depth: vibratoAnalysis.depth,
                quality: vibratoAnalysis.quality
              });
            }
            inVibratoRegion = false;
          }
        }
      } else {
        if (inVibratoRegion) {
          const duration = (i - currentRegionStart) / frameRate;
          if (duration >= minVibratoDuration) {
            const regionPitches = pitchTrack.slice(currentRegionStart, i);
            const regionConfidences = pitchConfidence.slice(currentRegionStart, i);
            const vibratoAnalysis = this.analyzeVibratoInRegion(regionPitches, regionConfidences);
            regions.push({
              start: currentRegionStart / frameRate,
              end: i / frameRate,
              rate: vibratoAnalysis.rate,
              depth: vibratoAnalysis.depth,
              quality: vibratoAnalysis.quality
            });
          }
          inVibratoRegion = false;
        }
      }
    }

    if (inVibratoRegion) {
      const duration = (pitchTrack.length - currentRegionStart) / frameRate;
      if (duration >= minVibratoDuration) {
        const regionPitches = pitchTrack.slice(currentRegionStart, pitchTrack.length);
        const regionConfidences = pitchConfidence.slice(currentRegionStart, pitchTrack.length);
        const vibratoAnalysis = this.analyzeVibratoInRegion(regionPitches, regionConfidences);
        regions.push({
          start: currentRegionStart / frameRate,
          end: pitchTrack.length / frameRate,
          rate: vibratoAnalysis.rate,
          depth: vibratoAnalysis.depth,
          quality: vibratoAnalysis.quality
        });
      }
    }

    return regions;
  }

  private analyzeVibratoInRegion(pitchTrack: number[], pitchConfidence: number[]): {
    rate: number;
    depth: number;
    quality: number;
  } {
    const validPitches = pitchTrack.filter(p => p > 0);
    if (validPitches.length < 20) {
      return { rate: 0, depth: 0, quality: 0 };
    }

    const avgPitch = validPitches.reduce((sum, p) => sum + p, 0) / validPitches.length;
    const stats = AudioUtils.calculateStatistics(validPitches);
    const depth = this.convertFrequencyDeviationToCents(stats.std, avgPitch);

    // Improved rate estimation using zero crossings
    const rate = this.calculateVibratoRate(validPitches);
    
    // Quality based on consistency with saxophone vibrato characteristics
    const quality = this.calculateVibratoQuality(rate, depth, pitchConfidence);

    return { rate, depth, quality };
  }

  private convertFrequencyDeviationToCents(freqStd: number, avgFreq: number): number {
    if (avgFreq <= 0) return 0;
    return 1200 * Math.log2((avgFreq + freqStd) / avgFreq);
  }

  private calculateVibratoRate(pitchTrack: number[]): number {
    if (pitchTrack.length < 20) return 0;
    
    const avgPitch = pitchTrack.reduce((sum, p) => sum + p, 0) / pitchTrack.length;
    const deviations = pitchTrack.map(p => p - avgPitch);
    
    // Count zero crossings to estimate rate
    let zeroCrossings = 0;
    for (let i = 1; i < deviations.length; i++) {
      const current = deviations[i];
      const prev = deviations[i - 1];
      
      if (current !== undefined && prev !== undefined && 
          ((current > 0 && prev < 0) || (current < 0 && prev > 0))) {
        zeroCrossings++;
      }
    }
    
    // Convert to Hz (assuming 100 frames per second)
    const rate = (zeroCrossings / 2) / (pitchTrack.length / 100);
    
    // Clamp to saxophone vibrato range
    return Math.max(this.vibratoConfig.minRate, 
      Math.min(this.vibratoConfig.maxRate, rate));
  }

  private calculateVibratoQuality(rate: number, depth: number, pitchConfidence: number[]): number {
    // Base quality on pitch tracking confidence
    const confidenceScore = AudioUtils.calculateStatistics(pitchConfidence).mean;
    
    // Penalize rates and depths outside optimal range
    const rateScore = 1 - Math.abs(rate - this.vibratoConfig.optimalRate) / this.vibratoConfig.optimalRate;
    const depthScore = 1 - Math.abs(depth - this.vibratoConfig.optimalDepth) / this.vibratoConfig.optimalDepth;
    
    // Combine scores
    const quality = (confidenceScore * 0.4) + (rateScore * 0.3) + (depthScore * 0.3);
    
    return Math.max(0, Math.min(1, quality));
  }
}
