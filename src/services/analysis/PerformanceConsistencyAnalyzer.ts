import { PerformanceConsistencyAnalysis, EssentiaAnalysisResult } from "../../types";
import { AudioUtils } from "./AudioUtils";

export class PerformanceConsistencyAnalyzer {
  async analyze(analysisResult: EssentiaAnalysisResult): Promise<PerformanceConsistencyAnalysis> {
    const { pitchTrack, beats, tempo, spectralCentroid, harmonics } = analysisResult;
    const numSections = 4;
    const sectionLength = Math.floor(pitchTrack.length / numSections);

    // 1. Pitch Consistency
    const pitchConsistency = this.analyzePitchConsistency(pitchTrack, numSections, sectionLength);

    // 2. Rhythm Consistency
    const rhythmConsistency = this.analyzeRhythmConsistency(beats, tempo);

    // 3. Tone Consistency
    const toneConsistency = this.analyzeToneConsistency(spectralCentroid, harmonics, numSections, sectionLength);

    // 4. Error Analysis
    const errorAnalysis = this.identifyErrors(pitchTrack, beats, tempo);

    // 5. Endurance Analysis
    const endurancePatterns = this.analyzeEndurancePatterns(pitchTrack, spectralCentroid, numSections, sectionLength);

    // 6. Recovery Speed Analysis
    const recoverySpeed = this.analyzeRecoverySpeed(errorAnalysis.errorDistribution);

    // 7. Error Frequency and Type Analysis
    const errorFrequencyAndType = this.analyzeErrorFrequencyAndType(errorAnalysis.errorDistribution, pitchTrack, beats);

    return {
      pitchConsistency,
      rhythmConsistency,
      toneConsistency,
      errorAnalysis,
      endurancePatterns,
      recoverySpeed,
      errorFrequencyAndType,
    };
  }

  private analyzePitchConsistency(pitchTrack: number[], numSections: number, sectionLength: number): PerformanceConsistencyAnalysis["pitchConsistency"] {
    const driftOverTime = this.calculatePitchDrift(pitchTrack);
    const standardDeviationBySection = [];
    for (let i = 0; i < numSections; i++) {
      const section = pitchTrack.slice(i * sectionLength, (i + 1) * sectionLength);
      standardDeviationBySection.push(AudioUtils.calculateStatistics(section).std);
    }
    const overallStability = 1 - (AudioUtils.calculateStatistics(pitchTrack).std / 50); // 50 cents is significant deviation

    return {
      driftOverTime,
      standardDeviationBySection,
      overallStability: Math.max(0, Math.min(1, overallStability))
    };
  }


  private calculatePitchDrift(pitchTrack: number[]): number[] {
    if (pitchTrack.length === 0) return [];
    const referencePitch = pitchTrack[0];
    return pitchTrack.map(pitch => {
      if (pitch <= 0 || !referencePitch || referencePitch <= 0) return 0;
      return 1200 * Math.log2(pitch / referencePitch); // Cents
    });
  }

  private analyzeRhythmConsistency(beats: number[], tempo: number): PerformanceConsistencyAnalysis["rhythmConsistency"] {
    const beatIntervals: number[] = [];
    for (let i = 1; i < beats.length; i++) {
      const currentBeat = beats[i];
      const previousBeat = beats[i-1];
      if (currentBeat !== undefined && previousBeat !== undefined) {
        beatIntervals.push(currentBeat - previousBeat);
      }
    }
    const expectedInterval = 60 / tempo;
    const tempoFluctuation = beatIntervals.map(interval => (60 / interval) - tempo);
    const beatTimingDeviation = beatIntervals.map(interval => (interval - expectedInterval) * 1000);
    const overallSteadiness = 1 - (AudioUtils.calculateStatistics(beatTimingDeviation).std / 100); // 100ms is significant deviation

    return {
      tempoFluctuation,
      beatTimingDeviation,
      overallSteadiness: Math.max(0, Math.min(1, overallSteadiness))
    };
  }

  private analyzeToneConsistency(spectralCentroid: number[], harmonics: number[], numSections: number, sectionLength: number): PerformanceConsistencyAnalysis["toneConsistency"] {
    const timbreVariability = [];
    const harmonicRichnessVariability = [];
    for (let i = 0; i < numSections; i++) {
      const centroidSection = spectralCentroid.slice(i * sectionLength, (i + 1) * sectionLength);
      const harmonicsSection = harmonics.slice(i * sectionLength, (i + 1) * sectionLength);
      timbreVariability.push(AudioUtils.calculateStatistics(centroidSection).std);
      harmonicRichnessVariability.push(AudioUtils.calculateStatistics(harmonicsSection).std);
    }
    const overallUniformity = 1 - (AudioUtils.calculateStatistics(spectralCentroid).std / (AudioUtils.calculateMean(spectralCentroid) || 1));

    return {
      timbreVariability,
      harmonicRichnessVariability,
      overallUniformity: Math.max(0, Math.min(1, overallUniformity))
    };
  }

  private identifyErrors(pitchTrack: number[], beats: number[], tempo: number): PerformanceConsistencyAnalysis["errorAnalysis"] {
    const errors = [];
    // Simplified error detection - look for large pitch deviations
    const pitchDeviations = this.calculatePitchDeviations(pitchTrack);
    for (let i = 0; i < pitchDeviations.length; i++) {
      const deviation = pitchDeviations[i];
      if (deviation !== undefined && Math.abs(deviation) > 50) { // 50 cents deviation
        errors.push({ time: i * (512/44100), type: "pitch", severity: Math.abs(deviation) / 100 });
      }
    }
    // Simplified rhythm error detection
    const beatIntervals: number[] = [];
    for (let i = 1; i < beats.length; i++) {
      const currentBeat = beats[i];
      const previousBeat = beats[i-1];
      if (currentBeat !== undefined && previousBeat !== undefined) {
        beatIntervals.push(currentBeat - previousBeat);
      }
    }
    const expectedInterval = 60 / tempo;
    for (let i = 0; i < beatIntervals.length; i++) {
      const interval = beatIntervals[i];
      const beat = beats[i];
      if (interval !== undefined && beat !== undefined) {
        const deviation = Math.abs(interval - expectedInterval);
        if (deviation > 0.1) { // 100ms deviation
          errors.push({ time: beat, type: "rhythm", severity: deviation / expectedInterval });
        }
      }
    }

    return { errorCount: errors.length, errorDistribution: errors as Array<{ time: number; type: "pitch" | "rhythm" | "tone"; severity: number; }> };
  }

  private calculatePitchDeviations(pitchTrack: number[]): number[] {
    if (pitchTrack.length === 0) return [];
    
    // Calculate deviations from equal temperament in cents
    return pitchTrack.map(freq => {
      if (freq <= 0) return 0;
      
      // Find closest semitone
      const semitonesFromA4 = 12 * Math.log2(freq / 440);
      const closestSemitone = Math.round(semitonesFromA4);
      const expectedFreq = 440 * Math.pow(2, closestSemitone / 12);
      
      // Convert to cents deviation
      const centsDeviation = 1200 * Math.log2(freq / expectedFreq);
      return centsDeviation;
    });
  }

  private analyzeEndurancePatterns(
    pitchTrack: number[], 
    spectralCentroid: number[], 
    numSections: number, 
    sectionLength: number
  ): PerformanceConsistencyAnalysis["endurancePatterns"] {
    // Calculate consistency degradation over time
    const sectionQualities = [];
    const fatigueIndicators: number[] = [];
    
    for (let i = 0; i < numSections; i++) {
      const pitchSection = pitchTrack.slice(i * sectionLength, (i + 1) * sectionLength);
      const centroidSection = spectralCentroid.slice(i * sectionLength, (i + 1) * sectionLength);
      
      // Calculate section quality metrics
      const pitchStability = 1 - (AudioUtils.calculateStatistics(pitchSection).std / 50);
      const toneStability = 1 - (AudioUtils.calculateStatistics(centroidSection).std / (AudioUtils.calculateMean(centroidSection) || 1));
      const sectionQuality = (pitchStability + toneStability) / 2;
      
      sectionQualities.push(Math.max(0, Math.min(1, sectionQuality)));
      
      // Detect fatigue indicators as quality degradation values
      if (i > 0) {
        const qualityDrop = (sectionQualities[i - 1] || 0) - sectionQuality;
        if (qualityDrop > 0.15) { // Significant quality drop
          fatigueIndicators.push(qualityDrop); // Store severity as number
        }
      }
    }
    
    // Calculate overall consistency throughout performance
    const consistencyThroughoutPerformance = sectionQualities.length > 1 
      ? 1 - (AudioUtils.calculateStatistics(sectionQualities).std / (AudioUtils.calculateMean(sectionQualities) || 1))
      : 0.5;
    
    return {
      consistencyThroughoutPerformance: Math.max(0, Math.min(1, consistencyThroughoutPerformance)),
      fatigueIndicators
    };
  }

  private analyzeRecoverySpeed(errorDistribution: Array<{ time: number; type: "pitch" | "rhythm" | "tone"; severity: number; }>): PerformanceConsistencyAnalysis["recoverySpeed"] {
    if (errorDistribution.length === 0) {
      return {
        recoveryEffectiveness: 0.8, // No errors means good recovery potential
        errorImpactOnSubsequentPerformance: 0.1
      };
    }
    
    let totalRecoveryTime = 0;
    let recoveryCount = 0;
    let errorImpactSum = 0;
    
    // Analyze recovery patterns after errors
    for (let i = 0; i < errorDistribution.length - 1; i++) {
      const currentError = errorDistribution[i];
      const nextError = errorDistribution[i + 1];
      
      if (currentError && nextError) {
        const timeBetweenErrors = nextError.time - currentError.time;
        
        // If time between errors is > 2 seconds, consider it a recovery
        if (timeBetweenErrors > 2.0) {
          totalRecoveryTime += timeBetweenErrors;
          recoveryCount++;
          
          // Calculate impact: how much did the previous error affect the next performance?
          const severityIncrease = nextError.severity - currentError.severity;
          errorImpactSum += Math.max(0, severityIncrease);
        }
      }
    }
    
    // Calculate recovery effectiveness
    const avgRecoveryTime = recoveryCount > 0 ? totalRecoveryTime / recoveryCount : 5.0;
    const recoveryEffectiveness = Math.max(0, Math.min(1, 1 - (avgRecoveryTime - 2.0) / 8.0)); // 2-10 second range
    
    // Calculate error impact on subsequent performance
    const avgErrorImpact = recoveryCount > 0 ? errorImpactSum / recoveryCount : 0;
    const errorImpactOnSubsequentPerformance = Math.min(1, avgErrorImpact);
    
    return {
      recoveryEffectiveness,
      errorImpactOnSubsequentPerformance
    };
  }

  private analyzeErrorFrequencyAndType(
    errorDistribution: Array<{ time: number; type: "pitch" | "rhythm" | "tone"; severity: number; }>,
    pitchTrack: number[],
    beats: number[]
  ): PerformanceConsistencyAnalysis["errorFrequencyAndType"] {
    let missedNotes = 0;
    let crackedNotes = 0;
    let timingSlips = 0;
    let otherErrors = 0;
    
    // Analyze pitch-based errors for missed/cracked notes
    const pitchDeviations = this.calculatePitchDeviations(pitchTrack);
    for (const deviation of pitchDeviations) {
      if (Math.abs(deviation) > 100) { // > 100 cents deviation
        if (deviation === 0) {
          missedNotes++; // Zero pitch often indicates missed note
        } else if (Math.abs(deviation) > 200) {
          crackedNotes++; // Very large deviations often indicate cracked notes
        }
      }
    }
    
    // Analyze rhythm-based errors for timing slips
    const beatIntervals: number[] = [];
    for (let i = 1; i < beats.length; i++) {
      const currentBeat = beats[i];
      const previousBeat = beats[i-1];
      if (currentBeat !== undefined && previousBeat !== undefined) {
        beatIntervals.push(currentBeat - previousBeat);
      }
    }
    
    if (beatIntervals.length > 0) {
      const expectedInterval = AudioUtils.calculateMean(beatIntervals);
      for (const interval of beatIntervals) {
        const deviation = Math.abs(interval - expectedInterval);
        if (deviation > 0.15 * expectedInterval) { // > 15% timing deviation
          timingSlips++;
        }
      }
    }
    
    // Count other errors from error distribution
    for (const error of errorDistribution) {
      if (error.type === "tone" || (error.type === "pitch" && error.severity < 1.0)) {
        otherErrors++;
      }
    }
    
    return {
      missedNotes,
      crackedNotes,
      timingSlips,
      otherErrors
    };
  }

}
