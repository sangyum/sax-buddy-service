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

    return {
      pitchConsistency,
      rhythmConsistency,
      toneConsistency,
      errorAnalysis,
      endurancePatterns: {
        consistencyThroughoutPerformance: 0.5,
        fatigueIndicators: [],
      },
      recoverySpeed: {
        recoveryEffectiveness: 0.5,
        errorImpactOnSubsequentPerformance: 0.5,
      },
      errorFrequencyAndType: {
        missedNotes: 0,
        crackedNotes: 0,
        timingSlips: 0,
        otherErrors: 0,
      },
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

}
