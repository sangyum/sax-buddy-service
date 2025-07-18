import { TimingRhythmAnalysis, EssentiaAnalysisResult } from "../../types";
import { AudioUtils } from "./AudioUtils";

export class TimingRhythmAnalyzer {
  async analyze(analysisResult: EssentiaAnalysisResult): Promise<TimingRhythmAnalysis> {
    const temporalMetrics = this.analyzeTemporalAccuracy(
      analysisResult.beats,
      analysisResult.onsets,
      analysisResult.tempo
    );

    const subdivisionMetrics = this.analyzeRhythmicSubdivisions(
      analysisResult.onsets,
      analysisResult.tempo
    );

    const grooveMetrics = this.analyzeGrooveConsistency(
      analysisResult.beats,
      analysisResult.tempo
    );

    const rubatoMetrics = this.analyzeRubatoControl(
      analysisResult.beats,
      analysisResult.tempo
    );

    return {
      temporalAccuracy: temporalMetrics,
      rhythmicSubdivisionPrecision: subdivisionMetrics,
      grooveConsistency: grooveMetrics,
      rubatoControl: rubatoMetrics
    };
  }

  private analyzeTemporalAccuracy(
    beats: number[],
    onsets: number[],
    tempo: number
  ): {
    metronomeDeviation: number[];
    backingTrackSync: number;
    overallTimingScore: number;
    rushTendency: number;
  } {
    if (beats.length < 2) {
      return {
        metronomeDeviation: [],
        backingTrackSync: 0.5,
        overallTimingScore: 0.5,
        rushTendency: 0
      };
    }

    const expectedBeatInterval = 60 / tempo;

    const beatIntervals: number[] = [];
    for (let i = 1; i < beats.length; i++) {
      const current = beats[i];
      const previous = beats[i - 1];
      if (current !== undefined && previous !== undefined) {
        beatIntervals.push(current - previous);
      }
    }

    const metronomeDeviation = beatIntervals.map(interval =>
      (interval - expectedBeatInterval) * 1000
    );

    const rushTendency = metronomeDeviation.length > 0
      ? metronomeDeviation.reduce((sum, dev) => sum + dev, 0) / metronomeDeviation.length / 1000
      : 0;

    const avgDeviation = metronomeDeviation.length > 0
      ? metronomeDeviation.reduce((sum, dev) => sum + Math.abs(dev), 0) / metronomeDeviation.length
      : 0;

    const overallTimingScore = Math.max(0, 1 - (avgDeviation / 100));

    const backingTrackSync = overallTimingScore;

    return {
      metronomeDeviation,
      backingTrackSync,
      overallTimingScore,
      rushTendency
    };
  }

  private analyzeRhythmicSubdivisions(
    onsets: number[],
    tempo: number
  ): {
    subdivisionAccuracy: Record<string, number>;
    complexPatternSuccess: number;
    polyrhythmStability: number;
  } {
    if (onsets.length < 4) {
      return {
        subdivisionAccuracy: {
          quarter: 0.5,
          eighth: 0.5,
          sixteenth: 0.5,
          triplets: 0.5
        },
        complexPatternSuccess: 0.5,
        polyrhythmStability: 0.5
      };
    }

    const intervals: number[] = [];
    for (let i = 1; i < onsets.length; i++) {
      const current = onsets[i];
      const previous = onsets[i - 1];
      if (current !== undefined && previous !== undefined) {
        intervals.push(current - previous);
      }
    }

    const beatDuration = 60 / tempo;
    const expectedSubdivisions = {
      quarter: beatDuration,
      eighth: beatDuration / 2,
      sixteenth: beatDuration / 4,
      triplets: beatDuration / 3
    };

    const subdivisionAccuracy: Record<string, number> = {};

    Object.entries(expectedSubdivisions).forEach(([subdivision, expectedDuration]) => {
      const matchingIntervals = intervals.filter(interval =>
        Math.abs(interval - expectedDuration) < expectedDuration * 0.2
      );

      if (matchingIntervals.length === 0) {
        subdivisionAccuracy[subdivision] = 0.5;
        return;
      }

      const deviations = matchingIntervals.map(interval =>
        Math.abs(interval - expectedDuration) / expectedDuration
      );
      const avgDeviation = deviations.reduce((sum, dev) => sum + dev, 0) / deviations.length;

      subdivisionAccuracy[subdivision] = Math.max(0, 1 - avgDeviation * 5);
    });

    const usedSubdivisions = Object.values(subdivisionAccuracy).filter(score => score > 0.6).length;
    const complexPatternSuccess = usedSubdivisions / 4;

    const intervalVariability = AudioUtils.calculateStatistics(intervals).std;
    const polyrhythmStability = Math.max(0, 1 - (intervalVariability / beatDuration));

    return {
      subdivisionAccuracy,
      complexPatternSuccess,
      polyrhythmStability
    };
  }

  private analyzeGrooveConsistency(
    beats: number[],
    tempo: number
  ): {
    swingRatio: number;
    grooveStability: number;
    microTimingVariations: number[];
    styleAdherence: number;
  } {
    if (beats.length < 4) {
      return {
        swingRatio: 0.5,
        grooveStability: 0.5,
        microTimingVariations: [],
        styleAdherence: 0.5
      };
    }

    const beatIntervals: number[] = [];
    for (let i = 1; i < beats.length; i++) {
      const current = beats[i];
      const previous = beats[i - 1];
      if (current !== undefined && previous !== undefined) {
        beatIntervals.push(current - previous);
      }
    }

    const intervalStats = AudioUtils.calculateStatistics(beatIntervals);

    let swingRatio = 0.5;
    if (beatIntervals.length >= 2) {
      const variationCoeff = intervalStats.std / intervalStats.mean;
      if (variationCoeff > 0.1 && variationCoeff < 0.3) {
        swingRatio = 0.67;
      }
    }

    const grooveStability = Math.max(0, 1 - (intervalStats.std / intervalStats.mean));

    const expectedInterval = 60 / tempo;
    const microTimingVariations = beatIntervals.map(interval =>
      (interval - expectedInterval) * 1000
    );

    let styleAdherence = 0.8;

    const avgVariation = microTimingVariations.reduce((sum, v) => sum + Math.abs(v), 0) / microTimingVariations.length;

    if (avgVariation > 2 && avgVariation < 25) {
      styleAdherence = Math.min(1, styleAdherence + 0.1);
    }

    if (grooveStability > 0.7 && avgVariation > 1) {
      styleAdherence = Math.min(1, styleAdherence + 0.05);
    }

    return {
      swingRatio,
      grooveStability,
      microTimingVariations,
      styleAdherence
    };
  }

  private analyzeRubatoControl(
    beats: number[],
    tempo: number
  ): {
    intentionalTempoChanges: Array<{ time: number; change: number; intentional: boolean }>;
    unintentionalFluctuations: number;
    phraseTiming: number[];
    expressiveTimingScore: number;
  } {
    if (beats.length < 6) {
      return {
        intentionalTempoChanges: [],
        unintentionalFluctuations: 0,
        phraseTiming: [],
        expressiveTimingScore: 0.5
      };
    }

    const tempoChanges: Array<{ time: number; change: number; intentional: boolean }> = [];
    const windowSize = 4;

    for (let i = 0; i <= beats.length - windowSize; i++) {
      const window = beats.slice(i, i + windowSize);
      const intervals = [];
      for (let j = 1; j < window.length; j++) {
        const current = window[j];
        const previous = window[j - 1];
        if (current !== undefined && previous !== undefined) {
          intervals.push(current - previous);
        }
      }

      const avgInterval = intervals.length > 0 ? intervals.reduce((sum, int) => sum + int, 0) / intervals.length : 1;
      const instantTempo = 60 / avgInterval;
      const tempoChange = instantTempo - tempo;

      const intentional = Math.abs(tempoChange) > 10 &&
                         (i === 0 || Math.abs(tempoChanges[tempoChanges.length - 1]?.change || 0) > 5);

      if (Math.abs(tempoChange) > 5) {
        tempoChanges.push({
          time: window[0] || 0,
          change: tempoChange,
          intentional
        });
      }
    }

    const unintentionalChanges = tempoChanges.filter(change => !change.intentional);
    const unintentionalFluctuations = unintentionalChanges.length > 0
      ? unintentionalChanges.reduce((sum, change) => sum + Math.abs(change.change), 0) / unintentionalChanges.length / tempo
      : 0;

    const phraseTiming: number[] = [];
    let phraseStart = beats[0] || 0;

    for (let i = 1; i < beats.length; i++) {
      const currentBeat = beats[i];
      const previousBeat = beats[i - 1];
      if (currentBeat === undefined || previousBeat === undefined) continue;

      const gap = currentBeat - previousBeat;
      const expectedGap = 60 / tempo;

      if (gap > expectedGap * 2) {
        const phraseLength = previousBeat - phraseStart;
        if (phraseLength > 1) {
          phraseTiming.push(phraseLength);
        }
        phraseStart = currentBeat;
      }
    }

    const lastBeat = beats[beats.length - 1];
    if (lastBeat !== undefined) {
      const finalPhraseLength = lastBeat - phraseStart;
      if (finalPhraseLength > 1) {
        phraseTiming.push(finalPhraseLength);
      }
    }

    const intentionalChanges = tempoChanges.filter(change => change.intentional);
    const hasExpressiveChanges = intentionalChanges.length > 0;
    const hasReasonableFluctuations = unintentionalFluctuations < 0.1;

    const expressiveTimingScore = (hasExpressiveChanges ? 0.6 : 0.4) +
                                 (hasReasonableFluctuations ? 0.4 : 0.2);

    return {
      intentionalTempoChanges: tempoChanges,
      unintentionalFluctuations,
      phraseTiming,
      expressiveTimingScore
    };
  }
}
