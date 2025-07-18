import { MusicalExpressionAnalysis, EssentiaAnalysisResult } from "../../types";
import { AudioUtils } from "./AudioUtils";

export interface MusicalExpressionConfig {
  sampleRate: number;
  onset: {
    threshold: number;
    minSilenceLength: number;
  };
}

export class MusicalExpressionAnalyzer {
  private config: MusicalExpressionConfig;  

  constructor(config: MusicalExpressionConfig) {
    this.config = config;
  }

  async analyze(analysisResult: EssentiaAnalysisResult): Promise<MusicalExpressionAnalysis> {
    const { pitchTrack, energy, vibratoCharacteristics, onsets } = analysisResult;

    // 1. Phrasing Analysis
    const phraseAnalysis = this.analyzePhrasing(onsets, pitchTrack, energy);

    // 2. Dynamics Analysis
    const dynamicAnalysis = this.analyzeDynamics(energy);

    // 3. Articulation Analysis
    const articulationAnalysis = this.analyzeArticulation(onsets, energy);

    // 4. Vibrato Analysis (using pre-computed vibrato data)
    const vibratoAnalysis = {
      rate: vibratoCharacteristics.vibratoRate,
      depth: vibratoCharacteristics.vibratoDepth,
      consistency: vibratoCharacteristics.vibratoControl,
      sections: vibratoCharacteristics.vibratoTiming.map(v => ({ startTime: v.start, endTime: v.end }))
    };

    return {
      phrasing: phraseAnalysis,
      dynamics: dynamicAnalysis,
      articulation: articulationAnalysis,
      vibrato: vibratoAnalysis,
      phrasingSophistication: {
        musicalSentenceStructure: 0.5,
        breathingLogic: 0.5,
        phraseShaping: 0.5,
      },
      dynamicContourComplexity: {
        dynamicShapingScore: 0.5,
        dynamicRangeUtilization: 0.5,
      },
      stylisticAuthenticity: {
        styleConsistency: 0.5,
        genreAdherence: 0.5,
      },
    };
    
  }

  private analyzeArticulation(onsets: number[], energy: number[]): MusicalExpressionAnalysis["articulation"] {
    const attackTransients = this.analyzeAttackTransients(onsets, energy);
    const attackClarity = this.calculateMean(attackTransients.map(t => t.clarity));
    const noteSeparation = this.determineNoteSeparation(onsets);

    return { attackClarity, noteSeparation };
  }

  private calculateMean(values: number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  private determineNoteSeparation(onsets: number[]): MusicalExpressionAnalysis["articulation"]["noteSeparation"] {
    if (onsets.length < 2) return "mixed";

    const interOnsetIntervals = [];
    for (let i = 1; i < onsets.length; i++) {
      const currentOnset = onsets[i];
      const previousOnset = onsets[i - 1];
      if (currentOnset !== undefined && previousOnset !== undefined) {
        interOnsetIntervals.push(currentOnset - previousOnset);
      }
    }

    const meanInterval = this.calculateMean(interOnsetIntervals);
    const stdDevInterval = AudioUtils.calculateStatistics(interOnsetIntervals).std;

    if (stdDevInterval < 0.05 * meanInterval) { // Very consistent intervals
      if (meanInterval < 0.2) return "legato"; // Short, consistent intervals
      else return "staccato"; // Longer, consistent intervals
    } else {
      return "mixed"; // Inconsistent intervals
    }
  }

  private analyzeAttackTransients(onsets: number[], energy: number[]): Array<{ time: number; clarity: number }> {
    const transients: Array<{ time: number; clarity: number }> = [];
    const windowSize = Math.floor(0.05 * this.config.sampleRate); // 50ms window

    for (const onsetTime of onsets) {
      const onsetIndex = Math.floor(onsetTime * this.config.sampleRate);
      const start = Math.max(0, onsetIndex - windowSize);
      const end = Math.min(energy.length, onsetIndex + windowSize);
      const window = energy.slice(start, end);

      if (window.length > 0) {
        const maxEnergy = Math.max(...window);
        const minEnergy = Math.min(...window);
        const clarity = maxEnergy > 0 ? (maxEnergy - minEnergy) / maxEnergy : 0; // Simple clarity metric
        transients.push({ time: onsetTime, clarity });
      }
    }
    return transients;
  }

  private analyzeDynamics(energy: number[]): MusicalExpressionAnalysis["dynamics"] {
    const energyDb = energy.map(e => 20 * Math.log10(Math.max(e, 1e-10)));
    const minDb = Math.min(...energyDb);
    const maxDb = Math.max(...energyDb);
    const dynamicRange = { minDecibels: minDb, maxDecibels: maxDb, overallRange: maxDb - minDb };
    const dynamicChanges = this.detectDynamicChanges(energyDb);
    const dynamicConsistency = 1 - (AudioUtils.calculateStatistics(energyDb).std / dynamicRange.overallRange);

    return {
      dynamicRange,
      dynamicChanges,
      dynamicConsistency: Math.max(0, Math.min(1, dynamicConsistency))
    };
  }

  private detectDynamicChanges(energyDb: number[]): MusicalExpressionAnalysis["dynamics"]["dynamicChanges"] {
    const changes: MusicalExpressionAnalysis["dynamics"]["dynamicChanges"] = [];
    const threshold = 6; // 6 dB change
    const windowSize = 5;

    for (let i = windowSize; i < energyDb.length - windowSize; i++) {
      const prevAvg = this.calculateMean(energyDb.slice(i - windowSize, i));
      const current = energyDb[i];
      const nextAvg = this.calculateMean(energyDb.slice(i + 1, i + 1 + windowSize));

      if (current && prevAvg && nextAvg) {
        if (current > prevAvg + threshold && current > nextAvg + threshold) {
          changes.push({ time: i * (512 / this.config.sampleRate), type: "sudden_accent", magnitude: current - prevAvg });
        } else if (nextAvg > current + threshold && prevAvg < current - threshold) {
          changes.push({ time: i * (512 / this.config.sampleRate), type: "crescendo", magnitude: nextAvg - current });
        } else if (nextAvg < current - threshold && prevAvg > current + threshold) {
          changes.push({ time: i * (512 / this.config.sampleRate), type: "decrescendo", magnitude: current - nextAvg });
        }
      }
    }
    return changes;
  }


  // Helper methods for rule-based analysis
  private analyzePhrasing(onsets: number[], pitchTrack: number[], energy: number[]): MusicalExpressionAnalysis["phrasing"] {
    const phraseBreaks = this.detectPhraseBreaks(onsets, this.config.onset.threshold);
    const phraseLengths = AudioUtils.calculatePhraseLengths(onsets, phraseBreaks);
    const phraseContour = this.analyzePhraseContours(onsets, pitchTrack, energy, phraseBreaks);

    return {
      phraseLengths,
      breathingPoints: phraseBreaks,
      phraseContour
    };
  }

  // Helper methods for rule-based analysis
  private detectPhraseBreaks(onsets: number[], threshold: number): number[] {
    const breaks: number[] = [];
    if (onsets.length < 2) return breaks;

    for (let i = 1; i < onsets.length; i++) {
      const currentOnset = onsets[i];
      const previousOnset = onsets[i - 1];
      if (currentOnset !== undefined && previousOnset !== undefined && currentOnset - previousOnset > threshold) {
        breaks.push(currentOnset);
      }
    }
    return breaks;
  }

  
  private analyzePhraseContours(onsets: number[], pitchTrack: number[], energy: number[], phraseBreaks: number[]): MusicalExpressionAnalysis["phrasing"]["phraseContour"] {
    const contours: MusicalExpressionAnalysis["phrasing"]["phraseContour"] = [];
    let lastBreakTime = 0;
    for (const breakTime of phraseBreaks) {
      const startIndex = Math.floor(lastBreakTime * (this.config.sampleRate / 512)); // Assuming 512 hop size
      const endIndex = Math.floor(breakTime * (this.config.sampleRate / 512));
      const phrasePitch = pitchTrack.slice(startIndex, endIndex);
      const phraseEnergy = energy.slice(startIndex, endIndex);

      let pitchContour: "ascending" | "descending" | "mixed" | "flat" = "flat";
      if (phrasePitch.length > 1) {
        const firstPitch = phrasePitch[0];
        const lastPitch = phrasePitch[phrasePitch.length - 1];
        if (firstPitch && lastPitch) {
          if (lastPitch > firstPitch + 10) pitchContour = "ascending"; // 10 Hz threshold
          else if (lastPitch < firstPitch - 10) pitchContour = "descending";
          else pitchContour = "flat";
        }
      }

      let dynamicContour: "crescendo" | "decrescendo" | "steady" | "mixed" = "steady";
      if (phraseEnergy.length > 1) {
        const firstEnergy = phraseEnergy[0];
        const lastEnergy = phraseEnergy[phraseEnergy.length - 1];
        if (firstEnergy && lastEnergy) {
          if (lastEnergy > firstEnergy * 1.5) dynamicContour = "crescendo"; // 50% increase
          else if (lastEnergy < firstEnergy * 0.5) dynamicContour = "decrescendo";
          else dynamicContour = "steady";
        }
      }

      contours.push({
        startTime: lastBreakTime,
        endTime: breakTime,
        pitchContour,
        dynamicContour,
      });
      lastBreakTime = breakTime;
    }
    return contours;
  }

}
