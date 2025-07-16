// Using Date instead of Firebase Timestamp for independence from firebase-admin
export type Timestamp = Date;

export interface PitchIntonationAnalysis {
  pitchAccuracyDistribution: {
    deviationStats: { 
      mean: number; 
      std: number; 
      range: [number, number]; 
    };
    frequencyRanges: Array<{ 
      range: string; 
      accuracy: number; 
    }>;
    targetFrequencyDeviations: number[];
  };
  intonationStability: {
    sustainedNoteConsistency: number;
    pitchVariationWithinNotes: number[];
    stabilityByRegister: Record<string, number>;
  };
  intervalPrecision: {
    melodicIntervalAccuracy: Array<{ 
      interval: string; 
      accuracy: number; 
    }>;
    harmonicIntervalAccuracy: Array<{ 
      interval: string; 
      accuracy: number; 
    }>;
    overallIntervalScore: number;
  };
  tuningDriftPatterns: {
    driftOverTime: number[];
    driftDirection: "flat" | "sharp" | "stable";
    driftRate: number;
    performanceRegions: Array<{ 
      time: number; 
      drift: number; 
    }>;
  };
}

export interface TimingRhythmAnalysis {
  temporalAccuracy: {
    metronomeDeviation: number[];
    backingTrackSync: number;
    overallTimingScore: number;
    rushTendency: number;
  };
  rhythmicSubdivisionPrecision: {
    subdivisionAccuracy: Record<string, number>;
    complexPatternSuccess: number;
    polyrhythmStability: number;
  };
  grooveConsistency: {
    swingRatio: number;
    grooveStability: number;
    microTimingVariations: number[];
    styleAdherence: number;
  };
  rubatoControl: {
    intentionalTempoChanges: Array<{ 
      time: number; 
      change: number; 
      intentional: boolean; 
    }>;
    unintentionalFluctuations: number;
    phraseTiming: number[];
    expressiveTimingScore: number;
  };
}

export interface ToneQualityTimbreAnalysis {
  harmonicContentAnalysis: {
    overtoneDistribution: number[];
    harmonicRichness: number;
    fundamentalToHarmonicRatio: number;
    spectralCentroid: number[];
  };
  dynamicRangeUtilization: {
    volumeRange: [number, number];
    dynamicVariationEffectiveness: number;
    crescendoDecrescendoControl: number;
    accentClarity: number;
  };
  timbralConsistency: {
    toneStabilityAcrossRegisters: Record<string, number>;
    toneStabilityAcrossDynamics: Record<string, number>;
    overallTimbralUniformity: number;
    colorVariationControl: number;
  };
  vibratoCharacteristics: {
    vibratoRate: number;
    vibratoDepth: number;
    vibratoControl: number;
    vibratoTiming: Array<{ 
      start: number; 
      end: number; 
      quality: number; 
    }>;
  };
}

export interface TechnicalExecutionAnalysis {
  articulationClarity: {
    tonguingPrecision: number;
    attackConsistency: number;
    articulationTypes: Record<string, number>;
    clarityByTempo: Record<string, number>;
  };
  fingerTechniqueEfficiency: {
    keyTransitionSmoothness: number;
    passageCleanness: number;
    fingeringAccuracy: number;
    technicalPassageSuccess: Record<string, number>;
  };
  breathManagementIndicators: {
    phraseLength: number[];
    breathingPlacement: Array<{ 
      time: number; 
      appropriateness: number; 
    }>;
    sustainCapacity: number;
    breathSupportConsistency: number;
  };
  extendedTechniqueMastery: {
    multiphonicsClarity: number;
    altissimoControl: number;
    growlExecution: number;
    bendAccuracy: number;
    otherTechniques: Record<string, number>;
  };
}

export interface MusicalExpressionAnalysis {
  phrasing: {
    phraseLengths: number[]; // in seconds
    breathingPoints: number[]; // timestamps in seconds
    phraseContour: Array<{
      startTime: number;
      endTime: number;
      pitchContour: "ascending" | "descending" | "mixed" | "flat";
      dynamicContour: "crescendo" | "decrescendo" | "steady" | "mixed";
    }>;
  };
  dynamics: {
    dynamicRange: {
      minDecibels: number;
      maxDecibels: number;
      overallRange: number; // in dB
    };
    dynamicChanges: Array<{
      time: number;
      type: "crescendo" | "decrescendo" | "sudden_accent";
      magnitude: number; // in dB
    }>;
    dynamicConsistency: number; // score 0-1
  };
  articulation: {
    attackClarity: number; // score 0-1
    noteSeparation: "legato" | "staccato" | "mixed";
  };
  vibrato: {
    rate: number; // in Hz
    depth: number; // in cents
    consistency: number; // score 0-1
    sections: Array<{ startTime: number; endTime: number }>;
  };
  phrasingSophistication: {
    musicalSentenceStructure: number;
    breathingLogic: number;
    phraseShaping: number;
  };
  dynamicContourComplexity: {
    dynamicShapingScore: number;
    dynamicRangeUtilization: number;
  };
  stylisticAuthenticity: {
    styleConsistency: number;
    genreAdherence: number;
  };
}

export interface PerformanceConsistencyAnalysis {
  pitchConsistency: {
    driftOverTime: number[]; // cents deviation from start
    standardDeviationBySection: number[]; // std dev for each quarter of the performance
    overallStability: number; // score 0-1
  };
  rhythmConsistency: {
    tempoFluctuation: number[]; // bpm deviation from average
    beatTimingDeviation: number[]; // ms deviation from grid
    overallSteadiness: number; // score 0-1
  };
  toneConsistency: {
    timbreVariability: number[]; // spectral centroid std dev by section
    harmonicRichnessVariability: number[]; // harmonic richness std dev by section
    overallUniformity: number; // score 0-1
  };
  errorAnalysis: {
    errorCount: number;
    errorDistribution: Array<{
      time: number;
      type: "pitch" | "rhythm" | "tone";
      severity: number; // 0-1
    }>;
  };
  endurancePatterns: {
    consistencyThroughoutPerformance: number;
    fatigueIndicators: number[]; // e.g., increase in error rate, decrease in dynamic range
  };
  recoverySpeed: {
    recoveryEffectiveness: number; // score 0-1
    errorImpactOnSubsequentPerformance: number; // score 0-1, lower is better
  };
  errorFrequencyAndType: {
    missedNotes: number;
    crackedNotes: number;
    timingSlips: number;
    otherErrors: number;
  };
}

export interface PerformanceScore {
  overallScore: number;
  categoryScores: {
    pitchIntonation: number;
    timingRhythm: number;
    toneQuality: number;
    technicalExecution: number;
    musicalExpression: number;
    consistency: number;
  };
  strengthAreas: string[];
  improvementAreas: string[];
  specificFeedback: string[];
  nextLevelRecommendations: string[];
}

export interface BasicAnalysis {
  tempo: {
    bpm: number;
    confidence: number;
  };
  rhythm: {
    beats: number[];
    onsets: number[];
  };
  pitch: {
    fundamentalFreq: number;
    melody: number[];
  };
  harmony: {
    key: string;
    chords: string[];
    chroma: number[];
  };
  spectral: {
    mfcc: number[];
    centroid: number;
    rolloff: number;
  };
  quality: {
    snr: number;
    clarity: number;
  };
}

export interface ExtendedAudioAnalysis extends BasicAnalysis {
  pitchIntonation: PitchIntonationAnalysis;
  timingRhythm: TimingRhythmAnalysis;
  toneQualityTimbre: ToneQualityTimbreAnalysis;
  technicalExecution: TechnicalExecutionAnalysis;
  musicalExpression: MusicalExpressionAnalysis;
  performanceConsistency: PerformanceConsistencyAnalysis;
  performanceScore: PerformanceScore;
  duration: number;
  processedAt: Timestamp;
  analysisVersion: string;
  confidenceScores: Record<string, number>;
}