export interface EssentiaAnalysisResult {
  tempo: number;
  confidence: number;
  beats: number[];
  pitchTrack: number[];
  pitchConfidence: number[];
  onsets: number[];
  mfcc: number[];
  spectralCentroid: number[];
  spectralRolloff: number[];
  spectralFlux: number[];
  harmonics: number[];
  energy: number[];
  zcr: number[];
  loudness: number[];
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

export interface ArticulationClarityAnalysis {
  tonguingPrecision: number;
  attackConsistency: number;
  articulationTypes: Record<string, number>;
  clarityByTempo: Record<string, number>;
}

export interface FingerTechniqueAnalysis {
  keyTransitionSmoothness: number;
  passageCleanness: number;
  fingeringAccuracy: number;
  technicalPassageSuccess: Record<string, number>;
}

export interface BreathManagementAnalysis {
  phraseLength: number[];
  breathingPlacement: { time: number; appropriateness: number }[];
  sustainCapacity: number;
  breathSupportConsistency: number;
}

export interface ExtendedTechniquesAnalysis {
  multiphonicsClarity: number;
  altissimoControl: number;
  growlExecution: number;
  bendAccuracy: number;
  otherTechniques: Record<string, number>;
}

export interface ArticulationTypesAnalysis extends Record<string, number> {
  legato: number;
  staccato: number;
  tenuto: number;
  overallArticulationScore: number;
}

export interface ClarityByTempoAnalysis extends Record<string, number> {
  slow: number;
  medium: number;
  fast: number;
  overall: number;
}

export interface TechnicalPassageAnalysis extends Record<string, number> {
  scales: number;
  arpeggios: number;
  chromatic: number;
  overall: number;
}

export interface OtherExtendedTechniquesAnalysis extends Record<string, number> {
  harmonicControl: number;
  noiseTextures: number;
  overallOtherScore: number;
}