declare module "essentia.js" {
  export class EssentiaWASM {
    initialize(): Promise<void>;
  }

  export class Essentia {
    constructor(wasmModule: EssentiaWASM);
    version?: string;
    algorithmNames?: string;
    
    // Audio processing methods
    arrayToVector(array: Float32Array): EssentiaVector;
    vectorToArray(vector: EssentiaVector): Float32Array;
    FrameGenerator(buffer: Float32Array, frameSize: number, hopSize: number): EssentiaVector[];
    RhythmExtractor2013(vector: EssentiaVector, maxTempo: number, method: string, minTempo: number): RhythmResult;
    YinFFT(spectrum: EssentiaVector, frameSize: number, sampleRate: number): PitchResult;
    MFCC(spectrum: EssentiaVector, numberBands: number, numberCoeffs: number, sampleRate: number): MfccResult;
    OnsetDetection(spectrum: EssentiaVector, method: string, sampleRate: number): OnsetResult;
    SpectralCentroid(spectrum: EssentiaVector, sampleRate: number): SpectralCentroidResult;
    SpectralRolloff(spectrum: EssentiaVector, sampleRate: number): SpectralRolloffResult;
    ChromaSTFT(spectrum: EssentiaVector, sampleRate: number): ChromaResult;
    Loudness(spectrum: EssentiaVector, sampleRate: number): LoudnessResult;
    Windowing(vector: EssentiaVector, type: string, size: number, normalized: boolean, normalizeGain: number, zeroPhase: boolean): WindowingResult;
    Spectrum(vector: EssentiaVector, size: number): SpectrumResult;
    Energy(vector: EssentiaVector): EnergyResult;
    ZeroCrossingRate(vector: Float32Array): ZcrResult;
    HarmonicPeaks(spectrum: EssentiaVector, pitch: number): HarmonicPeaksResult;
    Onsets(vector: EssentiaVector): OnsetResult;
    
    // Allow other methods
    [key: string]: unknown;
  }
  
  export interface EssentiaVector {
    delete?: () => void;
    [key: string]: unknown;
  }
  
  export interface RhythmResult {
    bpm: number;
    confidence: number;
    beats: number[];
  }
  
  export interface PitchResult {
    pitch: number;
    pitchConfidence: number;
  }
  
  export interface MfccResult {
    mfcc: number[];
  }
  
  export interface OnsetResult {
    onsets: number[];
  }
  
  export interface SpectralCentroidResult {
    spectralCentroid: number;
  }
  
  export interface SpectralRolloffResult {
    spectralRolloff: number;
  }
  
  export interface ChromaResult {
    chroma: number[];
  }
  
  export interface LoudnessResult {
    loudness: number;
  }
  
  export interface WindowingResult {
    frame: EssentiaVector;
  }
  
  export interface SpectrumResult {
    spectrum: EssentiaVector;
  }
  
  export interface EnergyResult {
    energy: number;
  }
  
  export interface ZcrResult {
    zeroCrossingRate: number;
  }
  
  export interface HarmonicPeaksResult {
    magnitudes: number[];
  }
}