declare module 'essentia.js' {
  export class EssentiaWASM {
    initialize(): Promise<void>;
  }

  export class Essentia {
    constructor(wasmModule: EssentiaWASM);
    
    YinFFT(audio: any): { pitch: any; pitchConfidence: any };
    MFCC(spectrum: any): { mfcc: any };
    SpectralCentroid(spectrum: any): { spectralCentroid: number };
    SpectralRolloff(spectrum: any): { spectralRolloff: number };
    ZeroCrossingRate(audio: any): { zeroCrossingRate: number };
    RhythmExtractor2013(audio: any): { bpm: number; beats: any; beatLoudness: any };
    BeatTrackerDegara(audio: any): { ticks: any };
    arrayToVector(array: Float32Array): any;
    vectorToArray(vector: any): Float32Array;
    Windowing(audio: any, options?: any): { frame: any };
    Spectrum(frame: any): { spectrum: any };
    AutoCorrelation(audio: any): { autoCorrelation: any };
    FrameGenerator(audio: any, options?: any): { frames: any };
  }
}