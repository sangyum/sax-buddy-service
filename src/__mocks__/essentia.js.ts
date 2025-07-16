// Mock implementation of essentia.js for testing
export class EssentiaWASM {
  async initialize(): Promise<void> {
    // Mock initialization - no-op
  }
}

export class Essentia {
  version = "0.1.3";
  algorithmNames = "mock-algorithms";

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  constructor(_: EssentiaWASM) {
    // Mock constructor
  }

  // Mock Essentia methods with realistic return values
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  arrayToVector(_array: Float32Array): unknown {
    return { delete: () => {} }; // Mock EssentiaVector with delete method
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  vectorToArray(_vector: unknown): Float32Array {
    // Return a mock array with some non-zero values to pass validation
    const array = new Float32Array(1024);
    for (let i = 0; i < 1024; i++) {
      array[i] = Math.sin(2 * Math.PI * 440 * i / 44100) * 0.5; // Mock audio data
    }
    return array;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  YinFFT(_spectrum: unknown, _frameSize: number, _sampleRate: number): unknown {
    return {
      pitch: 440.0,
      pitchConfidence: 0.8
    };
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  RhythmExtractor2013(_vector: unknown, _maxTempo: number, _method: string, _minTempo: number): unknown {
    return {
      bpm: 120.0,
      confidence: 0.9,
      beats: [0.5, 1.0, 1.5, 2.0]
    };
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  Onsets(_vector: unknown): unknown {
    return {
      onsets: [0.5, 1.0, 1.5, 2.0]
    };
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  MFCC(_spectrum: unknown, _numberBands: number, _numberCoeffs: number, _sampleRate: number): unknown {
    return {
      mfcc: new Array(13).fill(0).map((_, i) => Math.sin(i * 0.1))
    };
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  SpectralCentroid(_spectrum: unknown, _sampleRate: number): unknown {
    return {
      spectralCentroid: 0.6 // Normalized spectral centroid
    };
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  SpectralRolloff(_spectrum: unknown, _sampleRate: number): unknown {
    return {
      spectralRolloff: 0.7 // Normalized spectral rolloff
    };
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  SpectralFlux(): unknown {
    return {
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      compute: (_spectrum: Float32Array): number => {
        return 0.3; // Mock spectral flux
      }
    };
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  HarmonicPeaks(_spectrum: unknown, _pitch: number): unknown {
    return {
      magnitudes: [0.8, 0.4, 0.2] // Normalized harmonic magnitudes
    };
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  Energy(_vector: unknown): unknown {
    return {
      energy: 0.5
    };
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  ZeroCrossingRate(_vector: Float32Array): unknown {
    return {
      zeroCrossingRate: 0.1
    };
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  Loudness(_vector: unknown, _sampleRate: number): unknown {
    return {
      loudness: 0.7
    };
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  Windowing(_vector: unknown, _type: string, _size: number, _normalized: boolean, _normalizeGain: number, _zeroPhase: boolean): unknown {
    return {
      frame: { delete: () => {} }
    };
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  Spectrum(_vector: unknown, _size: number): unknown {
    return {
      spectrum: { delete: () => {} }
    };
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  FrameGenerator(_audio: Float32Array, _frameSize: number, _hopSize: number): unknown[] {
    const frames: unknown[] = [];
    for (let i = 0; i < 10; i++) {
      frames.push({ delete: () => {} });
    }
    return frames;
  }

  // Add any other Essentia methods that might be used
  [key: string]: unknown;
}