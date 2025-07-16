// Mock implementation of essentia.js for testing
export class EssentiaWASM {
  async initialize(): Promise<void> {
    // Mock initialization - no-op
  }
}

export class Essentia {
  version = "0.1.3";
  algorithmNames = "mock-algorithms";

  constructor(_wasmModule: EssentiaWASM) {
    // Mock constructor
  }

  // Mock Essentia methods with realistic return values
  arrayToVector(array: Float32Array): unknown {
    return array; // Just return the array for testing
  }

  vectorToArray(vector: unknown): Float32Array {
    return vector as Float32Array;
  }

  // Mock audio analysis algorithms
  PitchYin(params: unknown): unknown {
    return {
      compute: (frame: Float32Array): number[] => {
        // Return mock pitch values
        return [440.0, 0.8]; // [frequency, confidence]
      }
    };
  }

  RhythmExtractor2013(params: unknown): unknown {
    return {
      compute: (audio: Float32Array): number[] => {
        // Return mock rhythm data: [tempo, confidence, beats...]
        return [120.0, 0.9, 0.5, 1.0, 1.5, 2.0];
      }
    };
  }

  OnsetDetection(params: unknown): unknown {
    return {
      compute: (spectrum: Float32Array): number => {
        return 0.5; // Mock onset detection value
      }
    };
  }

  MFCC(params: unknown): unknown {
    return {
      compute: (spectrum: Float32Array): number[] => {
        // Return mock MFCC coefficients
        return new Array(13).fill(0).map((_, i) => Math.sin(i * 0.1));
      }
    };
  }

  SpectralCentroid(params: unknown): unknown {
    return {
      compute: (spectrum: Float32Array): number => {
        return 1000.0; // Mock spectral centroid
      }
    };
  }

  SpectralRolloff(params: unknown): unknown {
    return {
      compute: (spectrum: Float32Array): number => {
        return 2000.0; // Mock spectral rolloff
      }
    };
  }

  SpectralFlux(params: unknown): unknown {
    return {
      compute: (spectrum: Float32Array): number => {
        return 0.3; // Mock spectral flux
      }
    };
  }

  HarmonicPeaks(params: unknown): unknown {
    return {
      compute: (spectrum: Float32Array): number[] => {
        // Return mock harmonic peaks
        return [440.0, 880.0, 1320.0];
      }
    };
  }

  InstantPower(params: unknown): unknown {
    return {
      compute: (frame: Float32Array): number => {
        return 0.5; // Mock power value
      }
    };
  }

  ZeroCrossingRate(params: unknown): unknown {
    return {
      compute: (frame: Float32Array): number => {
        return 0.1; // Mock zero crossing rate
      }
    };
  }

  Loudness(params: unknown): unknown {
    return {
      compute: (frame: Float32Array): number => {
        return 0.7; // Mock loudness value
      }
    };
  }

  Windowing(params: unknown): unknown {
    return {
      compute: (frame: Float32Array): Float32Array => {
        return frame; // Return the frame as-is for testing
      }
    };
  }

  Spectrum(params: unknown): unknown {
    return {
      compute: (frame: Float32Array): Float32Array => {
        // Return mock spectrum
        const spectrum = new Float32Array(1024);
        for (let i = 0; i < spectrum.length; i++) {
          spectrum[i] = Math.random() * 0.5;
        }
        return spectrum;
      }
    };
  }

  FrameGenerator(params: unknown): unknown {
    return {
      compute: (audio: Float32Array): Float32Array[] => {
        // Return mock frames
        const frameSize = 1024;
        const frames: Float32Array[] = [];
        for (let i = 0; i < audio.length; i += frameSize) {
          const frame = audio.slice(i, i + frameSize);
          if (frame.length === frameSize) {
            frames.push(frame);
          }
        }
        return frames;
      }
    };
  }

  // Add any other Essentia methods that might be used
  [key: string]: unknown;
}