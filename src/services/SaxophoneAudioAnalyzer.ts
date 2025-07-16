import { 
  ExtendedAudioAnalysis, 
  PitchIntonationAnalysis,
  TimingRhythmAnalysis,
  ToneQualityTimbreAnalysis,
  TechnicalExecutionAnalysis,
  MusicalExpressionAnalysis,
  PerformanceConsistencyAnalysis,
  PerformanceScore,
  BasicAnalysis,
  EssentiaAnalysisResult,
  ArticulationClarityAnalysis,
  FingerTechniqueAnalysis,
  BreathManagementAnalysis,
  ExtendedTechniquesAnalysis,
  ArticulationTypesAnalysis,
  ClarityByTempoAnalysis,
  TechnicalPassageAnalysis,
  OtherExtendedTechniquesAnalysis
} from "../types";
import { Logger } from "../utils/logger";

// Import Essentia.js
import * as EssentiaJS from "essentia.js";
const { Essentia, EssentiaWASM } = EssentiaJS;


interface AnalysisConfig {
  sampleRate: number;
  frameSize: number;
  hopSize: number;
  pitchTracking: {
    smoothingFactor: number;
    medianFilterSize: number;
    confidenceThreshold: number;
    octaveErrorThreshold: number;
  };
  vibrato: {
    minRate: number;
    maxRate: number;
    minDepth: number;
    qualityThreshold: number;
  };
  onset: {
    threshold: number;
    minSilenceLength: number;
  };
  validation: {
    minDurationSec: number;
    maxDurationSec: number;
    minRms: number;
    maxAmplitude: number;
    minSnrDb: number;
  };
}

export class SaxophoneAudioAnalyzer {
  private logger = new Logger("SaxophoneAudioAnalyzer");
  private isInitialized = false;
  private essentia: EssentiaJS.Essentia | null = null;
  private essentiaWasm: EssentiaJS.EssentiaWASM | null = null;
  
  private config: AnalysisConfig = {
    sampleRate: 44100,
    frameSize: 2048,
    hopSize: 512,
    pitchTracking: {
      smoothingFactor: 0.3,
      medianFilterSize: 5,
      confidenceThreshold: 0.5,
      octaveErrorThreshold: 0.1
    },
    vibrato: {
      minRate: 3,
      maxRate: 10,
      minDepth: 10,
      qualityThreshold: 0.3
    },
    onset: {
      threshold: 1.5,
      minSilenceLength: 0.1
    },
    validation: {
      minDurationSec: 1.0,
      maxDurationSec: 300.0,
      minRms: 0.001,
      maxAmplitude: 0.99,
      minSnrDb: 20
    }
  };

  private get sampleRate(): number {
    return this.config.sampleRate;
  }

  updateConfig(partialConfig: Partial<AnalysisConfig>): void {
    this.config = {
      ...this.config,
      ...partialConfig,
      pitchTracking: {
        ...this.config.pitchTracking,
        ...partialConfig.pitchTracking
      },
      vibrato: {
        ...this.config.vibrato,
        ...partialConfig.vibrato
      },
      onset: {
        ...this.config.onset,
        ...partialConfig.onset
      },
      validation: {
        ...this.config.validation,
        ...partialConfig.validation
      }
    };
  }

  getConfig(): AnalysisConfig {
    return { ...this.config };
  }

  async initialize(): Promise<void> {
    if (!this.isInitialized) {
      this.logger.info("Initializing audio analyzer");
      
      try {
        // Initialize Essentia.js WASM backend
        this.essentiaWasm = new EssentiaWASM();
        await this.essentiaWasm.initialize();
        
        // Create Essentia.js instance
        this.essentia = new Essentia(this.essentiaWasm);
        
        this.isInitialized = true;
        this.logger.info("Audio analyzer initialized", {
          version: this.essentia?.version,
          algorithms: this.essentia?.algorithmNames?.split(",").length || 0
        });
      } catch (error) {
        this.logger.error("Failed to initialize Essentia.js", {
          error: error instanceof Error ? error.message : "Unknown error"
        });
        throw new Error(`Failed to initialize audio analyzer: ${error}`);
      }
    }
  }

  async analyzeExercise(
    audioBuffer: Float32Array
  ): Promise<ExtendedAudioAnalysis> {
    await this.initialize();
    
    // Validate input audio buffer
    const validationResult = this.validateAudioBuffer(audioBuffer);
    if (!validationResult.isValid) {
      throw new Error(`Invalid audio buffer: ${validationResult.error}`);
    }
    
    this.logger.info("Starting comprehensive audio analysis", {
      bufferLength: audioBuffer.length,
      duration: audioBuffer.length / this.sampleRate,
      snr: validationResult.snr,
      rms: validationResult.rms
    });

    try {
      // Perform Essentia analysis once
      const analysisResult = await this.performEssentiaAnalysis(audioBuffer);
      
      // Perform basic audio analysis using the result
      const basicAnalysis = this.performBasicAnalysisFromResult(analysisResult);
      
      // Perform extended analysis using the same result
      const extendedAnalysis: ExtendedAudioAnalysis = {
        ...basicAnalysis,
        pitchIntonation: await this.analyzePitchIntonation(analysisResult),
        timingRhythm: await this.analyzeTimingRhythm(analysisResult),
        toneQualityTimbre: await this.analyzeToneQualityTimbre(analysisResult),
        technicalExecution: await this.analyzeTechnicalExecution(analysisResult),
        musicalExpression: await this.analyzeMusicalExpression(analysisResult),
        performanceConsistency: await this.analyzePerformanceConsistency(analysisResult),
        performanceScore: {
          overallScore: 0,
          categoryScores: {
            pitchIntonation: 0,
            timingRhythm: 0,
            toneQuality: 0,
            technicalExecution: 0,
            musicalExpression: 0,
            consistency: 0,
          },
          strengthAreas: [],
          improvementAreas: [],
          specificFeedback: [],
          nextLevelRecommendations: [],
        },
        duration: audioBuffer.length / this.sampleRate,
        processedAt: new Date(),
        analysisVersion: "2.0.0",
        confidenceScores: {}
      };

      // Calculate performance score based on all analyses
      extendedAnalysis.performanceScore = this.calculatePerformanceScore(extendedAnalysis);
      extendedAnalysis.confidenceScores = this.calculateConfidenceScores(extendedAnalysis);

      this.logger.info("Audio analysis completed", {
        overallScore: extendedAnalysis.performanceScore.overallScore,
        duration: extendedAnalysis.duration
      });

      return extendedAnalysis;

    } catch (error) {
      this.logger.error("Audio analysis failed", {
        error: error instanceof Error ? error.message : "Unknown error",
        bufferLength: audioBuffer.length
      });
      throw error;
    }
  }

  private validateAudioBuffer(audioBuffer: Float32Array): {
    isValid: boolean;
    error?: string;
    snr?: number;
    rms?: number;
  } {
    // Check if buffer exists and is not empty
    if (!audioBuffer || audioBuffer.length === 0) {
      return { isValid: false, error: "Audio buffer is empty or null" };
    }

    // Check minimum duration
    const duration = audioBuffer.length / this.sampleRate;
    if (duration < this.config.validation.minDurationSec) {
      return { isValid: false, error: `Audio too short: ${duration.toFixed(2)}s (minimum ${this.config.validation.minDurationSec}s required)` };
    }

    // Check maximum duration
    if (duration > this.config.validation.maxDurationSec) {
      return { isValid: false, error: `Audio too long: ${duration.toFixed(2)}s (maximum ${this.config.validation.maxDurationSec}s)` };
    }

    // Check for NaN or infinite values
    for (let i = 0; i < audioBuffer.length; i++) {
      const sample = audioBuffer[i];
      if (sample === undefined || !isFinite(sample)) {
        return { isValid: false, error: `Invalid sample at index ${i}: ${sample}` };
      }
    }

    // Calculate RMS level
    const rms = Math.sqrt(
      audioBuffer.reduce((sum, sample) => sum + sample * sample, 0) / audioBuffer.length
    );

    // Check if signal is too quiet
    if (rms < this.config.validation.minRms) {
      return { isValid: false, error: `Signal too quiet: RMS = ${rms.toFixed(6)}` };
    }

    // Check if signal is clipping (avoid stack overflow with large arrays)
    let maxAmplitude = 0;
    for (let i = 0; i < audioBuffer.length; i++) {
      const sample = audioBuffer[i];
      if (sample !== undefined) {
        const absValue = Math.abs(sample);
        if (absValue > maxAmplitude) {
          maxAmplitude = absValue;
        }
      }
    }
    if (maxAmplitude > this.config.validation.maxAmplitude) {
      return { isValid: false, error: `Signal clipping detected: max amplitude = ${maxAmplitude.toFixed(3)}` };
    }

    // Estimate SNR (simple noise floor estimation)
    const sortedSamples = [...audioBuffer].map(Math.abs).sort((a, b) => a - b);
    const noiseFloor = sortedSamples[Math.floor(sortedSamples.length * 0.1)] || 1e-10; // 10th percentile
    const snr = 20 * Math.log10(rms / (noiseFloor + 1e-10));

    // Check SNR threshold
    if (snr < this.config.validation.minSnrDb) {
      return { isValid: false, error: `Poor signal quality: SNR = ${snr.toFixed(1)}dB (minimum ${this.config.validation.minSnrDb}dB)` };
    }

    return { isValid: true, snr, rms };
  }

  private performBasicAnalysisFromResult(analysisResult: EssentiaAnalysisResult): BasicAnalysis {
    return {
      tempo: {
        bpm: analysisResult.tempo,
        confidence: analysisResult.confidence
      },
      rhythm: {
        beats: analysisResult.beats,
        onsets: analysisResult.onsets
      },
      pitch: {
        fundamentalFreq: this.calculateMeanFrequency(analysisResult.pitchTrack),
        melody: analysisResult.pitchTrack
      },
      harmony: {
        key: this.estimateKey(analysisResult.pitchTrack),
        chords: this.estimateChords(analysisResult.pitchTrack),
        chroma: this.calculateChromaVector(analysisResult.pitchTrack)
      },
      spectral: {
        mfcc: analysisResult.mfcc,
        centroid: this.calculateMean(analysisResult.spectralCentroid),
        rolloff: this.calculateMean(analysisResult.spectralRolloff)
      },
      quality: {
        snr: 0.8, // Default value since we don't have audioBuffer
        clarity: this.calculateMean(analysisResult.harmonics)
      }
    };
  }

  private async performEssentiaAnalysis(audioBuffer: Float32Array): Promise<EssentiaAnalysisResult> {
    try {
      if (!this.essentia) {
        throw new Error("Essentia not initialized");
      }
      
      // Convert Float32Array to Essentia vector
      // Convert Float32Array to Essentia vector
      const audioVector = (this.essentia as EssentiaJS.Essentia).arrayToVector(audioBuffer);
      
      // 1. Rhythm and Tempo Analysis
      const rhythmResult = this.essentia.RhythmExtractor2013(
        audioVector as EssentiaJS.EssentiaVector,
        208,      // maxTempo
        "multifeature", // method
        40        // minTempo
      );

      // 2. Pitch Analysis using YinFFT
      const frameSize = this.config.frameSize;
      const hopSize = this.config.hopSize;
      const frames = (this.essentia as EssentiaJS.Essentia).FrameGenerator(audioBuffer, frameSize, hopSize);
      
      const pitchTrack: number[] = [];
      const pitchConfidence: number[] = [];
      const spectralFeatures = {
        centroid: [] as number[],
        rolloff: [] as number[],
        flux: [] as number[],
        mfcc: [] as number[][],
        energy: [] as number[],
        zcr: [] as number[],
        harmonics: [] as number[][]
      };

      // Process each frame
      const framesSize = frames?.length || 0;
      let processedFrames = 0;
      let skippedFrames = 0;
      
      for (let i = 0; i < framesSize; i++) {
        try {
          const frame = frames[i];
          if (!frame) {
            skippedFrames++;
            continue;
          }
          
          const frameArray = this.essentia.vectorToArray(frame as EssentiaJS.EssentiaVector);
          
          // Validate frame data
          if (!frameArray || frameArray.length === 0) {
            skippedFrames++;
            continue;
          }
          
          // Check for invalid values in frame
          const hasValidData = Array.from(frameArray).some(sample => isFinite(sample) && sample !== 0);
          if (!hasValidData) {
            skippedFrames++;
            continue;
          }
          
          // Apply windowing
          const windowedFrame = this.essentia.Windowing(
            this.essentia.arrayToVector(frameArray),
            "hann",
            frameSize,
            false, // normalized
            1.0,   // size
            true   // zeroPhase
          );

          // Compute spectrum
          const spectrumResult = this.essentia.Spectrum(
            windowedFrame.frame,
            frameSize
          );

          // Pitch detection
          const pitchResult = this.essentia.YinFFT(
            spectrumResult.spectrum,
            frameSize,
            this.sampleRate
          );
          
          // Validate pitch result
          const pitch = pitchResult.pitch;
          const confidence = pitchResult.pitchConfidence;
          
          if (isFinite(pitch) && pitch >= 0) {
            pitchTrack.push(pitch);
            pitchConfidence.push(isFinite(confidence) ? confidence : 0);
          } else {
            pitchTrack.push(0); // Unvoiced frame
            pitchConfidence.push(0);
          }

          // Spectral features
          try {
            const centroid = this.essentia.SpectralCentroid(spectrumResult.spectrum, this.sampleRate);
            spectralFeatures.centroid.push(isFinite(centroid.spectralCentroid) ? centroid.spectralCentroid : 0);
          } catch (error) {
            spectralFeatures.centroid.push(0);
          }

          try {
            const rolloff = this.essentia.SpectralRolloff(spectrumResult.spectrum, this.sampleRate);
            spectralFeatures.rolloff.push(isFinite(rolloff.spectralRolloff) ? rolloff.spectralRolloff : 0);
          } catch (error) {
            spectralFeatures.rolloff.push(0);
          }

          try {
            const energy = this.essentia.Energy ? this.essentia.Energy(windowedFrame.frame) : { energy: 0.5 };
            spectralFeatures.energy.push(isFinite(energy.energy) ? energy.energy : 0.5);
          } catch (error) {
            spectralFeatures.energy.push(0.5);
          }

          try {
            const zcr = this.essentia.ZeroCrossingRate(frameArray);
            spectralFeatures.zcr.push(isFinite(zcr.zeroCrossingRate) ? zcr.zeroCrossingRate : 0);
          } catch (error) {
            spectralFeatures.zcr.push(0);
          }

          try {
            const mfccResult = this.essentia.MFCC(spectrumResult.spectrum, 40, 13, this.sampleRate);
            const mfccArray = Array.from(mfccResult.mfcc || []);
            const validMfcc = mfccArray.map(coeff => isFinite(coeff) ? coeff : 0);
            spectralFeatures.mfcc.push(validMfcc);
          } catch (error) {
            spectralFeatures.mfcc.push(new Array(13).fill(0));
          }

          if (pitch > 0) {
            try {
              const harmonicPeaks = this.essentia.HarmonicPeaks(
                spectrumResult.spectrum,
                pitch
              );
              const harmonicsArray = Array.from(harmonicPeaks.magnitudes || []);
              const validHarmonics = harmonicsArray.map(mag => isFinite(mag) ? mag : 0);
              spectralFeatures.harmonics.push(validHarmonics);
            } catch (error) {
              spectralFeatures.harmonics.push([]);
            }
          }
          
          processedFrames++;
          
        } catch (error) {
          // Log frame processing error but continue
          this.logger.error(`Frame processing error at index ${i}`, {
            error: error instanceof Error ? error.message : "Unknown error",
            frameIndex: i,
            totalFrames: framesSize
          });
          
          // Add default values for failed frame
          pitchTrack.push(0);
          pitchConfidence.push(0);
          spectralFeatures.centroid.push(0);
          spectralFeatures.rolloff.push(0);
          spectralFeatures.energy.push(0.5);
          spectralFeatures.zcr.push(0);
          spectralFeatures.mfcc.push(new Array(13).fill(0));
          
          skippedFrames++;
        }
      }
      
      // Log processing statistics
      this.logger.info("Frame processing completed", {
        totalFrames: framesSize,
        processedFrames,
        skippedFrames,
        successRate: processedFrames / framesSize
      });
      
      // Validate minimum processed frames
      if (processedFrames < framesSize * 0.5) {
        throw new Error(`Too many frames failed processing: ${processedFrames}/${framesSize} successful`);
      }

      // 3. Onset Detection
      const onsetResult = this.essentia.Onsets(audioVector as EssentiaJS.EssentiaVector);

      // Calculate vibrato characteristics
      const vibratoCharacteristics = this.analyzeVibratoCharacteristics(pitchTrack, pitchConfidence);

      // Clean up
      if (audioVector && typeof audioVector.delete === "function") {
        audioVector.delete();
      }
      if (frames) {
        frames.forEach(frame => {
          if (frame && typeof frame.delete === "function") {
            frame.delete();
          }
        });
      }

      const meanMfcc = this.calculateMeanVector(spectralFeatures.mfcc);
      const meanHarmonics = this.calculateMeanVector(spectralFeatures.harmonics);

      // Apply pitch tracking smoothing
      const smoothedPitchTrack = this.smoothPitchTrack(pitchTrack, pitchConfidence);
      
      return {
        tempo: rhythmResult.bpm || 120,
        confidence: rhythmResult.confidence || 0.5,
        beats: Array.from(rhythmResult.beats || []),
        pitchTrack: smoothedPitchTrack,
        pitchConfidence,
        onsets: Array.from(onsetResult.onsets || []),
        mfcc: meanMfcc,
        spectralCentroid: spectralFeatures.centroid,
        spectralRolloff: spectralFeatures.rolloff,
        spectralFlux: spectralFeatures.flux,
        harmonics: meanHarmonics,
        energy: spectralFeatures.energy,
        zcr: spectralFeatures.zcr,
        loudness: spectralFeatures.energy, // Use energy as proxy for loudness
        vibratoCharacteristics: vibratoCharacteristics
      };

    } catch (error) {
      this.logger.error("Essentia analysis failed", {
        error: error instanceof Error ? error.message : "Unknown error"
      });
      
      // Re-throw the error so higher-level functions can handle retry logic
      if (error instanceof Error) {
        throw new Error(`Essentia.js analysis failed: ${error.message}`);
      } else {
        throw new Error("Essentia.js analysis failed with unknown error");
      }
    }
  }

  private calculateMeanVector(vectors: number[][]): number[] {
    if (vectors.length === 0) return [];
    const vectorLength = vectors[0]?.length || 0;
    if (vectorLength === 0) return [];

    const meanVector = new Array(vectorLength).fill(0);
    for (const vector of vectors) {
      for (let i = 0; i < vectorLength; i++) {
        meanVector[i] += vector[i] || 0;
      }
    }

    for (let i = 0; i < vectorLength; i++) {
      meanVector[i] /= vectors.length;
    }

    return meanVector;
  }

  private smoothPitchTrack(pitchTrack: number[], pitchConfidence: number[]): number[] {
    if (pitchTrack.length === 0) return [];
    
    // Step 1: Remove octave errors
    const octaveCorrectedTrack = this.correctOctaveErrors(pitchTrack, pitchConfidence);
    
    // Step 2: Apply median filter for outlier removal
    const medianFilteredTrack = this.applyMedianFilter(octaveCorrectedTrack, this.config.pitchTracking.medianFilterSize);
    
    // Step 3: Apply confidence-weighted smoothing
    const smoothedTrack = this.applyConfidenceWeightedSmoothing(medianFilteredTrack, pitchConfidence);
    
    // Step 4: Fill gaps with interpolation
    const interpolatedTrack = this.interpolatePitchGaps(smoothedTrack);
    
    return interpolatedTrack;
  }

  private correctOctaveErrors(pitchTrack: number[], pitchConfidence: number[]): number[] {
    if (pitchTrack.length < 3) return pitchTrack;
    
    const correctedTrack = [...pitchTrack];
    const windowSize = 5;
    
    for (let i = windowSize; i < correctedTrack.length - windowSize; i++) {
      const current = correctedTrack[i];
      const currentConfidence = pitchConfidence[i] || 0;
      
      if (current === undefined || current <= 0) continue;
      
      // Only consider correction if current confidence is below threshold
      // High confidence measurements are less likely to be octave errors
      if (currentConfidence > this.config.pitchTracking.confidenceThreshold) {
        continue;
      }
      
      // Get surrounding valid pitches and their confidences
      const surrounding = [];
      const surroundingConfidences = [];
      
      for (let j = Math.max(0, i - windowSize); j <= Math.min(correctedTrack.length - 1, i + windowSize); j++) {
        const value = correctedTrack[j];
        const confidence = pitchConfidence[j] || 0;
        
        if (j !== i && value !== undefined && value > 0 && confidence > 0) {
          surrounding.push(value);
          surroundingConfidences.push(confidence);
        }
      }
      
      if (surrounding.length < 3) continue;
      
      // Calculate confidence-weighted median of surrounding pitches
      const weightedMedian = this.calculateConfidenceWeightedMedian(surrounding, surroundingConfidences);
      
      // Check for octave errors (2x or 0.5x frequency)
      const ratioUp = current / weightedMedian;
      const ratioDown = weightedMedian / current;
      const threshold = this.config.pitchTracking.octaveErrorThreshold;
      
      // Calculate average confidence of surrounding measurements
      const avgSurroundingConfidence = surroundingConfidences.reduce((sum, conf) => sum + conf, 0) / surroundingConfidences.length;
      
      // Only apply correction if:
      // 1. The octave ratio is detected
      // 2. The current measurement has low confidence
      // 3. The surrounding measurements have higher average confidence
      const shouldCorrect = avgSurroundingConfidence > currentConfidence + 0.1;
      
      if (shouldCorrect) {
        if (ratioUp > (2.0 - threshold) && ratioUp < (2.0 + threshold)) {
          // Octave too high - apply correction with confidence weighting
          const correctionStrength = Math.min(1, avgSurroundingConfidence - currentConfidence);
          correctedTrack[i] = current / (1 + correctionStrength);
        } else if (ratioDown > (2.0 - threshold) && ratioDown < (2.0 + threshold)) {
          // Octave too low - apply correction with confidence weighting
          const correctionStrength = Math.min(1, avgSurroundingConfidence - currentConfidence);
          correctedTrack[i] = current * (1 + correctionStrength);
        }
      }
    }
    
    return correctedTrack;
  }

  private applyMedianFilter(pitchTrack: number[], windowSize: number): number[] {
    if (pitchTrack.length === 0) return [];
    
    const filtered = [...pitchTrack];
    const halfWindow = Math.floor(windowSize / 2);
    
    for (let i = halfWindow; i < pitchTrack.length - halfWindow; i++) {
      const window = [];
      for (let j = i - halfWindow; j <= i + halfWindow; j++) {
        const value = pitchTrack[j];
        if (value !== undefined && value > 0) {
          window.push(value);
        }
      }
      
      if (window.length >= 3) {
        filtered[i] = this.calculateMedian(window);
      }
    }
    
    return filtered;
  }

  private applyConfidenceWeightedSmoothing(pitchTrack: number[], pitchConfidence: number[]): number[] {
    if (pitchTrack.length === 0) return [];
    
    const smoothed = [...pitchTrack];
    const alpha = this.config.pitchTracking.smoothingFactor;
    
    for (let i = 1; i < pitchTrack.length; i++) {
      const currentPitch = pitchTrack[i];
      const previousPitch = smoothed[i - 1];
      
      if (currentPitch === undefined || currentPitch <= 0) continue;
      
      const currentConfidence = pitchConfidence[i] || 0;
      const previousConfidence = pitchConfidence[i - 1] || 0;
      
      // Weight the smoothing based on confidence
      const weight = Math.min(currentConfidence, previousConfidence);
      const effectiveAlpha = alpha * weight;
      
      if (previousPitch !== undefined && previousPitch > 0) {
        smoothed[i] = effectiveAlpha * previousPitch + (1 - effectiveAlpha) * currentPitch;
      }
    }
    
    return smoothed;
  }

  private interpolatePitchGaps(pitchTrack: number[]): number[] {
    if (pitchTrack.length === 0) return [];
    
    const interpolated = [...pitchTrack];
    
    for (let i = 1; i < pitchTrack.length - 1; i++) {
      const currentPitch = pitchTrack[i];
      if (currentPitch === undefined || currentPitch <= 0) {
        // Find the nearest valid pitches before and after
        let beforeIndex = -1;
        let afterIndex = -1;
        
        for (let j = i - 1; j >= 0; j--) {
          const value = pitchTrack[j];
          if (value !== undefined && value > 0) {
            beforeIndex = j;
            break;
          }
        }
        
        for (let j = i + 1; j < pitchTrack.length; j++) {
          const value = pitchTrack[j];
          if (value !== undefined && value > 0) {
            afterIndex = j;
            break;
          }
        }
        
        // Interpolate if we have both before and after values
        if (beforeIndex !== -1 && afterIndex !== -1) {
          const beforePitch = pitchTrack[beforeIndex];
          const afterPitch = pitchTrack[afterIndex];
          
          if (beforePitch !== undefined && afterPitch !== undefined) {
            const ratio = (i - beforeIndex) / (afterIndex - beforeIndex);
            
            // Linear interpolation in log space (geometric mean)
            interpolated[i] = Math.exp(
              Math.log(beforePitch) + ratio * (Math.log(afterPitch) - Math.log(beforePitch))
            );
          }
        }
      }
    }
    
    return interpolated;
  }

  private calculateMedian(values: number[]): number {
    if (values.length === 0) return 0;
    
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    
    if (sorted.length % 2 === 0) {
      const left = sorted[middle - 1];
      const right = sorted[middle];
      return (left !== undefined && right !== undefined) ? (left + right) / 2 : 0;
    } else {
      return sorted[middle] || 0;
    }
  }

  private calculateConfidenceWeightedMedian(values: number[], confidences: number[]): number {
    if (values.length === 0 || values.length !== confidences.length) return 0;
    
    // Create weighted pairs and sort by value
    const weightedPairs = values.map((value, index) => ({
      value,
      confidence: confidences[index] || 0
    })).sort((a, b) => a.value - b.value);
    
    // Calculate cumulative confidence weights
    const totalWeight = weightedPairs.reduce((sum, pair) => sum + pair.confidence, 0);
    if (totalWeight === 0) return this.calculateMedian(values);
    
    const targetWeight = totalWeight / 2;
    let cumulativeWeight = 0;
    
    for (let i = 0; i < weightedPairs.length; i++) {
      const pair = weightedPairs[i];
      if (pair) {
        cumulativeWeight += pair.confidence;
        if (cumulativeWeight >= targetWeight) {
          return pair.value;
        }
      }
    }
    
    // Fallback to regular median
    return this.calculateMedian(values);
  }


  private async analyzePitchIntonation(
    analysisResult: EssentiaAnalysisResult
  ): Promise<PitchIntonationAnalysis> {
    
    // Calculate pitch statistics
    const validPitches = analysisResult.pitchTrack.filter(p => p > 0);
    const pitchDeviations = this.calculatePitchDeviations(validPitches);
    const deviationStats = this.calculateStatistics(pitchDeviations);
    
    // Analyze pitch stability
    const stabilityMetrics = this.analyzePitchStability(
      analysisResult.pitchTrack,
      analysisResult.pitchConfidence
    );
    
    // Calculate interval precision
    const intervalMetrics = this.analyzeIntervalPrecision(validPitches);
    
    // Analyze tuning drift
    const driftMetrics = this.analyzeTuningDrift(validPitches);
    
    return {
      pitchAccuracyDistribution: {
        deviationStats: {
          mean: deviationStats.mean,
          std: deviationStats.std,
          range: [deviationStats.min, deviationStats.max]
        },
        frequencyRanges: this.categorizeByFrequencyRanges(validPitches, pitchDeviations),
        targetFrequencyDeviations: pitchDeviations
      },
      intonationStability: {
        sustainedNoteConsistency: stabilityMetrics.consistency,
        pitchVariationWithinNotes: stabilityMetrics.variations,
        stabilityByRegister: stabilityMetrics.registerStability
      },
      intervalPrecision: {
        melodicIntervalAccuracy: intervalMetrics.melodic,
        harmonicIntervalAccuracy: intervalMetrics.harmonic,
        overallIntervalScore: intervalMetrics.overall
      },
      tuningDriftPatterns: {
        driftOverTime: driftMetrics.driftOverTime,
        driftDirection: driftMetrics.direction,
        driftRate: driftMetrics.rate,
        performanceRegions: driftMetrics.regions
      }
    };
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

  private calculateStatistics(values: number[]): { mean: number; std: number; min: number; max: number } {
    if (values.length === 0) {
      return { mean: 0, std: 0, min: 0, max: 0 };
    }
    
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    const std = Math.sqrt(variance);
    const min = Math.min(...values);
    const max = Math.max(...values);
    
    return { mean, std, min, max };
  }

  private analyzePitchStability(
    pitchTrack: number[], 
    confidences: number[]
  ): {
    consistency: number;
    variations: number[];
    registerStability: Record<string, number>;
  } {
    const validPitches = pitchTrack.filter(p => p > 0);
    
    // Calculate overall consistency as average confidence
    const consistency = confidences.length > 0 
      ? confidences.reduce((sum, conf) => sum + conf, 0) / confidences.length
      : 0.5;
    
    // Calculate pitch variations (sliding window standard deviation)
    const windowSize = 10;
    const variations: number[] = [];
    
    for (let i = 0; i <= validPitches.length - windowSize; i++) {
      const window = validPitches.slice(i, i + windowSize);
      const stats = this.calculateStatistics(window);
      variations.push(stats.std);
    }
    
    // Analyze stability by register
    const lowRegister = validPitches.filter(p => p < 300);
    const midRegister = validPitches.filter(p => p >= 300 && p < 600);
    const highRegister = validPitches.filter(p => p >= 600);
    
    const registerStability = {
      low: this.calculateRegisterStability(lowRegister),
      middle: this.calculateRegisterStability(midRegister),
      high: this.calculateRegisterStability(highRegister)
    };
    
    return { consistency, variations, registerStability };
  }

  private calculateRegisterStability(pitches: number[]): number {
    if (pitches.length < 3) return 0.5;
    
    const stats = this.calculateStatistics(pitches);
    // Convert pitch standard deviation to a 0-1 stability score
    // Lower standard deviation = higher stability
    const stability = Math.max(0, 1 - (stats.std / 50)); // Normalize by 50 Hz
    return Math.min(1, stability);
  }

  private analyzeIntervalPrecision(pitchTrack: number[]): {
    melodic: Array<{ interval: string; accuracy: number }>;
    harmonic: Array<{ interval: string; accuracy: number }>;
    overall: number;
  } {
    if (pitchTrack.length < 2) {
      return {
        melodic: [],
        harmonic: [],
        overall: 0.5
      };
    }
    
    // Calculate melodic intervals
    const intervals: number[] = [];
    for (let i = 1; i < pitchTrack.length; i++) {
      const current = pitchTrack[i];
      const previous = pitchTrack[i - 1];
      if (current && current > 0 && previous && previous > 0) {
        const semitones = 12 * Math.log2(current / previous);
        intervals.push(semitones);
      }
    }
    
    // Analyze common interval accuracies
    const intervalTypes = [
      { name: "m2", semitones: 1 },
      { name: "M2", semitones: 2 },
      { name: "m3", semitones: 3 },
      { name: "M3", semitones: 4 },
      { name: "P4", semitones: 5 },
      { name: "P5", semitones: 7 }
    ];
    
    const melodic = intervalTypes.map(intervalType => {
      const matchingIntervals = intervals.filter(interval => 
        Math.abs(interval - intervalType.semitones) < 0.5
      );
      
      const accuracy = matchingIntervals.length > 0
        ? this.calculateIntervalAccuracy(matchingIntervals, intervalType.semitones)
        : 0.5;
      
      return { interval: intervalType.name, accuracy };
    });
    
    // For harmonic intervals, a true analysis requires polyphonic pitch detection,
    // which is not directly available from the current monophonic pitchTrack.
    // As a placeholder, we'll use the overall melodic interval accuracy.
    // In a real-world scenario, this would involve more advanced chord detection or
    // polyphonic analysis.
    const harmonic = melodic.map(item => ({ ...item })); // Copy melodic intervals as a placeholder
    
    const overall = [...melodic, ...harmonic]
      .reduce((sum, item) => sum + item.accuracy, 0) / (melodic.length + harmonic.length);
    
    return { melodic, harmonic, overall };
  }

  private calculateIntervalAccuracy(intervals: number[], targetSemitones: number): number {
    const deviations = intervals.map(interval => Math.abs(interval - targetSemitones));
    const avgDeviation = deviations.reduce((sum, dev) => sum + dev, 0) / deviations.length;
    
    // Convert deviation to accuracy score (0-1)
    // Perfect intervals (deviation = 0) get score 1
    // Deviations > 0.5 semitones get lower scores
    return Math.max(0, 1 - (avgDeviation / 0.5));
  }

  private analyzeTuningDrift(pitchTrack: number[]): {
    driftOverTime: number[];
    direction: "flat" | "sharp" | "stable";
    rate: number;
    regions: Array<{ time: number; drift: number }>;
  } {
    if (pitchTrack.length < 10) {
      return {
        driftOverTime: [],
        direction: "stable",
        rate: 0,
        regions: []
      };
    }
    
    // Calculate drift as deviation from initial tuning
    const referencePitch = pitchTrack[0];
    const driftOverTime = pitchTrack.map(pitch => {
      if (pitch <= 0 || !referencePitch || referencePitch <= 0) return 0;
      return 1200 * Math.log2(pitch / referencePitch); // Cents
    });
    
    // Determine overall drift direction
    const finalDrift = driftOverTime[driftOverTime.length - 1] || 0;
    const direction: "flat" | "sharp" | "stable" = 
      Math.abs(finalDrift) < 10 ? "stable" :
        finalDrift < 0 ? "flat" : "sharp";
    
    // Calculate drift rate (cents per second)
    const durationSeconds = pitchTrack.length * 0.01; // Assuming 100Hz analysis rate
    const rate = finalDrift / durationSeconds;
    
    // Create drift regions (every 10 seconds)
    const regions: Array<{ time: number; drift: number }> = [];
    const regionSize = Math.max(1, Math.floor(pitchTrack.length / 10));
    
    for (let i = 0; i < pitchTrack.length; i += regionSize) {
      const regionEnd = Math.min(i + regionSize, pitchTrack.length - 1);
      const time = (i / pitchTrack.length) * durationSeconds;
      const drift = driftOverTime[regionEnd] || 0;
      regions.push({ time, drift });
    }
    
    return { driftOverTime, direction, rate, regions };
  }

  private categorizeByFrequencyRanges(
    pitches: number[], 
    deviations: number[]
  ): Array<{ range: string; accuracy: number }> {
    const ranges = [
      { name: "low", min: 0, max: 300 },
      { name: "middle", min: 300, max: 600 },
      { name: "high", min: 600, max: 2000 }
    ];
    
    return ranges.map(range => {
      const rangeIndices = pitches
        .map((pitch, index) => ({ pitch, index }))
        .filter(({ pitch }) => pitch >= range.min && pitch < range.max)
        .map(({ index }) => index);
      
      if (rangeIndices.length === 0) {
        return { range: range.name, accuracy: 0.5 };
      }
      
      const rangeDeviations = rangeIndices.map(i => Math.abs(deviations[i] || 0));
      const avgDeviation = rangeDeviations.reduce((sum, dev) => sum + dev, 0) / rangeDeviations.length;
      
      // Convert average deviation to accuracy (0-1)
      const accuracy = Math.max(0, 1 - (avgDeviation / 50)); // 50 cents = 0 accuracy
      
      return { range: range.name, accuracy };
    });
  }

  private async analyzeTimingRhythm(
    analysisResult: EssentiaAnalysisResult
  ): Promise<TimingRhythmAnalysis> {
    
    // Analyze temporal accuracy
    const temporalMetrics = this.analyzeTemporalAccuracy(
      analysisResult.beats,
      analysisResult.onsets,
      analysisResult.tempo
    );
    
    // Analyze rhythmic subdivisions
    const subdivisionMetrics = this.analyzeRhythmicSubdivisions(
      analysisResult.onsets,
      analysisResult.tempo
    );
    
    // Analyze groove consistency
    const grooveMetrics = this.analyzeGrooveConsistency(
      analysisResult.beats,
      analysisResult.tempo
    );
    
    // Analyze rubato control
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
    
    // Calculate expected beat intervals
    const expectedBeatInterval = 60 / tempo; // seconds
    
    // Calculate actual beat intervals
    const beatIntervals: number[] = [];
    for (let i = 1; i < beats.length; i++) {
      const current = beats[i];
      const previous = beats[i - 1];
      if (current !== undefined && previous !== undefined) {
        beatIntervals.push(current - previous);
      }
    }
    
    // Calculate deviations from expected timing (in milliseconds)
    const metronomeDeviation = beatIntervals.map(interval => 
      (interval - expectedBeatInterval) * 1000
    );
    
    // Calculate rush tendency (positive = rushing, negative = dragging)
    const rushTendency = metronomeDeviation.length > 0
      ? metronomeDeviation.reduce((sum, dev) => sum + dev, 0) / metronomeDeviation.length / 1000
      : 0;
    
    // Calculate overall timing score based on consistency
    const avgDeviation = metronomeDeviation.length > 0
      ? metronomeDeviation.reduce((sum, dev) => sum + Math.abs(dev), 0) / metronomeDeviation.length
      : 0;
    
    const overallTimingScore = Math.max(0, 1 - (avgDeviation / 100)); // 100ms = score 0
    
    // Calculate backing track sync (use detected tempo as baseline)
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
    
    // Calculate inter-onset intervals
    const intervals: number[] = [];
    for (let i = 1; i < onsets.length; i++) {
      const current = onsets[i];
      const previous = onsets[i - 1];
      if (current !== undefined && previous !== undefined) {
        intervals.push(current - previous);
      }
    }
    
    // Expected subdivision durations (in seconds)
    const beatDuration = 60 / tempo;
    const expectedSubdivisions = {
      quarter: beatDuration,
      eighth: beatDuration / 2,
      sixteenth: beatDuration / 4,
      triplets: beatDuration / 3
    };
    
    // Calculate accuracy for each subdivision
    const subdivisionAccuracy: Record<string, number> = {};
    
    Object.entries(expectedSubdivisions).forEach(([subdivision, expectedDuration]) => {
      // Find intervals that match this subdivision (within 20% tolerance)
      const matchingIntervals = intervals.filter(interval => 
        Math.abs(interval - expectedDuration) < expectedDuration * 0.2
      );
      
      if (matchingIntervals.length === 0) {
        subdivisionAccuracy[subdivision] = 0.5;
        return;
      }
      
      // Calculate accuracy based on timing precision
      const deviations = matchingIntervals.map(interval => 
        Math.abs(interval - expectedDuration) / expectedDuration
      );
      const avgDeviation = deviations.reduce((sum, dev) => sum + dev, 0) / deviations.length;
      
      subdivisionAccuracy[subdivision] = Math.max(0, 1 - avgDeviation * 5); // Scale deviation
    });
    
    // Calculate complex pattern success based on variety of subdivisions used
    const usedSubdivisions = Object.values(subdivisionAccuracy).filter(score => score > 0.6).length;
    const complexPatternSuccess = usedSubdivisions / 4; // Normalized by total subdivisions
    
    // Calculate polyrhythm stability (consistency of irregular patterns)
    const intervalVariability = this.calculateStatistics(intervals).std;
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
    
    // Calculate beat-to-beat intervals
    const beatIntervals: number[] = [];
    for (let i = 1; i < beats.length; i++) {
      const current = beats[i];
      const previous = beats[i - 1];
      if (current !== undefined && previous !== undefined) {
        beatIntervals.push(current - previous);
      }
    }
    
    // Calculate groove stability (consistency of timing)
    const intervalStats = this.calculateStatistics(beatIntervals);
    
    // Calculate swing ratio (detect automatically from timing patterns)
    let swingRatio = 0.5; // Default straight timing
    if (beatIntervals.length >= 2) {
      // Simplified swing detection based on eighth note timing patterns
      // Analyze for long-short patterns typical of swing
      const variationCoeff = intervalStats.std / intervalStats.mean;
      if (variationCoeff > 0.1 && variationCoeff < 0.3) {
        swingRatio = 0.67; // Detected swing pattern
      }
    }
    
    const grooveStability = Math.max(0, 1 - (intervalStats.std / intervalStats.mean));
    
    // Calculate micro-timing variations (in milliseconds)
    const expectedInterval = 60 / tempo;
    const microTimingVariations = beatIntervals.map(interval => 
      (interval - expectedInterval) * 1000
    );
    
    // Calculate style adherence based on timing characteristics
    let styleAdherence = 0.8; // Base score
    
    // Analyze micro-timing characteristics without requiring style metadata
    const avgVariation = microTimingVariations.reduce((sum, v) => sum + Math.abs(v), 0) / microTimingVariations.length;
    
    // Good timing has some natural variation but not too much
    if (avgVariation > 2 && avgVariation < 25) { // 2-25ms is natural musical timing
      styleAdherence = Math.min(1, styleAdherence + 0.1);
    }
    
    // Bonus for consistent but not mechanical timing
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
    
    // Calculate instantaneous tempo changes
    const tempoChanges: Array<{ time: number; change: number; intentional: boolean }> = [];
    const windowSize = 4; // Analyze tempo over 4-beat windows
    
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
      
      // Classify as intentional if change is significant and sustained
      const intentional = Math.abs(tempoChange) > 10 && 
                         (i === 0 || Math.abs(tempoChanges[tempoChanges.length - 1]?.change || 0) > 5);
      
      if (Math.abs(tempoChange) > 5) { // Only record significant changes
        tempoChanges.push({
          time: window[0] || 0,
          change: tempoChange,
          intentional
        });
      }
    }
    
    // Calculate unintentional fluctuations
    const unintentionalChanges = tempoChanges.filter(change => !change.intentional);
    const unintentionalFluctuations = unintentionalChanges.length > 0
      ? unintentionalChanges.reduce((sum, change) => sum + Math.abs(change.change), 0) / unintentionalChanges.length / tempo
      : 0;
    
    // Estimate phrase timing (periods of sustained playing)
    const phraseTiming: number[] = [];
    let phraseStart = beats[0] || 0;
    
    for (let i = 1; i < beats.length; i++) {
      const currentBeat = beats[i];
      const previousBeat = beats[i - 1];
      if (currentBeat === undefined || previousBeat === undefined) continue;
      
      const gap = currentBeat - previousBeat;
      const expectedGap = 60 / tempo;
      
      // If gap is significantly longer than expected, it might be a phrase boundary
      if (gap > expectedGap * 2) {
        const phraseLength = previousBeat - phraseStart;
        if (phraseLength > 1) { // Only count phrases longer than 1 second
          phraseTiming.push(phraseLength);
        }
        phraseStart = currentBeat;
      }
    }
    
    // Add final phrase
    const lastBeat = beats[beats.length - 1];
    if (lastBeat !== undefined) {
      const finalPhraseLength = lastBeat - phraseStart;
      if (finalPhraseLength > 1) {
        phraseTiming.push(finalPhraseLength);
      }
    }
    
    // Calculate expressive timing score
    const intentionalChanges = tempoChanges.filter(change => change.intentional);
    const hasExpressiveChanges = intentionalChanges.length > 0;
    const hasReasonableFluctuations = unintentionalFluctuations < 0.1; // Less than 10% tempo variation
    
    const expressiveTimingScore = (hasExpressiveChanges ? 0.6 : 0.4) + 
                                 (hasReasonableFluctuations ? 0.4 : 0.2);
    
    return {
      intentionalTempoChanges: tempoChanges,
      unintentionalFluctuations,
      phraseTiming,
      expressiveTimingScore
    };
  }

  private async analyzeToneQualityTimbre(
    analysisResult: EssentiaAnalysisResult
  ): Promise<ToneQualityTimbreAnalysis> {
    
    // Analyze harmonic content
    const harmonicMetrics = this.analyzeHarmonicContent(
      analysisResult.harmonics,
      analysisResult.spectralCentroid,
      analysisResult.pitchTrack
    );
    
    // Analyze dynamic range
    const dynamicMetrics = this.analyzeDynamicRange(
      analysisResult.energy,
      analysisResult.loudness
    );
    
    // Analyze timbral consistency
    const timbralMetrics = this.analyzeTimbralConsistency(
      analysisResult.spectralCentroid,
      analysisResult.spectralRolloff,
      analysisResult.mfcc,
      analysisResult.pitchTrack
    );
    
    // Analyze vibrato characteristics
    const vibratoMetrics = this.analyzeVibratoCharacteristics(
      analysisResult.pitchTrack,
      analysisResult.pitchConfidence
    );
    
    return {
      harmonicContentAnalysis: harmonicMetrics,
      dynamicRangeUtilization: dynamicMetrics,
      timbralConsistency: timbralMetrics,
      vibratoCharacteristics: vibratoMetrics
    };
  }

  private analyzeHarmonicContent(
    harmonics: number[],
    spectralCentroid: number[],
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    pitchTrack: number[]
  ): {
    overtoneDistribution: number[];
    harmonicRichness: number;
    fundamentalToHarmonicRatio: number;
    spectralCentroid: number[];
  } {
    // Calculate harmonic richness as the sum of harmonic strengths
    const harmonicRichness = harmonics.length > 0
      ? harmonics.reduce((sum, harmonic) => sum + harmonic, 0) / harmonics.length
      : 0.5;
    
    // Calculate fundamental to harmonic ratio
    const fundamentalStrength = harmonics[0] || 1;
    const harmonicSum = harmonics.slice(1).reduce((sum, h) => sum + h, 0);
    const fundamentalToHarmonicRatio = harmonicSum > 0 ? fundamentalStrength / harmonicSum : 1;
    
    return {
      overtoneDistribution: harmonics,
      harmonicRichness,
      fundamentalToHarmonicRatio,
      spectralCentroid
    };
  }

  private analyzeDynamicRange(
    energy: number[],
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    loudness: number[]
  ): {
    volumeRange: [number, number];
    dynamicVariationEffectiveness: number;
    crescendoDecrescendoControl: number;
    accentClarity: number;
  } {
    if (energy.length === 0) {
      return {
        volumeRange: [-60, -20],
        dynamicVariationEffectiveness: 0.5,
        crescendoDecrescendoControl: 0.5,
        accentClarity: 0.5
      };
    }
    
    // Convert energy to dB scale
    const energyDB = energy.map(e => 20 * Math.log10(Math.max(e, 1e-10)));
    const minDB = Math.min(...energyDB);
    const maxDB = Math.max(...energyDB);
    const volumeRange: [number, number] = [minDB, maxDB];
    
    // Calculate dynamic variation effectiveness
    const dynamicRange = maxDB - minDB;
    const dynamicVariationEffectiveness = Math.min(1, dynamicRange / 40); // 40dB = excellent range
    
    // Detect crescendos/decrescendos (smoothed energy trends)
    const smoothedEnergy = this.smoothArray(energyDB, 5);
    const crescendoDecrescendoControl = this.detectDynamicControl(smoothedEnergy);
    
    // Detect accents (sudden energy increases)
    const accentClarity = this.detectAccents(energyDB);
    
    return {
      volumeRange,
      dynamicVariationEffectiveness,
      crescendoDecrescendoControl,
      accentClarity
    };
  }

  private smoothArray(array: number[], windowSize: number): number[] {
    const smoothed: number[] = [];
    const halfWindow = Math.floor(windowSize / 2);
    
    for (let i = 0; i < array.length; i++) {
      const start = Math.max(0, i - halfWindow);
      const end = Math.min(array.length, i + halfWindow + 1);
      const window = array.slice(start, end);
      const avg = window.reduce((sum, val) => sum + val, 0) / window.length;
      smoothed.push(avg);
    }
    
    return smoothed;
  }

  private detectDynamicControl(smoothedEnergy: number[]): number {
    if (smoothedEnergy.length < 10) return 0.5;
    
    // Look for sustained crescendos and decrescendos
    let controlScore = 0;
    let gradualChanges = 0;
    
    for (let i = 5; i < smoothedEnergy.length - 5; i++) {
      const before = smoothedEnergy.slice(i - 5, i);
      const after = smoothedEnergy.slice(i, i + 5);
      
      const beforeAvg = before.reduce((sum, val) => sum + val, 0) / before.length;
      const afterAvg = after.reduce((sum, val) => sum + val, 0) / after.length;
      
      const change = Math.abs(afterAvg - beforeAvg);
      if (change > 3) { // Significant dynamic change (3dB)
        gradualChanges++;
        
        // Check if change is gradual (good control)
        const gradient = (afterAvg - beforeAvg) / 5;
        const expectedValues = before.map((_, idx) => beforeAvg + gradient * idx);
        const actualValues = after;
        const error = expectedValues.reduce((sum, expected, idx) => 
          sum + Math.abs(expected - (actualValues[idx] || 0)), 0) / expectedValues.length;
        
        if (error < 2) { // Good gradual control
          controlScore += 1;
        }
      }
    }
    
    return gradualChanges > 0 ? Math.min(1, controlScore / gradualChanges) : 0.5;
  }

  private detectAccents(energyDB: number[]): number {
    if (energyDB.length < 5) return 0.5;
    
    let accents = 0;
    let clearAccents = 0;
    
    for (let i = 2; i < energyDB.length - 2; i++) {
      const current = energyDB[i];
      const surrounding = [
        energyDB[i - 2], energyDB[i - 1],
        energyDB[i + 1], energyDB[i + 2]
      ].filter((val): val is number => val !== undefined);
      
      if (surrounding.length === 4 && current !== undefined) {
        const avgSurrounding = surrounding.reduce((sum, val) => sum + val, 0) / surrounding.length;
        
        // Check if current value is significantly higher (accent)
        if (current > avgSurrounding + 6) { // 6dB accent threshold
          accents++;
          
          // Check if it's a clear, isolated accent
          if (current > Math.max(...surrounding) + 3) {
            clearAccents++;
          }
        }
      }
    }
    
    return accents > 0 ? clearAccents / accents : 0.5;
  }

  private analyzeTimbralConsistency(
    spectralCentroid: number[],
    spectralRolloff: number[],
    mfcc: number[],
    pitchTrack: number[]
  ): {
    toneStabilityAcrossRegisters: Record<string, number>;
    toneStabilityAcrossDynamics: Record<string, number>;
    overallTimbralUniformity: number;
    colorVariationControl: number;
  } {
    // Analyze stability across registers
    const registerStability = this.analyzeTimbralStabilityByRegister(
      spectralCentroid, 
      pitchTrack
    );
    
    // Analyze stability across dynamics (simplified - would need dynamic level info)
    const dynamicStability = {
      pp: 0.75,
      p: 0.80,
      mf: 0.85,
      f: 0.80,
      ff: 0.75
    };
    
    // Calculate overall timbral uniformity
    const centroidStats = this.calculateStatistics(spectralCentroid);
    const rolloffStats = this.calculateStatistics(spectralRolloff);
    
    const centroidConsistency = centroidStats.mean > 0 ? 1 - (centroidStats.std / centroidStats.mean) : 0.5;
    const rolloffConsistency = rolloffStats.mean > 0 ? 1 - (rolloffStats.std / rolloffStats.mean) : 0.5;
    
    const overallTimbralUniformity = (centroidConsistency + rolloffConsistency) / 2;
    
    // Calculate color variation control (ability to vary timbre intentionally)
    const colorVariationControl = Math.min(1, centroidStats.std / 500); // Some variation is good
    
    return {
      toneStabilityAcrossRegisters: registerStability,
      toneStabilityAcrossDynamics: dynamicStability,
      overallTimbralUniformity: Math.max(0, Math.min(1, overallTimbralUniformity)),
      colorVariationControl: Math.max(0, Math.min(1, colorVariationControl))
    };
  }

  private analyzeTimbralStabilityByRegister(
    spectralCentroid: number[],
    pitchTrack: number[]
  ): Record<string, number> {
    const registers = {
      low: { min: 0, max: 300, centroids: [] as number[] },
      middle: { min: 300, max: 600, centroids: [] as number[] },
      high: { min: 600, max: 2000, centroids: [] as number[] }
    };
    
    // Group centroids by register
    pitchTrack.forEach((pitch, i) => {
      if (i < spectralCentroid.length && pitch > 0) {
        const centroid = spectralCentroid[i];
        
        if (centroid !== undefined) {
          if (pitch >= registers.low.min && pitch < registers.low.max) {
            registers.low.centroids.push(centroid);
          } else if (pitch >= registers.middle.min && pitch < registers.middle.max) {
            registers.middle.centroids.push(centroid);
          } else if (pitch >= registers.high.min && pitch < registers.high.max) {
            registers.high.centroids.push(centroid);
          }
        }
      }
    });
    
    // Calculate stability for each register
    const stability: Record<string, number> = {};
    
    Object.entries(registers).forEach(([register, data]) => {
      if (data.centroids.length < 3) {
        stability[register] = 0.5;
        return;
      }
      
      const stats = this.calculateStatistics(data.centroids);
      const consistency = stats.mean > 0 ? 1 - (stats.std / stats.mean) : 0.5;
      stability[register] = Math.max(0, Math.min(1, consistency));
    });
    
    return stability;
  }

  private analyzeVibratoCharacteristics(
    pitchTrack: number[],
    pitchConfidence: number[]
  ): {
    vibratoRate: number;
    vibratoDepth: number;
    vibratoControl: number;
    vibratoTiming: Array<{ start: number; end: number; quality: number }>;
  } {
    if (pitchTrack.length < 20) {
      return {
        vibratoRate: 0,
        vibratoDepth: 0,
        vibratoControl: 0,
        vibratoTiming: []
      };
    }
    
    // Detect vibrato regions (periods of periodic pitch oscillation)
    const vibratoRegions = this.detectVibratoRegions(pitchTrack, pitchConfidence);
    
    if (vibratoRegions.length === 0) {
      return {
        vibratoRate: 0,
        vibratoDepth: 0,
        vibratoControl: 0.5, // Neutral score for no vibrato
        vibratoTiming: []
      };
    }
    
    // Calculate average vibrato characteristics
    let totalRate = 0;
    let totalDepth = 0;
    let totalQuality = 0;
    
    vibratoRegions.forEach(region => {
      totalRate += region.rate;
      totalDepth += region.depth;
      totalQuality += region.quality;
    });
    
    const avgRate = totalRate / vibratoRegions.length;
    const avgDepth = totalDepth / vibratoRegions.length;
    const avgControl = totalQuality / vibratoRegions.length;
    
    const vibratoTiming = vibratoRegions.map(region => ({
      start: region.start * 0.01, // Convert to seconds (assuming 100Hz analysis)
      end: region.end * 0.01,
      quality: region.quality
    }));
    
    return {
      vibratoRate: avgRate,
      vibratoDepth: avgDepth,
      vibratoControl: avgControl,
      vibratoTiming
    };
  }

  private detectVibratoRegions(
    pitchTrack: number[],
    pitchConfidence: number[]
  ): Array<{ start: number; end: number; rate: number; depth: number; quality: number }> {
    const regions: Array<{ start: number; end: number; rate: number; depth: number; quality: number }> = [];
    // const minVibratoLength = 10; // Minimum 10 frames for vibrato
    const windowSize = 20; // Analysis window size
    
    for (let i = 0; i <= pitchTrack.length - windowSize; i++) {
      const window = pitchTrack.slice(i, i + windowSize);
      const confidenceWindow = pitchConfidence.slice(i, i + windowSize);
      
      // Skip if too many unvoiced frames
      const validFrames = window.filter((p, idx) => p > 0 && (confidenceWindow[idx] || 0) > 0.5);
      if (validFrames.length < windowSize * 0.7) continue;
      
      // Detect periodic oscillation
      const vibratoMetrics = this.analyzeVibratoInWindow(validFrames);
      
      if (vibratoMetrics.isVibrato) {
        regions.push({
          start: i,
          end: i + windowSize,
          rate: vibratoMetrics.rate,
          depth: vibratoMetrics.depth,
          quality: vibratoMetrics.quality
        });
      }
    }
    
    // Merge overlapping regions
    return this.mergeVibratoRegions(regions);
  }

  private analyzeVibratoInWindow(pitchWindow: number[]): {
    isVibrato: boolean;
    rate: number;
    depth: number;
    quality: number;
  } {
    if (pitchWindow.length < 10) {
      return { isVibrato: false, rate: 0, depth: 0, quality: 0 };
    }
    
    // Calculate autocorrelation to find periodic patterns
    const autocorr = this.calculateAutocorrelation(pitchWindow);
    
    // Find peaks in autocorrelation (excluding lag 0)
    const peaks = this.findPeaks(autocorr.slice(1));
    
    if (peaks.length === 0) {
      return { isVibrato: false, rate: 0, depth: 0, quality: 0 };
    }
    
    // Find the most prominent peak (highest correlation)
    const bestPeak = peaks.reduce((best, peak) => 
      (autocorr[peak.index + 1] || 0) > (autocorr[best.index + 1] || 0) ? peak : best
    );
    
    // Convert lag to rate (assuming 100Hz analysis rate)
    const rate = 100 / (bestPeak.index + 1); // Hz
    
    // Check if rate is in typical vibrato range
    if (rate < this.config.vibrato.minRate || rate > this.config.vibrato.maxRate) {
      return { isVibrato: false, rate: 0, depth: 0, quality: 0 };
    }
    
    // Calculate vibrato depth (peak-to-peak variation in cents)
    const pitchStats = this.calculateStatistics(pitchWindow);
    const depth = (pitchStats.max - pitchStats.min) / pitchStats.mean * 1200; // Convert to cents
    
    // Calculate quality based on regularity
    const quality = autocorr[bestPeak.index + 1] || 0; // Higher correlation = better quality
    
    return {
      isVibrato: depth > this.config.vibrato.minDepth && quality > this.config.vibrato.qualityThreshold,
      rate,
      depth,
      quality
    };
  }

  private calculateAutocorrelation(signal: number[]): number[] {
    const n = signal.length;
    const result: number[] = [];
    
    for (let lag = 0; lag < n / 2; lag++) {
      let sum = 0;
      let count = 0;
      
      for (let i = 0; i < n - lag; i++) {
        const current = signal[i];
        const lagged = signal[i + lag];
        if (current !== undefined && lagged !== undefined) {
          sum += current * lagged;
          count++;
        }
      }
      
      result.push(count > 0 ? sum / count : 0);
    }
    
    // Normalize by zero-lag autocorrelation
    const norm = result[0] || 1;
    return norm > 0 ? result.map(r => r / norm) : result;
  }

  private findPeaks(signal: number[]): Array<{ index: number; value: number }> {
    const peaks: Array<{ index: number; value: number }> = [];
    
    for (let i = 1; i < signal.length - 1; i++) {
      const current = signal[i];
      const prev = signal[i - 1];
      const next = signal[i + 1];
      
      if (current !== undefined && prev !== undefined && next !== undefined &&
          current > prev && current > next && current > 0.1) {
        peaks.push({ index: i, value: current });
      }
    }
    
    return peaks.sort((a, b) => b.value - a.value); // Sort by peak height
  }

  private mergeVibratoRegions(
    regions: Array<{ start: number; end: number; rate: number; depth: number; quality: number }>
  ): Array<{ start: number; end: number; rate: number; depth: number; quality: number }> {
    if (regions.length <= 1) return regions;
    
    const merged: Array<{ start: number; end: number; rate: number; depth: number; quality: number }> = [];
    let current = { ...regions[0] as { start: number; end: number; rate: number; depth: number; quality: number } };
    
    for (let i = 1; i < regions.length; i++) {
      const next = regions[i];
      
      if (!next) continue;
      
      // If regions overlap or are close together, merge them
      if (next.start <= current.end + 5) {
        current.end = Math.max(current.end, next.end);
        // Average the characteristics
        current.rate = (current.rate + next.rate) / 2;
        current.depth = (current.depth + next.depth) / 2;
        current.quality = (current.quality + next.quality) / 2;
      } else {
        merged.push(current);
        current = { ...next };
      }
    }
    
    merged.push(current);
    return merged;
  }

  private async analyzeTechnicalExecution(
    analysisResult: EssentiaAnalysisResult
  ): Promise<TechnicalExecutionAnalysis> {
    
    // Analyze articulation clarity using onset detection and spectral analysis
    const articulationClarity = await this.analyzeArticulationClarity(analysisResult);
    
    // Analyze finger technique using pitch tracking and timing analysis
    const fingerTechnique = await this.analyzeFingerTechnique(analysisResult);
    
    // Analyze breath management using loudness and onset patterns
    const breathManagement = await this.analyzeBreathManagement(analysisResult);
    
    // Analyze extended techniques using spectral features
    const extendedTechniques = await this.analyzeExtendedTechniques(analysisResult);
    
    return {
      articulationClarity,
      fingerTechniqueEfficiency: fingerTechnique,
      breathManagementIndicators: breathManagement,
      extendedTechniqueMastery: extendedTechniques
    };
  }

  private async analyzeMusicalExpression(
    analysisResult: EssentiaAnalysisResult
  ): Promise<MusicalExpressionAnalysis> {
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

  private async analyzePerformanceConsistency(
    analysisResult: EssentiaAnalysisResult
  ): Promise<PerformanceConsistencyAnalysis> {
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

  private calculatePerformanceScore(analysis: ExtendedAudioAnalysis): PerformanceScore {
    // Calculate category scores
    const categoryScores = {
      pitchIntonation: this.scorePitchIntonation(analysis.pitchIntonation),
      timingRhythm: this.scoreTimingRhythm(analysis.timingRhythm),
      toneQuality: this.scoreToneQuality(analysis.toneQualityTimbre),
      technicalExecution: this.scoreTechnicalExecution(analysis.technicalExecution),
      musicalExpression: this.scoreMusicalExpression(analysis.musicalExpression),
      consistency: this.scoreConsistency(analysis.performanceConsistency)
    };

    // Calculate overall score as weighted average
    const weights = {
      pitchIntonation: 0.20,
      timingRhythm: 0.20,
      toneQuality: 0.15,
      technicalExecution: 0.20,
      musicalExpression: 0.15,
      consistency: 0.10
    };

    const overallScore = Object.entries(categoryScores).reduce((sum, [category, score]) => {
      return sum + score * weights[category as keyof typeof weights];
    }, 0);

    return {
      overallScore: Math.round(overallScore),
      categoryScores,
      strengthAreas: this.identifyStrengthAreas(categoryScores),
      improvementAreas: this.identifyImprovementAreas(categoryScores),
      specificFeedback: this.generateSpecificFeedback(analysis),
      nextLevelRecommendations: this.generateRecommendations(analysis)
    };
  }

  private scorePitchIntonation(analysis: PitchIntonationAnalysis): number {
    const accuracy = analysis.intervalPrecision.overallIntervalScore;
    const stability = analysis.intonationStability.sustainedNoteConsistency;
    return Math.round((accuracy * 0.6 + stability * 0.4) * 100);
  }

  private scoreTimingRhythm(analysis: TimingRhythmAnalysis): number {
    const timing = analysis.temporalAccuracy.overallTimingScore;
    const rhythm = analysis.rhythmicSubdivisionPrecision.complexPatternSuccess;
    const groove = analysis.grooveConsistency.grooveStability;
    return Math.round((timing * 0.4 + rhythm * 0.3 + groove * 0.3) * 100);
  }

  private scoreToneQuality(analysis: ToneQualityTimbreAnalysis): number {
    const richness = analysis.harmonicContentAnalysis.harmonicRichness;
    const consistency = analysis.timbralConsistency.overallTimbralUniformity;
    const dynamics = analysis.dynamicRangeUtilization.dynamicVariationEffectiveness;
    return Math.round((richness * 0.4 + consistency * 0.4 + dynamics * 0.2) * 100);
  }

  private scoreTechnicalExecution(analysis: TechnicalExecutionAnalysis): number {
    const articulation = analysis.articulationClarity.tonguingPrecision;
    const fingers = analysis.fingerTechniqueEfficiency.keyTransitionSmoothness;
    const breath = analysis.breathManagementIndicators.breathSupportConsistency;
    return Math.round((articulation * 0.4 + fingers * 0.4 + breath * 0.2) * 100);
  }

  private scoreMusicalExpression(analysis: MusicalExpressionAnalysis): number {
    const phrasing = analysis.phrasingSophistication.phraseShaping;
    const dynamics = analysis.dynamicContourComplexity.dynamicShapingScore;
    const style = analysis.stylisticAuthenticity.styleConsistency;
    return Math.round((phrasing * 0.4 + dynamics * 0.3 + style * 0.3) * 100);
  }

  private scoreConsistency(analysis: PerformanceConsistencyAnalysis): number {
    const consistency = analysis.endurancePatterns.consistencyThroughoutPerformance;
    const recovery = analysis.recoverySpeed.recoveryEffectiveness;
    const errorImpact = 1 - analysis.recoverySpeed.errorImpactOnSubsequentPerformance;
    return Math.round((consistency * 0.5 + recovery * 0.3 + errorImpact * 0.2) * 100);
  }

  private identifyStrengthAreas(scores: Record<string, number>): string[] {
    const entries = Object.entries(scores);
    const topScores = entries
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      .filter(([_, score]) => score >= 80)
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      .sort(([_, a], [__, b]) => b - a)
      .slice(0, 3)
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      .map(([category, _]) => this.formatCategoryName(category))
      .filter((name): name is string => name !== undefined);
    
    return topScores;
  }

  private identifyImprovementAreas(scores: Record<string, number>): string[] {
    const entries = Object.entries(scores);
    const lowScores = entries
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      .filter(([_, score]) => score < 75)
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      .sort(([_, a], [__, b]) => a - b)
      .slice(0, 3)
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      .map(([category, _]) => this.formatCategoryName(category))
      .filter((name): name is string => name !== undefined);
    
    return lowScores;
  }

  private formatCategoryName(category: string): string {
    return category.replace(/([A-Z])/g, " $1").trim().toLowerCase()
      .split(" ")
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  }

  private generateSpecificFeedback(analysis: ExtendedAudioAnalysis): string[] {
    const feedback: string[] = [];
    
    if (analysis.pitchIntonation.tuningDriftPatterns.driftDirection === "flat") {
      feedback.push("Watch for pitch dropping throughout the performance");
    }
    
    if (analysis.timingRhythm.temporalAccuracy.rushTendency > 0.05) {
      feedback.push("Tendency to rush - focus on steady tempo");
    }
    
    if (analysis.toneQualityTimbre.vibratoCharacteristics.vibratoControl < 0.8) {
      feedback.push("Work on vibrato control and consistency");
    }
    
    return feedback.slice(0, 5);
  }

  private generateRecommendations(analysis: ExtendedAudioAnalysis): string[] {
    const recommendations: string[] = [];
    
    if (analysis.performanceScore.categoryScores.pitchIntonation < 75) {
      recommendations.push("Practice with tuner for intonation improvement");
    }
    
    if (analysis.performanceScore.categoryScores.technicalExecution < 80) {
      recommendations.push("Focus on technical exercises and scales");
    }
    
    return recommendations.slice(0, 3);
  }

  private calculateConfidenceScores(analysis: ExtendedAudioAnalysis): Record<string, number> {
    // Calculate confidence scores based on actual analysis data quality and reliability
    
    // Pitch confidence based on pitch track quality and consistency
    const pitchConfidence = this.calculatePitchConfidence(analysis.pitchIntonation, analysis.pitch);
    
    // Timing confidence based on onset detection quality and rhythm consistency
    const timingConfidence = this.calculateTimingConfidence(analysis.timingRhythm, analysis.rhythm);
    
    // Tone quality confidence based on spectral analysis stability
    const toneQualityConfidence = this.calculateToneQualityConfidence(analysis.toneQualityTimbre, analysis.spectral);
    
    // Technical execution confidence based on analysis consistency
    const technicalConfidence = this.calculateTechnicalConfidence(analysis.technicalExecution);
    
    // Musical expression confidence based on feature extraction quality
    const musicalConfidence = this.calculateMusicalConfidence(analysis.musicalExpression);
    
    // Consistency confidence based on performance stability
    const consistencyConfidence = this.calculateConsistencyConfidence(analysis.performanceConsistency);
    
    return {
      pitchIntonation: pitchConfidence,
      timingRhythm: timingConfidence,
      toneQuality: toneQualityConfidence,
      technicalExecution: technicalConfidence,
      musicalExpression: musicalConfidence,
      consistency: consistencyConfidence
    };
  }

  // Helper methods for confidence score calculation
  private calculatePitchConfidence(pitchAnalysis: PitchIntonationAnalysis, pitchData: { fundamentalFreq: number; melody: number[] }): number {
    // Base confidence from pitch tracking algorithm - use melody length as indicator
    const pitchDataQuality = pitchData.melody.filter(p => p > 0).length / pitchData.melody.length;
    
    // Factor in analysis quality indicators
    const stabilityFactor = 1 - (pitchAnalysis.pitchAccuracyDistribution.deviationStats.std / 100);
    const consistencyFactor = 1 - Math.abs(pitchAnalysis.tuningDriftPatterns.driftRate);
    
    // Combine factors with weights
    const confidence = (pitchDataQuality * 0.5) + (stabilityFactor * 0.3) + (consistencyFactor * 0.2);
    
    return Math.min(0.95, Math.max(0.5, confidence));
  }
  
  private calculateTimingConfidence(timingAnalysis: TimingRhythmAnalysis, rhythmData: { beats: number[]; onsets: number[] }): number {
    // Base confidence from timing analysis consistency
    const timingAccuracy = timingAnalysis.temporalAccuracy.overallTimingScore / 100;
    const rhythmStability = 1 - Math.abs(timingAnalysis.temporalAccuracy.rushTendency);
    
    // Factor in onset detection quality
    const onsetQuality = rhythmData.onsets.length > 0 ? Math.min(1, rhythmData.onsets.length / (rhythmData.beats.length * 2)) : 0.5;
    
    const confidence = (timingAccuracy * 0.5) + (rhythmStability * 0.3) + (onsetQuality * 0.2);
    
    return Math.min(0.95, Math.max(0.5, confidence));
  }
  
  private calculateToneQualityConfidence(toneAnalysis: ToneQualityTimbreAnalysis, spectralData: { mfcc: number[]; centroid: number; rolloff: number }): number {
    // Base confidence from spectral analysis stability
    const timbralStability = 1 - (toneAnalysis.vibratoCharacteristics.vibratoRate / 10); // Use vibrato as timbral indicator
    const harmonicQuality = toneAnalysis.harmonicContentAnalysis.harmonicRichness;
    
    // Factor in spectral data quality
    const spectralQuality = spectralData.mfcc.length > 0 ? Math.min(1, spectralData.mfcc.filter(m => !isNaN(m)).length / spectralData.mfcc.length) : 0.5;
    
    const confidence = (timbralStability * 0.4) + (harmonicQuality * 0.4) + (spectralQuality * 0.2);
    
    return Math.min(0.95, Math.max(0.5, confidence));
  }
  
  private calculateTechnicalConfidence(technicalAnalysis: TechnicalExecutionAnalysis): number {
    // Base confidence from technical execution consistency
    const articulationConfidence = technicalAnalysis.articulationClarity.tonguingPrecision;
    const fingeringConfidence = technicalAnalysis.fingerTechniqueEfficiency.fingeringAccuracy;
    const breathingConfidence = technicalAnalysis.breathManagementIndicators.breathSupportConsistency;
    
    const confidence = (articulationConfidence + fingeringConfidence + breathingConfidence) / 3;
    
    return Math.min(0.95, Math.max(0.5, confidence));
  }
  
  private calculateMusicalConfidence(musicalAnalysis: MusicalExpressionAnalysis): number {
    // Base confidence from musical expression consistency
    const phrasingConfidence = (musicalAnalysis.phrasingSophistication.musicalSentenceStructure + 
                              musicalAnalysis.phrasingSophistication.breathingLogic) / 2;
    const dynamicConfidence = musicalAnalysis.dynamicContourComplexity.dynamicShapingScore;
    const styleConfidence = musicalAnalysis.stylisticAuthenticity.styleConsistency;
    
    const confidence = (phrasingConfidence + dynamicConfidence + styleConfidence) / 3;
    
    return Math.min(0.95, Math.max(0.5, confidence));
  }
  
  private calculateConsistencyConfidence(consistencyAnalysis: PerformanceConsistencyAnalysis): number {
    // Base confidence from performance consistency metrics
    const errorRate = (consistencyAnalysis.errorFrequencyAndType.missedNotes + 
                      consistencyAnalysis.errorFrequencyAndType.crackedNotes +
                      consistencyAnalysis.errorFrequencyAndType.timingSlips) / 3;
    const recoveryEffectiveness = consistencyAnalysis.recoverySpeed.recoveryEffectiveness;
    const enduranceStability = consistencyAnalysis.endurancePatterns.consistencyThroughoutPerformance;
    
    // Lower error rate = higher confidence
    const errorConfidence = Math.max(0.3, 1 - (errorRate / 10));
    
    const confidence = (errorConfidence * 0.4) + (recoveryEffectiveness * 0.3) + (enduranceStability * 0.3);
    
    return Math.min(0.95, Math.max(0.5, confidence));
  }

  // Helper methods for rule-based analysis
  private analyzePhrasing(onsets: number[], pitchTrack: number[], energy: number[]): MusicalExpressionAnalysis["phrasing"] {
    const phraseBreaks = this.detectPhraseBreaks(onsets, this.config.onset.threshold);
    const phraseLengths = this.calculatePhraseLengths(onsets, phraseBreaks);
    const phraseContour = this.analyzePhraseContours(onsets, pitchTrack, energy, phraseBreaks);

    return {
      phraseLengths,
      breathingPoints: phraseBreaks,
      phraseContour
    };
  }

  private analyzeDynamics(energy: number[]): MusicalExpressionAnalysis["dynamics"] {
    const energyDb = energy.map(e => 20 * Math.log10(Math.max(e, 1e-10)));
    const minDb = Math.min(...energyDb);
    const maxDb = Math.max(...energyDb);
    const dynamicRange = { minDecibels: minDb, maxDecibels: maxDb, overallRange: maxDb - minDb };
    const dynamicChanges = this.detectDynamicChanges(energyDb);
    const dynamicConsistency = 1 - (this.calculateStatistics(energyDb).std / dynamicRange.overallRange);

    return {
      dynamicRange,
      dynamicChanges,
      dynamicConsistency: Math.max(0, Math.min(1, dynamicConsistency))
    };
  }

  private analyzeArticulation(onsets: number[], energy: number[]): MusicalExpressionAnalysis["articulation"] {
    const attackTransients = this.analyzeAttackTransients(onsets, energy);
    const attackClarity = this.calculateMean(attackTransients.map(t => t.clarity));
    const noteSeparation = this.determineNoteSeparation(onsets);

    return { attackClarity, noteSeparation };
  }

  private analyzePitchConsistency(pitchTrack: number[], numSections: number, sectionLength: number): PerformanceConsistencyAnalysis["pitchConsistency"] {
    const driftOverTime = this.calculatePitchDrift(pitchTrack);
    const standardDeviationBySection = [];
    for (let i = 0; i < numSections; i++) {
      const section = pitchTrack.slice(i * sectionLength, (i + 1) * sectionLength);
      standardDeviationBySection.push(this.calculateStatistics(section).std);
    }
    const overallStability = 1 - (this.calculateStatistics(pitchTrack).std / 50); // 50 cents is significant deviation

    return {
      driftOverTime,
      standardDeviationBySection,
      overallStability: Math.max(0, Math.min(1, overallStability))
    };
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
    const overallSteadiness = 1 - (this.calculateStatistics(beatTimingDeviation).std / 100); // 100ms is significant deviation

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
      timbreVariability.push(this.calculateStatistics(centroidSection).std);
      harmonicRichnessVariability.push(this.calculateStatistics(harmonicsSection).std);
    }
    const overallUniformity = 1 - (this.calculateStatistics(spectralCentroid).std / (this.calculateMean(spectralCentroid) || 1));

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

  private calculatePhraseLengths(onsets: number[], phraseBreaks: number[]): number[] {
    const lengths: number[] = [];
    let lastBreakTime = 0;
    for (const breakTime of phraseBreaks) {
      const phraseOnsets = onsets.filter(onset => onset > lastBreakTime && onset <= breakTime);
      if (phraseOnsets.length > 0) {
        lengths.push(breakTime - lastBreakTime);
      }
      lastBreakTime = breakTime;
    }
    // Add the last phrase
    const lastOnset = onsets[onsets.length - 1];
    if (lastOnset && lastOnset > lastBreakTime) {
      lengths.push(lastOnset - lastBreakTime);
    }
    return lengths;
  }

  private analyzePhraseContours(onsets: number[], pitchTrack: number[], energy: number[], phraseBreaks: number[]): MusicalExpressionAnalysis["phrasing"]["phraseContour"] {
    const contours: MusicalExpressionAnalysis["phrasing"]["phraseContour"] = [];
    let lastBreakTime = 0;
    for (const breakTime of phraseBreaks) {
      const startIndex = Math.floor(lastBreakTime * (this.sampleRate / 512)); // Assuming 512 hop size
      const endIndex = Math.floor(breakTime * (this.sampleRate / 512));
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
          changes.push({ time: i * (512 / this.sampleRate), type: "sudden_accent", magnitude: current - prevAvg });
        } else if (nextAvg > current + threshold && prevAvg < current - threshold) {
          changes.push({ time: i * (512 / this.sampleRate), type: "crescendo", magnitude: nextAvg - current });
        } else if (nextAvg < current - threshold && prevAvg > current + threshold) {
          changes.push({ time: i * (512 / this.sampleRate), type: "decrescendo", magnitude: current - nextAvg });
        }
      }
    }
    return changes;
  }

  private analyzeAttackTransients(onsets: number[], energy: number[]): Array<{ time: number; clarity: number }> {
    const transients: Array<{ time: number; clarity: number }> = [];
    const windowSize = Math.floor(0.05 * this.sampleRate); // 50ms window

    for (const onsetTime of onsets) {
      const onsetIndex = Math.floor(onsetTime * this.sampleRate);
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
    const stdDevInterval = this.calculateStatistics(interOnsetIntervals).std;

    if (stdDevInterval < 0.05 * meanInterval) { // Very consistent intervals
      if (meanInterval < 0.2) return "legato"; // Short, consistent intervals
      else return "staccato"; // Longer, consistent intervals
    } else {
      return "mixed"; // Inconsistent intervals
    }
  }

  private calculatePitchDrift(pitchTrack: number[]): number[] {
    if (pitchTrack.length === 0) return [];
    const referencePitch = pitchTrack[0];
    return pitchTrack.map(pitch => {
      if (pitch <= 0 || !referencePitch || referencePitch <= 0) return 0;
      return 1200 * Math.log2(pitch / referencePitch); // Cents
    });
  }

  // Utility methods

  private calculateMean(values: number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  private calculateMeanFrequency(pitchTrack: number[]): number {
    return pitchTrack.reduce((sum, freq) => sum + freq, 0) / pitchTrack.length;
  }

  private estimateKey(pitchTrack: number[]): string {
    const keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    
    // Filter out invalid pitch values
    const validPitches = pitchTrack.filter(pitch => pitch > 0 && pitch < 5000);
    
    if (validPitches.length === 0) {
      return "C"; // Default fallback
    }
    
    // Convert frequencies to pitch classes (0-11)
    const pitchClasses = validPitches.map(freq => {
      // Convert frequency to MIDI note number
      const midiNote = 12 * Math.log2(freq / 440) + 69;
      // Get pitch class (0-11)
      return Math.round(midiNote) % 12;
    });
    
    // Count occurrences of each pitch class
    const pitchClassCounts = new Array(12).fill(0);
    pitchClasses.forEach(pc => {
      if (pc >= 0 && pc < 12) {
        pitchClassCounts[pc]++;
      }
    });
    
    // Find the most common pitch class as the key center
    const mostCommonPitchClass = pitchClassCounts.indexOf(Math.max(...pitchClassCounts));
    
    return keys[mostCommonPitchClass] || "C";
  }

  private estimateChords(pitchTrack: number[]): string[] {
    // Filter out invalid pitch values
    const validPitches = pitchTrack.filter(pitch => pitch > 0 && pitch < 5000);
    
    if (validPitches.length === 0) {
      return ["Cmaj7"]; // Default fallback
    }
    
    // Divide the pitch track into segments for chord analysis
    const segmentSize = Math.max(1, Math.floor(validPitches.length / 4));
    const chords: string[] = [];
    
    for (let i = 0; i < validPitches.length; i += segmentSize) {
      const segment = validPitches.slice(i, i + segmentSize);
      const chord = this.analyzeChordInSegment(segment);
      chords.push(chord);
      
      // Limit to reasonable number of chords
      if (chords.length >= 8) break;
    }
    
    return chords.length > 0 ? chords : ["Cmaj7"];
  }

  private calculateChromaVector(pitchTrack: number[]): number[] {
    // Filter out invalid pitch values
    const validPitches = pitchTrack.filter(pitch => pitch > 0 && pitch < 5000);
    
    if (validPitches.length === 0) {
      return new Array(12).fill(0.083); // Equal distribution fallback
    }
    
    // Initialize chroma vector (12 pitch classes)
    const chromaVector = new Array(12).fill(0);
    
    // Convert frequencies to pitch classes and accumulate
    validPitches.forEach(freq => {
      // Convert frequency to MIDI note number
      const midiNote = 12 * Math.log2(freq / 440) + 69;
      // Get pitch class (0-11)
      const pitchClass = Math.round(midiNote) % 12;
      
      if (pitchClass >= 0 && pitchClass < 12) {
        chromaVector[pitchClass]++;
      }
    });
    
    // Normalize to get probability distribution
    const total = chromaVector.reduce((sum, count) => sum + count, 0);
    if (total > 0) {
      return chromaVector.map(count => count / total);
    }
    
    return new Array(12).fill(0.083); // Equal distribution fallback
  }
  
  private analyzeChordInSegment(pitchSegment: number[]): string {
    if (pitchSegment.length === 0) return "Cmaj7";
    
    // Convert frequencies to pitch classes
    const pitchClasses = pitchSegment.map(freq => {
      const midiNote = 12 * Math.log2(freq / 440) + 69;
      return Math.round(midiNote) % 12;
    });
    
    // Count pitch class occurrences
    const pitchClassCounts = new Array(12).fill(0);
    pitchClasses.forEach(pc => {
      if (pc >= 0 && pc < 12) {
        pitchClassCounts[pc]++;
      }
    });
    
    // Find the most prominent pitch classes
    const sortedPitchClasses = pitchClassCounts
      .map((count, index) => ({ pitchClass: index, count }))
      .filter(item => item.count > 0)
      .sort((a, b) => b.count - a.count);
    
    if (sortedPitchClasses.length === 0) return "Cmaj7";
    
    // Get the root pitch class
    const rootPitchClass = sortedPitchClasses[0]?.pitchClass ?? 0;
    
    // Map pitch classes to note names
    const noteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    const rootNote = noteNames[rootPitchClass];
    
    // Simple chord estimation based on prominent pitch classes
    const prominentPitchClasses = sortedPitchClasses.slice(0, 3).map(item => item.pitchClass);
    
    // Check for common chord patterns
    if (prominentPitchClasses.length >= 3) {
      // Sort for pattern matching
      const sortedPCs = prominentPitchClasses.sort((a, b) => a - b);
      const intervals = [];
      
      for (let i = 1; i < sortedPCs.length; i++) {
        const current = sortedPCs[i];
        const root = sortedPCs[0];
        if (current !== undefined && root !== undefined) {
          intervals.push((current - root + 12) % 12);
        }
      }
      
      // Major chord pattern (root, major third, perfect fifth)
      if (intervals.includes(4) && intervals.includes(7)) {
        return `${rootNote}maj7`;
      }
      // Minor chord pattern (root, minor third, perfect fifth)
      if (intervals.includes(3) && intervals.includes(7)) {
        return `${rootNote}m7`;
      }
      // Dominant chord pattern
      if (intervals.includes(4) && intervals.includes(10)) {
        return `${rootNote}7`;
      }
    }
    
    // Default to major 7th chord
    return `${rootNote}maj7`;
  }

  

  // Technical Execution Analysis Methods

  private async analyzeArticulationClarity(analysisResult: EssentiaAnalysisResult): Promise<ArticulationClarityAnalysis> {
    const { onsets, loudness, spectralCentroid } = analysisResult;
    
    // Analyze attack consistency using onset detection
    const attackConsistency = this.calculateAttackConsistency(onsets, loudness);
    
    // Analyze tonguing precision using spectral analysis
    const tonguingPrecision = this.calculateTonguingPrecision(onsets, spectralCentroid);
    
    // Analyze different articulation types based on onset patterns
    const articulationTypes = this.analyzeArticulationTypes(onsets, loudness);
    
    // Analyze clarity by tempo (estimated from beat intervals)
    const clarityByTempo = this.analyzeClarityByTempo(onsets, analysisResult.tempo);
    
    return {
      tonguingPrecision,
      attackConsistency,
      articulationTypes,
      clarityByTempo
    };
  }

  private async analyzeFingerTechnique(analysisResult: EssentiaAnalysisResult): Promise<FingerTechniqueAnalysis> {
    const { pitchTrack, onsets, spectralFlux } = analysisResult;
    
    // Analyze key transition smoothness using pitch continuity
    const keyTransitionSmoothness = this.calculateKeyTransitionSmoothness(pitchTrack, onsets);
    
    // Analyze passage cleanness using spectral flux
    const passageCleanness = this.calculatePassageCleanness(spectralFlux, onsets);
    
    // Analyze fingering accuracy using pitch stability
    const fingeringAccuracy = this.calculateFingeringAccuracy(pitchTrack);
    
    // Analyze technical passage success based on exercise type
    const technicalPassageSuccess = this.analyzeTechnicalPassageSuccess(
      pitchTrack, onsets
    );
    
    return {
      keyTransitionSmoothness,
      passageCleanness,
      fingeringAccuracy,
      technicalPassageSuccess
    };
  }

  private async analyzeBreathManagement(analysisResult: EssentiaAnalysisResult): Promise<BreathManagementAnalysis> {
    const { loudness, onsets, energy } = analysisResult;
    
    // Analyze phrase lengths using silence detection
    const phraseLength = this.calculatePhraseLengths(loudness, onsets);
    
    // Analyze breathing placement appropriateness
    const breathingPlacement = this.analyzeBreathingPlacement(loudness, onsets);
    
    // Analyze sustain capacity using energy consistency
    const sustainCapacity = this.calculateSustainCapacity(energy);
    
    // Analyze breath support consistency using loudness stability
    const breathSupportConsistency = this.calculateBreathSupportConsistency(loudness);
    
    return {
      phraseLength,
      breathingPlacement,
      sustainCapacity,
      breathSupportConsistency
    };
  }

  private async analyzeExtendedTechniques(analysisResult: EssentiaAnalysisResult): Promise<ExtendedTechniquesAnalysis> {
    const { spectralCentroid, harmonics, zcr, pitchTrack } = analysisResult;
    
    // Analyze multiphonics using harmonic content
    const multiphonicsClarity = this.detectMultiphonics(harmonics, spectralCentroid);
    
    // Analyze altissimo control using high-frequency content
    const altissimoControl = this.analyzeAltissimoControl(pitchTrack, spectralCentroid);
    
    // Analyze growl execution using spectral noise characteristics
    const growlExecution = this.detectGrowlTechnique(spectralCentroid, zcr);
    
    // Analyze pitch bending accuracy
    const bendAccuracy = this.analyzePitchBending(pitchTrack);
    
    // Analyze other extended techniques
    const otherTechniques = this.analyzeOtherExtendedTechniques(spectralCentroid, zcr);
    
    return {
      multiphonicsClarity,
      altissimoControl,
      growlExecution,
      bendAccuracy,
      otherTechniques: otherTechniques as Record<string, number>
    };
  }

  // Helper methods for articulation analysis
  private calculateAttackConsistency(onsets: number[], loudness: number[]): number {
    if (onsets.length < 2) return 0.5;
    
    // Analyze the consistency of attack strength at onset times
    const attackStrengths = onsets.map(onset => {
      const frameIndex = Math.floor(onset * 100); // Assuming 100Hz analysis rate
      return frameIndex < loudness.length ? loudness[frameIndex] || 0 : 0;
    });
    
    const avgStrength = attackStrengths.reduce((sum, s) => sum + s, 0) / attackStrengths.length;
    const variance = attackStrengths.reduce((sum, s) => sum + Math.pow(s - avgStrength, 2), 0) / attackStrengths.length;
    
    // Convert variance to consistency score (lower variance = higher consistency)
    return Math.max(0, 1 - (Math.sqrt(variance) / avgStrength));
  }

  private calculateTonguingPrecision(onsets: number[], spectralCentroid: number[]): number {
    if (onsets.length < 2) return 0.5;
    
    // Analyze spectral centroid changes at onset times for tonguing clarity
    let precisionScore = 0;
    let validOnsets = 0;
    
    for (const onset of onsets) {
      const frameIndex = Math.floor(onset * 100);
      if (frameIndex < spectralCentroid.length - 5) {
        // Look for sharp spectral changes indicating clear tonguing
        const beforeCentroid = spectralCentroid[frameIndex - 1] || 0;
        const atCentroid = spectralCentroid[frameIndex] || 0;
        const afterCentroid = spectralCentroid[frameIndex + 1] || 0;
        
        const clarityIndicator = Math.abs(atCentroid - beforeCentroid) + Math.abs(afterCentroid - atCentroid);
        precisionScore += Math.min(1, clarityIndicator / 500); // Normalize
        validOnsets++;
      }
    }
    
    return validOnsets > 0 ? precisionScore / validOnsets : 0.5;
  }

  private analyzeArticulationTypes(onsets: number[], loudness: number[]): ArticulationTypesAnalysis {
    // Simplified analysis based on onset patterns and loudness characteristics
    const staccato = this.detectStaccatoArticulation(onsets, loudness);
    const legato = this.detectLegatoArticulation(onsets, loudness);
    const tenuto = this.detectTenutoArticulation(onsets, loudness);
    
    return { 
      staccato, 
      legato, 
      tenuto,
      overallArticulationScore: (staccato + legato + tenuto) / 3
    };
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  private detectStaccatoArticulation(onsets: number[], loudness: number[]): number {
    // Staccato characterized by short, detached notes
    let staccatoScore = 0;
    for (let i = 0; i < onsets.length - 1; i++) {
      const currentOnset = onsets[i];
      const nextOnset = onsets[i + 1];
      if (currentOnset !== undefined && nextOnset !== undefined) {
        const noteDuration = nextOnset - currentOnset;
        if (noteDuration < 0.3) { // Short notes indicate staccato
          staccatoScore += 1;
        }
      }
    }
    return onsets.length > 1 ? staccatoScore / (onsets.length - 1) : 0.5;
  }

  private detectLegatoArticulation(onsets: number[], loudness: number[]): number {
    // Legato characterized by smooth connections between notes
    let legatoScore = 0;
    for (let i = 0; i < onsets.length - 1; i++) {
      const currentOnset = onsets[i];
      const nextOnset = onsets[i + 1];
      if (currentOnset !== undefined && nextOnset !== undefined) {
        const startFrame = Math.floor(currentOnset * 100);
        const endFrame = Math.floor(nextOnset * 100);
      
        // Check for sustained loudness between notes
        let sustainedFrames = 0;
        for (let frame = startFrame; frame < endFrame && frame < loudness.length; frame++) {
          if ((loudness[frame] || 0) > 0.1) sustainedFrames++;
        }
        
        const sustainRatio = sustainedFrames / (endFrame - startFrame);
        if (sustainRatio > 0.7) legatoScore += 1;
      }
    }
    return onsets.length > 1 ? legatoScore / (onsets.length - 1) : 0.5;
  }

  private detectTenutoArticulation(onsets: number[], loudness: number[]): number {
    // Tenuto characterized by held, stressed notes
    let tenutoScore = 0;
    for (let i = 0; i < onsets.length - 1; i++) {
      const currentOnset = onsets[i];
      const nextOnset = onsets[i + 1];
      if (currentOnset !== undefined && nextOnset !== undefined) {
        const noteDuration = nextOnset - currentOnset;
        const frameIndex = Math.floor(currentOnset * 100);
        const noteStrength = frameIndex < loudness.length ? loudness[frameIndex] || 0 : 0;
      
        // Tenuto: longer notes with sustained strength
        if (noteDuration > 0.5 && noteStrength > 0.3) {
          tenutoScore += 1;
        }
      }
    }
    return onsets.length > 1 ? tenutoScore / (onsets.length - 1) : 0.5;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  private analyzeClarityByTempo(onsets: number[], tempo: number): ClarityByTempoAnalysis {
    // Analyze articulation clarity at different tempos
    const firstOnset = onsets[0];
    const lastOnset = onsets[onsets.length - 1];
    const avgInterval = onsets.length > 1 && firstOnset !== undefined && lastOnset !== undefined ? 
      (lastOnset - firstOnset) / (onsets.length - 1) : 1;
    
    const estimatedTempo = 60 / avgInterval;
    
    // Score clarity based on tempo - generally harder to articulate clearly at faster tempos
    let slow = 0.9, moderate = 0.8, fast = 0.7;
    
    if (estimatedTempo < 80) {
      slow = 0.95;
      moderate = 0.85;
      fast = 0.75;
    } else if (estimatedTempo > 140) {
      slow = 0.85;
      moderate = 0.75;
      fast = 0.65;
    }
    
    return { 
      slow, 
      medium: moderate, 
      fast,
      overall: (slow + moderate + fast) / 3
    };
  }

  // Helper methods for finger technique analysis
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  private calculateKeyTransitionSmoothness(pitchTrack: number[], onsets: number[]): number {
    if (pitchTrack.length < 10) return 0.5;
    
    let smoothnessScore = 0;
    let transitions = 0;
    
    // Analyze pitch transitions for smoothness
    for (let i = 1; i < pitchTrack.length; i++) {
      const currentPitch = pitchTrack[i];
      const prevPitch = pitchTrack[i - 1];
      
      if (currentPitch && prevPitch && currentPitch > 0 && prevPitch > 0) {
        const pitchChange = Math.abs(currentPitch - prevPitch);
        // Smooth transitions have gradual pitch changes
        if (pitchChange > 10) { // Significant pitch change
          const smoothness = Math.max(0, 1 - (pitchChange / 100));
          smoothnessScore += smoothness;
          transitions++;
        }
      }
    }
    
    return transitions > 0 ? smoothnessScore / transitions : 0.8;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  private calculatePassageCleanness(spectralFlux: number[], onsets: number[]): number {
    if (spectralFlux.length === 0) return 0.5;
    
    // Clean passages have low spectral flux (less spectral noise)
    const avgFlux = spectralFlux.reduce((sum, flux) => sum + flux, 0) / spectralFlux.length;
    
    // Lower average flux indicates cleaner playing
    return Math.max(0, 1 - (avgFlux / 0.5)); // Normalize to 0-1 range
  }

  private calculateFingeringAccuracy(pitchTrack: number[]): number {
    if (pitchTrack.length < 10) return 0.5;
    
    let accuracyScore = 0;
    let validNotes = 0;
    
    // Analyze pitch stability for each note region
    for (let i = 5; i < pitchTrack.length - 5; i += 10) {
      const window = pitchTrack.slice(i - 5, i + 5);
      const validPitches = window.filter(p => p > 0);
      
      if (validPitches.length > 5) {
        const avgPitch = validPitches.reduce((sum, p) => sum + p, 0) / validPitches.length;
        const stability = validPitches.reduce((sum, p) => sum + Math.abs(p - avgPitch), 0) / validPitches.length;
        
        // Lower pitch deviation indicates better fingering accuracy
        accuracyScore += Math.max(0, 1 - (stability / 10));
        validNotes++;
      }
    }
    
    return validNotes > 0 ? accuracyScore / validNotes : 0.8;
  }

  private analyzeTechnicalPassageSuccess(pitchTrack: number[], onsets: number[]): TechnicalPassageAnalysis {
    // Analyze success based on exercise type
    const scales = this.analyzeScaleSuccess(pitchTrack, onsets);
    const arpeggios = this.analyzeArpeggioSuccess(pitchTrack, onsets);
    const chromatic = this.analyzeChromaticSuccess(pitchTrack, onsets);
    
    return { 
      scales, 
      arpeggios, 
      chromatic, 
      overall: (scales + arpeggios + chromatic) / 3 
    };
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  private analyzeScaleSuccess(pitchTrack: number[], _onsets: number[]): number {
    // Scales have stepwise motion - analyze for consistent intervals
    if (pitchTrack.length < 8) return 0.5;
    
    let scaleScore = 0;
    let intervals = 0;
    
    for (let i = 1; i < pitchTrack.length; i++) {
      const current = pitchTrack[i];
      const prev = pitchTrack[i - 1];
      
      if (current && prev && current > 0 && prev > 0) {
        const semitones = Math.abs(12 * Math.log2(current / prev));
        // Scale steps are typically 1-2 semitones
        if (semitones >= 0.5 && semitones <= 2.5) {
          scaleScore += 1;
        }
        intervals++;
      }
    }
    
    return intervals > 0 ? scaleScore / intervals : 0.7;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  private analyzeArpeggioSuccess(pitchTrack: number[], _onsets: number[]): number {
    // Arpeggios have larger, chord-tone intervals
    if (pitchTrack.length < 6) return 0.5;
    
    let arpeggioScore = 0;
    let intervals = 0;
    
    for (let i = 1; i < pitchTrack.length; i++) {
      const current = pitchTrack[i];
      const prev = pitchTrack[i - 1];
      
      if (current && prev && current > 0 && prev > 0) {
        const semitones = Math.abs(12 * Math.log2(current / prev));
        // Arpeggio intervals are typically 3-4 semitones (thirds/fourths)
        if (semitones >= 2.5 && semitones <= 5.5) {
          arpeggioScore += 1;
        }
        intervals++;
      }
    }
    
    return intervals > 0 ? arpeggioScore / intervals : 0.6;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  private analyzeChromaticSuccess(pitchTrack: number[], _onsets: number[]): number {
    // Chromatic passages have consistent semitone steps
    if (pitchTrack.length < 6) return 0.5;
    
    let chromaticScore = 0;
    let intervals = 0;
    
    for (let i = 1; i < pitchTrack.length; i++) {
      const current = pitchTrack[i];
      const prev = pitchTrack[i - 1];
      
      if (current && prev && current > 0 && prev > 0) {
        const semitones = Math.abs(12 * Math.log2(current / prev));
        // Chromatic steps are exactly 1 semitone
        if (semitones >= 0.8 && semitones <= 1.2) {
          chromaticScore += 1;
        }
        intervals++;
      }
    }
    
    return intervals > 0 ? chromaticScore / intervals : 0.6;
  }


  // Helper methods for breath management analysis
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  private calculatePhraseLengthsFromLoudness(loudness: number[], _onsets: number[]): number[] {
    const phraseLengths: number[] = [];
    let phraseStart = 0;
    
    // Detect phrase boundaries based on silence periods
    for (let i = 1; i < loudness.length; i++) {
      const currentLoudness = loudness[i] || 0;
      const prevLoudness = loudness[i - 1] || 0;
      
      // Detect end of phrase (drop to silence)
      if (prevLoudness > 0.1 && currentLoudness < 0.05) {
        const phraseLength = (i - phraseStart) * 0.01; // Convert frames to seconds
        if (phraseLength > 1) { // Only count significant phrases
          phraseLengths.push(phraseLength);
        }
        phraseStart = i;
      }
    }
    
    return phraseLengths.length > 0 ? phraseLengths : [4.0, 5.5, 3.8]; // Default values
  }

  private analyzeBreathingPlacement(loudness: number[], onsets: number[]): Array<{ time: number; appropriateness: number }> {
    const breathingPlacements: Array<{ time: number; appropriateness: number }> = [];
    
    // Find silence periods that indicate breathing
    for (let i = 0; i < loudness.length - 10; i++) {
      const silenceWindow = loudness.slice(i, i + 10);
      const avgLoudness = silenceWindow.reduce((sum, l) => sum + (l || 0), 0) / silenceWindow.length;
      
      if (avgLoudness < 0.02) { // Silence detected
        const time = i * 0.01; // Convert to seconds
        
        // Assess appropriateness based on musical context
        // For now, use a simple heuristic based on timing
        const appropriateness = this.assessBreathingAppropriateness(time, onsets);
        
        breathingPlacements.push({ time, appropriateness });
      }
    }
    
    return breathingPlacements.length > 0 ? breathingPlacements : [
      { time: 8.5, appropriateness: 0.9 },
      { time: 16.2, appropriateness: 0.85 }
    ];
  }

  private assessBreathingAppropriateness(breathTime: number, onsets: number[]): number {
    // Simple heuristic: breathing is more appropriate at phrase boundaries
    // Find the closest onset before and after the breath
    let beforeOnset = 0;
    let afterOnset = breathTime + 10;
    
    for (const onset of onsets) {
      if (onset <= breathTime && onset > beforeOnset) {
        beforeOnset = onset;
      }
      if (onset > breathTime && onset < afterOnset) {
        afterOnset = onset;
      }
    }
    
    const timeSinceLastNote = breathTime - beforeOnset;
    const timeToNextNote = afterOnset - breathTime;
    
    // More appropriate if there's sufficient time before and after
    const appropriateness = Math.min(1, (timeSinceLastNote + timeToNextNote) / 2);
    return Math.max(0.5, appropriateness);
  }

  private calculateSustainCapacity(energy: number[]): number {
    if (energy.length === 0) return 0.5;
    
    // Analyze energy consistency over time
    const avgEnergy = energy.reduce((sum, e) => sum + e, 0) / energy.length;
    const energyVariance = energy.reduce((sum, e) => sum + Math.pow(e - avgEnergy, 2), 0) / energy.length;
    
    // Lower variance indicates better sustain capacity
    return Math.max(0, 1 - (Math.sqrt(energyVariance) / avgEnergy));
  }

  private calculateBreathSupportConsistency(loudness: number[]): number {
    if (loudness.length === 0) return 0.5;
    
    // Analyze loudness stability during sustained notes
    const validLoudness = loudness.filter(l => l > 0.1);
    if (validLoudness.length === 0) return 0.5;
    
    const avgLoudness = validLoudness.reduce((sum, l) => sum + l, 0) / validLoudness.length;
    const variance = validLoudness.reduce((sum, l) => sum + Math.pow(l - avgLoudness, 2), 0) / validLoudness.length;
    
    // Lower variance indicates more consistent breath support
    return Math.max(0, 1 - (Math.sqrt(variance) / avgLoudness));
  }

  // Helper methods for extended techniques analysis
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  private detectMultiphonics(harmonics: number[], _spectralCentroid: number[]): number {
    if (harmonics.length === 0) return 0.3;
    
    // Multiphonics show multiple strong frequency peaks
    const strongHarmonics = harmonics.filter(h => h > 0.3);
    const multiphonicsIndicator = strongHarmonics.length / harmonics.length;
    
    return Math.min(1, multiphonicsIndicator * 1.5);
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  private analyzeAltissimoControl(pitchTrack: number[], _spectralCentroid: number[]): number {
    if (pitchTrack.length === 0) return 0.4;
    
    // Altissimo characterized by very high pitches (above normal range)
    const highPitches = pitchTrack.filter(p => p > 800); // Above high B♭
    const altissimoRatio = highPitches.length / pitchTrack.filter(p => p > 0).length;
    
    if (altissimoRatio > 0.1) {
      // Analyze stability of high pitches
      const avgHighPitch = highPitches.reduce((sum, p) => sum + p, 0) / highPitches.length;
      const stability = highPitches.reduce((sum, p) => sum + Math.abs(p - avgHighPitch), 0) / highPitches.length;
      
      return Math.max(0.3, 1 - (stability / avgHighPitch));
    }
    
    return 0.4;
  }

  private detectGrowlTechnique(spectralCentroid: number[], zcr: number[]): number {
    if (spectralCentroid.length === 0 || zcr.length === 0) return 0.3;
    
    // Growl technique shows high zero-crossing rate and spectral noise
    const avgZCR = zcr.reduce((sum, z) => sum + z, 0) / zcr.length;
    const avgCentroid = spectralCentroid.reduce((sum, sc) => sum + sc, 0) / spectralCentroid.length;
    
    // High ZCR and spectral centroid indicate growl-like noise
    const growlIndicator = (avgZCR / 0.2) * (avgCentroid / 2000);
    
    return Math.min(1, growlIndicator);
  }

  private analyzePitchBending(pitchTrack: number[]): number {
    if (pitchTrack.length < 10) return 0.5;
    
    let bendingScore = 0;
    let bends = 0;
    
    // Look for smooth pitch glides (bending)
    for (let i = 2; i < pitchTrack.length - 2; i++) {
      const window = pitchTrack.slice(i - 2, i + 3);
      const validPitches = window.filter(p => p > 0);
      
      if (validPitches.length >= 4) {
        // Check for monotonic pitch changes (smooth bending)
        let isSmooth = true;
        const firstPitch = validPitches[0];
        const secondPitch = validPitches[1];
        if (firstPitch !== undefined && secondPitch !== undefined) {
          const direction = secondPitch - firstPitch;
          
          for (let j = 1; j < validPitches.length - 1; j++) {
            const currentPitch = validPitches[j];
            const nextPitch = validPitches[j + 1];
            if (currentPitch !== undefined && nextPitch !== undefined) {
              const change = nextPitch - currentPitch;
              if (Math.sign(change) !== Math.sign(direction)) {
                isSmooth = false;
                break;
              }
            }
          }
        
          if (isSmooth && Math.abs(direction) > 2) {
            bendingScore += 1;
            bends++;
          }
        }
      }
    }
    
    return bends > 0 ? bendingScore / bends : 0.6;
  }

  private analyzeOtherExtendedTechniques(spectralCentroid: number[], zcr: number[], /* harmonics: number[] */): OtherExtendedTechniquesAnalysis {
    // Analyze various extended techniques
    const slapTongue = this.detectSlapTongue(spectralCentroid, zcr);
    const flutterTongue = this.detectFlutterTongue(spectralCentroid);
    
    return {
      harmonicControl: slapTongue,
      noiseTextures: flutterTongue,
      overallOtherScore: (slapTongue + flutterTongue) / 2
    };
  }

  private detectSlapTongue(spectralCentroid: number[], zcr: number[]): number {
    // Slap tongue creates sudden spectral attacks
    let slapScore = 0;
    let attacks = 0;
    
    for (let i = 1; i < spectralCentroid.length; i++) {
      const centroidChange = Math.abs((spectralCentroid[i] || 0) - (spectralCentroid[i - 1] || 0));
      const zcrChange = Math.abs((zcr[i] || 0) - (zcr[i - 1] || 0));
      
      // Sharp changes in both centroid and ZCR indicate slap attacks
      if (centroidChange > 200 && zcrChange > 0.05) {
        slapScore += 1;
        attacks++;
      }
    }
    
    return attacks > 0 ? Math.min(1, slapScore / (spectralCentroid.length * 0.1)) : 0.4;
  }

  private detectFlutterTongue(spectralCentroid: number[]): number {
    // Flutter tongue creates rapid spectral modulation
    if (spectralCentroid.length < 20) return 0.4;
    
    let modulationScore = 0;
    const windowSize = 10;
    
    for (let i = 0; i < spectralCentroid.length - windowSize; i += windowSize) {
      const window = spectralCentroid.slice(i, i + windowSize);
      const avgCentroid = window.reduce((sum, sc) => sum + (sc || 0), 0) / window.length;
      const variance = window.reduce((sum, sc) => sum + Math.pow((sc || 0) - avgCentroid, 2), 0) / window.length;
      
      // High variance indicates flutter-like modulation
      if (variance > 1000) {
        modulationScore += 1;
      }
    }
    
    const windowCount = Math.floor(spectralCentroid.length / windowSize);
    return windowCount > 0 ? modulationScore / windowCount : 0.4;
  }
}