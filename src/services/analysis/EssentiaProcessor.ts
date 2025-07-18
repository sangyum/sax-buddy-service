import * as EssentiaJS from "essentia.js";
import { EssentiaAnalysisResult } from "../../types";
import { Logger } from "../../utils/logger";
import { AudioUtils } from "./AudioUtils";
import { SAXOPHONE_CONFIG } from "./SaxophoneConstants";

const { Essentia, EssentiaWASM } = EssentiaJS;

export interface EssentiaConfig {
  sampleRate: number;
  frameSize: number;
  hopSize: number;
  vibrato: {
    minRate: number;
    maxRate: number;
    minDepth: number;
    qualityThreshold: number;
  };
  validation?: {
    minSampleRate?: number;
    maxSampleRate?: number;
    preferredSampleRate?: number;
  };
}

export class EssentiaProcessor {
  private logger = new Logger("EssentiaProcessor");
  private essentia: EssentiaJS.Essentia | null = null;
  private essentiaWasm: EssentiaJS.EssentiaWASM | null = null;
  private config: EssentiaConfig;

  constructor(config: EssentiaConfig) {
    this.config = {
      ...config,
      validation: {
        minSampleRate: config.validation?.minSampleRate ?? SAXOPHONE_CONFIG.SAMPLE_RATE.MINIMUM,
        maxSampleRate: config.validation?.maxSampleRate ?? SAXOPHONE_CONFIG.SAMPLE_RATE.MAXIMUM,
        preferredSampleRate: config.validation?.preferredSampleRate ?? SAXOPHONE_CONFIG.SAMPLE_RATE.PREFERRED
      }
    };
  }

  async initialize(): Promise<void> {
    if (!this.essentia) {
      this.logger.info("Initializing Essentia.js");
      try {
        this.essentiaWasm = new EssentiaWASM();
        await this.essentiaWasm.initialize();
        this.essentia = new Essentia(this.essentiaWasm);
        this.logger.info("Essentia.js initialized", {
          version: this.essentia?.version,
          algorithms: this.essentia?.algorithmNames?.split(",").length || 0
        });
      } catch (error) {
        this.logger.error("Failed to initialize Essentia.js", {
          error: error instanceof Error ? error.message : "Unknown error"
        });
        throw new Error(`Failed to initialize Essentia.js: ${error}`);
      }
    }
  }

  async performEssentiaAnalysis(audioBuffer: Float32Array): Promise<EssentiaAnalysisResult> {
    try {
      if (!this.essentia) {
        throw new Error("Essentia not initialized");
      }
      
      // Validate sample rate for saxophone analysis
      this.validateSampleRate();

      const audioVector = (this.essentia as EssentiaJS.Essentia).arrayToVector(audioBuffer);

      const rhythmResult = this.essentia.RhythmExtractor2013(
        audioVector as EssentiaJS.EssentiaVector,
        208,
        "multifeature",
        40
      );

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

          if (!frameArray || frameArray.length === 0) {
            skippedFrames++;
            continue;
          }

          const hasValidData = Array.from(frameArray).some(sample => isFinite(sample) && sample !== 0);
          if (!hasValidData) {
            skippedFrames++;
            continue;
          }

          const windowedFrame = this.essentia.Windowing(
            this.essentia.arrayToVector(frameArray),
            "hann",
            frameSize,
            false,
            1.0,
            true
          );

          const spectrumResult = this.essentia.Spectrum(
            windowedFrame.frame,
            frameSize
          );

          // Use saxophone-optimized pitch tracking parameters
          const pitchResult = this.essentia.YinFFT(
            spectrumResult.spectrum,
            frameSize,
            this.config.sampleRate
          );

          const pitch = pitchResult.pitch;
          const confidence = pitchResult.pitchConfidence;

          if (isFinite(pitch) && pitch >= 0) {
            pitchTrack.push(pitch);
            pitchConfidence.push(isFinite(confidence) ? confidence : 0);
          } else {
            pitchTrack.push(0);
            pitchConfidence.push(0);
          }

          try {
            const centroid = this.essentia.SpectralCentroid(spectrumResult.spectrum, this.config.sampleRate);
            // Filter out spectral centroid values outside saxophone range
            const validCentroid = isFinite(centroid.spectralCentroid) && 
                                 centroid.spectralCentroid >= SAXOPHONE_CONFIG.DEFAULT_RANGE.LOWEST && 
                                 centroid.spectralCentroid <= SAXOPHONE_CONFIG.DEFAULT_RANGE.HIGHEST * 5 ? 
                                 centroid.spectralCentroid : 0;
            spectralFeatures.centroid.push(validCentroid);
          } catch (error) {
            spectralFeatures.centroid.push(0);
          }

          try {
            const rolloff = this.essentia.SpectralRolloff(spectrumResult.spectrum, this.config.sampleRate);
            // Filter rolloff values for saxophone frequency range
            const validRolloff = isFinite(rolloff.spectralRolloff) && 
                                rolloff.spectralRolloff >= SAXOPHONE_CONFIG.DEFAULT_RANGE.LOWEST && 
                                rolloff.spectralRolloff <= SAXOPHONE_CONFIG.DEFAULT_RANGE.HIGHEST * 8 ? 
                                rolloff.spectralRolloff : 0;
            spectralFeatures.rolloff.push(validRolloff);
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
            // Use saxophone-optimized MFCC parameters
          const mfccResult = this.essentia.MFCC(
            spectrumResult.spectrum, 
            40, 
            13, 
            this.config.sampleRate
          );
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
          this.logger.error(`Frame processing error at index ${i}`, {
            error: error instanceof Error ? error.message : "Unknown error",
            frameIndex: i,
            totalFrames: framesSize
          });

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

      this.logger.info("Frame processing completed", {
        totalFrames: framesSize,
        processedFrames,
        skippedFrames,
        successRate: processedFrames / framesSize
      });

      if (processedFrames < framesSize * 0.5) {
        throw new Error(`Too many frames failed processing: ${processedFrames}/${framesSize} successful`);
      }

      const onsetResult = this.essentia.Onsets(audioVector as EssentiaJS.EssentiaVector);

      const vibratoCharacteristics = this.analyzeVibratoCharacteristics(pitchTrack, pitchConfidence);

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

      const meanMfcc = AudioUtils.calculateMeanVector(spectralFeatures.mfcc);
      const meanHarmonics = AudioUtils.calculateMeanVector(spectralFeatures.harmonics);

      return {
        tempo: rhythmResult.bpm || 120,
        confidence: rhythmResult.confidence || 0.5,
        beats: Array.from(rhythmResult.beats || []),
        pitchTrack,
        pitchConfidence,
        onsets: Array.from(onsetResult.onsets || []),
        mfcc: meanMfcc,
        spectralCentroid: spectralFeatures.centroid,
        spectralRolloff: spectralFeatures.rolloff,
        spectralFlux: spectralFeatures.flux,
        harmonics: meanHarmonics,
        energy: spectralFeatures.energy,
        zcr: spectralFeatures.zcr,
        loudness: spectralFeatures.energy,
        vibratoCharacteristics: vibratoCharacteristics
      };

    } catch (error) {
      this.logger.error("Essentia analysis failed", {
        error: error instanceof Error ? error.message : "Unknown error"
      });

      if (error instanceof Error) {
        throw new Error(`Essentia.js analysis failed: ${error.message}`);
      } else {
        throw new Error("Essentia.js analysis failed with unknown error");
      }
    }
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

    const vibratoRegions = this.detectVibratoRegions(pitchTrack, pitchConfidence);

    if (vibratoRegions.length === 0) {
      return {
        vibratoRate: 0,
        vibratoDepth: 0,
        vibratoControl: 0.5,
        vibratoTiming: []
      };
    }

    let totalRate = 0;
    let totalDepth = 0;
    let totalQuality = 0;

    for (const region of vibratoRegions) {
      totalRate += region.rate;
      totalDepth += region.depth;
      totalQuality += region.quality;
    }

    const vibratoRate = totalRate / vibratoRegions.length;
    const vibratoDepth = totalDepth / vibratoRegions.length;
    const vibratoControl = totalQuality / vibratoRegions.length;

    const vibratoTiming = vibratoRegions.map(region => ({
      start: region.start,
      end: region.end,
      quality: region.quality
    }));

    return {
      vibratoRate,
      vibratoDepth,
      vibratoControl,
      vibratoTiming
    };
  }

  private detectVibratoRegions(pitchTrack: number[], pitchConfidence: number[]): Array<{
    start: number;
    end: number;
    rate: number;
    depth: number;
    quality: number;
  }> {
    const regions: Array<{
      start: number;
      end: number;
      rate: number;
      depth: number;
      quality: number;
    }> = [];
    const minVibratoDuration = 0.5; // seconds
    const frameRate = 100; // Assuming 100 frames per second

    let inVibratoRegion = false;
    let currentRegionStart = 0;

    for (let i = 0; i < pitchTrack.length; i++) {
      const pitch = pitchTrack[i] || 0;
      const confidence = pitchConfidence[i] || 0;

      if (pitch > 0 && confidence > this.config.vibrato.qualityThreshold) {
        const window = pitchTrack.slice(Math.max(0, i - 10), i + 1);
        const stats = AudioUtils.calculateStatistics(window.filter(p => p > 0));

        // Use saxophone-specific vibrato detection parameters
        const vibratoDepthCents = this.convertFrequencyDeviationToCents(stats.std, pitch);
        if (vibratoDepthCents > SAXOPHONE_CONFIG.VIBRATO.MIN_DEPTH && 
            vibratoDepthCents < SAXOPHONE_CONFIG.VIBRATO.MAX_DEPTH) {
          if (!inVibratoRegion) {
            currentRegionStart = i;
            inVibratoRegion = true;
          }
        } else {
          if (inVibratoRegion) {
            const duration = (i - currentRegionStart) / frameRate;
            if (duration >= minVibratoDuration) {
              const regionPitches = pitchTrack.slice(currentRegionStart, i);
              const regionConfidences = pitchConfidence.slice(currentRegionStart, i);
              const vibratoAnalysis = this.analyzeVibratoInRegion(regionPitches, regionConfidences);
              regions.push({
                start: currentRegionStart / frameRate,
                end: i / frameRate,
                rate: vibratoAnalysis.rate,
                depth: vibratoAnalysis.depth,
                quality: vibratoAnalysis.quality
              });
            }
            inVibratoRegion = false;
          }
        }
      } else {
        if (inVibratoRegion) {
          const duration = (i - currentRegionStart) / frameRate;
          if (duration >= minVibratoDuration) {
            const regionPitches = pitchTrack.slice(currentRegionStart, i);
            const regionConfidences = pitchConfidence.slice(currentRegionStart, i);
            const vibratoAnalysis = this.analyzeVibratoInRegion(regionPitches, regionConfidences);
            regions.push({
              start: currentRegionStart / frameRate,
              end: i / frameRate,
              rate: vibratoAnalysis.rate,
              depth: vibratoAnalysis.depth,
              quality: vibratoAnalysis.quality
            });
          }
          inVibratoRegion = false;
        }
      }
    }

    if (inVibratoRegion) {
      const duration = (pitchTrack.length - currentRegionStart) / frameRate;
      if (duration >= minVibratoDuration) {
        const regionPitches = pitchTrack.slice(currentRegionStart, pitchTrack.length);
        const regionConfidences = pitchConfidence.slice(currentRegionStart, pitchTrack.length);
        const vibratoAnalysis = this.analyzeVibratoInRegion(regionPitches, regionConfidences);
        regions.push({
          start: currentRegionStart / frameRate,
          end: pitchTrack.length / frameRate,
          rate: vibratoAnalysis.rate,
          depth: vibratoAnalysis.depth,
          quality: vibratoAnalysis.quality
        });
      }
    }

    return regions;
  }

  private analyzeVibratoInRegion(pitchTrack: number[], pitchConfidence: number[]): {
    rate: number;
    depth: number;
    quality: number;
  } {
    const validPitches = pitchTrack.filter(p => p > 0);
    if (validPitches.length < 20) {
      return { rate: 0, depth: 0, quality: 0 };
    }

    const stats = AudioUtils.calculateStatistics(validPitches);
    const depth = stats.std;

    let peaks = 0;
    let troughs = 0;
    for (let i = 1; i < validPitches.length - 1; i++) {
      const current = validPitches[i];
      const prev = validPitches[i - 1];
      const next = validPitches[i + 1];
      
      if (current !== undefined && prev !== undefined && next !== undefined) {
        if (current > prev && current > next) {
          peaks++;
        }
        if (current < prev && current < next) {
          troughs++;
        }
      }
    }
    const rate = (peaks + troughs) / 2 / (validPitches.length / 100);

    const quality = AudioUtils.calculateStatistics(pitchConfidence).mean;

    return { rate, depth, quality };
  }
  
  private validateSampleRate(): void {
    if (!this.config.validation) return;
    
    const { minSampleRate, maxSampleRate, preferredSampleRate } = this.config.validation;
    
    if (this.config.sampleRate < minSampleRate!) {
      this.logger.warn("Sample rate below minimum for saxophone analysis", {
        current: this.config.sampleRate,
        minimum: minSampleRate,
        preferred: preferredSampleRate
      });
    }
    
    if (this.config.sampleRate > maxSampleRate!) {
      this.logger.warn("Sample rate above maximum practical limit", {
        current: this.config.sampleRate,
        maximum: maxSampleRate,
        preferred: preferredSampleRate
      });
    }
    
    if (this.config.sampleRate !== preferredSampleRate) {
      this.logger.info("Using non-preferred sample rate for saxophone analysis", {
        current: this.config.sampleRate,
        preferred: preferredSampleRate
      });
    }
  }
  
  private convertFrequencyDeviationToCents(freqStd: number, avgFreq: number): number {
    if (avgFreq <= 0) return 0;
    return 1200 * Math.log2((avgFreq + freqStd) / avgFreq);
  }
}
