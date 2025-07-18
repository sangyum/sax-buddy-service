import { TechnicalExecutionAnalysis, EssentiaAnalysisResult, ArticulationClarityAnalysis, ClarityByTempoAnalysis, ArticulationTypesAnalysis, FingerTechniqueAnalysis, TechnicalPassageAnalysis, BreathManagementAnalysis, ExtendedTechniquesAnalysis, OtherExtendedTechniquesAnalysis } from "../../types";
import { AudioUtils } from "./AudioUtils";
import { 
  SAXOPHONE_CONFIG, 
  detectSaxophoneType, 
  getSaxophoneRange, 
  isAltissimoRange, 
  getRegisterDifficulty 
} from "./SaxophoneConstants";


export class TechnicalExecutionAnalyzer {
  private saxophoneType: keyof typeof SAXOPHONE_CONFIG.FREQUENCY_RANGES = "ALTO";
  
  async analyze(analysisResult: EssentiaAnalysisResult): Promise<TechnicalExecutionAnalysis> {
    // Detect saxophone type from pitch range
    this.saxophoneType = detectSaxophoneType(analysisResult.pitchTrack);

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
    const clarityByTempo = this.analyzeClarityByTempo(onsets, analysisResult.tempo, analysisResult);
    
    return {
      tonguingPrecision,
      attackConsistency,
      articulationTypes,
      clarityByTempo
    };
  }

  private analyzeClarityByTempo(onsets: number[], tempo: number, analysisResult: EssentiaAnalysisResult): ClarityByTempoAnalysis {
    if (onsets.length < 2) {
      return {
        slow: 0.5,
        medium: 0.5,
        fast: 0.5,
        overall: 0.5
      };
    }
    
    // Use the provided tempo parameter for more accurate analysis
    const actualTempo = tempo > 0 ? tempo : this.estimateTempoFromOnsets(onsets);
    
    // Analyze onset consistency and precision for tempo-based clarity scoring
    const onsetConsistency = this.calculateOnsetConsistency(onsets);
    const articulationPrecision = this.calculateArticulationPrecision(onsets, actualTempo);
    
    // Calculate clarity scores based on tempo ranges and performance characteristics
    const clarityScores = this.calculateTempoBasedClarityScores(
      actualTempo, 
      onsetConsistency, 
      articulationPrecision
    );
    
    // Analyze tempo-specific challenges for saxophone articulation
    const tempoSpecificFactors = this.analyzeTempoSpecificFactors(actualTempo, onsets, analysisResult.pitchTrack);
    
    // Apply tempo-specific adjustments to clarity scores
    const adjustedScores = this.applyTempoSpecificAdjustments(
      clarityScores, 
      tempoSpecificFactors
    );
    
    return {
      slow: adjustedScores.slow,
      medium: adjustedScores.medium,
      fast: adjustedScores.fast,
      overall: (adjustedScores.slow + adjustedScores.medium + adjustedScores.fast) / 3
    };
  }

  private analyzeTempoSpecificFactors(tempo: number, onsets: number[], pitchTrack: number[]): {
    difficultyMultiplier: number;
    enduranceEffect: number;
    articulationChallenge: number;
  } {
    // Analyze saxophone-specific challenges at different tempos
    let difficultyMultiplier = 1.0;
    let enduranceEffect = 1.0;
    let articulationChallenge = 1.0;
    
    // Use saxophone-specific tempo analysis
    const saxRange = getSaxophoneRange(this.saxophoneType);
    const validPitches = pitchTrack.filter((p: number) => p > 0);
    const avgPitch = validPitches.length > 0 ? 
      validPitches.reduce((sum: number, p: number) => sum + p, 0) / validPitches.length : 440;
    const registerDifficulty = getRegisterDifficulty(avgPitch, this.saxophoneType);
    
    // Calculate range-specific adjustments
    const rangeAdjustments = this.calculateRangeAdjustments(avgPitch, saxRange);
    
    if (tempo < 80) {
      // Very slow tempo challenges: maintaining steady air support, avoiding dragging
      difficultyMultiplier = 0.95 * registerDifficulty * rangeAdjustments.slowTempo;
      enduranceEffect = 0.98 * rangeAdjustments.endurance;
      articulationChallenge = 0.92 * rangeAdjustments.articulation;
    } else if (tempo < 120) {
      // Moderate-slow tempo: good balance, generally easier
      difficultyMultiplier = 1.05 * registerDifficulty * rangeAdjustments.slowTempo;
      enduranceEffect = 1.02 * rangeAdjustments.endurance;
      articulationChallenge = 1.0 * rangeAdjustments.articulation;
    } else if (tempo < 160) {
      // Moderate-fast tempo: requires good technique but manageable
      difficultyMultiplier = 1.0 * registerDifficulty * rangeAdjustments.moderateTempo;
      enduranceEffect = 0.98 * rangeAdjustments.endurance;
      articulationChallenge = 0.95 * rangeAdjustments.articulation;
    } else if (tempo < 200) {
      // Fast tempo: technical challenges increase, especially in altissimo
      difficultyMultiplier = 0.9 * registerDifficulty * rangeAdjustments.fastTempo;
      enduranceEffect = 0.92 * rangeAdjustments.endurance;
      articulationChallenge = 0.85 * rangeAdjustments.articulation;
    } else {
      // Very fast tempo: significant challenges for clarity
      difficultyMultiplier = 0.8 * registerDifficulty * rangeAdjustments.fastTempo;
      enduranceEffect = 0.85 * rangeAdjustments.endurance;
      articulationChallenge = 0.75 * rangeAdjustments.articulation;
    }
    
    // Analyze performance duration effect on endurance
    const performanceDuration = onsets.length > 1 ? 
      (onsets[onsets.length - 1] ?? 0) - (onsets[0] ?? 0) : 0;
    
    if (performanceDuration > 60) { // More than 1 minute
      enduranceEffect *= 0.95;
    }
    if (performanceDuration > 180) { // More than 3 minutes
      enduranceEffect *= 0.9;
    }
    
    return {
      difficultyMultiplier,
      enduranceEffect,
      articulationChallenge
    };
  }
  
  private applyTempoSpecificAdjustments(
    clarityScores: { slow: number; medium: number; fast: number },
    factors: { difficultyMultiplier: number; enduranceEffect: number; articulationChallenge: number }
  ): { slow: number; medium: number; fast: number } {
    return {
      slow: Math.max(0, Math.min(1, 
        clarityScores.slow * factors.difficultyMultiplier * factors.enduranceEffect
      )),
      medium: Math.max(0, Math.min(1, 
        clarityScores.medium * factors.difficultyMultiplier * factors.enduranceEffect * factors.articulationChallenge
      )),
      fast: Math.max(0, Math.min(1, 
        clarityScores.fast * factors.difficultyMultiplier * factors.enduranceEffect * factors.articulationChallenge * 0.9
      ))
    };
  }

  private estimateTempoFromOnsets(onsets: number[]): number {
    if (onsets.length < 2) return 120; // Default tempo
    
    const firstOnset = onsets[0];
    const lastOnset = onsets[onsets.length - 1];
    
    if (firstOnset === undefined || lastOnset === undefined) return 120;
    
    const avgInterval = (lastOnset - firstOnset) / (onsets.length - 1);
    return 60 / avgInterval;
  }
  
  private calculateOnsetConsistency(onsets: number[]): number {
    if (onsets.length < 3) return 0.5;
    
    // Calculate intervals between consecutive onsets
    const intervals: number[] = [];
    for (let i = 1; i < onsets.length; i++) {
      const current = onsets[i];
      const previous = onsets[i - 1];
      if (current !== undefined && previous !== undefined) {
        intervals.push(current - previous);
      }
    }
    
    if (intervals.length === 0) return 0.5;
    
    // Calculate coefficient of variation (std dev / mean) for consistency
    const mean = intervals.reduce((sum, val) => sum + val, 0) / intervals.length;
    const variance = intervals.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / intervals.length;
    const stdDev = Math.sqrt(variance);
    
    const coefficientOfVariation = mean > 0 ? stdDev / mean : 1;
    
    // Convert to consistency score (lower variation = higher consistency)
    return Math.max(0, Math.min(1, 1 - coefficientOfVariation));
  }
  
  private calculateArticulationPrecision(onsets: number[], tempo: number): number {
    if (onsets.length < 2) return 0.5;
    
    // Calculate expected beat interval from tempo
    const expectedInterval = 60 / tempo;
    
    // Analyze how closely onsets align with expected beat grid
    let precisionScore = 0;
    let validOnsets = 0;
    
    for (let i = 1; i < onsets.length; i++) {
      const current = onsets[i];
      const previous = onsets[i - 1];
      
      if (current !== undefined && previous !== undefined) {
        const actualInterval = current - previous;
        
        // Find the closest multiple of expected interval
        const beatMultiple = Math.round(actualInterval / expectedInterval);
        const expectedTime = beatMultiple * expectedInterval;
        
        // Calculate timing precision (closer to expected = higher precision)
        const timingError = Math.abs(actualInterval - expectedTime);
        const precision = Math.max(0, 1 - (timingError / (expectedInterval * 0.5)));
        
        precisionScore += precision;
        validOnsets++;
      }
    }
    
    return validOnsets > 0 ? precisionScore / validOnsets : 0.5;
  }
  
  private calculateTempoBasedClarityScores(
    tempo: number, 
    onsetConsistency: number, 
    articulationPrecision: number
  ): { slow: number; medium: number; fast: number } {
    // Define tempo ranges for saxophone articulation
    const tempoRanges = {
      slow: { min: 0, max: 100 },
      medium: { min: 100, max: 150 },
      fast: { min: 150, max: 300 }
    };
    
    // Base clarity scores - slower tempos generally allow better clarity
    const baseScores = {
      slow: 0.85,
      medium: 0.75,
      fast: 0.65
    };
    
    // Adjust scores based on actual performance characteristics
    const consistencyWeight = 0.4;
    const precisionWeight = 0.6;
    
    // Calculate weighted performance score
    const performanceScore = (onsetConsistency * consistencyWeight) + 
                           (articulationPrecision * precisionWeight);
    
    // Apply tempo-specific adjustments
    const adjustedScores = {
      slow: baseScores.slow,
      medium: baseScores.medium,
      fast: baseScores.fast
    };
    
    // Determine which tempo range the actual tempo falls into
    const actualRange = tempo <= tempoRanges.slow.max ? "slow" :
      tempo <= tempoRanges.medium.max ? "medium" : "fast";
    
    // Apply performance-based adjustments more heavily to the current tempo range
    Object.keys(adjustedScores).forEach(range => {
      const isCurrentRange = range === actualRange;
      const adjustmentFactor = isCurrentRange ? 0.8 : 0.3;
      
      adjustedScores[range as keyof typeof adjustedScores] = 
        baseScores[range as keyof typeof baseScores] * (1 - adjustmentFactor) +
        performanceScore * adjustmentFactor;
    });
    
    return adjustedScores;
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
      
        // Tenuto: longer notes with sustained strength (saxophone-specific duration)
        if (noteDuration > SAXOPHONE_CONFIG.ARTICULATION.STACCATO_MAX_DURATION * 2 && noteStrength > 0.3) {
          tenutoScore += 1;
        }
      }
    }
    return onsets.length > 1 ? tenutoScore / (onsets.length - 1) : 0.5;
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
      
        // Check for sustained loudness between notes using saxophone-specific threshold
        let sustainedFrames = 0;
        for (let frame = startFrame; frame < endFrame && frame < loudness.length; frame++) {
          if ((loudness[frame] || 0) > SAXOPHONE_CONFIG.ARTICULATION.SLUR_DETECTION_THRESHOLD) sustainedFrames++;
        }
        
        const sustainRatio = sustainedFrames / (endFrame - startFrame);
        if (sustainRatio > 0.7) legatoScore += 1;
      }
    }
    return onsets.length > 1 ? legatoScore / (onsets.length - 1) : 0.5;
  }

  private detectStaccatoArticulation(onsets: number[], loudness: number[]): number {
    if (onsets.length < 2 || loudness.length === 0) {
      return 0.5;
    }
    
    // Convert loudness to dB for analysis
    const loudnessDB = loudness.map(l => 20 * Math.log10(Math.max(l, 1e-10)));
    
    let staccatoScore = 0;
    let validNotes = 0;
    
    for (let i = 0; i < onsets.length - 1; i++) {
      const currentOnset = onsets[i];
      const nextOnset = onsets[i + 1];
      
      if (currentOnset !== undefined && nextOnset !== undefined) {
        const noteDuration = nextOnset - currentOnset;
        
        // Calculate loudness characteristics for staccato detection
        const loudnessCharacteristics = this.analyzeLoudnessForStaccato(
          currentOnset, 
          nextOnset, 
          loudnessDB
        );
        
        // Staccato scoring based on multiple characteristics
        let noteStaccatoScore = 0;
        
        // 1. Duration-based scoring using saxophone-specific staccato duration
        if (noteDuration < SAXOPHONE_CONFIG.ARTICULATION.STACCATO_MAX_DURATION * 0.5) {
          noteStaccatoScore += 0.4; // Very short notes
        } else if (noteDuration < SAXOPHONE_CONFIG.ARTICULATION.STACCATO_MAX_DURATION) {
          noteStaccatoScore += 0.3; // Short notes
        } else if (noteDuration < SAXOPHONE_CONFIG.ARTICULATION.STACCATO_MAX_DURATION * 1.5) {
          noteStaccatoScore += 0.1; // Medium-short notes
        }
        
        // 2. Loudness envelope characteristics
        noteStaccatoScore += loudnessCharacteristics.attackDecayRatio * 0.3;
        noteStaccatoScore += loudnessCharacteristics.rapidDecayScore * 0.2;
        noteStaccatoScore += loudnessCharacteristics.separationClarity * 0.1;
        
        staccatoScore += Math.min(1, noteStaccatoScore);
        validNotes++;
      }
    }
    
    return validNotes > 0 ? staccatoScore / validNotes : 0.5;
  }

  private analyzeLoudnessForStaccato(
    startTime: number, 
    endTime: number, 
    loudnessDB: number[]
  ): {
    attackDecayRatio: number;
    rapidDecayScore: number;
    separationClarity: number;
  } {
    // Estimate frame indices from time (assuming ~100 frames per second)
    const frameRate = loudnessDB.length / (loudnessDB.length * 0.01); // Approximate frame rate
    const startFrame = Math.floor(startTime * frameRate);
    const endFrame = Math.floor(endTime * frameRate);
    
    // Extract loudness segment for this note
    const noteSegment = loudnessDB.slice(
      Math.max(0, startFrame), 
      Math.min(loudnessDB.length, endFrame)
    );
    
    if (noteSegment.length < 3) {
      return {
        attackDecayRatio: 0.5,
        rapidDecayScore: 0.5,
        separationClarity: 0.5
      };
    }
    
    // 1. Analyze attack/decay ratio - staccato has sharp attack and rapid decay
    const attackDecayRatio = this.calculateAttackDecayRatio(noteSegment);
    
    // 2. Analyze rapid decay characteristic - staccato should decay quickly
    const rapidDecayScore = this.calculateRapidDecayScore(noteSegment);
    
    // 3. Analyze separation clarity - staccato notes should have clear gaps
    const separationClarity = this.calculateSeparationClarity(
      startFrame, 
      endFrame, 
      loudnessDB
    );
    
    return {
      attackDecayRatio,
      rapidDecayScore,
      separationClarity
    };
  }

  private calculateAttackDecayRatio(noteSegment: number[]): number {
    if (noteSegment.length < 4) return 0.5;
    
    // Find peak and analyze attack/decay phases
    const maxLoudness = Math.max(...noteSegment);
    const peakIndex = noteSegment.indexOf(maxLoudness);
    
    // Calculate attack slope (beginning to peak)
    const attackPhase = noteSegment.slice(0, peakIndex + 1);
    const attackSlope = attackPhase.length > 1 && attackPhase[0] !== undefined ? 
      (maxLoudness - attackPhase[0]) / attackPhase.length : 0;
    
    // Calculate decay slope (peak to end)
    const decayPhase = noteSegment.slice(peakIndex);
    const lastDecayValue = decayPhase[decayPhase.length - 1];
    const decaySlope = decayPhase.length > 1 && lastDecayValue !== undefined ? 
      Math.abs(lastDecayValue - maxLoudness) / decayPhase.length : 0;
    
    // Staccato should have relatively fast attack and even faster decay
    const attackDecayRatio = attackSlope > 0 ? decaySlope / attackSlope : 0;
    
    // Normalize to 0-1 range (higher values indicate more staccato-like)
    return Math.min(1, attackDecayRatio / 2);
  }
  
  private calculateRapidDecayScore(noteSegment: number[]): number {
    if (noteSegment.length < 3) return 0.5;
    
    const maxLoudness = Math.max(...noteSegment);
    const peakIndex = noteSegment.indexOf(maxLoudness);
    
    // Focus on the decay portion after peak
    const decayPhase = noteSegment.slice(peakIndex);
    
    if (decayPhase.length < 2) return 0.5;
    
    // Calculate how quickly the loudness drops after peak
    const decayRate = decayPhase.map((loudness, i) => {
      if (i === 0) return 0;
      const prevLoudness = decayPhase[i - 1];
      return prevLoudness !== undefined ? prevLoudness - loudness : 0; // Positive values indicate decay
    });
    
    const avgDecayRate = decayRate.reduce((sum, rate) => sum + rate, 0) / decayRate.length;
    
    // Higher decay rate indicates more staccato-like behavior
    // Normalize to 0-1 range (3dB per frame is considered rapid decay)
    return Math.min(1, Math.max(0, avgDecayRate / 3));
  }
  
  private calculateSeparationClarity(
    startFrame: number, 
    endFrame: number, 
    loudnessDB: number[]
  ): number {
    // Analyze loudness before and after the note to detect clear separation
    const beforeSegmentStart = Math.max(0, startFrame - 5);
    const beforeSegment = loudnessDB.slice(beforeSegmentStart, startFrame);
    
    const afterSegmentEnd = Math.min(loudnessDB.length, endFrame + 5);
    const afterSegment = loudnessDB.slice(endFrame, afterSegmentEnd);
    
    if (beforeSegment.length === 0 || afterSegment.length === 0) {
      return 0.5;
    }
    
    // Calculate average loudness before and after the note
    const beforeAvg = beforeSegment.reduce((sum, val) => sum + val, 0) / beforeSegment.length;
    const afterAvg = afterSegment.reduce((sum, val) => sum + val, 0) / afterSegment.length;
    
    // Calculate loudness during the note
    const noteSegment = loudnessDB.slice(startFrame, endFrame);
    const noteAvg = noteSegment.reduce((sum, val) => sum + val, 0) / noteSegment.length;
    
    // Staccato should have clear separation (lower loudness before/after)
    const separationBefore = Math.max(0, noteAvg - beforeAvg);
    const separationAfter = Math.max(0, noteAvg - afterAvg);
    
    // Average separation in dB - normalize to 0-1 range
    const avgSeparation = (separationBefore + separationAfter) / 2;
    
    // 6dB separation is considered good staccato separation
    return Math.min(1, Math.max(0, avgSeparation / 6));
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
        
        // Use saxophone-specific pitch stability threshold
        const stabilityThreshold = avgPitch * SAXOPHONE_CONFIG.TECHNICAL_EXECUTION.FINGER_TRANSITION_TIME;
        accuracyScore += Math.max(0, 1 - (stability / stabilityThreshold));
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

  private calculatePassageCleanness(spectralFlux: number[], onsets: number[]): number {
    if (spectralFlux.length === 0 || onsets.length === 0) return 0.5;
    
    // Analyze spectral cleanness (traditional approach)
    const spectralCleanness = this.analyzeSpectralCleanness(spectralFlux);
    
    // Analyze onset-based cleanness characteristics
    const onsetCleanness = this.analyzeOnsetBasedCleanness(onsets, spectralFlux);
    
    // Analyze passage flow and continuity using onsets
    const passageFlow = this.analyzePassageFlow(onsets);
    
    // Analyze timing precision for passage cleanness
    const timingPrecision = this.analyzeTimingPrecision(onsets);
    
    // Combine multiple cleanness factors with appropriate weights
    const combinedCleanness = this.combineCleannessFactor(
      spectralCleanness,
      onsetCleanness,
      passageFlow,
      timingPrecision
    );
    
    return Math.max(0, Math.min(1, combinedCleanness));
  }

  private combineCleannessFactor(
    spectralCleanness: number,
    onsetCleanness: number,
    passageFlow: number,
    timingPrecision: number
  ): number {
    // Combine different cleanness factors with appropriate weights
    const spectralWeight = 0.25;    // Spectral stability
    const onsetWeight = 0.25;       // Clean attacks
    const flowWeight = 0.30;        // Passage continuity
    const timingWeight = 0.20;      // Timing precision
    
    const combinedCleanness = (spectralCleanness * spectralWeight) +
                             (onsetCleanness * onsetWeight) +
                             (passageFlow * flowWeight) +
                             (timingPrecision * timingWeight);
    
    return combinedCleanness;
  }

  private analyzeTimingPrecision(onsets: number[]): number {
    if (onsets.length < 2) return 0.5;
    
    // Analyze the precision of onset timing for passage cleanness
    const intervals = this.calculateOnsetIntervals(onsets);
    
    if (intervals.length === 0) return 0.5;
    
    // Estimate the intended beat interval
    const medianInterval = AudioUtils.calculateMedian(intervals);
    
    // Calculate how precisely onsets align with the intended rhythm
    let precisionScore = 0;
    
    for (const interval of intervals) {
      // Calculate deviation from intended interval
      const deviation = Math.abs(interval - medianInterval);
      const precision = Math.max(0, 1 - (deviation / (medianInterval * 0.3))); // 30% tolerance
      precisionScore += precision;
    }
    
    return precisionScore / intervals.length;
  }

  private calculateOnsetIntervals(onsets: number[]): number[] {
    const intervals: number[] = [];
    for (let i = 1; i < onsets.length; i++) {
      const current = onsets[i];
      const previous = onsets[i - 1];
      if (current !== undefined && previous !== undefined) {
        intervals.push(current - previous);
      }
    }
    return intervals;
  }

  private analyzePassageFlow(onsets: number[]): number {
    if (onsets.length < 3) return 0.5;
    
    // Analyze the flow and continuity of the passage based on onset patterns
    const intervals = this.calculateOnsetIntervals(onsets);
    
    if (intervals.length === 0) return 0.5;
    
    // Analyze interval consistency for smooth passage flow
    const intervalConsistency = this.calculateIntervalConsistency(intervals);
    
    // Analyze passage momentum - good flow has appropriate pacing
    const passageMomentum = this.analyzePasageMomentum(intervals);
    
    // Analyze for hesitations or rushes that indicate poor cleanness
    const hesitationScore = this.analyzeHesitationAndRushes(intervals);
    
    // Combine flow factors
    const passageFlow = (intervalConsistency * 0.4) + 
                       (passageMomentum * 0.3) + 
                       (hesitationScore * 0.3);
    
    return Math.max(0, Math.min(1, passageFlow));
  }

  private calculateIntervalConsistency(intervals: number[]): number {
    if (intervals.length < 2) return 0.5;
    
    // Calculate coefficient of variation for interval consistency
    const mean = intervals.reduce((sum, val) => sum + val, 0) / intervals.length;
    const variance = intervals.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / intervals.length;
    const stdDev = Math.sqrt(variance);
    
    const coefficientOfVariation = mean > 0 ? stdDev / mean : 1;
    
    // Clean passages have consistent intervals
    return Math.max(0, 1 - coefficientOfVariation);
  }
  
  private analyzePasageMomentum(intervals: number[]): number {
    if (intervals.length < 3) return 0.5;
    
    // Analyze gradual changes in interval length for natural momentum
    let momentumScore = 0;
    let validTransitions = 0;
    
    for (let i = 2; i < intervals.length; i++) {
      const current = intervals[i];
      const previous = intervals[i - 1];
      const prevPrev = intervals[i - 2];
      
      if (current !== undefined && previous !== undefined && prevPrev !== undefined) {
        // Calculate momentum (gradual vs. sudden changes)
        const change1 = Math.abs(previous - prevPrev);
        const change2 = Math.abs(current - previous);
        
        // Good momentum has gradual changes
        const momentumFactor = Math.max(0, 1 - Math.abs(change2 - change1) / 0.2);
        momentumScore += momentumFactor;
        validTransitions++;
      }
    }
    
    return validTransitions > 0 ? momentumScore / validTransitions : 0.5;
  }
  
  private analyzeHesitationAndRushes(intervals: number[]): number {
    if (intervals.length < 2) return 0.5;
    
    // Analyze for hesitations (unusually long intervals) and rushes (unusually short intervals)
    const meanInterval = intervals.reduce((sum, val) => sum + val, 0) / intervals.length;
    const stdDev = Math.sqrt(intervals.reduce((sum, val) => sum + Math.pow(val - meanInterval, 2), 0) / intervals.length);
    
    let hesitationPenalty = 0;
    let rushPenalty = 0;
    
    for (const interval of intervals) {
      if (interval > meanInterval + 2 * stdDev) {
        hesitationPenalty += 0.1; // Penalty for hesitation
      } else if (interval < meanInterval - 2 * stdDev) {
        rushPenalty += 0.1; // Penalty for rushing
      }
    }
    
    const totalPenalty = hesitationPenalty + rushPenalty;
    const hesitationScore = Math.max(0, 1 - totalPenalty);
    
    return hesitationScore;
  }

  private analyzeSpectralCleanness(spectralFlux: number[]): number {
    if (spectralFlux.length === 0) return 0.5;
    
    // Clean passages have low spectral flux (less spectral noise)
    const avgFlux = spectralFlux.reduce((sum, flux) => sum + flux, 0) / spectralFlux.length;
    
    // Analyze spectral flux variance - cleaner passages have more stable spectral content
    const fluxVariance = this.calculateVariance(spectralFlux);
    const fluxStability = Math.max(0, 1 - (fluxVariance / 0.1)); // Normalize variance
    
    // Combine average flux and stability for spectral cleanness
    const avgFluxScore = Math.max(0, 1 - (avgFlux / 0.5));
    const spectralCleanness = (avgFluxScore * 0.6) + (fluxStability * 0.4);
    
    return spectralCleanness;
  }

  private calculateVariance(values: number[]): number {
    if (values.length === 0) return 0;
    
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    
    return variance;
  }
  
  private analyzeOnsetBasedCleanness(onsets: number[], spectralFlux: number[]): number {
    if (onsets.length < 2) return 0.5;
    
    // Analyze spectral flux around onset points for clean attacks
    let onsetCleanness = 0;
    let validOnsets = 0;
    
    for (let i = 0; i < onsets.length; i++) {
      const onsetTime = onsets[i];
      if (onsetTime === undefined) continue;
      
      // Convert onset time to spectral flux frame index (assuming ~100 fps)
      const frameIndex = Math.floor(onsetTime * 100);
      
      // Analyze spectral flux in the vicinity of each onset
      const onsetFluxAnalysis = this.analyzeOnsetSpectralFlux(spectralFlux, frameIndex);
      onsetCleanness += onsetFluxAnalysis;
      validOnsets++;
    }
    
    return validOnsets > 0 ? onsetCleanness / validOnsets : 0.5;
  }

  private analyzeOnsetSpectralFlux(spectralFlux: number[], centerFrame: number): number {
    // Analyze spectral flux around the onset for clean attack characteristics
    const windowSize = 3; // Small window around onset
    const startFrame = Math.max(0, centerFrame - windowSize);
    const endFrame = Math.min(spectralFlux.length, centerFrame + windowSize + 1);
    
    const onsetWindow = spectralFlux.slice(startFrame, endFrame);
    
    if (onsetWindow.length === 0) return 0.5;
    
    // Clean onsets should have controlled spectral flux
    const avgFluxInWindow = onsetWindow.reduce((sum, flux) => sum + flux, 0) / onsetWindow.length;
    const fluxCleanness = Math.max(0, 1 - (avgFluxInWindow / 0.4));
    
    return fluxCleanness;
  }

  private async analyzeBreathManagement(analysisResult: EssentiaAnalysisResult): Promise<BreathManagementAnalysis> {
    const { loudness, onsets, energy } = analysisResult;
    
    // Analyze phrase lengths using saxophone-specific breath management
    const phraseLength = this.calculateSaxophonePhraseLengths(loudness, onsets);
    
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

  private analyzeBreathingPlacement(loudness: number[], onsets: number[]): Array<{ time: number; appropriateness: number }> {
    const breathingPlacements: Array<{ time: number; appropriateness: number }> = [];
    
    // Find silence periods that indicate breathing
    for (let i = 0; i < loudness.length - 10; i++) {
      const silenceWindow = loudness.slice(i, i + 10);
      const avgLoudness = silenceWindow.reduce((sum, l) => sum + (l || 0), 0) / silenceWindow.length;
      
      if (avgLoudness < SAXOPHONE_CONFIG.BREATH_MANAGEMENT.BREATH_NOISE_THRESHOLD) { // Silence detected
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
    
    // Use saxophone-specific altissimo range
    const altissimoPitches = pitchTrack.filter(p => isAltissimoRange(p, this.saxophoneType));
    const altissimoRatio = altissimoPitches.length / pitchTrack.filter(p => p > 0).length;
    
    if (altissimoRatio > 0.1) {
      // Analyze stability of altissimo pitches
      const avgAltissimoPitch = altissimoPitches.reduce((sum, p) => sum + p, 0) / altissimoPitches.length;
      const stability = altissimoPitches.reduce((sum, p) => sum + Math.abs(p - avgAltissimoPitch), 0) / altissimoPitches.length;
      
      // Altissimo is more difficult on saxophone, so adjust scoring
      const stabilityScore = Math.max(0.3, 1 - (stability / avgAltissimoPitch));
      return stabilityScore * SAXOPHONE_CONFIG.TECHNICAL_EXECUTION.ALTISSIMO_DIFFICULTY_MULTIPLIER;
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
  
  private calculateSaxophonePhraseLengths(loudness: number[], onsets: number[]): number[] {
    // Use saxophone-specific breath detection threshold
    const breathThreshold = SAXOPHONE_CONFIG.BREATH_MANAGEMENT.BREATH_NOISE_THRESHOLD;
    const minBreathDuration = SAXOPHONE_CONFIG.BREATH_MANAGEMENT.MIN_BREATH_DURATION;
    const minPhraseDuration = 1.0; // Minimum phrase length in seconds
    
    // Method 1: Onset-based phrase detection (primary method)
    const onsetPhraseLengths = this.detectPhrasesFromOnsets(onsets, minPhraseDuration);
    
    // Method 2: Loudness-based phrase detection (secondary method)
    const loudnessPhraseLengths = this.detectPhrasesFromLoudness(loudness, breathThreshold, minBreathDuration);
    
    // Method 3: Combined approach - use onsets to refine loudness-based detection
    const combinedPhraseLengths = this.combineOnsetAndLoudnessDetection(
      onsets, 
      loudness, 
      breathThreshold, 
      minBreathDuration, 
      minPhraseDuration
    );
    
    // Choose the best method based on data quality
    let finalPhraseLengths = combinedPhraseLengths;
    
    // Fallback to onset-based if combined method fails
    if (finalPhraseLengths.length === 0 && onsetPhraseLengths.length > 0) {
      finalPhraseLengths = onsetPhraseLengths;
    }
    
    // Fallback to loudness-based if onset-based fails
    if (finalPhraseLengths.length === 0 && loudnessPhraseLengths.length > 0) {
      finalPhraseLengths = loudnessPhraseLengths;
    }
    
    // Final fallback to typical saxophone phrase lengths
    if (finalPhraseLengths.length === 0) {
      return [SAXOPHONE_CONFIG.BREATH_MANAGEMENT.TYPICAL_PHRASE_LENGTH];
    }
    
    return finalPhraseLengths;
  }
  
  private detectPhrasesFromOnsets(onsets: number[], minPhraseDuration: number): number[] {
    if (onsets.length < 2) return [];
    
    const phraseLengths: number[] = [];
    let phraseStart = onsets[0] || 0;
    
    for (let i = 1; i < onsets.length; i++) {
      const currentOnset = onsets[i];
      const prevOnset = onsets[i - 1];
      
      if (currentOnset !== undefined && prevOnset !== undefined) {
        const gap = currentOnset - prevOnset;
        
        // If gap is longer than typical breath duration, it's likely a phrase boundary
        if (gap > SAXOPHONE_CONFIG.BREATH_MANAGEMENT.MIN_BREATH_DURATION * 2) {
          const phraseLength = prevOnset - phraseStart;
          if (phraseLength > minPhraseDuration) {
            phraseLengths.push(phraseLength);
          }
          phraseStart = currentOnset;
        }
      }
    }
    
    // Add the final phrase
    const lastOnset = onsets[onsets.length - 1];
    if (lastOnset !== undefined && phraseStart !== undefined) {
      const finalPhraseLength = lastOnset - phraseStart;
      if (finalPhraseLength > minPhraseDuration) {
        phraseLengths.push(finalPhraseLength);
      }
    }
    
    return phraseLengths;
  }
  
  private detectPhrasesFromLoudness(loudness: number[], breathThreshold: number, minBreathDuration: number): number[] {
    const phraseLengths: number[] = [];
    let phraseStart = 0;
    let inSilence = false;
    let silenceStart = 0;
    
    for (let i = 1; i < loudness.length; i++) {
      const currentLoudness = loudness[i] || 0;
      const prevLoudness = loudness[i - 1] || 0;
      
      // Detect start of silence (potential breath)
      if (!inSilence && prevLoudness > breathThreshold && currentLoudness < breathThreshold) {
        silenceStart = i;
        inSilence = true;
      }
      
      // Detect end of silence (end of breath)
      if (inSilence && prevLoudness < breathThreshold && currentLoudness > breathThreshold) {
        const silenceDuration = (i - silenceStart) * 0.01; // Convert frames to seconds
        
        // If silence is long enough to be a breath, end the phrase
        if (silenceDuration >= minBreathDuration) {
          const phraseLength = (silenceStart - phraseStart) * 0.01;
          if (phraseLength > 1) { // Only count significant phrases
            phraseLengths.push(phraseLength);
          }
          phraseStart = i;
        }
        inSilence = false;
      }
    }
    
    // Add final phrase if it exists
    if (phraseStart < loudness.length - 1) {
      const finalPhraseLength = (loudness.length - 1 - phraseStart) * 0.01;
      if (finalPhraseLength > 1) {
        phraseLengths.push(finalPhraseLength);
      }
    }
    
    return phraseLengths;
  }
  
  private combineOnsetAndLoudnessDetection(
    onsets: number[], 
    loudness: number[], 
    breathThreshold: number, 
    minBreathDuration: number, 
    minPhraseDuration: number
  ): number[] {
    if (onsets.length < 2) return [];
    
    const phraseLengths: number[] = [];
    let phraseStart = onsets[0] || 0;
    
    for (let i = 1; i < onsets.length; i++) {
      const currentOnset = onsets[i];
      const prevOnset = onsets[i - 1];
      
      if (currentOnset !== undefined && prevOnset !== undefined) {
        const gap = currentOnset - prevOnset;
        
        // Check if there's a significant gap between onsets
        if (gap > SAXOPHONE_CONFIG.BREATH_MANAGEMENT.MIN_BREATH_DURATION) {
          // Verify with loudness data that this gap contains silence (breath)
          const gapStartFrame = Math.floor(prevOnset * 100); // Convert to frame index
          const gapEndFrame = Math.floor(currentOnset * 100);
          
          if (gapStartFrame < loudness.length && gapEndFrame < loudness.length) {
            const gapLoudness = loudness.slice(gapStartFrame, gapEndFrame);
            const avgGapLoudness = gapLoudness.reduce((sum, l) => sum + (l || 0), 0) / gapLoudness.length;
            
            // If the gap is quiet enough and long enough, it's likely a breath
            if (avgGapLoudness < breathThreshold && gap >= minBreathDuration) {
              const phraseLength = prevOnset - phraseStart;
              if (phraseLength > minPhraseDuration) {
                phraseLengths.push(phraseLength);
              }
              phraseStart = currentOnset;
            }
          }
        }
      }
    }
    
    // Add the final phrase
    const lastOnset = onsets[onsets.length - 1];
    if (lastOnset !== undefined && phraseStart !== undefined) {
      const finalPhraseLength = lastOnset - phraseStart;
      if (finalPhraseLength > minPhraseDuration) {
        phraseLengths.push(finalPhraseLength);
      }
    }
    
    return phraseLengths;
  }
  
  private calculateRangeAdjustments(avgPitch: number, saxRange: ReturnType<typeof getSaxophoneRange>): {
    slowTempo: number;
    moderateTempo: number;
    fastTempo: number;
    endurance: number;
    articulation: number;
  } {
    // Determine which register the average pitch falls into
    const isLowRegister = avgPitch < saxRange.COMFORTABLE_LOW;
    const isHighRegister = avgPitch > saxRange.COMFORTABLE_HIGH;
    const isAltissimo = avgPitch >= saxRange.ALTISSIMO_START;
    
    let slowTempo = 1.0;
    let moderateTempo = 1.0;
    let fastTempo = 1.0;
    let endurance = 1.0;
    let articulation = 1.0;
    
    if (isAltissimo) {
      // Altissimo range is most challenging
      slowTempo = 0.9;     // Requires more air support even at slow tempos
      moderateTempo = 0.85; // More difficult to maintain control
      fastTempo = 0.75;    // Very challenging at fast tempos
      endurance = 0.8;     // More tiring to play
      articulation = 0.8;  // Harder to articulate clearly
    } else if (isHighRegister) {
      // High register (but not altissimo)
      slowTempo = 0.95;
      moderateTempo = 0.9;
      fastTempo = 0.85;
      endurance = 0.9;
      articulation = 0.9;
    } else if (isLowRegister) {
      // Low register can be challenging for different reasons
      slowTempo = 0.95;    // Requires good air support
      moderateTempo = 1.0; // Generally easier
      fastTempo = 0.9;     // Fast low notes can be muddy
      endurance = 0.95;    // Requires more air volume
      articulation = 0.95; // Slightly harder to articulate clearly
    }
    // Middle register (comfortable range) uses default values (1.0)
    
    return {
      slowTempo,
      moderateTempo,
      fastTempo,
      endurance,
      articulation
    };
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
