export class AudioUtils {
  static calculateMeanVector(vectors: number[][]): number[] {
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

  static calculateMedian(values: number[]): number {
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

  static calculateMean(values: number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  static calculateMeanFrequency(pitchTrack: number[]): number {
    return pitchTrack.reduce((sum, freq) => sum + freq, 0) / pitchTrack.length;
  }  

  static calculateConfidenceWeightedMedian(values: number[], confidences: number[]): number {
    if (values.length === 0 || values.length !== confidences.length) return 0;
    
    // Create weighted pairs and sort by value
    const weightedPairs = values.map((value, index) => ({
      value,
      confidence: confidences[index] || 0
    })).sort((a, b) => a.value - b.value);
    
    // Calculate cumulative confidence weights
    const totalWeight = weightedPairs.reduce((sum, pair) => sum + pair.confidence, 0);
    if (totalWeight === 0) return AudioUtils.calculateMedian(values);
    
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
    return AudioUtils.calculateMedian(values);
  }

  static calculateStatistics(values: number[]): { mean: number; std: number; min: number; max: number } {
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

  static smoothArray(array: number[], windowSize: number): number[] {
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

  static calculatePhraseLengths(onsets: number[], phraseBreaks: number[]): number[] {
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

}
