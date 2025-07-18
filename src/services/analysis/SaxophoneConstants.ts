export const SAXOPHONE_CONFIG = {
  // Saxophone frequency ranges (Hz)
  FREQUENCY_RANGES: {
    // Alto Saxophone (Eb) - most common
    ALTO: {
      LOWEST: 138.59,      // Db3
      HIGHEST: 830.61,     // Ab5
      ALTISSIMO_START: 698.46,  // F5
      ALTISSIMO_END: 1396.91,   // F6
      COMFORTABLE_LOW: 174.61,  // F3
      COMFORTABLE_HIGH: 587.33, // D5
    },
    // Tenor Saxophone (Bb)
    TENOR: {
      LOWEST: 103.83,      // Ab2
      HIGHEST: 622.25,     // Eb5
      ALTISSIMO_START: 523.25,  // C5
      ALTISSIMO_END: 1046.50,   // C6
      COMFORTABLE_LOW: 130.81,  // C3
      COMFORTABLE_HIGH: 440.00, // A4
    },
    // Soprano Saxophone (Bb)
    SOPRANO: {
      LOWEST: 207.65,      // Ab3
      HIGHEST: 1244.51,    // Eb6
      ALTISSIMO_START: 1046.50, // C6
      ALTISSIMO_END: 2093.00,   // C7
      COMFORTABLE_LOW: 261.63,  // C4
      COMFORTABLE_HIGH: 880.00, // A5
    },
    // Baritone Saxophone (Eb)
    BARITONE: {
      LOWEST: 69.30,       // Db2
      HIGHEST: 415.30,     // Ab4
      ALTISSIMO_START: 349.23,  // F4
      ALTISSIMO_END: 698.46,    // F5
      COMFORTABLE_LOW: 87.31,   // F2
      COMFORTABLE_HIGH: 293.66, // D4
    }
  },

  // Default to Alto Saxophone (most common)
  DEFAULT_RANGE: {
    LOWEST: 138.59,
    HIGHEST: 830.61,
    ALTISSIMO_START: 698.46,
    ALTISSIMO_END: 1396.91,
    COMFORTABLE_LOW: 174.61,
    COMFORTABLE_HIGH: 587.33,
  },

  // Pitch analysis parameters
  PITCH_ANALYSIS: {
    SMOOTHING_FACTOR: 0.4,        // Slightly more aggressive smoothing for saxophone
    MEDIAN_FILTER_SIZE: 7,        // Larger filter for saxophone vibrato
    CONFIDENCE_THRESHOLD: 0.6,    // Higher threshold for saxophone clarity
    OCTAVE_ERROR_THRESHOLD: 0.15, // More tolerance for saxophone register breaks
    PITCH_STABILITY_THRESHOLD: 0.08, // Cents deviation for stable pitch
    INTONATION_TOLERANCE: 25,     // Cents tolerance for good intonation
    VIBRATO_PITCH_DEVIATION: 50,  // Cents deviation that indicates vibrato
  },

  // Vibrato characteristics for saxophone
  VIBRATO: {
    MIN_RATE: 4.0,               // Hz - slower than violin
    MAX_RATE: 8.0,               // Hz - faster vibrato is rare on saxophone
    OPTIMAL_RATE: 6.0,           // Hz - ideal saxophone vibrato rate
    MIN_DEPTH: 15,               // Cents - minimum for noticeable vibrato
    MAX_DEPTH: 80,               // Cents - maximum before it sounds excessive
    OPTIMAL_DEPTH: 30,           // Cents - ideal saxophone vibrato depth
    QUALITY_THRESHOLD: 0.4,      // Lower threshold for saxophone vibrato
    CONSISTENCY_THRESHOLD: 0.7,   // Consistency metric for good vibrato
  },

  // Articulation parameters
  ARTICULATION: {
    STACCATO_MAX_DURATION: 0.3,   // seconds
    LEGATO_MIN_OVERLAP: 0.05,     // seconds
    TONGUE_ATTACK_TIME: 0.02,     // seconds - typical saxophone attack
    RELEASE_TIME: 0.05,           // seconds - typical saxophone release
    SLUR_DETECTION_THRESHOLD: 0.15, // amplitude threshold for slur detection
    ACCENT_THRESHOLD: 1.5,        // amplitude multiplier for accent detection
  },

  // Dynamic range (dB)
  DYNAMICS: {
    PIANISSIMO: -40,
    PIANO: -30,
    MEZZO_PIANO: -20,
    MEZZO_FORTE: -10,
    FORTE: 0,
    FORTISSIMO: 10,
    RANGE: 50,                    // Total dynamic range capability
    COMFORTABLE_RANGE: 35,        // Comfortable playing range
  },

  // Breath management
  BREATH_MANAGEMENT: {
    TYPICAL_PHRASE_LENGTH: 8.0,   // seconds
    MAX_PHRASE_LENGTH: 20.0,      // seconds
    MIN_BREATH_DURATION: 0.5,     // seconds
    BREATH_NOISE_THRESHOLD: 0.02, // amplitude threshold for breath detection
    CIRCULAR_BREATHING_THRESHOLD: 25.0, // seconds - indicates circular breathing
  },

  // Tone quality parameters
  TONE_QUALITY: {
    FUNDAMENTAL_PROMINENCE: 0.7,   // Fundamental should be prominent
    HARMONIC_ROLLOFF: 0.5,        // Expected harmonic rolloff rate
    BRIGHTNESS_THRESHOLD: 0.6,     // Spectral centroid threshold for brightness
    ROUGHNESS_THRESHOLD: 0.3,      // Spectral irregularity threshold
    WARMTH_FREQUENCY: 1000,        // Hz - frequency below which defines warmth
  },

  // Technical execution
  TECHNICAL_EXECUTION: {
    FINGER_TRANSITION_TIME: 0.05,  // seconds - clean finger transitions
    REGISTER_BREAK_TOLERANCE: 0.2, // seconds - time allowed for register breaks
    ALTISSIMO_DIFFICULTY_MULTIPLIER: 1.5, // Difficulty scaling for altissimo
    FAST_PASSAGE_THRESHOLD: 8.0,   // notes per second
    TIMING_PRECISION_THRESHOLD: 0.02, // seconds - acceptable timing deviation
  },

  // Audio validation thresholds
  VALIDATION: {
    MIN_DURATION: 0.5,            // seconds
    MAX_DURATION: 300.0,          // seconds
    MIN_RMS: 0.002,               // Minimum RMS level for saxophone
    MAX_AMPLITUDE: 0.95,          // Prevent clipping
    MIN_SNR_DB: 15,               // dB - lower than speech due to breath noise
    SILENCE_THRESHOLD: 0.005,     // RMS level considered silence
    PEAK_TO_AVERAGE_RATIO: 6.0,   // dB - typical for saxophone
  },

  // Sample rate preferences
  SAMPLE_RATE: {
    PREFERRED: 44100,             // Hz - standard for audio
    MINIMUM: 22050,               // Hz - minimum for saxophone analysis
    MAXIMUM: 96000,               // Hz - maximum practical rate
  },

  // Analysis window parameters
  ANALYSIS_WINDOWS: {
    PITCH_FRAME_SIZE: 4096,       // samples - larger for better pitch resolution
    PITCH_HOP_SIZE: 1024,         // samples - 25% overlap
    SPECTRAL_FRAME_SIZE: 2048,    // samples - good for spectral analysis
    SPECTRAL_HOP_SIZE: 512,       // samples - 75% overlap
    ONSET_FRAME_SIZE: 1024,       // samples - smaller for better onset detection
    ONSET_HOP_SIZE: 256,          // samples - high overlap for onset detection
  },

  // Onset detection
  ONSET_DETECTION: {
    THRESHOLD: 1.2,               // Lower threshold for saxophone subtleties
    MIN_SILENCE_LENGTH: 0.08,     // seconds - minimum gap between notes
    PRE_EMPHASIS: 0.97,           // Pre-emphasis filter coefficient
    SPECTRAL_DIFF_THRESHOLD: 0.3, // Spectral difference threshold
  },

  // Musical expression
  MUSICAL_EXPRESSION: {
    PHRASING_SEGMENT_LENGTH: 2.0, // seconds - typical phrase segment
    RUBATO_THRESHOLD: 0.1,        // Tempo deviation threshold
    CRESCENDO_THRESHOLD: 3.0,     // dB change for crescendo detection
    DIMINUENDO_THRESHOLD: -3.0,   // dB change for diminuendo detection
    ACCENT_PROMINENCE: 1.8,       // Amplitude multiplier for accent
  },

  // Performance consistency
  CONSISTENCY: {
    INTONATION_CONSISTENCY_WINDOW: 5.0, // seconds - window for consistency analysis
    TIMING_CONSISTENCY_WINDOW: 4.0,     // seconds - window for timing analysis
    TONE_CONSISTENCY_WINDOW: 3.0,       // seconds - window for tone analysis
    ERROR_RECOVERY_TIME: 1.0,           // seconds - time to recover from error
    FATIGUE_ANALYSIS_WINDOW: 30.0,      // seconds - window for fatigue detection
  }
};

// Helper function to get saxophone type from frequency range
export function detectSaxophoneType(pitchTrack: number[]): keyof typeof SAXOPHONE_CONFIG.FREQUENCY_RANGES {
  const validPitches = pitchTrack.filter(p => p > 0);
  if (validPitches.length === 0) return "ALTO";

  const avgPitch = validPitches.reduce((sum, p) => sum + p, 0) / validPitches.length;
  const minPitch = Math.min(...validPitches);
  const maxPitch = Math.max(...validPitches);

  // Determine saxophone type based on range, using maxPitch for better accuracy
  // Soprano: highest range (C4-F#6, with altissimo up to F7)
  if (maxPitch > 1000 && avgPitch > 500) return "SOPRANO";
  
  // Baritone: lowest range (Bb1-Eb5, with altissimo up to Bb5)
  if (maxPitch < 400 && avgPitch < 200) return "BARITONE";
  
  // Tenor: mid-low range (Bb2-F#5, with altissimo up to Bb6)
  if (minPitch < 120 && maxPitch < 800) return "TENOR";
  
  // Alto: standard range (Db3-Ab5, with altissimo up to F6) - most common
  return "ALTO";
}

// Helper function to get appropriate frequency range
export function getSaxophoneRange(saxType?: keyof typeof SAXOPHONE_CONFIG.FREQUENCY_RANGES) {
  return saxType ? SAXOPHONE_CONFIG.FREQUENCY_RANGES[saxType] : SAXOPHONE_CONFIG.DEFAULT_RANGE;
}

// Helper function to validate if a frequency is within saxophone range
export function isValidSaxophoneFrequency(frequency: number, saxType?: keyof typeof SAXOPHONE_CONFIG.FREQUENCY_RANGES): boolean {
  const range = getSaxophoneRange(saxType);
  return frequency >= range.LOWEST * 0.9 && frequency <= range.ALTISSIMO_END * 1.1;
}

// Helper function to determine if frequency is in altissimo range
export function isAltissimoRange(frequency: number, saxType?: keyof typeof SAXOPHONE_CONFIG.FREQUENCY_RANGES): boolean {
  const range = getSaxophoneRange(saxType);
  return frequency >= range.ALTISSIMO_START && frequency <= range.ALTISSIMO_END;
}

// Helper function to get difficulty multiplier based on register
export function getRegisterDifficulty(frequency: number, saxType?: keyof typeof SAXOPHONE_CONFIG.FREQUENCY_RANGES): number {
  const range = getSaxophoneRange(saxType);
  
  if (frequency >= range.ALTISSIMO_START) {
    return SAXOPHONE_CONFIG.TECHNICAL_EXECUTION.ALTISSIMO_DIFFICULTY_MULTIPLIER;
  }
  
  if (frequency < range.COMFORTABLE_LOW || frequency > range.COMFORTABLE_HIGH) {
    return 1.2; // Slightly more difficult outside comfortable range
  }
  
  return 1.0; // Normal difficulty
}