import * as admin from "firebase-admin";

type Timestamp = admin.firestore.Timestamp;
import { ExtendedAudioAnalysis } from "./analysis";

export type ProcessingStatus = "pending" | "processing" | "completed" | "failed";

export interface Exercise {
  id: string;
  recordingUrl: string;
  analysisStatus: ProcessingStatus;
  analysis?: ExtendedAudioAnalysis;
  error?: string;
  metadata?: ExerciseMetadata;
}

export interface AssessmentSession {
  id: string;
  userId: string;
  exercises: Exercise[];
  overallStatus: ProcessingStatus;
  createdAt: Timestamp;
  analyzedAt?: Timestamp;
  summary?: AssessmentSummary;
}

export interface ExerciseMetadata {
  exerciseType: "scale" | "arpeggio" | "etude" | "improvisation" | "sight-reading" | "technical-study";
  targetKey?: string;
  targetTempo?: number;
  difficulty: "beginner" | "intermediate" | "advanced" | "professional";
  musicalStyle: "classical" | "jazz" | "contemporary" | "latin" | "blues" | "other";
  hasBackingTrack: boolean;
  expectedDuration: number;
  technicalFocus: string[];
  instructions?: string;
  scoreReference?: string;
}

export interface AssessmentSummary {
  overallPerformanceScore: number;
  totalExercises: number;
  completedExercises: number;
  failedExercises: number;
  averageScores: {
    pitchIntonation: number;
    timingRhythm: number;
    toneQuality: number;
    technicalExecution: number;
    musicalExpression: number;
    consistency: number;
  };
  strengthAreas: string[];
  improvementAreas: string[];
  overallFeedback: string[];
  recommendedNextSteps: string[];
  processingTime: number;
}