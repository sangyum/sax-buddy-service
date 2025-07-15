import { ProcessingStatus, AssessmentSummary } from "./assessment";
import { ExtendedAudioAnalysis } from "./analysis";

export interface AnalysisRequest {
  userId: string;
  assessmentId: string;
}

export interface AnalysisResponse {
  success: boolean;
  assessmentId: string;
  processedExercises: number;
  failedExercises: number;
  results: Record<string, ExerciseResult>;
  summary?: AssessmentSummary;
  processingTime: number;
  message?: string;
}

export interface ExerciseResult {
  status: ProcessingStatus;
  analysis?: ExtendedAudioAnalysis;
  error?: string;
}

export interface HealthCheckResponse {
  status: "ok" | "error";
  timestamp: string;
  version: string;
  services: {
    firestore: boolean;
    storage: boolean;
    essentia: boolean;
  };
}

export interface ErrorResponse {
  error: string;
  code: string;
  timestamp: string;
  requestId?: string;
}