import { 
  Exercise, 
  ProcessingStatus, 
  AssessmentSummary,
  ExerciseResult,
  AnalysisResponse 
} from "../types";
import { SaxophoneAudioAnalyzer } from "./SaxophoneAudioAnalyzer";
import { FirestoreService } from "./FirestoreService";
import { StorageService } from "./StorageService";
import { Logger } from "../utils/logger";

export class AssessmentProcessor {
  private audioAnalyzer: SaxophoneAudioAnalyzer;
  private firestoreService: FirestoreService;
  private storageService: StorageService;
  private logger: Logger;
  private requestId: string;

  constructor(requestId: string) {
    this.requestId = requestId;
    this.logger = new Logger(`AssessmentProcessor-${requestId}`);
    this.audioAnalyzer = new SaxophoneAudioAnalyzer();
    this.firestoreService = new FirestoreService();
    this.storageService = new StorageService();
  }

  async processAssessment(
    userId: string, 
    assessmentId: string
  ): Promise<AnalysisResponse> {
    const startTime = Date.now();
    
    this.logger.info("Starting assessment processing", { 
      userId, 
      assessmentId 
    });

    try {
      // Get assessment session from Firestore
      const session = await this.firestoreService.getAssessmentSession(
        userId, 
        assessmentId
      );

      if (!session) {
        throw new Error(`Assessment session not found: ${assessmentId}`);
      }

      if (session.exercises.length === 0) {
        throw new Error("No exercises found in assessment session");
      }

      // Update status to processing
      await this.firestoreService.updateAssessmentStatus(
        session.id, 
        "processing"
      );

      this.logger.info("Processing exercises", { 
        exerciseCount: session.exercises.length 
      });

      // Process each exercise
      const results = await this.processExercises(session.exercises);

      // Calculate summary statistics
      const summary = this.calculateAssessmentSummary(
        results, 
        Date.now() - startTime
      );

      // Update final status
      const overallStatus = this.determineOverallStatus(results);
      await this.firestoreService.updateAssessmentStatusAndSummary(
        session.id, 
        overallStatus,
        summary
      );

      this.logger.info("Assessment processing completed", {
        overallStatus,
        processedExercises: summary.completedExercises,
        failedExercises: summary.failedExercises,
        processingTime: summary.processingTime
      });

      return {
        success: overallStatus === "completed",
        assessmentId,
        processedExercises: summary.completedExercises,
        failedExercises: summary.failedExercises,
        results,
        summary,
        processingTime: summary.processingTime,
        message: overallStatus === "completed" 
          ? "Assessment analysis completed successfully"
          : "Assessment analysis completed with some failures"
      };

    } catch (error) {
      this.logger.error("Assessment processing failed", { 
        error: error instanceof Error ? error.message : "Unknown error",
        userId,
        assessmentId
      });

      // Update status to failed
      await this.firestoreService.updateAssessmentStatus(
        assessmentId, 
        "failed"
      );

      throw error;
    }
  }

  private async processExercises(
    exercises: Exercise[]
  ): Promise<Record<string, ExerciseResult>> {
    const results: Record<string, ExerciseResult> = {};
    
    for (const exercise of exercises) {
      this.logger.info("Processing exercise", { exerciseId: exercise.id });
      
      try {
        // Update exercise status to processing
        await this.firestoreService.updateExerciseStatus(
          exercise.id, 
          "processing"
        );
        
        // Download audio file
        this.logger.debug("Downloading audio", { 
          exerciseId: exercise.id,
          recordingUrl: exercise.recordingUrl 
        });
        
        const audioBuffer = await this.storageService.downloadAudio(
          exercise.recordingUrl
        );
        
        // Perform audio analysis
        this.logger.debug("Starting audio analysis", { 
          exerciseId: exercise.id,
          bufferLength: audioBuffer.length 
        });
        
        const analysis = await this.audioAnalyzer.analyzeExercise(
          audioBuffer
        );
        
        // Save analysis results
        await this.firestoreService.saveExerciseAnalysis(
          exercise.id, 
          analysis
        );
        
        // Update exercise status to completed
        await this.firestoreService.updateExerciseStatus(
          exercise.id, 
          "completed"
        );
        
        results[exercise.id] = {
          status: "completed",
          analysis
        };

        this.logger.info("Exercise processing completed", { 
          exerciseId: exercise.id,
          overallScore: analysis.performanceScore.overallScore 
        });
        
      } catch (error) {
        const errorMessage = error instanceof Error 
          ? error.message 
          : "Unknown error";
        
        this.logger.error("Exercise processing failed", { 
          exerciseId: exercise.id,
          error: errorMessage 
        });
        
        // Update exercise status to failed
        await this.firestoreService.updateExerciseStatus(
          exercise.id, 
          "failed"
        );
        
        results[exercise.id] = {
          status: "failed",
          error: errorMessage
        };
      }
    }
    
    return results;
  }

  private calculateAssessmentSummary(
    results: Record<string, ExerciseResult>,
    processingTime: number
  ): AssessmentSummary {
    const exercises = Object.values(results);
    const completedExercises = exercises.filter(
      result => result.status === "completed"
    );
    const failedExercises = exercises.filter(
      result => result.status === "failed"
    );

    // Calculate average scores from completed exercises
    const averageScores = {
      pitchIntonation: 0,
      timingRhythm: 0,
      toneQuality: 0,
      technicalExecution: 0,
      musicalExpression: 0,
      consistency: 0,
    };

    let overallPerformanceScore = 0;

    if (completedExercises.length > 0) {
      const scores: ({ pitchIntonation: number; timingRhythm: number; toneQuality: number; technicalExecution: number; musicalExpression: number; consistency: number; } | undefined)[] = completedExercises.map(result => 
        result.analysis?.performanceScore?.categoryScores
      );

      const filteredScores = scores.filter((s): s is { pitchIntonation: number; timingRhythm: number; toneQuality: number; technicalExecution: number; musicalExpression: number; consistency: number; } => s !== undefined && s !== null);

      averageScores.pitchIntonation = this.calculateAverage(
        filteredScores.map(s => s.pitchIntonation)
      );
      averageScores.timingRhythm = this.calculateAverage(
        filteredScores.map(s => s.timingRhythm)
      );
      averageScores.toneQuality = this.calculateAverage(
        filteredScores.map(s => s.toneQuality)
      );
      averageScores.technicalExecution = this.calculateAverage(
        filteredScores.map(s => s.technicalExecution)
      );
      averageScores.musicalExpression = this.calculateAverage(
        filteredScores.map(s => s.musicalExpression)
      );
      averageScores.consistency = this.calculateAverage(
        filteredScores.map(s => s.consistency)
      );

      overallPerformanceScore = this.calculateAverage(
        completedExercises.map(result => 
          result.analysis?.performanceScore.overallScore
        ).filter((s): s is number => s !== undefined && s !== null)
      );
    }

    // Aggregate feedback and recommendations
    const allFeedback = completedExercises.flatMap(result => 
      result.analysis?.performanceScore?.specificFeedback || []
    );
    const allRecommendations = completedExercises.flatMap(result => 
      result.analysis?.performanceScore?.nextLevelRecommendations || []
    );

    return {
      overallPerformanceScore,
      totalExercises: exercises.length,
      completedExercises: completedExercises.length,
      failedExercises: failedExercises.length,
      averageScores,
      strengthAreas: this.aggregateStrengthAreas(completedExercises),
      improvementAreas: this.aggregateImprovementAreas(completedExercises),
      overallFeedback: this.deduplicateArray(allFeedback).slice(0, 10),
      recommendedNextSteps: this.deduplicateArray(allRecommendations).slice(0, 5),
      processingTime
    };
  }

  private determineOverallStatus(
    results: Record<string, ExerciseResult>
  ): ProcessingStatus {
    const statuses = Object.values(results).map(result => result.status);
    
    if (statuses.every(status => status === "completed")) {
      return "completed";
    }
    
    if (statuses.some(status => status === "completed")) {
      return "completed"; // Partial success still counts as completed
    }
    
    return "failed";
  }

  private calculateAverage(numbers: number[]): number {
    if (numbers.length === 0) return 0;
    return numbers.reduce((sum, num) => sum + num, 0) / numbers.length;
  }

  private aggregateStrengthAreas(
    completedExercises: ExerciseResult[]
  ): string[] {
    const allStrengths = completedExercises.flatMap(result => 
      result.analysis?.performanceScore.strengthAreas || []
    );
    return this.deduplicateArray(allStrengths).slice(0, 5);
  }

  private aggregateImprovementAreas(
    completedExercises: ExerciseResult[]
  ): string[] {
    const allImprovements = completedExercises.flatMap(result => 
      result.analysis?.performanceScore.improvementAreas || []
    );
    return this.deduplicateArray(allImprovements).slice(0, 5);
  }

  private deduplicateArray(array: string[]): string[] {
    return Array.from(new Set(array));
  }
}