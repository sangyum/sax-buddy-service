import * as admin from "firebase-admin";
import { 
  AssessmentSession, 
  ProcessingStatus, 
  AssessmentSummary, 
  ExtendedAudioAnalysis 
} from "../types";
import { Logger } from "../utils/logger";

export class FirestoreService {
  private db = admin.firestore();
  private logger = new Logger("FirestoreService");

  async getAssessmentSession(
    userId: string, 
    assessmentId: string
  ): Promise<AssessmentSession | null> {
    try {
      this.logger.debug("Fetching assessment session", { 
        userId, 
        assessmentId 
      });

      const docRef = this.db
        .collection("users")
        .doc(userId)
        .collection("assessmentSessions")
        .doc(assessmentId);

      const doc = await docRef.get();

      if (!doc.exists) {
        this.logger.warn("Assessment session not found", { 
          userId, 
          assessmentId 
        });
        return null;
      }

      const data = doc.data() as AssessmentSession;
      
      this.logger.info("Assessment session retrieved", { 
        userId, 
        assessmentId,
        exerciseCount: data.exercises?.length || 0,
        status: data.overallStatus
      });

      return {
        ...data,
        id: doc.id
      };

    } catch (error) {
      this.logger.error("Failed to fetch assessment session", { 
        userId, 
        assessmentId,
        error: error instanceof Error ? error.message : "Unknown error"
      });
      throw new Error(`Failed to fetch assessment session: ${error}`);
    }
  }

  async updateAssessmentStatus(
    assessmentId: string, 
    status: ProcessingStatus
  ): Promise<void> {
    try {
      this.logger.debug("Updating assessment status", { 
        assessmentId, 
        status 
      });

      // Find the assessment across all users (if needed)
      // For now, assuming we have a direct path or can query
      const assessmentQuery = await this.db
        .collectionGroup("assessmentSessions")
        .where("id", "==", assessmentId)
        .limit(1)
        .get();

      if (assessmentQuery.empty || !assessmentQuery.docs[0]) {
        throw new Error(`Assessment not found: ${assessmentId}`);
      }

      const docRef = assessmentQuery.docs[0].ref;
      
      const updateData: any = {
        overallStatus: status
      };

      if (status === "completed" || status === "failed") {
        updateData.analyzedAt = admin.firestore.Timestamp.now();
      }

      await docRef.update(updateData);

      this.logger.info("Assessment status updated", { 
        assessmentId, 
        status 
      });

    } catch (error) {
      this.logger.error("Failed to update assessment status", { 
        assessmentId, 
        status,
        error: error instanceof Error ? error.message : "Unknown error"
      });
      throw new Error(`Failed to update assessment status: ${error}`);
    }
  }

  async updateAssessmentStatusAndSummary(
    assessmentId: string, 
    status: ProcessingStatus,
    summary: AssessmentSummary
  ): Promise<void> {
    try {
      this.logger.debug("Updating assessment status and summary", { 
        assessmentId, 
        status 
      });

      const assessmentQuery = await this.db
        .collectionGroup("assessmentSessions")
        .where("id", "==", assessmentId)
        .limit(1)
        .get();

      if (assessmentQuery.empty || !assessmentQuery.docs[0]) {
        throw new Error(`Assessment not found: ${assessmentId}`);
      }

      const docRef = assessmentQuery.docs[0].ref;
      
      const updateData: any = {
        overallStatus: status,
        summary,
        analyzedAt: admin.firestore.Timestamp.now()
      };

      await docRef.update(updateData);

      this.logger.info("Assessment status and summary updated", { 
        assessmentId, 
        status,
        overallScore: summary.overallPerformanceScore
      });

    } catch (error) {
      this.logger.error("Failed to update assessment status and summary", { 
        assessmentId, 
        status,
        error: error instanceof Error ? error.message : "Unknown error"
      });
      throw new Error(`Failed to update assessment: ${error}`);
    }
  }

  async updateExerciseStatus(
    exerciseId: string, 
    status: ProcessingStatus
  ): Promise<void> {
    try {
      this.logger.debug("Updating exercise status", { 
        exerciseId, 
        status 
      });

      // Query for the exercise across all assessment sessions
      const assessmentQuery = await this.db
        .collectionGroup("assessmentSessions")
        .where("exercises", "array-contains-any", [{ id: exerciseId }])
        .limit(1)
        .get();

      if (assessmentQuery.empty || !assessmentQuery.docs[0]) {
        throw new Error(`Exercise not found: ${exerciseId}`);
      }

      const docRef = assessmentQuery.docs[0].ref;
      const sessionData = assessmentQuery.docs[0].data() as AssessmentSession;

      // Update the specific exercise in the exercises array
      const updatedExercises = sessionData.exercises.map(exercise => {
        if (exercise.id === exerciseId) {
          return { ...exercise, analysisStatus: status };
        }
        return exercise;
      });

      await docRef.update({ exercises: updatedExercises });

      this.logger.info("Exercise status updated", { 
        exerciseId, 
        status 
      });

    } catch (error) {
      this.logger.error("Failed to update exercise status", { 
        exerciseId, 
        status,
        error: error instanceof Error ? error.message : "Unknown error"
      });
      throw new Error(`Failed to update exercise status: ${error}`);
    }
  }

  async saveExerciseAnalysis(
    exerciseId: string, 
    analysis: ExtendedAudioAnalysis
  ): Promise<void> {
    try {
      this.logger.debug("Saving exercise analysis", { 
        exerciseId,
        overallScore: analysis.performanceScore.overallScore 
      });

      // Query for the exercise across all assessment sessions
      const assessmentQuery = await this.db
        .collectionGroup("assessmentSessions")
        .where("exercises", "array-contains-any", [{ id: exerciseId }])
        .limit(1)
        .get();

      if (assessmentQuery.empty || !assessmentQuery.docs[0]) {
        throw new Error(`Exercise not found: ${exerciseId}`);
      }

      const docRef = assessmentQuery.docs[0].ref;
      const sessionData = assessmentQuery.docs[0].data() as AssessmentSession;

      // Update the specific exercise with analysis results
      const updatedExercises = sessionData.exercises.map(exercise => {
        if (exercise.id === exerciseId) {
          return { 
            ...exercise, 
            analysis,
            analysisStatus: "completed" as ProcessingStatus
          };
        }
        return exercise;
      });

      await docRef.update({ exercises: updatedExercises });

      // Also save a separate detailed analysis document for easier querying
      await this.saveDetailedAnalysis(exerciseId, analysis);

      this.logger.info("Exercise analysis saved", { 
        exerciseId,
        overallScore: analysis.performanceScore.overallScore
      });

    } catch (error) {
      this.logger.error("Failed to save exercise analysis", { 
        exerciseId,
        error: error instanceof Error ? error.message : "Unknown error"
      });
      throw new Error(`Failed to save exercise analysis: ${error}`);
    }
  }

  private async saveDetailedAnalysis(
    exerciseId: string,
    analysis: ExtendedAudioAnalysis
  ): Promise<void> {
    try {
      const analysisRef = this.db
        .collection("exerciseAnalyses")
        .doc(exerciseId);

      await analysisRef.set({
        exerciseId,
        analysis,
        createdAt: new Date(),
        version: analysis.analysisVersion
      });

      this.logger.debug("Detailed analysis saved", { exerciseId });

    } catch (error) {
      this.logger.warn("Failed to save detailed analysis", { 
        exerciseId,
        error: error instanceof Error ? error.message : "Unknown error"
      });
      // Don't throw here as this is supplementary storage
    }
  }

  async getExerciseAnalysis(exerciseId: string): Promise<ExtendedAudioAnalysis | null> {
    try {
      const docRef = this.db
        .collection("exerciseAnalyses")
        .doc(exerciseId);

      const doc = await docRef.get();

      if (!doc.exists) {
        return null;
      }

      const data = doc.data();
      return data?.analysis || null;

    } catch (error) {
      this.logger.error("Failed to fetch exercise analysis", { 
        exerciseId,
        error: error instanceof Error ? error.message : "Unknown error"
      });
      return null;
    }
  }
}