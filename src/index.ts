import { onRequest } from "firebase-functions/v2/https";
import { setGlobalOptions } from "firebase-functions/v2";
import { initializeApp } from "firebase-admin/app";
import { Request, Response } from "express";
import { AnalysisRequest, AnalysisResponse, HealthCheckResponse, ErrorResponse } from "./types/api";
import { AssessmentProcessor } from "./services/AssessmentProcessor";
import { Logger } from "./utils/logger";

// Initialize Firebase Admin SDK
initializeApp();

// Set global options for all functions
setGlobalOptions({
  region: "us-central1",
  memory: "2GiB",
  timeoutSeconds: 540,
  maxInstances: 10,
});

const logger = new Logger("MainFunction");

export const analyzeAssessmentAudio = onRequest(
  { 
    cors: true,
    memory: "2GiB",
    timeoutSeconds: 540,
  },
  async (req: Request, res: Response): Promise<void> => {
    const requestId = generateRequestId();
    logger.info("Starting assessment analysis", { requestId, method: req.method });

    try {
      // Validate HTTP method
      if (req.method !== "POST") {
        const errorResponse: ErrorResponse = {
          error: "Method not allowed. Use POST.",
          code: "METHOD_NOT_ALLOWED",
          timestamp: new Date().toISOString(),
          requestId,
        };
        res.status(405).json(errorResponse);
        return;
      }

      // Validate request body
      const { userId, assessmentId }: AnalysisRequest = req.body;
      
      if (!userId || !assessmentId) {
        const errorResponse: ErrorResponse = {
          error: "Missing required fields: userId and assessmentId",
          code: "INVALID_REQUEST",
          timestamp: new Date().toISOString(),
          requestId,
        };
        res.status(400).json(errorResponse);
        return;
      }

      logger.info("Processing assessment", { 
        requestId, 
        userId, 
        assessmentId 
      });

      // Process the assessment
      const processor = new AssessmentProcessor(requestId);
      const result: AnalysisResponse = await processor.processAssessment(
        userId, 
        assessmentId
      );

      logger.info("Assessment analysis completed", { 
        requestId, 
        success: result.success,
        processedExercises: result.processedExercises,
        failedExercises: result.failedExercises
      });

      res.status(200).json(result);

    } catch (error) {
      logger.error("Assessment analysis failed", { 
        requestId, 
        error: error instanceof Error ? error.message : "Unknown error",
        stack: error instanceof Error ? error.stack : undefined
      });

      const errorResponse: ErrorResponse = {
        error: error instanceof Error ? error.message : "Internal server error",
        code: "INTERNAL_ERROR",
        timestamp: new Date().toISOString(),
        requestId,
      };

      res.status(500).json(errorResponse);
    }
  }
);

export const healthCheck = onRequest(
  { cors: true },
  async (req: Request, res: Response): Promise<void> => {
    try {
      // Basic health check - could be expanded to test dependencies
      const response: HealthCheckResponse = {
        status: "ok",
        timestamp: new Date().toISOString(),
        version: "1.0.0",
        services: {
          firestore: true, // Could test actual connection
          storage: true,   // Could test actual connection
          essentia: true,  // Could test Essentia.js initialization
        },
      };

      res.status(200).json(response);
    } catch (error) {
      const errorResponse: HealthCheckResponse = {
        status: "error",
        timestamp: new Date().toISOString(),
        version: "1.0.0",
        services: {
          firestore: false,
          storage: false,
          essentia: false,
        },
      };

      res.status(500).json(errorResponse);
    }
  }
);

function generateRequestId(): string {
  return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}