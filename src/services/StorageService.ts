import * as admin from "firebase-admin";
import { Logger } from "../utils/logger";

export class StorageService {
  private storage = admin.storage();
  private logger = new Logger("StorageService");

  async downloadAudio(recordingUrl: string): Promise<Float32Array> {
    try {
      this.logger.debug("Starting audio download", { recordingUrl });

      // Extract bucket and file path from the recording URL
      const { bucketName, filePath } = this.parseStorageUrl(recordingUrl);
      
      const bucket = this.storage.bucket(bucketName);
      const file = bucket.file(filePath);

      // Check if file exists
      const [exists] = await file.exists();
      if (!exists) {
        throw new Error(`Audio file not found: ${filePath}`);
      }

      // Get file metadata
      const [metadata] = await file.getMetadata();
      const fileSize = parseInt(metadata.size || "0");
      
      this.logger.debug("File metadata retrieved", { 
        filePath,
        fileSize,
        contentType: metadata.contentType 
      });

      // Validate file size (max 100MB)
      const maxSize = 100 * 1024 * 1024; // 100MB
      if (fileSize > maxSize) {
        throw new Error(`Audio file too large: ${fileSize} bytes (max: ${maxSize})`);
      }

      // Download file as buffer
      const [buffer] = await file.download();
      
      this.logger.debug("Audio file downloaded", { 
        filePath,
        bufferSize: buffer.length 
      });

      // Convert audio buffer to Float32Array for audio processing
      const audioBuffer = await this.convertToAudioBuffer(buffer, metadata.contentType);

      this.logger.info("Audio conversion completed", { 
        filePath,
        originalSize: buffer.length,
        audioBufferLength: audioBuffer.length 
      });

      return audioBuffer;

    } catch (error) {
      this.logger.error("Audio download failed", { 
        recordingUrl,
        error: error instanceof Error ? error.message : "Unknown error"
      });
      throw new Error(`Failed to download audio: ${error}`);
    }
  }

  private parseStorageUrl(url: string): { bucketName: string; filePath: string } {
    try {
      // Handle different URL formats:
      // gs://bucket-name/path/to/file
      // https://firebasestorage.googleapis.com/v0/b/bucket/o/path%2Fto%2Ffile
      // https://storage.googleapis.com/bucket/path/to/file

      if (url.startsWith("gs://")) {
        const urlParts = url.substring(5).split("/");
        const bucketName = urlParts[0];
        if (!bucketName) {
          throw new Error(`Invalid gs:// URL format: ${url}`);
        }
        const filePath = urlParts.slice(1).join("/");
        return { bucketName, filePath };
      }

      if (url.includes("firebasestorage.googleapis.com")) {
        const urlObj = new URL(url);
        const pathParts = urlObj.pathname.split("/");
        const bucketIndex = pathParts.indexOf("b") + 1;
        const objectIndex = pathParts.indexOf("o") + 1;
        
        if (bucketIndex > 0 && objectIndex > 0) {
          const bucketName = pathParts[bucketIndex];
          if (!bucketName) {
            throw new Error(`Invalid Firebase Storage URL format: ${url}`);
          }
          const encodedFilePath = pathParts.slice(objectIndex).join("/");
          const filePath = decodeURIComponent(encodedFilePath);
          return { bucketName, filePath };
        }
      }

      if (url.includes("storage.googleapis.com")) {
        const urlObj = new URL(url);
        const pathParts = urlObj.pathname.substring(1).split("/");
        const bucketName = pathParts[0];
        if (!bucketName) {
          throw new Error(`Invalid Google Storage URL format: ${url}`);
        }
        const filePath = pathParts.slice(1).join("/");
        return { bucketName, filePath };
      }

      throw new Error(`Unsupported storage URL format: ${url}`);

    } catch (error) {
      throw new Error(`Failed to parse storage URL: ${url}`);
    }
  }

  private async convertToAudioBuffer(
    buffer: Buffer, 
    contentType?: string
  ): Promise<Float32Array> {
    try {
      // For now, we'll implement a basic conversion
      // In a real implementation, you'd want to use a proper audio decoding library
      // like node-web-audio-api or similar to handle different audio formats
      
      this.logger.debug("Converting buffer to audio", { 
        bufferSize: buffer.length,
        contentType 
      });

      // Basic implementation: assume 16-bit PCM and convert to Float32Array
      // This is a simplified approach - in practice you'd want proper audio decoding
      
      if (contentType?.includes("wav") || contentType?.includes("audio")) {
        return this.convertPCMToFloat32(buffer);
      } else {
        // For other formats, you'd need proper audio decoding
        // For now, we'll throw an error
        throw new Error(`Unsupported audio format: ${contentType}`);
      }

    } catch (error) {
      this.logger.error("Audio conversion failed", { 
        bufferSize: buffer.length,
        contentType,
        error: error instanceof Error ? error.message : "Unknown error"
      });
      throw new Error(`Failed to convert audio: ${error}`);
    }
  }

  private convertPCMToFloat32(buffer: Buffer): Float32Array {
    // Skip WAV header (44 bytes) if present
    let dataStart = 0;
    if (buffer.length > 44 && 
        buffer.subarray(0, 4).toString() === "RIFF" &&
        buffer.subarray(8, 12).toString() === "WAVE") {
      dataStart = 44;
    }

    const dataBuffer = buffer.subarray(dataStart);
    const samples = new Int16Array(dataBuffer.buffer, dataBuffer.byteOffset, dataBuffer.length / 2);
    const floatSamples = new Float32Array(samples.length);

    // Convert 16-bit integers to floating point values between -1 and 1
    for (let i = 0; i < samples.length; i++) {
      floatSamples[i] = (samples[i] || 0) / 32768.0;
    }

    return floatSamples;
  }

  async uploadAnalysisResults(
    exerciseId: string, 
    results: any, 
    format: "json" | "csv" = "json"
  ): Promise<string> {
    try {
      this.logger.debug("Uploading analysis results", { 
        exerciseId, 
        format 
      });

      const fileName = `analysis-results/${exerciseId}.${format}`;
      const bucket = this.storage.bucket();
      const file = bucket.file(fileName);

      let content: string;
      let contentType: string;

      if (format === "json") {
        content = JSON.stringify(results, null, 2);
        contentType = "application/json";
      } else {
        content = this.convertToCSV(results);
        contentType = "text/csv";
      }

      await file.save(content, {
        metadata: {
          contentType,
          metadata: {
            exerciseId,
            uploadedAt: new Date().toISOString()
          }
        }
      });

      // Get the download URL
      const [url] = await file.getSignedUrl({
        action: "read",
        expires: Date.now() + 365 * 24 * 60 * 60 * 1000, // 1 year
      });

      this.logger.info("Analysis results uploaded", { 
        exerciseId, 
        fileName,
        downloadUrl: url 
      });

      return url;

    } catch (error) {
      this.logger.error("Failed to upload analysis results", { 
        exerciseId,
        format,
        error: error instanceof Error ? error.message : "Unknown error"
      });
      throw new Error(`Failed to upload analysis results: ${error}`);
    }
  }

  private convertToCSV(data: any): string {
    // Basic CSV conversion - in practice you'd want a more robust implementation
    if (Array.isArray(data)) {
      if (data.length === 0) return "";
      
      const headers = Object.keys(data[0]);
      const csvRows = [
        headers.join(","),
        ...data.map(row => 
          headers.map(header => 
            JSON.stringify(row[header] ?? "")
          ).join(",")
        )
      ];
      
      return csvRows.join("\n");
    } else {
      // Convert object to key-value CSV
      const entries = Object.entries(data);
      return "Key,Value\n" + entries.map(([key, value]) => 
        `"${key}","${JSON.stringify(value)}"`
      ).join("\n");
    }
  }
}