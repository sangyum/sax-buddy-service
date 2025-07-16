export class Logger {
  private context: string;

  constructor(context: string) {
    this.context = context;
  }

  info(message: string, metadata?: Record<string, unknown>): void {
    console.log(JSON.stringify({
      level: "info",
      context: this.context,
      message,
      timestamp: new Date().toISOString(),
      ...metadata,
    }));
  }

  warn(message: string, metadata?: Record<string, unknown>): void {
    console.warn(JSON.stringify({
      level: "warn",
      context: this.context,
      message,
      timestamp: new Date().toISOString(),
      ...metadata,
    }));
  }

  error(message: string, metadata?: Record<string, unknown>): void {
    console.error(JSON.stringify({
      level: "error",
      context: this.context,
      message,
      timestamp: new Date().toISOString(),
      ...metadata,
    }));
  }

  debug(message: string, metadata?: Record<string, unknown>): void {
    if (process.env.NODE_ENV === "development") {
      console.debug(JSON.stringify({
        level: "debug",
        context: this.context,
        message,
        timestamp: new Date().toISOString(),
        ...metadata,
      }));
    }
  }
}