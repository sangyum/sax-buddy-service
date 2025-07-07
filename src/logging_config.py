"""Structured logging configuration using loguru."""

import os
import sys
from loguru import logger
from typing import Optional


def setup_logging(log_level: Optional[str] = None) -> None:
    """
    Configure structured logging with loguru.
    
    Args:
        log_level: Override the log level (defaults to environment variable or ERROR)
    """
    # Remove default logger
    logger.remove()
    
    # Get log level from environment or use provided level or default to ERROR
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "ERROR").upper()
    
    # Validate log level
    valid_levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in valid_levels:
        log_level = "ERROR"
    
    # Configure structured logging format
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level> | "
        "{extra}"
    )
    
    # Add console handler with structured format
    logger.add(
        sys.stderr,
        format=log_format,
        level=log_level,
        colorize=True,
        serialize=False,
        backtrace=True,
        diagnose=True,
        enqueue=False,
        catch=True
    )
    
    # Add file handler for production environments
    if os.getenv("ENVIRONMENT") == "production":
        logger.add(
            "logs/sax_buddy_{time:YYYY-MM-DD}.log",
            format=log_format,
            level=log_level,
            rotation="1 day",
            retention="30 days",
            compression="gz",
            serialize=True,  # JSON format for production logs
            backtrace=True,
            diagnose=False,  # Don't include sensitive info in production
            enqueue=True,
            catch=True
        )
    
    # Log the configuration
    logger.info(
        "Logging configured",
        log_level=log_level,
        environment=os.getenv("ENVIRONMENT", "development"),
        file_logging=os.getenv("ENVIRONMENT") == "production"
    )


def get_logger(name: str):
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logger.bind(service="sax_buddy", component=name)


# Context manager for request logging
class RequestContext:
    """Context manager for adding request-specific context to logs."""
    
    def __init__(self, request_id: str, user_id: Optional[str] = None, endpoint: Optional[str] = None):
        self.request_id = request_id
        self.user_id = user_id
        self.endpoint = endpoint
        self.context = {}
        
    def __enter__(self):
        self.context = {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "endpoint": self.endpoint
        }
        # Filter out None values
        self.context = {k: v for k, v in self.context.items() if v is not None}
        
        # Bind context to logger
        self.logger = logger.bind(**self.context)
        return self.logger
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.logger.error(
                "Request failed with exception",
                exception_type=exc_type.__name__,
                exception_message=str(exc_val)
            )


# Performance logging decorator
def log_performance(operation: str):
    """
    Decorator to log performance metrics for operations.
    
    Args:
        operation: Name of the operation being logged
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            operation_logger = logger.bind(operation=operation)
            operation_logger.debug(f"Starting {operation}")
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                operation_logger.info(
                    f"Completed {operation}",
                    duration_seconds=round(duration, 3),
                    success=True
                )
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                operation_logger.error(
                    f"Failed {operation}",
                    duration_seconds=round(duration, 3),
                    success=False,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
                
        return wrapper
    return decorator


# Database operation logging
def log_db_operation(table: str, operation: str):
    """
    Log database operations with structured context.
    
    Args:
        table: Database table/collection name
        operation: Type of operation (create, read, update, delete)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            db_logger = logger.bind(
                db_table=table,
                db_operation=operation,
                component="database"
            )
            
            db_logger.debug(f"Starting {operation} on {table}")
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                db_logger.info(
                    f"DB {operation} completed",
                    table=table,
                    duration_seconds=round(duration, 3),
                    success=True
                )
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                db_logger.error(
                    f"DB {operation} failed",
                    table=table,
                    duration_seconds=round(duration, 3),
                    success=False,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
                
        return wrapper
    return decorator