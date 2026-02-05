"""
Structured logging configuration for production observability.
Demonstrates enterprise logging best practices.
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Optional
from pathlib import Path



class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Outputs logs in JSON format for easy parsing by log aggregation systems
    (CloudWatch, Datadog, Elasticsearch, etc.).
    
    Demonstrates production logging standards.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = record.correlation_id
            
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
            
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        
        return json.dumps(log_data)


class SensitiveDataFilter(logging.Filter):
    """
    Filter to mask sensitive data in logs.
    
    Prevents API keys, passwords, and PII from being logged.
    Demonstrates security best practices.
    """
    
    SENSITIVE_PATTERNS = [
        "api_key",
        "password",
        "secret",
        "token",
        "authorization",
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Mask sensitive data in log messages.
        
        Args:
            record: Log record to filter
            
        Returns:
            Always True (don't drop records, just mask data)
        """
        message = record.getMessage().lower()
        
        # Check if message contains sensitive keywords
        for pattern in self.SENSITIVE_PATTERNS:
            if pattern in message:
                # Mask the sensitive data
                record.msg = record.msg.replace(
                    pattern,
                    f"{pattern}=***MASKED***"
                )
        
        return True


def setup_logging(
    log_level: str = "INFO",
    use_json: bool = False,
    log_file: Optional[Path] = None

) -> None:
    """
    Configure application logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_json: Whether to use JSON formatting
        log_file: Optional file path for log output
        
    Demonstrates:
    - Structured logging configuration
    - Multiple handlers (console + file)
    - Custom formatters
    - Security filters
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Choose formatter
    if use_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(SensitiveDataFilter())
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(SensitiveDataFilter())
        root_logger.addHandler(file_handler)
    
    # Set specific log levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    

class CorrelationIdLogger(logging.LoggerAdapter):
    """
    Logger adapter that adds correlation ID to all log messages.
    
    Enables request tracing across distributed systems.
    Demonstrates observability best practices.
    """
    
    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """
        Add correlation ID to log record.
        
        Args:
            msg: Log message
            kwargs: Additional kwargs
            
        Returns:
            Processed message and kwargs
        """
        # Add correlation_id to extra fields
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        
        kwargs["extra"]["correlation_id"] = self.extra.get("correlation_id", "N/A")
        
        return msg, kwargs


# Initialize logging with default configuration
setup_logging()
