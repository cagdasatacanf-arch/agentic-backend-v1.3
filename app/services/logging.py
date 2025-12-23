"""
Structured Logging Service
Configures loguru for JSON/Structured logging, handles correlation IDs,
and intercepts standard library logs.
"""

import sys
import logging
import json
from typing import Any, Dict
from loguru import logger
from contextvars import ContextVar

# Context variables for request-level tracing
correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")

def serialize(record):
    """
    Serializer for loguru to output structured JSON logs compatible with ELK/Datadog.
    """
    subset = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
        "correlation_id": correlation_id.get(),
        "extra": record["extra"]
    }
    
    # Handle exception info if present
    if record["exception"]:
        subset["exception"] = record["exception"]
        
    return json.dumps(subset)

def wrapping_sink(message):
    """Sink that prints records, ensuring exceptions are visible."""
    record = message.record
    if record.get("exception"):
        # If there's an exception, print it clearly
        print(f"EXCEPTION: {record['message']}\n{record['exception']}")
    
    serialized = serialize(record)
    print(serialized)

class InterceptHandler(logging.Handler):
    """
    Default handler from python standard logging to loguru.
    Allows us to capture logs from third-party libraries (e.g. uvicorn, fastapi).
    """
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_logging(json_logs: bool = False, level: str = "INFO"):
    """
    Initialize loguru with proper formatting and interceptors.
    """
    # Remove default handler
    logger.remove()
    
    if json_logs:
        # Structured JSON output for production
        logger.add(wrapping_sink, level=level)
    else:
        # Human-readable output for development
        fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{extra[correlation_id]}</cyan> - <level>{message}</level>"
        logger.add(sys.stdout, format=fmt, level=level, colorize=True)

    # Reconfigure standard logging to use Loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Intercept specific library loggers
    for name in ["uvicorn", "uvicorn.access", "fastapi"]:
        _logger = logging.getLogger(name)
        _logger.handlers = [InterceptHandler()]
        _logger.propagate = False

    logger.configure(extra={"correlation_id": ""})
    logger.info(f"Logging initialized (JSON={json_logs}, Level={level})")

def get_logger():
    """Returns the loguru logger with correlation ID context."""
    return logger.bind(correlation_id=correlation_id.get())
