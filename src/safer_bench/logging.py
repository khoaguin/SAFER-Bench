"""Logging configuration for SAFER-Bench using loguru."""

from pathlib import Path
from loguru import logger
import sys


def setup_logging(log_level: str = "INFO", log_dir: Path = Path("logs")):
    """
    Set up logging configuration with loguru following best practices.

    Args:
        log_level: Logging level for console output (default: INFO)
        log_dir: Directory for log files (default: logs/)
    """
    # Create log directory if it doesn't exist
    log_dir.mkdir(exist_ok=True)

    # Remove default handler
    logger.remove()

    # Add console handler with simplified, readable format
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    # Add file handler with detailed format, rotation, and retention
    logger.add(
        log_dir / "safer_bench_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="100 MB",  # Rotate when file reaches 100MB
        retention="30 days",  # Keep logs for 30 days
        compression="zip",  # Compress rotated logs
        enqueue=True,  # Thread-safe logging
        backtrace=True,  # Include traceback for errors
        diagnose=True,  # Include variable values in tracebacks
    )

    # Add JSON handler for structured logging (useful for log aggregation)
    logger.add(
        log_dir / "safer_bench_{time:YYYY-MM-DD}.json",
        format="{message}",
        level="INFO",
        rotation="100 MB",
        retention="7 days",
        compression="zip",
        serialize=True,  # Output as JSON
    )

    logger.info("Logger initialized")
    return logger


# Export logger for use in other modules
__all__ = ["logger", "setup_logging"]
