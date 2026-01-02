"""
Logger Utilities Module.

This module provides logging configuration and utilities
for consistent logging across the application.
"""

from typing import Optional
import logging
import sys


# Default format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Module-level logger cache
_loggers = {}


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level.
        format_string: Log message format.
        log_file: Optional file to log to.
    """
    # TODO: Implement logging setup
    format_string = format_string or DEFAULT_FORMAT
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        from pathlib import Path
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger.
    
    Args:
        name: Logger name (usually __name__).
        
    Returns:
        Logger instance.
    """
    # TODO: Implement logger caching
    if name not in _loggers:
        logger = logging.getLogger(name)
        _loggers[name] = logger
    return _loggers[name]


class LoggerMixin:
    """Mixin class to add logging capability to classes."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


def log_function_call(func):
    """
    Decorator to log function calls.
    
    Args:
        func: Function to decorate.
        
    Returns:
        Decorated function.
    """
    # TODO: Implement function call logging
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed: {e}")
            raise
    
    return wrapper


def create_file_logger(
    name: str,
    log_file: str,
    level: str = "DEBUG"
) -> logging.Logger:
    """
    Create a logger that writes to a specific file.
    
    Args:
        name: Logger name.
        log_file: Path to log file.
        level: Logging level.
        
    Returns:
        Logger instance.
    """
    # TODO: Implement file logger creation
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
    logger.addHandler(handler)
    
    return logger
