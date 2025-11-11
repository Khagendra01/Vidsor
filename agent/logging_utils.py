"""Logging utilities for the video clip agent."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class DualLogger:
    """Logger that writes to both file and console."""
    
    def __init__(self, log_file: Optional[str] = None, verbose: bool = True):
        """
        Initialize dual logger.
        
        Args:
            log_file: Path to log file (if None, only console logging)
            verbose: Whether to print to console
        """
        self.verbose = verbose
        self.log_file = log_file
        self.logger = logging.getLogger("video_clip_agent")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler (if verbose)
        if verbose:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler (if log_file provided)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # Log everything to file
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def print(self, message: str = ""):
        """Print message (for compatibility with existing print statements)."""
        if message:
            self.info(message)
        else:
            self.info("")
    
    def __call__(self, message: str):
        """Allow logger to be called directly."""
        self.info(message)


def create_log_file(query: str, output_dir: str = "logs") -> str:
    """
    Create a log file path based on query and timestamp.
    
    Args:
        query: User query (used in filename)
        output_dir: Directory for log files
        
    Returns:
        Path to log file
    """
    # Sanitize query for filename
    safe_query = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in query)
    safe_query = safe_query[:50].strip().replace(' ', '_')  # Limit length
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"agent_{safe_query}_{timestamp}.log"
    
    log_path = Path(output_dir) / log_filename
    return str(log_path)


def get_logger(
    log_file: Optional[str] = None,
    verbose: bool = True,
    query: Optional[str] = None
) -> DualLogger:
    """
    Create a DualLogger instance consistently.
    Provides a standard way to get a logger across all modules.
    
    Args:
        log_file: Optional path to log file (if None and query provided, auto-generates)
        verbose: Whether to print to console
        query: Optional query string (used to auto-generate log file if log_file is None)
        
    Returns:
        DualLogger instance
    """
    # Auto-generate log file if query provided but no log_file
    if query and not log_file:
        log_file = create_log_file(query)
    
    return DualLogger(log_file=log_file, verbose=verbose)


def get_log_helper(logger: Optional[DualLogger] = None, verbose: bool = False):
    """
    Get a consistent logging helper function.
    Returns a function that logs to logger if available, otherwise prints if verbose.
    
    Enhanced version that supports multiple log levels.
    
    Args:
        logger: Optional DualLogger instance
        verbose: Whether to print if logger is None
        
    Returns:
        Object with methods: info(), error(), warning(), debug()
    """
    class LogHelper:
        """Helper class for consistent logging across modules."""
        
        def __init__(self, logger: Optional[DualLogger], verbose: bool):
            self.logger = logger
            self.verbose = verbose
        
        def info(self, msg: str):
            """Log info message."""
            if self.logger:
                self.logger.info(msg)
            elif self.verbose:
                print(msg)
        
        def error(self, msg: str):
            """Log error message."""
            if self.logger:
                self.logger.error(msg)
            elif self.verbose:
                print(f"[ERROR] {msg}")
        
        def warning(self, msg: str):
            """Log warning message."""
            if self.logger:
                self.logger.warning(msg)
            elif self.verbose:
                print(f"[WARNING] {msg}")
        
        def debug(self, msg: str):
            """Log debug message."""
            if self.logger:
                self.logger.debug(msg)
            elif self.verbose:
                print(f"[DEBUG] {msg}")
        
        def __call__(self, msg: str):
            """Allow calling directly (defaults to info)."""
            self.info(msg)
    
    return LogHelper(logger, verbose)