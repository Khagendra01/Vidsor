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


def get_log_helper(logger: Optional[DualLogger] = None, verbose: bool = False):
    """
    Get a consistent logging helper function.
    Returns a function that logs to logger if available, otherwise prints if verbose.
    
    Args:
        logger: Optional DualLogger instance
        verbose: Whether to print if logger is None
        
    Returns:
        Function that takes a message string and logs/prints it
    """
    def log_info(msg: str):
        if logger:
            logger.info(msg)
        elif verbose:
            print(msg)
    
    return log_info
