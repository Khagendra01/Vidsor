"""Verbose progress logger for real-time UI updates."""

import re
import time
from typing import Optional, Callable, Dict, Any
from queue import Queue, Empty
from agent.utils.logging_utils import DualLogger


class VerboseProgressLogger(DualLogger):
    """Logger that extends DualLogger with real-time UI progress updates."""
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        verbose: bool = True,
        ui_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize verbose progress logger.
        
        Args:
            log_file: Path to log file (if None, only console logging)
            verbose: Whether to print to console
            ui_callback: Callback function to update UI (called from background thread)
        """
        super().__init__(log_file, verbose)
        self.ui_callback = ui_callback
        self.start_time = time.time()
        self.current_phase = None
        self.phase_start_time = None
        self.message_queue = Queue()
        self.last_progress_message = ""
        
    def _format_progress_message(self, message: str, level: str = "info") -> Optional[str]:
        """
        Parse and format log message for UI display.
        Returns formatted message or None if message should be filtered out.
        """
        # Remove timestamp and log level prefix if present
        message = re.sub(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - (INFO|DEBUG|WARNING|ERROR) - ', '', message)
        message = message.strip()
        
        if not message:
            return None
            
        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        
        # Parse different message patterns and format them
        formatted = self._parse_and_format(message, elapsed)
        
        return formatted
    
    def _parse_and_format(self, message: str, elapsed: float) -> str:
        """Parse message and format with icons - simplified high-level only."""
        
        # Phase markers - only show major phases
        if "ORCHESTRATOR AGENT RUNNER" in message or "CALLING ORCHESTRATOR" in message:
            self.current_phase = "initialization"
            self.phase_start_time = time.time()
            return f"⏳ Initializing... ({elapsed:.1f}s)"
        
        if "[OPERATION CLASSIFICATION]" in message:
            self.current_phase = "classification"
            return f"🔍 Analyzing query... ({elapsed:.1f}s)"
        
        if "Operation:" in message and "Confidence:" in message:
            op_match = re.search(r'Operation: (\w+)', message)
            if op_match:
                op = op_match.group(1)
                return f"✓ Operation: {op}"
        
        if "PLANNER AGENT:" in message or "[OPERATION]" in message:
            self.current_phase = "planner"
            op_match = re.search(r'\[OPERATION\] (\w+)', message)
            if op_match:
                op = op_match.group(1)
                return f"📊 Executing: {op}... ({elapsed:.1f}s)"
            return f"📊 Planning search strategy... ({elapsed:.1f}s)"
        
        if "[STEP 2] Executing search" in message or "[PARALLEL SEARCH]" in message:
            return f"🔎 Searching video content... ({elapsed:.1f}s)"
        
        if "[PARALLEL SEARCH] All searches completed" in message:
            total_match = re.search(r'Total results: (\d+)', message)
            if total_match:
                total = total_match.group(1)
                return f"✓ Found {total} matches"
            return f"✓ Search complete"
        
        if "[STEP 3] Scoring all seconds" in message or "[FILTERING]" in message:
            return f"📊 Analyzing and filtering results... ({elapsed:.1f}s)"
        
        if "[FILTERING] Filtered to" in message:
            count_match = re.search(r'Filtered to (\d+) seconds', message)
            if count_match:
                count = count_match.group(1)
                return f"✓ Filtered to {count} highlights"
        
        if "[STEP 4] Selecting best highlights" in message or "[MERGING]" in message:
            return f"🎯 Selecting best moments... ({elapsed:.1f}s)"
        
        if "Created" in message and "chunks" in message:
            count_match = re.search(r'Created (\d+)', message)
            if count_match:
                count = count_match.group(1)
                return f"✓ Created {count} clips"
        
        if "[MULTI-STEP PLANNING]" in message:
            step_match = re.search(r'executing (\d+) steps', message)
            if step_match:
                num_steps = step_match.group(1)
                return f"🔄 Multi-step plan: {num_steps} steps ({elapsed:.1f}s)"
        
        if "[STEP" in message and "]" in message and ":" in message:
            step_match = re.search(r'\[STEP (\d+)\] (.+?):', message)
            if step_match:
                step_num = step_match.group(1)
                step_desc = step_match.group(2)[:30]  # Truncate long descriptions
                return f"Step {step_num}: {step_desc}... ({elapsed:.1f}s)"
        
        if "ORCHESTRATOR RESULTS" in message or "QUERY PROCESSING COMPLETED" in message:
            self.current_phase = "completion"
            return f"✅ Complete! ({elapsed:.1f}s)"
        
        if "Chunks created:" in message:
            count_match = re.search(r'Chunks created: (\d+)', message)
            if count_match:
                count = count_match.group(1)
                return f"✓ {count} clips added to timeline"
        
        # Default: return message as-is if no pattern matches
        # Filter out very verbose technical messages
        skip_patterns = [
            "Sample keywords:",
            "Sample descriptions:",
            "Error: HTTPConnectionPool",
            "Detections: Total unique tracks",
            "Top 10 scored seconds:",
            "Top 10 selected seconds:",
            "Time ranges:",
            "Merged ranges with",
            "Range",
        ]
        
        for pattern in skip_patterns:
            if pattern in message:
                return None
        
        # Return simplified version of other messages
        if message.startswith("  ") or message.startswith("    "):
            # Indented messages - keep them
            return message
        elif len(message) > 100:
            # Very long messages - truncate
            return message[:97] + "..."
        else:
            return message
    
    def _update_ui(self, formatted_message: str):
        """Update UI with formatted message."""
        if formatted_message and self.ui_callback:
            try:
                self.ui_callback(formatted_message)
            except Exception as e:
                # Don't let UI update errors break logging
                if self.verbose:
                    print(f"[Progress Logger Error] {e}")
    
    def info(self, message: str):
        """Log info message and update UI."""
        super().info(message)
        formatted = self._format_progress_message(message, "info")
        if formatted:
            self._update_ui(formatted)
    
    def error(self, message: str):
        """Log error message and update UI."""
        super().error(message)
        formatted = self._format_progress_message(message, "error")
        if formatted:
            # Add error icon
            error_msg = f"❌ {formatted}" if not formatted.startswith("❌") else formatted
            self._update_ui(error_msg)
    
    def warning(self, message: str):
        """Log warning message and update UI."""
        super().warning(message)
        formatted = self._format_progress_message(message, "warning")
        if formatted:
            # Add warning icon
            warn_msg = f"⚠️ {formatted}" if not formatted.startswith("⚠️") else formatted
            self._update_ui(warn_msg)
    
    def debug(self, message: str):
        """Log debug message (usually not shown in UI)."""
        super().debug(message)
        # Only show debug messages for important phases
        if any(phase in message for phase in ["[OPERATION]", "[STEP", "[PHASE", "Complete", "Error"]):
            formatted = self._format_progress_message(message, "debug")
            if formatted:
                self._update_ui(formatted)

