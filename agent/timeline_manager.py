"""Timeline management utilities for loading, saving, and manipulating timeline.json."""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class TimelineManager:
    """Manages timeline.json loading, saving, and basic operations."""
    
    def __init__(self, timeline_path: str, verbose: bool = False):
        """
        Initialize timeline manager.
        
        Args:
            timeline_path: Path to timeline.json file
            verbose: Whether to print verbose output
        """
        self.timeline_path = timeline_path
        self.verbose = verbose
        self.timeline_data: Optional[Dict] = None
        self.chunks: List[Dict] = []
        
    def load(self) -> Dict:
        """
        Load timeline.json from disk.
        
        Returns:
            Timeline data dictionary
            
        Raises:
            FileNotFoundError: If timeline.json doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """
        if not os.path.exists(self.timeline_path):
            if self.verbose:
                print(f"[TIMELINE] Creating new timeline at {self.timeline_path}")
            # Create empty timeline
            self.timeline_data = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "chunks": []
            }
            self.chunks = []
            return self.timeline_data
        
        if self.verbose:
            print(f"[TIMELINE] Loading timeline from {self.timeline_path}")
        
        # Check if file is empty or whitespace only
        try:
            with open(self.timeline_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    # File is empty, create empty timeline
                    if self.verbose:
                        print(f"[TIMELINE] File is empty, creating new timeline")
                    self.timeline_data = {
                        "version": "1.0",
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                        "chunks": []
                    }
                    self.chunks = []
                    return self.timeline_data
            
            # Parse JSON
            with open(self.timeline_path, 'r', encoding='utf-8') as f:
                self.timeline_data = json.load(f)
            
            # Validate structure
            if not isinstance(self.timeline_data, dict):
                if self.verbose:
                    print(f"[TIMELINE] Invalid structure, creating new timeline")
                self.timeline_data = {
                    "version": "1.0",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "chunks": []
                }
                self.chunks = []
                return self.timeline_data
            
            # Extract chunks
            self.chunks = self.timeline_data.get("chunks", [])
            
            if self.verbose:
                print(f"[TIMELINE] Loaded {len(self.chunks)} chunks")
            
            return self.timeline_data
            
        except json.JSONDecodeError as e:
            # Invalid JSON, create empty timeline
            if self.verbose:
                print(f"[TIMELINE] Invalid JSON: {e}, creating new timeline")
            self.timeline_data = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "chunks": []
            }
            self.chunks = []
            return self.timeline_data
        except Exception as e:
            # Other errors, re-raise
            if self.verbose:
                print(f"[TIMELINE] Error loading timeline: {e}")
            raise
    
    def save(self) -> bool:
        """
        Save timeline.json to disk.
        
        Returns:
            True if successful, False otherwise
        """
        if self.timeline_data is None:
            if self.verbose:
                print("[TIMELINE] Warning: No timeline data to save")
            return False
        
        # Update metadata
        self.timeline_data["updated_at"] = datetime.now().isoformat()
        self.timeline_data["chunks"] = self.chunks
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.timeline_path) if os.path.dirname(self.timeline_path) else ".", exist_ok=True)
        
        if self.verbose:
            print(f"[TIMELINE] Saving timeline to {self.timeline_path}")
        
        try:
            with open(self.timeline_path, 'w', encoding='utf-8') as f:
                json.dump(self.timeline_data, f, indent=2, ensure_ascii=False)
            
            if self.verbose:
                print(f"[TIMELINE] Saved {len(self.chunks)} chunks")
            
            return True
        except Exception as e:
            if self.verbose:
                print(f"[TIMELINE] Error saving timeline: {e}")
            return False
    
    def get_chunk(self, index: int) -> Optional[Dict]:
        """
        Get chunk by timeline index.
        
        Args:
            index: Timeline index (0-based)
            
        Returns:
            Chunk dictionary or None if index is invalid
        """
        if 0 <= index < len(self.chunks):
            return self.chunks[index]
        return None
    
    def get_chunks(self, indices: List[int]) -> List[Dict]:
        """
        Get multiple chunks by timeline indices.
        
        Args:
            indices: List of timeline indices
            
        Returns:
            List of chunk dictionaries (None for invalid indices)
        """
        return [self.get_chunk(i) for i in indices if self.get_chunk(i) is not None]
    
    def validate_indices(self, indices: List[int]) -> Tuple[bool, Optional[str]]:
        """
        Validate timeline indices.
        
        Args:
            indices: List of timeline indices to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not indices:
            return False, "No indices provided"
        
        for idx in indices:
            if not isinstance(idx, int):
                return False, f"Index {idx} is not an integer"
            if idx < 0:
                return False, f"Index {idx} is negative"
            if idx >= len(self.chunks):
                return False, f"Index {idx} is out of range (timeline has {len(self.chunks)} chunks)"
        
        return True, None
    
    def calculate_timeline_duration(self) -> float:
        """
        Calculate total duration of timeline.
        
        Returns:
            Total duration in seconds
        """
        if not self.chunks:
            return 0.0
        
        # Find the maximum end_time
        max_end = max(chunk.get("end_time", 0.0) for chunk in self.chunks)
        return max_end
    
    def get_timeline_range(self, indices: List[int]) -> Optional[Tuple[float, float]]:
        """
        Get source video time range for given timeline indices.
        
        Args:
            indices: List of timeline indices
            
        Returns:
            Tuple of (min_original_start_time, max_original_end_time) or None if invalid
        """
        chunks = self.get_chunks(indices)
        if not chunks:
            return None
        
        start_times = [chunk.get("original_start_time") for chunk in chunks if chunk.get("original_start_time") is not None]
        end_times = [chunk.get("original_end_time") for chunk in chunks if chunk.get("original_end_time") is not None]
        
        if not start_times or not end_times:
            return None
        
        return (min(start_times), max(end_times))
    
    def validate_chunk(self, chunk: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate a chunk structure.
        
        Args:
            chunk: Chunk dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ["start_time", "end_time", "original_start_time", "original_end_time"]
        
        for field in required_fields:
            if field not in chunk:
                return False, f"Missing required field: {field}"
            if not isinstance(chunk[field], (int, float)):
                return False, f"Field {field} must be a number"
        
        # Validate time ranges
        if chunk["start_time"] < 0 or chunk["end_time"] < 0:
            return False, "Timeline times cannot be negative"
        
        if chunk["original_start_time"] < 0 or chunk["original_end_time"] < 0:
            return False, "Original times cannot be negative"
        
        if chunk["end_time"] <= chunk["start_time"]:
            return False, "end_time must be greater than start_time"
        
        if chunk["original_end_time"] <= chunk["original_start_time"]:
            return False, "original_end_time must be greater than original_start_time"
        
        return True, None
    
    def validate_timeline(self) -> Tuple[bool, List[str]]:
        """
        Validate entire timeline structure.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not isinstance(self.chunks, list):
            errors.append("Chunks must be a list")
            return False, errors
        
        # Check for overlaps in timeline
        sorted_chunks = sorted(self.chunks, key=lambda c: c.get("start_time", 0))
        for i in range(len(sorted_chunks) - 1):
            current = sorted_chunks[i]
            next_chunk = sorted_chunks[i + 1]
            
            if current.get("end_time", 0) > next_chunk.get("start_time", 0):
                errors.append(f"Overlap detected: chunk ending at {current.get('end_time')} overlaps with chunk starting at {next_chunk.get('start_time')}")
        
        # Validate each chunk
        for i, chunk in enumerate(self.chunks):
            is_valid, error = self.validate_chunk(chunk)
            if not is_valid:
                errors.append(f"Chunk {i}: {error}")
        
        return len(errors) == 0, errors
    
    def get_chunk_count(self) -> int:
        """Get number of chunks in timeline."""
        return len(self.chunks)
    
    def is_empty(self) -> bool:
        """Check if timeline is empty."""
        return len(self.chunks) == 0

