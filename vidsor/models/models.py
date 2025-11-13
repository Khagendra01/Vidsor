"""
Data models for Vidsor video editor.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set


@dataclass
class Chunk:
    """Represents a video chunk with metadata."""
    start_time: float
    end_time: float
    chunk_type: str  # "normal", "fast_forward", "highlight"
    speed: float = 1.0  # Playback speed (1.0 = normal, 2.0 = 2x speed)
    description: str = ""
    score: float = 0.0  # Interest score
    # Metadata for agent-extracted clips
    original_start_time: Optional[float] = None  # Original timing in source video
    original_end_time: Optional[float] = None
    unified_description: Optional[str] = None  # Visual description
    audio_description: Optional[str] = None  # Audio transcription
    clip_path: Optional[str] = None  # Path to extracted clip file


@dataclass
class EditState:
    """Current editing state."""
    chunks: List[Chunk]
    selected_chunks: Set[int] = field(default_factory=set)  # Multiple selected chunks
    selected_chunk: Optional[int] = None  # Deprecated: kept for backward compatibility
    preview_time: float = 0.0
    is_playing: bool = False
    has_started_playback: bool = False  # Track if playback has started (for resume button)

